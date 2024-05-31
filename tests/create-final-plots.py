#!/usr/bin/env python3
from palimpzest.profiler import Profiler, StatsProcessor
import palimpzest as pz

from palimpzest.execution import graphicEmit, flatten_nested_tuples
from palimpzest.operators import InduceFromCandidateOp

from PIL import Image
from sklearn.metrics import precision_recall_fscore_support
from tabulate import tabulate

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import argparse
import json
import shutil
import subprocess
import time
import os
import pdb


def get_color(workload, result_dict, plan_idx):
    color = "black"
    if workload != "real-estate" and len(set(filter(None, result_dict['plan_info']['models']))) > 1:
        color = "green"
    
    elif workload != "real-estate" and len(set(filter(None, result_dict['plan_info']['models']))) > 2:
        color = "green"

    elif "codegen-with-fallback" in result_dict['plan_info']['query_strategies']:
        color = "green"

    elif any([budget is not None and budget < 1.0 for budget in result_dict['plan_info']['token_budgets']]):
        color = "green"

    elif workload == "real-estate":
        # give color to logical re-orderings on real-estate
        if result_dict['plan_info']['models'][1] == "gpt-4-vision-preview":
            color = "green"
    
    elif workload == "enron" and plan_idx == 0:
        color = "green"

    return color


def get_pareto_indices(result_dicts, col):
    pareto_indices = []
    for idx, result_dict in enumerate(result_dicts):
        col_i, quality_i = result_dict[col], result_dict["f1_score"]
        paretoFrontier = True

        # check if any other plan dominates plan i
        for j, _result_dict in enumerate(result_dicts):
            col_j, quality_j = _result_dict[col], _result_dict["f1_score"]
            if idx == j:
                continue

            # if plan i is dominated by plan j, set paretoFrontier = False and break
            if col_j <= col_i and quality_j >= quality_i:
                paretoFrontier = False
                break

        # add plan i to pareto frontier if it's not dominated
        if paretoFrontier:
            pareto_indices.append(idx)

    return pareto_indices


def plot_runtime_cost_vs_quality(results):
    # create figure
    fig_text, axs_text = plt.subplots(nrows=2, ncols=3, figsize=(15,6))
    fig_clean_mc, axs_clean_mc = plt.subplots(nrows=2, ncols=3, figsize=(15,6))

    workload_to_col = {"enron": 0, "real-estate": 1, "biofabric": 2}

    # parse results into fields
    for workload, result_tuples in results.items():
        col = workload_to_col[workload]

        for plan_idx, result_dict in result_tuples:
            runtime = result_dict["runtime"]
            cost = result_dict["cost"]
            f1_score = result_dict["f1_score"]

            text = f"{plan_idx}"

            # set label and color
            color = get_color(workload, result_dict, plan_idx)
            marker = 'D' if color == "black" else None
            # mcolor = "black" if color == "black" else "#87bc45"

            # plot runtime vs. f1_score and cost vs. f1_score
            axs_text[0][col].scatter(f1_score, runtime, alpha=0.6, color=color)
            axs_text[1][col].scatter(f1_score, cost, alpha=0.6, color=color)
            axs_clean_mc[0][col].scatter(f1_score, runtime, alpha=0.6, color=color, marker=marker)
            axs_clean_mc[1][col].scatter(f1_score, cost, alpha=0.6, color=color, marker=marker)

            # add annotations
            axs_text[0][col].annotate(text, (f1_score, runtime))
            axs_text[1][col].annotate(text, (f1_score, cost))

        # compute pareto frontiers across all optimizations
        all_result_dicts = list(map(lambda tup: tup[1], result_tuples))
        cost_pareto_lst_indices = get_pareto_indices(all_result_dicts, "cost")
        runtime_pareto_lst_indices = get_pareto_indices(all_result_dicts, "runtime")

        # plot line for pareto frontiers
        cost_pareto_qualities = [all_result_dicts[idx]['f1_score'] for idx in cost_pareto_lst_indices]
        pareto_costs = [all_result_dicts[idx]['cost'] for idx in cost_pareto_lst_indices]
        cost_pareto_curve = zip(cost_pareto_qualities, pareto_costs)
        cost_pareto_curve = sorted(cost_pareto_curve, key=lambda tup: tup[0])
        if workload == "biofabric":
            cost_pareto_curve = list(filter(lambda tup: tup[0] > 0.3, cost_pareto_curve))
        pareto_cost_xs, pareto_cost_ys = zip(*cost_pareto_curve)

        runtime_pareto_qualities = [all_result_dicts[idx]['f1_score'] for idx in runtime_pareto_lst_indices]
        pareto_runtimes = [all_result_dicts[idx]['runtime'] for idx in runtime_pareto_lst_indices]
        runtime_pareto_curve = zip(runtime_pareto_qualities, pareto_runtimes)
        runtime_pareto_curve = sorted(runtime_pareto_curve, key=lambda tup: tup[0])
        if workload == "biofabric":
            runtime_pareto_curve = list(filter(lambda tup: tup[0] > 0.3, runtime_pareto_curve))
        pareto_runtime_xs, pareto_runtime_ys = zip(*runtime_pareto_curve)

        axs_text[0][col].plot(pareto_runtime_xs, pareto_runtime_ys, color="#ef9b20", linestyle='--')
        axs_text[1][col].plot(pareto_cost_xs, pareto_cost_ys, color="#ef9b20", linestyle='--')
        axs_clean_mc[0][col].plot(pareto_runtime_xs, pareto_runtime_ys, color="#ef9b20", linestyle='--')
        axs_clean_mc[1][col].plot(pareto_cost_xs, pareto_cost_ys, color="#ef9b20", linestyle='--')

        # set x,y-lim for each workload
        left, right = -0.05, 1.05
        if workload == "real-estate":
            left = 0.7
            right = 0.85
        elif workload == "biofabric":
            left = 0.15
            right = 0.55

        axs_text[0][col].set_xlim(left, right)
        axs_text[0][col].set_ylim(ymin=0)
        axs_text[1][col].set_xlim(left, right)
        if workload != "biofabric":
            axs_text[1][col].set_ylim(ymin=0)

        axs_clean_mc[0][col].set_xlim(left, right)
        axs_clean_mc[0][col].set_ylim(ymin=0)
        axs_clean_mc[1][col].set_xlim(left, right)
        if workload != "biofabric":
            axs_clean_mc[1][col].set_ylim(ymin=0)

        # turn on grid lines
        axs_text[0][col].grid(True, alpha=0.4)
        axs_text[1][col].grid(True, alpha=0.4)
        axs_clean_mc[0][col].grid(True, alpha=0.4)
        axs_clean_mc[1][col].grid(True, alpha=0.4)

    # remove ticks from first row
    for idx in range(3):
        axs_text[0][idx].set_xticklabels([])
        axs_clean_mc[0][idx].set_xticklabels([])

    # savefigs
    workload_to_title = {
        "enron": "Legal Discovery",
        "real-estate": "Real Estate Search",
        "biofabric": "Medical Schema Matching"
    }
    for workload, title in workload_to_title.items():
        idx = 0 if workload == "enron" else (1 if workload == "real-estate" else 2)
        axs_text[0][idx].set_title(f"{title}", fontsize=12)
        axs_clean_mc[0][idx].set_title(f"{title}", fontsize=12)

    axs_text[0][0].set_ylabel("Single-Threaded\nRuntime (seconds)", fontsize=12)
    axs_text[1][0].set_ylabel("Cost (USD)", fontsize=12)
    for idx in range(3):
        axs_text[1][idx].set_xlabel("F1 Score", fontsize=12)

    axs_clean_mc[0][0].set_ylabel("Single-Threaded\nRuntime (seconds)", fontsize=12)
    axs_clean_mc[1][0].set_ylabel("Cost (USD)", fontsize=12)
    for idx in range(3):
        axs_clean_mc[1][idx].set_xlabel("F1 Score", fontsize=12)

    fig_text.savefig(f"final-eval-results/plots/all-text.png", dpi=500, bbox_inches="tight")
    fig_clean_mc.savefig(f"final-eval-results/plots/all-clean-mc.png", dpi=500, bbox_inches="tight")


def plot_reopt(results, workload):
    fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(16,8))

    # parse results into fields
    results_df = pd.DataFrame(results)
    plan_to_ord = {"Baseline": 0, "PZ": 1, "Best": 2}    
    results_df['plan_ord'] = results_df.plan.apply(lambda plan: plan_to_ord[plan])

    plots = [
        ("enron", "runtime", 0, 0),
        ("enron", "cost", 0, 1),
        ("enron", "f1_score", 0, 2),
        ("real-estate", "runtime", 1, 0),
        ("real-estate", "cost", 1, 1),
        ("real-estate", "f1_score", 1, 2),
        ("biofabric", "runtime", 2, 0),
        ("biofabric", "cost", 2, 1),
        ("biofabric", "f1_score", 2, 2),
    ]

    for workload, metric, row, col in plots:
        data_df = results_df[(results_df.workload==workload) & (results_df.metric==metric)]

        if metric == "runtime":
            data_df['value'] = data_df.value / 60.0

        policy_to_label_col = {
            "max-quality-at-fixed-cost": "Policy A",
            "max-quality-at-fixed-runtime": "Policy B",
            "min-cost-at-fixed-quality": "Policy C",
        }
        data_df['label_col'] = data_df.apply(lambda row: policy_to_label_col[row['policy']] if row['plan'] == 'PZ' else 'Baseline', axis=1)
        label_col_to_ord = {"Baseline": 0, "Policy A": 1, "Policy B": 2, "Policy C": 3}
        data_df['label_col_ord'] = data_df.label_col.apply(lambda label: label_col_to_ord[label])

        # drop duplicates for baseline, which is replicated across policies
        data_df.drop_duplicates(subset=['label_col'], inplace=True)
        data_df.sort_values(by=['plan_ord', 'label_col_ord'], inplace=True)

        # rename Baseline --> GPT-4 for plot
        data_df['label_col'] = data_df.label_col.apply(lambda label_col: "GPT-4" if label_col == "Baseline" else label_col)

        g = sns.barplot(
            data=data_df, # kind="bar",
            x="value", y="label_col", hue="plan",
            alpha=.6, # palette=, height=6,
            ax=axs[row][col],
        )
        # if col == 0 and row == 0:
        #     g.legend_.set_title(None)
        # else:
        #     g.legend_.remove()
        g.set_ylabel(None)

        # set x-labels
        if row == 2:
            xlabel = "Cost (USD)"
            if metric == "runtime":
                xlabel = "Single-Threaded Runtime (minutes)"
            elif metric == "f1_score":
                xlabel = "F1-Score"
            g.set_xlabel(xlabel, fontsize=12)
        else:
            g.set_xlabel(None)

        # remove legends and tick labels
        if col > 0:
            g.set_yticklabels([])
        if col > 0 or row > 0:
            g.legend_.remove()

        # set x-limits for F1 score
        if col == 2:
            axs[row][col].set_xlim(0, 1.05)

        # # add constraints
        # xs = [
        #     [20, 166.7, 0.8],
        #     [3, 10, 0.8],
        #     [2, 16.7, 0.4],
        # ]
        # ys = [
        #     [(0.52, 0.73), (0.27, 0.48), (0.02, 0.23)],
        #     [(0.52, 0.73), (0.27, 0.48), (0.02, 0.23)],
        #     [(0.52, 0.73), (0.27, 0.48), (0.02, 0.23)],
        # ]
        # ls = [
        #     ["-", "-", "-"],
        #     ["-", "-", "--"],
        #     ["--", "-", "-"]
        # ]
        # axs[row][col].axvline(x=xs[row][col], ymin=ys[row][col][0], ymax=ys[row][col][1], linestyle=ls[row][col], color="black")

    # axs[0][0].set_title("Cost (USD)", fontsize=10)
    # axs[0][1].set_title("Single-Threaded Runtime (minutes)", fontsize=10)
    # axs[0][2].set_title("F1 Score", fontsize=10)
    axs[0][0].set_ylabel("Legal Discovery", fontsize=12)
    axs[1][0].set_ylabel("Real Estate Search", fontsize=12)
    axs[2][0].set_ylabel("Medical Schema Matching", fontsize=12)
    axs[0][1].set_title("Palimpzest Selected Plans vs. GPT-4 Baseline", fontsize=15)

    fig.savefig(f"final-eval-results/plots/reopt.png", dpi=500, bbox_inches="tight")


if __name__ == "__main__":
    # parse arguments
    startTime = time.time()
    parser = argparse.ArgumentParser(description='Run the evaluation(s) for the paper')
    parser.add_argument('--all', default=False, action='store_true', help='')
    parser.add_argument('--reopt', default=False, action='store_true', help='')
    # parser.add_argument('--workload', type=str, help='The workload: one of ["biofabric", "enron", "real-estate"]')
    # parser.add_argument('--opt' , type=str, help='The optimization: one of ["model", "codegen", "token-reduction"]')

    args = parser.parse_args()

    # create directory for intermediate results
    os.makedirs(f"final-eval-results/plots", exist_ok=True)

    if args.all:
        # # opt and workload to # of plots
        # opt_workload_to_num_plans = {
        #     "model": {
        #         "enron": 11,
        #         "real-estate": 10,
        #         "biofabric": 14,
        #     },
        #     "codegen": {
        #         "enron": 6,
        #         "real-estate": 7,
        #         "biofabric": 6,
        #     },
        #     "token-reduction": {
        #         "enron": 12,
        #         "real-estate": 16,
        #         "biofabric": 16,
        #     },
        # }

        # # read results file(s) generated by evaluate_pz_plans
        # results = {
        #     "enron": {
        #         "model": [],
        #         "codegen": [],
        #         "token-reduction": [],
        #     },
        #     "real-estate": {
        #         "model": [],
        #         "codegen": [],
        #         "token-reduction": [],
        #     },
        #     "biofabric": {
        #         "model": [],
        #         "codegen": [],
        #         "token-reduction": [],
        #     },
        # }
        # for workload in ["enron", "real-estate", "biofabric"]:
        #     for opt in ["model", "codegen", "token-reduction"]:
        #         num_plans = opt_workload_to_num_plans[opt][workload]
        #         for plan_idx in range(num_plans):
        #             if workload == "biofabric" and opt == "codegen" and plan_idx == 3:
        #                 continue

        #             with open(f"final-eval-results/{opt}/{workload}/results-{plan_idx}.json", 'r') as f:
        #                 result = json.load(f)
        #                 results[workload][opt].append((plan_idx, result))
        # opt and workload to # of plots
        workload_to_num_plans = {
            "enron": 20,
            "real-estate": 20,
            "biofabric": 20,
        }

        # read results file(s) generated by evaluate_pz_plans
        results = {
            "enron": [],
            "real-estate": [],
            "biofabric": [],
        }
        for workload in ["enron", "real-estate", "biofabric"]:
            num_plans = workload_to_num_plans[workload]
            for plan_idx in range(num_plans):
                with open(f"final-eval-results/{workload}/results-{plan_idx}.json", 'r') as f:
                    result = json.load(f)
                    results[workload].append((plan_idx, result))

        plot_runtime_cost_vs_quality(results)

    if args.reopt:
        policy_to_plan = {
            ### max quality
            "max-quality-at-fixed-cost": {
                "enron": "final-eval-results/enron/results-1.json",
                "real-estate": "final-eval-results/reoptimization/real-estate/max-quality-at-fixed-cost.json",
                "biofabric": "final-eval-results/reoptimization/biofabric/max-quality-at-fixed-cost.json",
            },
            ### max quality
            "max-quality-at-fixed-runtime": {
                "enron": "final-eval-results/enron/results-2.json",
                # NOTE: if reopt. picked plan which we already had results for from scatter, we simply used those
                "real-estate": "final-eval-results/real-estate/results-0.json",
                "biofabric": "final-eval-results/reoptimization/biofabric/max-quality-at-fixed-runtime.json",
            },
            ### min cost
            "min-cost-at-fixed-quality": {
                "enron": "final-eval-results/enron/results-9.json",
                # NOTE: if reopt. picked plan which we already had results for from scatter, we simply used those
                "real-estate": "final-eval-results/real-estate/results-3.json",
                "biofabric": "final-eval-results/reoptimization/biofabric/min-cost-at-fixed-quality.json",
            },
        }
        policy_to_naive_plan = {  # TODO: change to GPT-4 baseline everywhere
            ### max quality
            "max-quality-at-fixed-cost": {
                "enron": 18,
                "real-estate": 8,
                "biofabric": 2,
            },
            ### max quality
            "max-quality-at-fixed-runtime": {
                "enron": 18,
                "real-estate": 8,
                "biofabric": 2,
            },
            ### min cost
            "min-cost-at-fixed-quality": {
                "enron": 18,
                "real-estate": 8,
                "biofabric": 2,
            },
        }
        results = []
        for workload in ["enron", "real-estate", "biofabric"]:
            for policy in ["max-quality-at-fixed-cost", "max-quality-at-fixed-runtime", "min-cost-at-fixed-quality"]: #, "min-runtime-at-fixed-quality"]:
            # for workload in ["enron", "real-estate", "biofabric"]:
                fp = policy_to_plan[policy][workload]
                with open(fp, 'r') as f:
                    result_dict = json.load(f)
                    # results.append({"plan": "PZ", "policy": policy, "workload": workload, "f1_score": result_dict["f1_score"], "cost": result_dict["cost"], "runtime": result_dict["runtime"]})
                    results.append({"plan": "PZ", "policy": policy, "workload": workload, "metric": "f1_score", "value": result_dict["f1_score"]})
                    results.append({"plan": "PZ", "policy": policy, "workload": workload, "metric": "cost", "value": result_dict["cost"]})
                    results.append({"plan": "PZ", "policy": policy, "workload": workload, "metric": "runtime", "value": result_dict["runtime"]})

                # best_plan_opt, best_plan_idx = policy_to_best_plan[policy][workload]
                # with open(f'final-eval-results/{best_plan_opt}/{workload}/results-{best_plan_idx}.json', 'r') as f:
                #     result_dict = json.load(f)
                #     results.append({"plan": "Best", "policy": policy, "workload": workload, "f1_score": result_dict["f1_score"], "cost": result_dict["cost"], "runtime": result_dict["runtime"]})
                
                naive_plan_idx = policy_to_naive_plan[policy][workload]
                with open(f'final-eval-results/{workload}/results-{naive_plan_idx}.json', 'r') as f:
                    result_dict = json.load(f)
                    # results.append({"plan": "Baseline", "policy": policy, "workload": workload, "f1_score": result_dict["f1_score"], "cost": result_dict["cost"], "runtime": result_dict["runtime"]})
                    results.append({"plan": "Baseline", "policy": policy, "workload": workload, "metric": "f1_score", "value": result_dict["f1_score"]})
                    results.append({"plan": "Baseline", "policy": policy, "workload": workload, "metric": "cost", "value": result_dict["cost"]})
                    results.append({"plan": "Baseline", "policy": policy, "workload": workload, "metric": "runtime", "value": result_dict["runtime"]})

        # plot_reopt(results, policy)
        plot_reopt(results, workload)
