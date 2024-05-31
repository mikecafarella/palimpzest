#!/usr/bin/env python3
import context
from palimpzest.profiler import Profiler, StatsProcessor
import palimpzest as pz

from tabulate import tabulate
from PIL import Image


from palimpzest.constants import Model
from palimpzest.execution import Execution, graphicEmit, flatten_nested_tuples
from palimpzest.elements import DataRecord, GroupBySig

import matplotlib.pyplot as plt
import pandas as pd

import argparse
import json
import shutil
import time
import os
import pandas as pd
import pdb
from sklearn.metrics import precision_recall_fscore_support

IN_DIR= "testdata/biofabric-matching/"

# def evaluate_matching():

class CaseData(pz.Schema):
    """An individual row extracted from a table containing medical study data."""
    case_submitter_id = pz.Field(desc="The ID of the case", required=True)
    age_at_diagnosis = pz.Field(desc="The age of the patient at the time of diagnosis", required=False)
    race = pz.Field(desc="An arbitrary classification of a taxonomic group that is a division of a species.", required=False)
    ethnicity = pz.Field(desc="Whether an individual describes themselves as Hispanic or Latino or not.", required=False)
    gender = pz.Field(desc="Text designations that identify gender.", required=False)
    vital_status = pz.Field(desc="The vital status of the patient", required=False)
    ajcc_pathologic_t = pz.Field(desc="The AJCC pathologic T", required=False)
    ajcc_pathologic_n = pz.Field(desc="The AJCC pathologic N", required=False)
    ajcc_pathologic_stage = pz.Field(desc="The AJCC pathologic stage", required=False)
    tumor_grade = pz.Field(desc="The tumor grade", required=False)
    tumor_focality = pz.Field(desc="The tumor focality", required=False)
    tumor_largest_dimension_diameter = pz.Field(desc="The tumor largest dimension diameter", required=False)
    primary_diagnosis = pz.Field(desc="The primary diagnosis", required=False)
    morphology = pz.Field(desc="The morphology", required=False)
    tissue_or_organ_of_origin = pz.Field(desc="The tissue or organ of origin", required=False)
    # tumor_code = pz.Field(desc="The tumor code", required=False)
    filename = pz.Field(desc="The name of the file the record was extracted from", required=False)
    study = pz.Field(desc="The last name of the author of the study, from the table name", required=False)

def buildNestedStr(node, indent=0, buildStr=""):
    elt, child = node
    indentation = " " * indent
    buildStr =  f"{indentation}{elt}" if indent == 0 else buildStr + f"\n{indentation}{elt}"
    if child is not None:
        return buildNestedStr(child, indent=indent+2, buildStr=buildStr)
    else:
        return buildStr

def compute_label(physicalTree, label_idx):
    """
    Map integer to physical plan.
    """
    physicalOps = physicalTree.dumpPhysicalTree()
    label = buildNestedStr(physicalOps)
    print(f"LABEL {label_idx}: {label}")

    flat = flatten_nested_tuples(physicalOps)
    ops = [op for op in flat if not op.is_hardcoded()]
    label = "-".join([repr(op.model) for op in ops])
    return f"PZ-{label_idx}-{label}"


def score_biofabric_plans(result_dir) -> float:
    """
    Computes the results of all biofabric plans
    """

    results = []
    for file in os.listdir(result_dir):
        if not file.endswith(".json"):
            continue

        stats_dict = json.load(open(f"{result_dir}/{file}"))
        records = stats_dict["records"]

        # parse records
        exclude_keys = ["filename", "op_id", "uuid", "parent_uuid", "stats"]
        output_rows = []
        for rec in records:
            dct = {k:v for k,v in rec.items() if k not in exclude_keys}
            filename = os.path.basename(rec["filename"])
            dct["study"] = filename.split("_")[0]
            output_rows.append(dct) 
        records_df = pd.DataFrame(output_rows)

        if records_df.empty:
            stats_dict["f1_score"] = 0
            results.append(stats_dict)
            continue

        output = records_df
        index = [x for x in output.columns if x != "study"]
        target_matching = pd.read_csv(os.path.join(RESULT_PATH, "target_matching.csv"), index_col=0).reindex(index)

        studies = output["study"].unique()
        # Group by output by the "study" column and split it into many dataframes indexed by the "study" column
        df = pd.DataFrame(columns=target_matching.columns, index = index)
        cols = output.columns
        predicted = []
        targets = []

        for study in studies:
            output_study = output[output["study"] == study]
            study = study.split(".xlsx")[0]
            try:
                input_df = pd.read_excel(os.path.join(IN_DIR, f"{study}.xlsx"))
            except:
                print("Cannot find the study", study)
                targets += [study]*5 
                predicted += ["missing"]*5
                continue
            # for every column in output_study, check which column in input_df is the closest, i.e. the one with the highest number of matching values
            for col in cols:
                if col == "study":
                    continue
                max_matches = 0
                max_col = "missing"
                for input_col in input_df.columns:
                    try:
                        matches = sum([1 for idx,x in enumerate(output_study[col]) if x == input_df[input_col]
                        [idx]])
                    except:
                        pdb.set_trace()
                    if matches > max_matches:
                        max_matches = matches
                        max_col = input_col
                df.loc[col, study] = max_col

                # build a matrix that has the study on the columns and the predicted column names on the rows
            df.fillna("missing", inplace=True)

            targets += list(target_matching[study].values)
            predicted += list(df[study].values)

        # print(df)
        p,r,f1,sup = precision_recall_fscore_support(targets, predicted, average="micro", zero_division=0)
    
        stats_dict["f1_score"] = f1
        print("Plan", stats_dict["plan_label"], "F1", f1, "runtime", stats_dict["runtime"], "cost", stats_dict["total_usd"])
        results.append(stats_dict)

    return results



def stub_scores(records_df):

    if records_df.empty:
        return 0.0, 0.0, 0.0

    output = records_df
    index = [x for x in output.columns if x != "study"]
    target_matching = pd.read_csv(os.path.join(RESULT_PATH, "target_matching.csv"), index_col=0).reindex(index)

    studies = output["study"].unique()
    # Group by output by the "study" column and split it into many dataframes indexed by the "study" column
    df = pd.DataFrame(columns=target_matching.columns, index = index)
    cols = output.columns
    predicted = []
    targets = []

    for study in studies:
        output_study = output[output["study"] == study]
        study = study.split(".xlsx")[0]
        try:
            input_df = pd.read_excel(os.path.join(IN_DIR, f"{study}.xlsx"))
        except:
            targets += [study]*5 
            predicted += ["missing"]*5
            continue
        # for every column in output_study, check which column in input_df is the closest, i.e. the one with the highest number of matching values
        for col in cols:
            if col == "study":
                continue
            max_matches = 0
            max_col = "missing"
            for input_col in input_df.columns:
                try:
                    matches = sum([1 for idx,x in enumerate(output_study[col]) if x == input_df[input_col]
                    [idx]])
                except:
                    pdb.set_trace()
                if matches > max_matches:
                    max_matches = matches
                    max_col = input_col
            df.loc[col, study] = max_col

            # build a matrix that has the study on the columns and the predicted column names on the rows
        df.fillna("missing", inplace=True)
        df.to_csv(RESULT_PATH+"/predicted_matching.csv", index=False)

        targets += list(target_matching[study].values)
        predicted += list(df[study].values)

    # print(df)
    # get groundtruth
    p,r,f1,sup = precision_recall_fscore_support(targets, predicted, average="micro", zero_division=0)

    print(f"Precision {p} recall {r} F1 {f1}")


def execute_biofabric_pz(datasetid, result_dir, reoptimize=False, limit=None):
    """
    This creates the PZ set of plans
    """

    # pz reg --name biofabric-matching-tiny --path testdata/biofabric-matching-tiny
    # xls = pz.Dataset('biofabric-matching-tiny', schema=pz.XLSFile)
    xls = pz.Dataset(datasetid, schema=pz.XLSFile)
    patient_tables = xls.convert(pz.Table, desc="All tables in the file", cardinality="oneToMany")
    patient_tables = patient_tables.filterByStr("The rows of the table contain the patient age")
    case_data = patient_tables.convert(CaseData, desc="The patient data in the table",cardinality="oneToMany")

    logicalTree = case_data.getLogicalTree()

    allow_codegen = "codegen" in datasetid
    candidatePlans = logicalTree.createPhysicalPlanCandidates(max=limit, allow_codegen=allow_codegen, shouldProfile=True)

    num_plans = len(candidatePlans)
    # for idx, (totalTimeInitEst, totalCostInitEst, qualityInitEst, plan) in enumerate(candidatePlans):
    for idx in range(num_plans):

        if os.path.exists(f'{result_dir}/profiling-{idx}.json'):
            continue
        candidatePlans = logicalTree.createPhysicalPlanCandidates(max=limit, allow_codegen=allow_codegen, shouldProfile=True)
        _, _, _, plan = candidatePlans[idx]

        print("----------------------")
        ops = plan.dumpPhysicalTree()
        flatten_ops = flatten_nested_tuples(ops)
        print(f"Plan: {graphicEmit(flatten_ops)}")
        print("---")

        # Running the plan
        start_time = time.time()
        records = [r for r in plan]
        # try:
            # records = [r for r in plan]
        # except Exception as e:
            # print(f"Error running plan {idx}: {e}")
            # continue
        runtime = time.time() - start_time
        profileData = plan.getProfilingData()

        sp = StatsProcessor(profileData)
        profile_dict = sp.profiling_data.to_dict()
        profile_dict["runtime"] = runtime
        profile_dict["plan_idx"] = idx
        profile_dict["plan_label"] = compute_label(plan, idx)

        profile_dict["models"] = [op.model.value for op in flatten_ops if not op.is_hardcoded()]
        profile_dict["op_names"] = []
        profile_dict["generated_fields"] = []
        profile_dict["query_strategies"] = []

        stats = sp.profiling_data
        cost=0
        while stats is not None:
            cost += stats.total_usd
            stats = stats.source_op_stats
            if stats is not None:
                profile_dict["op_names"].append(stats.op_name)
                profile_dict["generated_fields"].append(stats.generated_fields)
            # profile_dict["query_strategies"].append(stats.query_strategy)
            


        profile_dict["total_usd"] = cost

        with open(f'{result_dir}/profiling-{idx}.json', 'w') as f:
            json.dump(profile_dict, f)

        # workaround to disabling cache: delete all cached generations after each plan
        dspy_cache_dir = os.path.join(os.path.expanduser("~"), "cachedir_joblib/joblib/dsp/")
        if os.path.exists(dspy_cache_dir):
            shutil.rmtree(dspy_cache_dir)


def plot_runtime_cost_vs_quality(results, result_dir):
    # create figure
    fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True)

    # parse results into fields2
    for result_dict in results:
        runtime = result_dict["runtime"]
        cost = result_dict["total_usd"]
        f1_score = result_dict["f1_score"]
        models = [mod for mod in result_dict["models"] if mod is not None]

        text = None
        if len(set(models)) == 1:
            mod = Model(models[0])
            text = f"ALL-{repr(mod)}"
        elif len(set(models)) > 1:
            text = "-".join(map(lambda x: repr(Model(x)),models))

        text = text.replace("GEMINI_1", "GEMINI")
        text = text.replace("GPT_4", "GPT-4")
        text = text.replace("GPT_3_5", "GPT-3.5")

        # set label and color
        color = None
        marker = None
        # marker = "*" if "PZ" in label else "^"

        axs[0].scatter(f1_score, runtime, alpha=0.4, color=color, marker=marker) 
        axs[1].scatter(f1_score, cost, alpha=0.4, color=color, marker=marker)

        # add annotations
        if text is not None:
            ha, va = 'right', 'top'
            if text == "MIXTRAL-GPT-4":
                va = 'bottom'
            elif text in ["GEMINI-GPT-4","GEMINI-GPT-3.5"]:
                ha = 'right'
                continue
            if text == 'GPT-3.5-GEMINI':
                ha = 'right'
                va = 'top'
                # continue
            elif text == "ALL-GPT-3.5":
                ha = 'left'
                continue
            elif text == "ALL-GPT-4":
                ha = 'right'
            elif text == "ALL-GEMINI":
                va = 'top'
                continue
            elif text == "GPT-3.5-MIXTRAL":
                ha = 'left'
            elif text == "GPT-3.5-GPT-4":
                va = 'center'
                # continue
            elif text == "GEMINI-MIXTRAL":
                ha = 'left'
                va = 'bottom'

            axs[0].annotate(text, (f1_score, runtime), ha=ha, va=va)
            axs[1].annotate(text, (f1_score, cost), ha=ha, va=va)

    # savefig
    axs[0].set_title("Runtime and Cost vs. F1 Score")
    axs[0].set_ylabel("runtime (seconds)")
    axs[1].set_ylabel("cost (USD)")
    axs[1].set_xlabel("F1 Score")

    axs[0].grid(True)
    axs[1].grid(True)
    # axs[0].legend(bbox_to_anchor=(1.03, 1.0))
    fig.savefig(f"{result_dir}/biofabric.png", bbox_inches="tight")



if __name__ == "__main__":
    # parse arguments
    startTime = time.time()
    parser = argparse.ArgumentParser(description='Run the evaluation(s) for the paper')
    parser.add_argument('--datasetid', type=str, help='The dataset id')
    parser.add_argument('--limit' , type=int, help='The number of plans to consider')
    parser.add_argument('--no-cache', action='store_true', help='Do not use cached results', default=False)
    parser.add_argument('--dry-run', action='store_true', help='Do not use cached results', default=False)

    args = parser.parse_args()
    no_cache = args.no_cache
    datasetid = args.datasetid
   
    if no_cache:
        pz.DataDirectory().clearCache(keep_registry=True)

    # records_df = pd.read_csv("eval-results/biofabric/clean_output.csv")
    # stub_scores(records_df)

    datasetid = "biofabric-medium"
    # datasetid = "biofabric-medium-codegen"
    RESULT_PATH = f"eval-results/{datasetid}/"
    # get PZ plan metrics
    print("Running PZ Plans")
    print("----------------")
    if not args.dry_run:
        execute_biofabric_pz(datasetid, limit=args.limit, result_dir=RESULT_PATH)
    pz_results = score_biofabric_plans(result_dir=RESULT_PATH)

    plot_runtime_cost_vs_quality(pz_results, RESULT_PATH)