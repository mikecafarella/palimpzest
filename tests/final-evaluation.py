#!/usr/bin/env python3
from palimpzest.profiler import Profiler, StatsProcessor
import palimpzest as pz
import context 

from palimpzest.execution import graphicEmit, flatten_nested_tuples
from palimpzest.operators import InduceFromCandidateOp

from PIL import Image
from sklearn.metrics import precision_recall_fscore_support
from tabulate import tabulate

import matplotlib.pyplot as plt
import pandas as pd

import argparse
import json
import shutil
import subprocess
import time
import os
import pdb


class Email(pz.TextFile):
    """Represents an email, which in practice is usually from a text file"""
    sender = pz.Field(desc="The email address of the sender", required=True)
    subject = pz.Field(desc="The subject of the email", required=True)
    # to = pz.ListField(element_type=pz.StringField, desc="The email address(es) of the recipient(s)", required=True)
    # cced = pz.ListField(element_type=pz.StringField, desc="The email address(es) CC'ed on the email", required=True)

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

# TODO: it might not be obvious to a new user how to write/split up a schema for multimodal file data;
#       under our current setup, we have one schema which represents a file (e.g. pz.File), so the equivalent
#       here is to have a schema which represents the different (sets of) files, but I feel like users
#       will naturally just want to define the fields they wish to extract from the underlying (set of) files
#       and have PZ take care of the rest
class RealEstateListingFiles(pz.Schema):
    """The source text and image data for a real estate listing."""
    listing = pz.StringField(desc="The name of the listing", required=True)
    text_content = pz.StringField(desc="The content of the listing's text description", required=True)
    image_contents = pz.ListField(element_type=pz.BytesField, desc="A list of the contents of each image of the listing", required=True)

# TODO: longer-term we will want to support one or more of the following:
#       0. allow use of multimodal models on text + image inputs
#
#       1. allow users to define fields and specify which source fields they
#          should be converted from (e.g. text_content or image_contents);
#          PZ can then re-order these separate conversion steps with downstream
#          filters automatically to minimize execution cost
#      
class TextRealEstateListing(RealEstateListingFiles):
    """Represents a real estate listing with specific fields extracted from its text."""
    address = pz.StringField(desc="The address of the property")
    price = pz.NumericField(desc="The listed price of the property")
    # sq_ft = pz.NumericField(desc="The square footage (sq. ft.) of the property")
    # year_built = pz.NumericField(desc="The year in which the property was built")
    # bedrooms = pz.NumericField(desc="The number of bedrooms")
    # bathrooms = pz.NumericField(desc="The number of bathrooms")

# class CodeGenEasyTextRealEstateListing(RealEstateListingFiles):
#     """Represents a real estate listing with specific fields extracted from its text."""
#     address = pz.StringField(desc="The address of the property")
#     price = pz.NumericField(desc="The listed price of the property")
#     sq_ft = pz.NumericField(desc="The square footage (sq. ft.) of the property")
#     bedrooms = pz.NumericField(desc="The number of bedrooms")
#     bathrooms = pz.NumericField(desc="The number of bathrooms")

# class CodeGenHardTextRealEstateListing(RealEstateListingFiles):
#     """Represents a real estate listing with specific fields extracted from its text."""
#     has_walk_in_closet = pz.BooleanField(desc="True if the property has a walk-in closet and False otherwise")
#     garage_spaces = pz.NumericField(desc="The number of garage spaces the property has")
#     has_city_view = pz.BooleanField(desc="True if the propery has a view of the city and False otherwise")

class ImageRealEstateListing(RealEstateListingFiles):
    """Represents a real estate listing with specific fields extracted from its text and images."""
    is_modern_and_attractive = pz.BooleanField(desc="True if the home interior is modern and attractive and False otherwise")
    has_natural_sunlight = pz.BooleanField(desc="True if the home interior has lots of natural sunlight and False otherwise")

class RealEstateListingSource(pz.UserSource):
    def __init__(self, datasetId, listings_dir):
        super().__init__(RealEstateListingFiles, datasetId)
        self.listings_dir = listings_dir
        self.idx = 0

    def userImplementedIterator(self):
        for root, _, files in os.walk(self.listings_dir):
            if root == self.listings_dir:
                continue

            # create data record
            dr = pz.DataRecord(self.schema, scan_idx=self.idx)
            dr.listing = root.split("/")[-1]
            dr.image_contents = []
            for file in files:
                bytes_data = open(os.path.join(root, file), "rb").read()
                if file.endswith('.txt'):
                    dr.text_content = bytes_data.decode("utf-8")
                    # dr.text_content = str(bytes_data)
                elif file.endswith('.png'):
                    dr.image_contents.append(bytes_data)
            yield dr

            self.idx += 1


def buildNestedStr(node, indent=0, buildStr=""):
    elt, child = node
    indentation = " " * indent
    buildStr =  f"{indentation}{elt}" if indent == 0 else buildStr + f"\n{indentation}{elt}"
    if child is not None:
        return buildNestedStr(child, indent=indent+2, buildStr=buildStr)
    else:
        return buildStr


def get_models_from_physical_plan(plan) -> list:
    models = []
    while plan is not None:
        model = getattr(plan, "model", None)
        models.append(model.value if model is not None else None)
        plan = plan.source

    return models


def compute_label(physicalTree, label_idx):
    """
    Map integer to physical plan.
    """
    physicalOps = physicalTree.dumpPhysicalTree()
    label = buildNestedStr(physicalOps)
    # print(f"LABEL {label_idx}: {label}")

    flat = flatten_nested_tuples(physicalOps)
    ops = [op for op in flat if not op.is_hardcoded()]
    label = "-".join([
        f"{repr(op.model)}_{op.query_strategy if isinstance(op, InduceFromCandidateOp) else None}_{op.token_budget if isinstance(op, InduceFromCandidateOp) else None}"
        for op in ops
    ])
    return f"PZ-{label_idx}-{label}"


def score_biofabric_plans(opt, workload, records, plan_idx) -> float:
    """
    Computes the results of all biofabric plans
    """
    # parse records
    exclude_keys = ["filename", "op_id", "uuid", "parent_uuid", "stats"]
    output_rows = []
    for rec in records:
        dct = {k:v for k,v in rec.asDict().items() if k not in exclude_keys}
        filename = os.path.basename(rec.asDict()["filename"])
        dct["study"] = filename.split("_")[0]
        output_rows.append(dct)

    records_df = pd.DataFrame(output_rows)
    records_df.to_csv(f'final-eval-results/{opt}/{workload}/preds-{plan_idx}.csv', index=False)

    if records_df.empty:
        return 0.0

    output = records_df
    index = [x for x in output.columns if x != "study"]
    # target_matching = pd.read_csv(os.path.join(f'final-eval-results/{opt}/{workload}/', "target_matching.csv"), index_col=0).reindex(index)
    target_matching = pd.read_csv(os.path.join(f'testdata/', "target_matching.csv"), index_col=0).reindex(index)

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
            input_df = pd.read_excel(os.path.join("testdata/biofabric-matching/", f"{study}.xlsx"))
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

    return f1


def score_plan(opt, workload, records, plan_idx) -> float:
    """
    Computes the F1 score of the plan
    """
    # special handling for biofabric workload
    if workload == "biofabric":
        return score_biofabric_plans(opt, workload, records, plan_idx)

    # parse records
    records = [
        {
            key: record.__dict__[key]
            for key in record.__dict__
            if not key.startswith('_') and key not in ["image_contents"]
        }
        for record in records
    ]
    records_df = pd.DataFrame(records)

    # save predictions for this plan
    records_df.to_csv(f'final-eval-results/{opt}/{workload}/preds-{plan_idx}.csv', index=False)

    if records_df.empty:
        return 0.0

    # get list of predictions
    preds = None
    if workload == "enron":
        preds = records_df.filename.apply(lambda fn: os.path.basename(fn)).tolist()
    elif workload == "real-estate":
        preds = list(records_df.listing)

    # get list of groundtruth answers
    targets = None
    if workload == "enron":
        gt_df = pd.read_csv("testdata/groundtruth/enron-eval.csv")
        targets = list(gt_df[gt_df.label == 1].filename)
    elif workload == "real-estate":
        gt_df = pd.read_csv("testdata/groundtruth/real-estate-eval-30.csv")
        targets = list(gt_df[gt_df.label == 1].listing)

    # compute true and false positives
    tp, fp = 0, 0
    for pred in preds:
        if pred in targets:
            tp += 1
        else:
            fp += 1

    # compute false negatives
    fn = 0
    for target in targets:
        if target not in preds:
            fn += 1

    # compute precision, recall, f1 score
    precision = tp/(tp + fp) if tp + fp > 0 else 0.0
    recall = tp/(tp + fn) if tp + fn > 0 else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

    return f1_score


def run_pz_plan(opt, workload, plan, plan_idx):
    """
    I'm placing this in a separate file from evaluate_pz_plans to see if this prevents
    an error where the DSPy calls to Gemini (and other models?) opens too many files.
    My hope is that placing this inside a separate function will cause the file descriptors
    to be cleaned up once the function returns.
    """
    # TODO: eventually get runtime from profiling data
    # execute plan to get records and runtime;
    start_time = time.time()
    records = [r for r in plan]
    runtime = time.time() - start_time

    # get profiling data for plan and compute its cost
    profileData = plan.getProfilingData()
    sp = StatsProcessor(profileData)

    # TODO: debug profiling issue w/conventional query stats for per-field stats
    # with open(f'eval-results/{datasetid}-profiling-{idx}.json', 'w') as f:
    #     json.dump(sp.profiling_data.to_dict(), f)

    # score plan based on its output records
    f1_score = score_plan(opt, workload, records, plan_idx)

    plan_info = {
        "plan_idx": plan_idx,
        "plan_label": compute_label(plan, plan_idx),
        "models": [],
        "op_names": [],
        "generated_fields": [],
        "query_strategies": [],
        "token_budgets": []
    }
    cost = 0.0
    stats = sp.profiling_data
    while stats is not None:
        cost += stats.total_usd
        plan_info["models"].append(stats.model_name)
        plan_info["op_names"].append(stats.op_name)
        plan_info["generated_fields"].append(stats.generated_fields)
        plan_info["query_strategies"].append(stats.query_strategy)
        plan_info["token_budgets"].append(stats.token_budget)
        stats = stats.source_op_stats

    # construct and return result_dict
    result_dict = {
        "runtime": runtime,
        "cost": cost,
        "f1_score": f1_score,
        "plan_info": plan_info,
    }

    return result_dict


def get_logical_tree(workload):
    """
    This assumes you have preregistered the enron and biofabric datasets:

    $ pz reg --path testdata/enron-eval --name enron-eval
    $ pz reg --path testdata/biofabric-medium --name biofabric-medium
    """
    if workload == "enron":
        emails = pz.Dataset("enron-eval", schema=Email)
        emails = emails.filterByStr("The email is not quoting from a news article or an article written by someone outside of Enron")
        emails = emails.filterByStr("The email refers to a fraudulent scheme (i.e., \"Raptor\", \"Deathstar\", \"Chewco\", and/or \"Fat Boy\")")
        # emails = emails.filterByStr("The email chain (including metadata) refers to Jeffrey Skilling (Jeff) and/or Andy Fastow (Andy)")
        return emails.getLogicalTree()

    if workload == "real-estate":
        def within_two_miles_of_mit(record):
            # NOTE: I'm using this hard-coded function so that folks w/out a
            #       Geocoding API key from google can still run this example
            try:
                far_away_addrs = ["Melcher St", "Sleeper St", "437 D St", "Seaport", "Liberty", "Telegraph St"]
                if any([street.lower() in record.address.lower() for street in far_away_addrs]):
                    return False
                return True
            except:
                return False

        def in_price_range(record):
            try:
                price = record.price
                if type(price) == str:
                    price = price.strip()
                    price = int(price.replace("$","").replace(",",""))
                return 6e5 < price and price <= 2e6
            except:
                return False

        listings = pz.Dataset(workload, schema=RealEstateListingFiles)
        listings = listings.convert(TextRealEstateListing, depends_on="text_content")
        listings = listings.convert(ImageRealEstateListing, image_conversion=True, depends_on="image_contents")
        listings = listings.filterByStr(
            "The interior is modern and attractive, and has lots of natural sunlight",
            depends_on=["is_modern_and_attractive", "has_natural_sunlight"]
        )
        listings = listings.filterByFn(within_two_miles_of_mit, depends_on="address")
        listings = listings.filterByFn(in_price_range, depends_on="price")
        return listings.getLogicalTree()

    if workload == "biofabric":
        xls = pz.Dataset("biofabric-medium", schema=pz.XLSFile)
        patient_tables = xls.convert(pz.Table, desc="All tables in the file", cardinality="oneToMany")
        patient_tables = patient_tables.filterByStr("The rows of the table contain the patient age")
        case_data = patient_tables.convert(CaseData, desc="The patient data in the table",cardinality="oneToMany")

        return case_data.getLogicalTree()

    return None


def evaluate_pz_plans(opt, workload, dry_run=False):
    """
    This creates the PZ set of plans for the Enron email evaluation.

    Make sure to pre-register the dataset(s) with:

    $ pz reg --path testdata/enron-eval --name enron-eval

    (Note that the real-estate dataset is registered dynamically.)
    """
    # # turn off DSPy cache
    # os.environ["DSP_CACHEBOOL"] = "FALSE"

    # TODO: we can expand these datasets, but they're good enough for now
    logicalTree = get_logical_tree(workload)

    # NOTE: the following weird iteration over physical plans by idx is intentional and necessary
    #       at the moment in order for stats collection to work properly. For some yet-to-be-discovered
    #       reason, `createPhysicalPlanCandidates` is creating physical plans which share the same
    #       copy of some operators. This means that if we naively iterate over the plans and execute them
    #       some plans' profilers will count 2x (or 3x or 4x etc.) the number of records processed,
    #       dollars spent, time spent, etc. This workaround recreates the physical plans on each
    #       iteration to ensure that they are new.

    # get total number of plans
    allow_model_selection = (opt == "model")
    allow_codegen = (opt == "codegen")
    allow_token_reduction = (opt == "token-reduction")
    num_plans = len(logicalTree.createPhysicalPlanCandidates(
        allow_model_selection=allow_model_selection,
        allow_codegen=allow_codegen,
        allow_token_reduction=allow_token_reduction,
        pareto_optimal=False if opt == "codegen" and workload == "enron" else True,
        shouldProfile=True,
    ))

    # remove codegen samples from previous dataset from cache
    if allow_codegen:
        cache = pz.DataDirectory().getCacheService()
        cache.rmCachedData("codeEnsemble")
        cache.rmCachedData("codeSamples")

    for plan_idx in range(num_plans):
    # for plan_idx, (totalTimeInitEst, totalCostInitEst, qualityInitEst, plan) in enumerate(candidatePlans):
        if os.path.exists(f'final-eval-results/{opt}/{workload}/results-{plan_idx}.json'):
            continue

        # TODO: for now, re-create candidate plans until we debug duplicate profiler issue
        candidatePlans = logicalTree.createPhysicalPlanCandidates(
            allow_model_selection=allow_model_selection,
            allow_codegen=allow_codegen,
            allow_token_reduction=allow_token_reduction,
            pareto_optimal=False if opt == "codegen" and workload == "enron" else True,
            shouldProfile=True,
        )
        _, _, _, plan, _ = candidatePlans[plan_idx]

        # workaround to disabling cache: delete all cached generations after each plan
        bad_files = ["testdata/enron-eval/assertion.log", "testdata/enron-eval/azure_openai_usage.log", "testdata/enron-eval/openai_usage.log"]
        for file in bad_files:
            if os.path.exists(file):
                os.remove(file)

        # display the plan output
        print("----------------------")
        ops = plan.dumpPhysicalTree()
        flatten_ops = flatten_nested_tuples(ops)
        print(f"Plan {plan_idx}:")
        graphicEmit(flatten_ops)
        print("---")

        if dry_run:
            continue

        # run the plan
        result_dict = run_pz_plan(opt, workload, plan, plan_idx)
        print(f"Plan: {result_dict['plan_info']['plan_label']}")
        print(f"  F1: {result_dict['f1_score']}")
        print(f"  rt: {result_dict['runtime']}")
        print(f"  $$: {result_dict['cost']}")
        print("---")

        # write result json object
        with open(f'final-eval-results/{opt}/{workload}/results-{plan_idx}.json', 'w') as f:
            json.dump(result_dict, f)

        # workaround to disabling cache: delete all cached generations after each plan
        dspy_cache_dir = os.path.join(os.path.expanduser("~"), "cachedir_joblib/joblib/dsp/")
        if os.path.exists(dspy_cache_dir):
            shutil.rmtree(dspy_cache_dir)

    return num_plans


def plot_runtime_cost_vs_quality(results, opt, workload):
    # create figure
    fig_text, axs_text = plt.subplots(nrows=2, ncols=1, sharex=True)
    fig_clean, axs_clean = plt.subplots(nrows=2, ncols=1, sharex=True)

    # parse results into fields
    for plan_idx, result_dict in results:
        runtime = result_dict["runtime"]
        cost = result_dict["cost"]
        f1_score = result_dict["f1_score"]
        text = plan_idx

        # set label and color
        color = None
        marker = None

        # plot runtime vs. f1_score and cost vs. f1_score
        axs_text[0].scatter(f1_score, runtime, alpha=0.4, color=color, marker=marker) 
        axs_text[1].scatter(f1_score, cost, alpha=0.4, color=color, marker=marker)
        axs_clean[0].scatter(f1_score, runtime, alpha=0.4, color=color, marker=marker) 
        axs_clean[1].scatter(f1_score, cost, alpha=0.4, color=color, marker=marker)

        # add annotations
        axs_text[0].annotate(text, (f1_score, runtime))
        axs_text[1].annotate(text, (f1_score, cost))

    # TODO:
    # set x,y-lim for each workload

    # turn on grid lines
    axs_text[0].grid(True)
    axs_text[1].grid(True)
    axs_clean[0].grid(True)
    axs_clean[1].grid(True)

    # savefigs
    workload_to_title = {
        "enron": "Enron Legal Discovery",
        "real-estate": "Real Estate Search",
        "biofabric": "Biofabric-medium"
    }
    axs_text[0].set_title(f"{workload_to_title[workload]}")
    axs_text[0].set_ylabel("Runtime (seconds)")
    axs_text[1].set_ylabel("Cost (USD)")
    axs_text[1].set_xlabel("F1 Score")
    fig_text.savefig(f"final-eval-results/{opt}/{workload}/{opt}-{workload}-text.png", dpi=500, bbox_inches="tight")

    axs_clean[0].set_title(f"{workload_to_title[workload]}")
    axs_clean[0].set_ylabel("Runtime (seconds)")
    axs_clean[1].set_ylabel("Cost (USD)")
    axs_clean[1].set_xlabel("F1 Score")
    fig_clean.savefig(f"final-eval-results/{opt}/{workload}/{opt}-{workload}-clean.png", dpi=500, bbox_inches="tight")


def plot_runtime_vs_dataset_size(all_results, plot_filename):
    # create figure
    fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True)

    # set up plot lists
    num_plans = len(all_results[0])
    plan_to_runtimes = {plan_idx: [] for plan_idx in range(num_plans)}
    plan_to_costs = {plan_idx: [] for plan_idx in range(num_plans)}
    plan_to_f1_scores = {plan_idx: [] for plan_idx in range(num_plans)}
    plan_to_text = {plan_idx: None for plan_idx in range(num_plans)}

    for results_idx, results in enumerate(all_results):
        for plan_idx, result_dict in enumerate(results):
            plan_to_runtimes[plan_idx].append(result_dict["runtime"])
            plan_to_costs[plan_idx].append(result_dict["cost"])
            plan_to_f1_scores[plan_idx].append(result_dict["f1_score"])

            if "codegen-easy" in datasetid and results_idx == 5:
                models = (
                    result_dict["models"]
                    if "models" in result_dict
                    else result_dict["plan_info"]["models"]
                )
                query_strategies = (
                    result_dict["plan_info"]["query_strategies"]
                    if "plan_info" in result_dict and "query_strategies" in result_dict["plan_info"]
                    else None
                )

                f1_score, runtime, cost = result_dict["f1_score"], result_dict["runtime"], result_dict["cost"]
                if all([model is None or "gpt-4" in model for model in models]):
                    # add text for ALL-GPT4
                    plan_to_text[plan_idx] = ("ALL-GPT4", f1_score, runtime, cost)
                if all([model is None or "mistralai" in model for model in models]):
                    # add text for ALL-MIXTRAL
                    plan_to_text[plan_idx] = ("ALL-MIXTRAL", f1_score, runtime, cost)
                if all([model is None or "gemini" in model for model in models]):
                    # add text for ALL-GEMINI
                    plan_to_text[plan_idx] = ("ALL-GEMINI", f1_score, runtime, cost)

                if query_strategies is not None and any([qs is not None and "codegen" in qs for qs in query_strategies]):
                    plan_to_text[plan_idx] = ("CODEGEN (GPT4)", f1_score, runtime, cost)

    # set label and color
    color = None
    marker = None

    # iterate over plans and add line plots (one-per-plan)
    for plan_idx in range(num_plans):
        # plot runtime vs. f1_score
        axs[0].plot([5, 10, 15, 20, 25, 30], plan_to_runtimes[plan_idx], alpha=0.4, color=color, marker=marker) 

        # plot cost vs. f1_score
        axs[1].plot([5, 10, 15, 20, 25, 30], plan_to_costs[plan_idx], alpha=0.4, color=color, marker=marker)

        # add annotations
        if plan_to_text[plan_idx] is not None:
            text, f1_score, runtime, cost = plan_to_text[plan_idx]
            runtime_x, cost_x = 30, 30
            if text == "ALL-GPT4":
                cost_x = 25
                cost = 0.3
            if text == "ALL-MIXTRAL":
                runtime_x = 25
                runtime = 250
            axs[0].annotate(text, (runtime_x, runtime), ha='right', va='bottom')
            axs[1].annotate(text, (cost_x, cost), ha='right', va='bottom')

    # savefig
    axs[0].set_title("Runtime and Cost vs. Dataset Size")
    axs[0].set_ylabel("runtime (seconds)")
    axs[1].set_ylabel("cost (USD)")
    axs[1].set_xlabel("Dataset Size (# of records)")
    # axs[0].legend(bbox_to_anchor=(1.03, 1.0))
    fig.savefig(f"final-eval-results/{opt}/{workload}/{plot_filename}.png", bbox_inches="tight")


def run_reoptimize_eval(datasetid):
    # set number of samples to draw
    num_samples=3

    # create query for enron dataset
    emails = pz.Dataset(datasetid, schema=Email, num_samples=num_samples, nocache=True)
    emails = emails.filterByStr("The email refers to a fraudulent scheme (i.e., \"Raptor\", \"Deathstar\", \"Chewco\", and/or \"Fat Boy\")")
    # emails = emails.filterByStr("The email is sent by Jeffrey Skilling (jeff.skilling@enron.com), or Andy Fastow (andy.fastow@enron.com), or refers to either one of them by name")
    emails = emails.filterByStr("The email is not quoting from a news article or an article written by someone outside of Enron")
    logicalTree = emails.getLogicalTree()

    # compute number of plans
    candidatePlans = logicalTree.createPhysicalPlanCandidates(shouldProfile=True)
    num_plans = len(candidatePlans)

    # identify initial est of best plan
    policy = pz.MaxQualityMinRuntime()
    best_plan, init_best_plan_idx = policy.choose(candidatePlans, return_idx=True)
    print(f"Initial best plan idx: {init_best_plan_idx}")
    print(f"Initial best plan: {buildNestedStr(best_plan[3].dumpPhysicalTree())}")

    # define helper function to get models for induce/filter operations that use LLMs;
    # this is a dirty hack for now, but we can easily return this info from createPhysicalPlanCandidates()
    def filter_for_llm_ops(models, limit=False):
        return models[:3]

    # compute all initial estimates
    best_models = None
    estimates_and_results = {"init_estimates": [], "v1_estimates": [], "v2_estimates": [], "results": []}
    for idx in range(num_plans):
        # TODO: for now, re-create candidate plans until we debug duplicate profiler issue
        totalTimeInitEst, totalCostInitEst, qualityInitEst, plan, _ = candidatePlans[idx]

        models = get_models_from_physical_plan(plan)
        models = filter_for_llm_ops(models)
        result_dict = {"runtime": totalTimeInitEst, "cost": totalCostInitEst, "f1_score": qualityInitEst, "models": models}
        estimates_and_results["init_estimates"].append(result_dict)
        with open(f'eval-results/reoptimize-enron-init-est-{idx}.json', 'w') as f:
            json.dump(result_dict, f)

        if idx == init_best_plan_idx:
            best_models = models

    # iterate over plans to get ones matching best_plan but w/different end models
    other_plan_idxs = []
    for idx in range(num_plans):
        totalTimeInitEst, totalCostInitEst, qualityInitEst, plan, _ = candidatePlans[idx]
        models = get_models_from_physical_plan(plan)
        models = filter_for_llm_ops(models)
        
        if (
            all([model == "mistralai/Mixtral-8x7B-Instruct-v0.1" for model in models])
            or all([model == "gemini-1.0-pro-001" for model in models])
        ):
        # if models[:-1] == best_models[:-1] and models[-1] != best_models[-1]:
        #     other_plan_idxs.append(idx)
    
            print(f"CONSIDERING OTHER PLAN: (PLAN IDX: {idx})")
            print("---")
            print(f"{buildNestedStr(plan.dumpPhysicalTree())}")
            print("-------")
            other_plan_idxs.append(idx)

    # run init_best_plan_idx + other_plan_idxs to get sample data
    all_cost_estimate_data = []
    for plan_idx in [init_best_plan_idx] + other_plan_idxs:
        candidatePlans = logicalTree.createPhysicalPlanCandidates(shouldProfile=True)
        _, _, _, plan, _ = candidatePlans[plan_idx]

        # workaround to disabling cache: delete all cached generations after each plan
        bad_files = ["testdata/enron-eval/assertion.log", "testdata/enron-eval/azure_openai_usage.log", "testdata/enron-eval/openai_usage.log"]
        for file in bad_files:
            if os.path.exists(file):
                os.remove(file)
        
        print("------------ABOUT TO RUN--------------")
        print(f"Plan IDX: {plan_idx}")
        print(f"Plan: {buildNestedStr(plan.dumpPhysicalTree())}")
        print("---")
    
        # execute plan to get records and runtime;
        start_time = time.time()
        records = [r for r in plan]
        runtime = time.time() - start_time

        # get profiling data for plan and compute its cost
        profileData = plan.getProfilingData()
        sp = StatsProcessor(profileData)
        cost_estimate_sample_data = sp.get_cost_estimate_sample_data()
        all_cost_estimate_data.extend(cost_estimate_sample_data)

    import pandas as pd
    df = pd.DataFrame(all_cost_estimate_data)
    df.to_csv("cost-est-data.csv", index=False)

    # create FULL query for enron dataset
    emails = pz.Dataset(datasetid, schema=Email, nocache=True, scan_start_idx=num_samples)
    emails = emails.filterByStr("The email refers to a fraudulent scheme (i.e., \"Raptor\", \"Deathstar\", \"Chewco\", and/or \"Fat Boy\")")
    # emails = emails.filterByStr("The email is sent by Jeffrey Skilling (jeff.skilling@enron.com), or Andy Fastow (andy.fastow@enron.com), or refers to either one of them by name")
    emails = emails.filterByStr("The email is not quoting from a news article or an article written by someone outside of Enron")
    logicalTree = emails.getLogicalTree()

    # re-compute best plan index using sample data
    print("-----------------------")
    print("-----------------------")
    print("-----------------------")
    candidatePlans = logicalTree.createPhysicalPlanCandidates(cost_estimate_sample_data=all_cost_estimate_data, shouldProfile=True)

    # identify new est of best plan
    policy = pz.MaxQualityMinRuntime()
    best_plan, new_best_plan_idx = policy.choose(candidatePlans, return_idx=True)
    print(f"NEW best plan idx: {new_best_plan_idx}")
    print(f"NEW best plan: {buildNestedStr(best_plan[3].dumpPhysicalTree())}")
    os.makedirs("cost-est", exist_ok=True)
    for idx, plan in enumerate(candidatePlans):
        print("--------------------")
        print(f"Plan IDX: {idx}")
        print(f"Plan: {buildNestedStr(plan[3].dumpPhysicalTree())}")
        print(f"time: {plan[0]}")
        print(f"cost: {plan[1]}")
        print(f"quality: {plan[2]}")
        print("---")
        with open(f'cost-est/plan-{idx}.json', 'w') as f:
            json.dump(plan[4], f)

    # create figure
    fig, axs = plt.subplots(nrows=1, ncols=2, sharey=True)

    # get groundtruth results from enron evaluation
    enron_eval_data = []
    for idx in range(17):
        with open(f"eval-results/enron-eval-results-{idx}.json", 'r') as f:
            result = json.load(f)
            enron_eval_data.append(result)

    # get initial estimates from this experiment
    init_est_data = []
    with idx in range(18):
        with open(f"eval-results/enron-eval-init-est-{idx}.json", 'r') as f:
            result = json.load(f)
            init_est_data.append(result)

    # get estimates after 3 samples
    sample_est_data = []
    with idx in range(12):
        with open(f"cost-est/plan-{idx}.json", 'r') as f:
            result = json.load(f)
            sample_est_data.append(result)

    # plot actual result data in background
    for idx in range(18):
        runtime = init_est_data[idx]["runtime"]
        f1_score = init_est_data[idx]["f1_score"]
        models = init_est_data[idx]["models"]

        text = None
        if all([model == "gpt-4-0125-preview" for model in models]):
            # add text for ALL-GPT4
            text = "ALL-GPT4"
        elif all([model == "mistralai/Mixtral-8x7B-Instruct-v0.1" for model in models]):
            # add text for ALL-MIXTRAL
            text = "ALL-MIXTRAL"
        elif datasetid == "enron-eval" and models == ["gpt-4-0125-preview"] * 2 + ["mistralai/Mixtral-8x7B-Instruct-v0.1"]:
            # add text for Mixtral-GPT4
            text = "MIXTRAL-GPT4"
        elif datasetid == "enron-eval" and models == ["gpt-4-0125-preview"] * 2 + ["gemini-1.0-pro-001"]:
            # add text for Gemini-GPT4
            text = "GEMINI-GPT4"
        
        if idx == init_best_plan_idx:
            text = "BEST-PLAN (ALL-GPT4)"
    
        # set label and color
        color = None
        marker = None

        # plot runtime vs. f1_score
        axs[0].scatter(f1_score, runtime, alpha=0.4, color=color, marker=marker) 

        # add annotations
        if text is not None:
            ha, va = 'right', 'bottom'
            if text == "ALL-GPT4":
                va = 'top'
            elif text == "MIXTRAL-GPT4":
                va = 'bottom'
            elif text == "GEMINI-GPT4":
                va = 'top'
            elif text == "ALL-MIXTRAL":
                va = 'bottom'
            axs[0].annotate(text, (f1_score, runtime), ha=ha, va=va)

    for idx in range(12):
        runtime = sample_est_data[idx]["runtime"]
        f1_score = sample_est_data[idx]["f1_score"]
        models = sample_est_data[idx]["models"]

        text = None
        if all([model == "gpt-4-0125-preview" for model in models]):
            # add text for ALL-GPT4
            text = "ALL-GPT4"
        elif all([model == "mistralai/Mixtral-8x7B-Instruct-v0.1" for model in models]):
            # add text for ALL-MIXTRAL
            text = "ALL-MIXTRAL"
        elif datasetid == "enron-eval" and models == ["gpt-4-0125-preview"] * 2 + ["mistralai/Mixtral-8x7B-Instruct-v0.1"]:
            # add text for Mixtral-GPT4
            text = "MIXTRAL-GPT4"
        elif datasetid == "enron-eval" and models == ["gpt-4-0125-preview"] * 2 + ["gemini-1.0-pro-001"]:
            # add text for Gemini-GPT4
            text = "GEMINI-GPT4"
        
        if idx == new_best_plan_idx:
            text = "BEST-PLAN (MIXTRAL-GPT4)"
    
        # set label and color
        color = None
        marker = None

        # plot runtime vs. f1_score
        axs[1].scatter(f1_score, runtime, alpha=0.4, color=color, marker=marker) 

        # add annotations
        if text is not None:
            ha, va = 'right', 'bottom'
            if text == "ALL-GPT4":
                va = 'top'
            elif text == "MIXTRAL-GPT4":
                va = 'bottom'
            elif text == "GEMINI-GPT4":
                va = 'top'
            elif text == "ALL-MIXTRAL":
                va = 'bottom'
            axs[1].annotate(text, (f1_score, runtime), ha=ha, va=va)

    # savefig
    axs[0].set_title("Runtime and Cost vs. Quality")
    axs[0].set_ylabel("Runtime (seconds)")
    axs[0].set_xlabel("Est. Quality")
    axs[1].set_xlabel("Est. Quality")
    # axs[0].legend(bbox_to_anchor=(1.03, 1.0))
    fig_name = f"eval-results/{datasetid}-reoptimize.png"
    fig.savefig(fig_name, bbox_inches="tight")


if __name__ == "__main__":
    # parse arguments
    startTime = time.time()
    parser = argparse.ArgumentParser(description='Run the evaluation(s) for the paper')
    parser.add_argument('--workload', type=str, help='The workload: one of ["biofabric", "enron", "real-estate"]')
    parser.add_argument('--opt' , type=str, help='The optimization: one of ["model", "codegen", "token-reduction"]')
    parser.add_argument('--listings-dir', default="testdata/real-estate-eval-30", type=str, help='The directory with real-estate listings')
    parser.add_argument('--reoptimize', default=False, action='store_true', help='Run reoptimization')
    parser.add_argument('--dry-run', default=False, action='store_true', help='Just print plans w/out actually running any')

    args = parser.parse_args()

    # create directory for intermediate results
    os.makedirs(f"final-eval-results/{args.opt}/{args.workload}", exist_ok=True)

    # The user has to indicate the evaluation to be run
    if args.opt is None or args.workload is None:
        print("Please provide an optimization (--opt) and a workload (--workload)")
        exit(1)

    # re-optimization is unique enough to warrant its own code path
    if args.reoptimize:
        run_reoptimize_eval(args.opt, args.workload)
        exit(1)

    # register real-estate workload if necessary
    if args.workload == "real-estate":
        print("Registering Datasource")
        pz.DataDirectory().registerUserSource(RealEstateListingSource(args.workload, args.listings_dir), args.workload)

    # get PZ plan metrics
    print("Running PZ Plans")
    print("----------------")
    num_plans = evaluate_pz_plans(args.opt, args.workload, args.dry_run)

    if args.dry_run:
        exit(1)

    # read results file(s) generated by evaluate_pz_plans
    results = []
    for plan_idx in range(num_plans):
        if args.workload == "real-estate" and args.opt in ["models", "token-reduction"] and plan_idx == 9:
            continue

        with open(f"final-eval-results/{args.opt}/{args.workload}/results-{plan_idx}.json", 'r') as f:
            result = json.load(f)
            results.append((plan_idx, result))

    plot_runtime_cost_vs_quality(results, args.opt, args.workload)
