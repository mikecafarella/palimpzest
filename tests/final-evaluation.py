#!/usr/bin/env python3
from palimpzest.profiler import Profiler, StatsProcessor
import palimpzest as pz
import context 

from palimpzest.execution import graphicEmit, flatten_nested_tuples
from palimpzest.operators import InduceFromCandidateOp

from PIL import Image
from sklearn.metrics import precision_recall_fscore_support
from tabulate import tabulate

from collections import defaultdict
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import argparse
import json
import shutil
import subprocess
import time
import os
import pdb

# Addresses far from MIT; we use a simple lookup like this to make the
# experiments re-producible w/out needed a Google API key for geocoding lookups
FAR_AWAY_ADDRS = [
    "Melcher St", "Sleeper St", "437 D St", "Seaport Blvd", "50 Liberty Dr",
    "Telegraph St", "Columbia Rd", "E 6th St", "E 7th St", "E 5th St",
]


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
    is_modern_and_attractive = pz.BooleanField(desc="True if the home interior design is modern and attractive and False otherwise")
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
                bytes_data = None
                with open(os.path.join(root, file), "rb") as f:
                    bytes_data = f.read()
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

    return models # back to front

def get_budgets_from_physical_plan(plan) -> list:
    budgets = []
    while plan is not None:
        token_budget = getattr(plan, "token_budget", None)
        budgets.append(token_budget)
        plan = plan.source

    return budgets # back to front


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


def score_biofabric_plans(workload, records, plan_idx, policy_str=None, reopt=False) -> float:
    """
    Computes the results of all biofabric plans
    """
    # parse records
    # exclude_keys = ["filename", "op_id", "uuid", "parent_uuid", "stats"]
    include_keys = ['age_at_diagnosis', 'ajcc_pathologic_n', 'ajcc_pathologic_stage',
       'ajcc_pathologic_t', 'case_submitter_id', 'ethnicity', 'gender',
       'morphology', 'primary_diagnosis', 'race',
       'tissue_or_organ_of_origin', 'tumor_focality', 'tumor_grade',
       'tumor_largest_dimension_diameter', 'vital_status']
    output_rows = []
    for rec in records:
        dct = {k: v for k, v in rec.items() if k in include_keys}
        # dct = {k:v for k,v in rec._asDict().items() if k not in exclude_keys}
        # filename = os.path.basename(rec._asDict()["filename"])
        dct["study"] = os.path.basename(rec['filename']).split("_")[0]
        output_rows.append(dct)

    records_df = pd.DataFrame(output_rows)
    if not reopt:
        records_df.to_csv(f'final-eval-results/{workload}/preds-{plan_idx}.csv', index=False)
    else:
        records_df.to_csv(f'final-eval-results/reoptimization/{workload}/{policy_str}.csv', index=False)

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
                    matches = sum([1 for idx,x in enumerate(output_study[col]) if x == input_df[input_col][idx]])
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


def score_plan(workload, records, plan_idx, policy_str=None, reopt=False) -> float:
    """
    Computes the F1 score of the plan
    """
    # special handling for biofabric workload
    if workload == "biofabric":
        return score_biofabric_plans(workload, records, plan_idx, policy_str, reopt)

    # parse records
    # records = [
    #     {
    #         key: record.__dict__[key]
    #         for key in record.__dict__
    #         if not key.startswith('_') and key not in ["image_contents"]
    #     }
    #     for record in records
    # ]
    records_df = pd.DataFrame(records)

    # save predictions for this plan
    if not reopt:
        records_df.to_csv(f'final-eval-results/{workload}/preds-{plan_idx}.csv', index=False)
    else:
        records_df.to_csv(f'final-eval-results/reoptimization/{workload}/{policy_str}.csv', index=False)

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
        gt_df = pd.read_csv("testdata/groundtruth/real-estate-eval-100.csv")
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


def run_pz_plan(workload, plan, plan_idx, total_sentinel_cost, total_sentinel_time, sentinel_records):
    """
    I'm placing this in a separate file from evaluate_pz_plans to see if this prevents
    an error where the DSPy calls to Gemini (and other models?) opens too many files.
    My hope is that placing this inside a separate function will cause the file descriptors
    to be cleaned up once the function returns.
    """
    # TODO: eventually get runtime from profiling data
    # execute plan to get records and runtime;
    start_time = time.time()
    new_records = [r for r in plan]
    runtime = total_sentinel_time + (time.time() - start_time)

    # parse new_records
    new_records = [
        {
            key: record.__dict__[key]
            for key in record.__dict__
            if not key.startswith('_') and key not in ["image_contents"]
        }
        for record in new_records
    ]
    all_records = sentinel_records + new_records

    # get profiling data for plan and compute its cost
    profileData = plan.getProfilingData()
    sp = StatsProcessor(profileData)

    # TODO: debug profiling issue w/conventional query stats for per-field stats
    # with open(f'eval-results/{datasetid}-profiling-{idx}.json', 'w') as f:
    #     json.dump(sp.profiling_data.to_dict(), f)

    # score plan based on its output records
    f1_score = score_plan(workload, all_records, plan_idx)

    plan_info = {
        "plan_idx": plan_idx,
        "plan_label": compute_label(plan, plan_idx),
        "models": [],
        "op_names": [],
        "generated_fields": [],
        "query_strategies": [],
        "token_budgets": []
    }
    cost = total_sentinel_cost
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


def get_logical_tree(workload, nocache: bool=True, num_samples: int=None, scan_start_idx: int=0):
    """
    This assumes you have preregistered the enron and biofabric datasets:

    $ pz reg --path testdata/enron-eval --name enron-eval
    $ pz reg --path testdata/biofabric-medium --name biofabric-medium
    """
    if workload == "enron":
        emails = pz.Dataset("enron-eval", schema=Email, nocache=nocache, num_samples=num_samples, scan_start_idx=scan_start_idx)
        emails = emails.filterByStr("The email is not quoting from a news article or an article written by someone outside of Enron")
        emails = emails.filterByStr("The email refers to a fraudulent scheme (i.e., \"Raptor\", \"Deathstar\", \"Chewco\", and/or \"Fat Boy\")")
        # emails = emails.filterByStr("The email chain (including metadata) refers to Jeffrey Skilling (Jeff) and/or Andy Fastow (Andy)")
        return emails.getLogicalTree()

    if workload == "real-estate":
        def within_two_miles_of_mit(record):
            # NOTE: I'm using this hard-coded function so that folks w/out a
            #       Geocoding API key from google can still run this example
            try:
                if any([street.lower() in record.address.lower() for street in FAR_AWAY_ADDRS]):
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

        listings = pz.Dataset(workload, schema=RealEstateListingFiles, nocache=nocache, num_samples=num_samples, scan_start_idx=scan_start_idx)
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
        xls = pz.Dataset("biofabric-medium", schema=pz.XLSFile, nocache=nocache, num_samples=num_samples, scan_start_idx=scan_start_idx)
        patient_tables = xls.convert(pz.Table, desc="All tables in the file", cardinality="oneToMany")
        patient_tables = patient_tables.filterByStr("The rows of the table contain the patient age")
        case_data = patient_tables.convert(CaseData, desc="The patient data in the table",cardinality="oneToMany")

        return case_data.getLogicalTree()

    return None

# function to run sentinel
def run_sentinel_plan(plan_idx, workload, num_samples):
    # get specified sentinel plan
    logicalTree = get_logical_tree(workload, nocache=True, num_samples=num_samples)
    sentinel_plans = logicalTree.createPhysicalPlanCandidates(sentinels=True)
    plan = sentinel_plans[plan_idx]

    # display the plan output
    print("----------------------")
    ops = plan.dumpPhysicalTree()
    flatten_ops = flatten_nested_tuples(ops)
    print(f"Sentinel Plan {plan_idx}:")
    graphicEmit(flatten_ops)
    print("---")

    # run the plan
    records = [r for r in plan]

    # parse records
    records = [
        {
            key: record.__dict__[key]
            for key in record.__dict__
            if not key.startswith('_') and key not in ["image_contents"]
        }
        for record in records
    ]

    # get profiling data for plan and compute its cost
    profileData = plan.getProfilingData()
    sp = StatsProcessor(profileData)
    cost_estimate_sample_data = sp.get_cost_estimate_sample_data()

    plan_info = {
        "plan_idx": plan_idx,
        "plan_label": compute_label(plan, f"s{plan_idx}"),
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
        "runtime": None,
        "cost": cost,
        "f1_score": None,
        "plan_info": plan_info,
    }

    return records, result_dict, cost_estimate_sample_data


def run_sentinel_plans(workload, num_samples, policy_str: str=None, parallel: bool=False):
    start_time = time.time()

    # create query for dataset
    logicalTree = get_logical_tree(workload, nocache=True, num_samples=num_samples)

    # compute number of plans
    sentinel_plans = logicalTree.createPhysicalPlanCandidates(sentinels=True)
    num_sentinel_plans = len(sentinel_plans)

    total_sentinel_cost, all_cost_estimate_data, return_records = 0.0, [], []
    with Pool(processes=num_sentinel_plans) as pool:
        results = pool.starmap(run_sentinel_plan, [(plan_idx, workload, num_samples) for plan_idx in range(num_sentinel_plans)])

        # write out result dict and samples collected for each sentinel
        for idx, (records, result_dict, cost_est_sample_data) in enumerate(results):
            fp = (
                f"final-eval-results/reoptimization/{workload}/sentinel-{idx}-{policy_str}-results.json"
                if policy_str is not None
                else f"final-eval-results/{workload}/sentinel-{idx}-results.json"
            )
            if parallel:
                fp.replace("sentinel", "parallel-sentinel")

            with open(fp, 'w') as f:
                json.dump(result_dict, f)

            csv_fp = fp.replace(".json", ".csv")
            sample_df = pd.DataFrame(cost_est_sample_data)
            sample_df.to_csv(csv_fp, index=False)

            # aggregate sentinel est. data
            all_cost_estimate_data.extend(cost_est_sample_data)

            # find GPT-4 plan records and add those to all_records
            if all([model is None or model in ["gpt-4-0125-preview", "gpt-4-vision-preview"] for model in result_dict['plan_info']['models']]):
                return_records = records

            # update total cost of running sentinels
            total_sentinel_cost += result_dict['cost']

    total_sentinel_time = time.time() - start_time

    # workaround to disabling cache: delete all cached generations after each plan
    dspy_cache_dir = os.path.join(os.path.expanduser("~"), "cachedir_joblib/joblib/dsp/")
    if os.path.exists(dspy_cache_dir):
        shutil.rmtree(dspy_cache_dir)

    return total_sentinel_cost, total_sentinel_time, all_cost_estimate_data, return_records


def evaluate_pz_plan(sentinel_data, workload, plan_idx):
    # if os.path.exists(f'final-eval-results/{opt}/{workload}/results-{plan_idx}.json'):
    #     return
    if os.path.exists(f"final-eval-results/{workload}/results-{plan_idx}.json"):
        return

    # unpack sentinel data
    total_sentinel_cost, total_sentinel_time, all_cost_estimate_data, sentinel_records, num_samples = sentinel_data

    # get logicalTree
    logicalTree = get_logical_tree(workload, nocache=True, scan_start_idx=num_samples)

    # TODO: for now, re-create candidate plans until we debug duplicate profiler issue
    plans = logicalTree.createPhysicalPlanCandidates(
        min=20,
        cost_estimate_sample_data=all_cost_estimate_data,
        allow_model_selection=True,
        allow_codegen=True,
        allow_token_reduction=True,
        # pareto_optimal=False if opt in ["codegen", "token-reduction"] and workload == "enron" else True,
        pareto_optimal=True,
        include_baselines=True,
        shouldProfile=True,
    )
    _, _, _, plan, _ = plans[plan_idx]
    plan.setPlanIdx(plan_idx)

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

    # run the plan
    result_dict = run_pz_plan(workload, plan, plan_idx, total_sentinel_cost, total_sentinel_time, sentinel_records)
    print(f"Plan: {result_dict['plan_info']['plan_label']}")
    print(f"  F1: {result_dict['f1_score']}")
    print(f"  rt: {result_dict['runtime']}")
    print(f"  $$: {result_dict['cost']}")
    print("---")

    # write result json object
    with open(f'final-eval-results/{workload}/results-{plan_idx}.json', 'w') as f:
        json.dump(result_dict, f)


def evaluate_pz_plans(workload, dry_run=False):
    """
    This creates the PZ set of plans for the Enron email evaluation.

    Make sure to pre-register the dataset(s) with:

    $ pz reg --path testdata/enron-eval --name enron-eval

    (Note that the real-estate dataset is registered dynamically.)

    Make sure to set DSP_CACHEBOOL=false.
    """
    workload_to_dataset_size = {"enron": 1000, "real-estate": 100, "biofabric": 11}
    dataset_size = workload_to_dataset_size[workload]
    # num_samples = min(10, int(0.05 * dataset_size)) if workload != "biofabric" else 0
    num_samples = int(0.05 * dataset_size) if workload != "biofabric" else 1

    # run sentinels
    output = run_sentinel_plans(workload, num_samples)
    total_sentinel_cost, total_sentinel_time, all_cost_estimate_data, sentinel_records = output

    # create query for dataset
    logicalTree = get_logical_tree(workload, nocache=True, scan_start_idx=num_samples)

    # get total number of plans
    compilation_start = time.time()
    plans = logicalTree.createPhysicalPlanCandidates(
        min=20,
        cost_estimate_sample_data=all_cost_estimate_data,
        allow_model_selection=True,
        allow_codegen=True,
        allow_token_reduction=True,
        # pareto_optimal=False if opt in ["codegen", "token-reduction"] and workload == "enron" else True,
        pareto_optimal=True,
        include_baselines=True,
        shouldProfile=True,
    )
    num_plans = len(plans)
    compilation_end = time.time()

    if dry_run:
        for plan_idx, (_, _, _, plan, _) in enumerate(plans):
            # display the plan output
            print("----------------------")
            ops = plan.dumpPhysicalTree()
            flatten_ops = flatten_nested_tuples(ops)
            print(f"Plan {plan_idx}:")
            graphicEmit(flatten_ops)
            print("---")
        print(f"COMPILATION TIME: {compilation_end - compilation_start:.2f}")
        return

    # remove codegen samples from previous dataset from cache
    cache = pz.DataDirectory().getCacheService()
    for plan_idx in range(num_plans):
        cache.rmCachedData(f"codeEnsemble{plan_idx}")
        cache.rmCachedData(f"codeSamples{plan_idx}")

    with Pool(processes=num_plans) as pool:
        sentinel_data = (total_sentinel_cost, total_sentinel_time, all_cost_estimate_data, sentinel_records, num_samples)
        _ = pool.starmap(evaluate_pz_plan, [(sentinel_data, workload, plan_idx) for plan_idx in range(num_plans)])

    # workaround to disabling cache: delete all cached generations after each plan
    dspy_cache_dir = os.path.join(os.path.expanduser("~"), "cachedir_joblib/joblib/dsp/")
    if os.path.exists(dspy_cache_dir):
        shutil.rmtree(dspy_cache_dir)

    return num_plans


def run_reoptimize_eval(workload, policy_str, parallel: bool=False):
    workload_to_fixed_cost = {
        "enron": 20.0,
        "real-estate": 3.0,
        "biofabric": 2.0,
    }
    workload_to_fixed_runtime = {
        "enron": 10000,
        "real-estate": 600,
        "biofabric": 1000,
    }
    workload_to_fixed_quality = {
        "enron": 0.8,
        "real-estate": 0.8,
        "biofabric": 0.40,
    }

    policy = pz.MaxHarmonicMean()
    if policy_str is not None:
        if policy_str == "max-quality-at-fixed-cost":
            policy = pz.MaxQualityAtFixedCost(fixed_cost=workload_to_fixed_cost[workload])
        elif policy_str == "max-quality-at-fixed-runtime":
            policy = pz.MaxQualityAtFixedRuntime(fixed_runtime=workload_to_fixed_runtime[workload])
        elif policy_str == "min-runtime-at-fixed-quality":
            policy = pz.MinRuntimeAtFixedQuality(fixed_quality=workload_to_fixed_quality[workload])
        elif policy_str == "min-cost-at-fixed-quality":
            policy = pz.MinCostAtFixedQuality(fixed_quality=workload_to_fixed_quality[workload])

    # TODO: in practice could move this inside of get_logical_tree w/flag indicating sentinel run;
    #       for now just manually set to make sure evaluation is accurate
    # set samples and size of dataset
    workload_to_dataset_size = {"enron": 1000, "real-estate": 100, "biofabric": 11}
    dataset_size = workload_to_dataset_size[workload]
    num_samples = int(0.05 * dataset_size) if workload != "biofabric" else 1

    # run sentinels
    start_time = time.time()
    output = run_sentinel_plans(workload, num_samples, policy_str=policy_str, parallel=parallel)
    total_sentinel_cost, _, all_cost_estimate_data, sentinel_records = output

    # # get cost estimates given current candidate plans
    # for plan_idx in range(num_plans):
    #     totalTimeInitEst, totalCostInitEst, qualityInitEst, plan, fullPlanCostEst = candidatePlans[plan_idx]

    #     models = get_models_from_physical_plan(plan)
    #     result_dict = {
    #         "plan_idx": plan_idx,
    #         "plan_label": compute_label(plan, plan_idx),
    #         "runtime": totalTimeInitEst,
    #         "cost": totalCostInitEst,
    #         "quality": qualityInitEst,
    #         "models": models,
    #         "full_plan_cost_est": fullPlanCostEst,
    #     }
    #     estimates[f"estimate_{estimate_iter}"].append(result_dict)

    # with open(f"final-eval-results/reoptimization/{opt}/{workload}/estimates.json", 'w') as f:
    #     estimates = dict(estimates)
    #     json.dump(estimates, f)

    # create new plan candidates based on current estimate data
    logicalTree = get_logical_tree(workload, nocache=True, scan_start_idx=num_samples)
    candidatePlans = logicalTree.createPhysicalPlanCandidates(
        cost_estimate_sample_data=all_cost_estimate_data,
        allow_model_selection=True,
        allow_codegen=True,
        allow_token_reduction=True,
        pareto_optimal=True,
        shouldProfile=True,
    )

    # choose best plan and execute it
    (_, _, _, plan, _), plan_idx = policy.choose(candidatePlans, return_idx=True)

    # display the plan output
    print("----------------------")
    ops = plan.dumpPhysicalTree()
    flatten_ops = flatten_nested_tuples(ops)
    print(f"Final Plan:")
    graphicEmit(flatten_ops)
    print("---")

    # run the plan
    new_records = [r for r in plan]
    runtime = time.time() - start_time

    # parse new_records
    new_records = [
        {
            key: record.__dict__[key]
            for key in record.__dict__
            if not key.startswith('_') and key not in ["image_contents"]
        }
        for record in new_records
    ]
    all_records = sentinel_records + new_records

    # get profiling data for plan and compute its cost
    profileData = plan.getProfilingData()
    sp = StatsProcessor(profileData)

    # workaround to disabling cache: delete all cached generations after each plan
    dspy_cache_dir = os.path.join(os.path.expanduser("~"), "cachedir_joblib/joblib/dsp/")
    if os.path.exists(dspy_cache_dir):
        shutil.rmtree(dspy_cache_dir)

    plan_info = {
        "plan_idx": None,
        "plan_label": compute_label(plan, plan_idx),
        "models": [],
        "op_names": [],
        "generated_fields": [],
        "query_strategies": [],
        "token_budgets": []
    }
    cost = total_sentinel_cost
    stats = sp.profiling_data
    while stats is not None:
        cost += stats.total_usd
        plan_info["models"].append(stats.model_name)
        plan_info["op_names"].append(stats.op_name)
        plan_info["generated_fields"].append(stats.generated_fields)
        plan_info["query_strategies"].append(stats.query_strategy)
        plan_info["token_budgets"].append(stats.token_budget)
        stats = stats.source_op_stats

    # score plan
    f1_score = score_plan(workload, all_records, None, policy_str=policy_str, reopt=True)

    # construct and return result_dict
    result_dict = {
        "runtime": runtime,
        "cost": cost,
        "f1_score": f1_score,
        "plan_info": plan_info,
    }

    fp = (
        f"final-eval-results/reoptimization/{workload}/{policy_str}.json"
        if not parallel
        else f"final-eval-results/reoptimization/{workload}/parallel-{policy_str}.json"
    )

    with open(fp, 'w') as f:
        json.dump(result_dict, f)


if __name__ == "__main__":
    # parse arguments
    startTime = time.time()
    parser = argparse.ArgumentParser(description='Run the evaluation(s) for the paper')
    parser.add_argument('--workload', type=str, help='The workload: one of ["biofabric", "enron", "real-estate"]')
    parser.add_argument('--opt' , type=str, help='The optimization: one of ["model", "codegen", "token-reduction"]')
    parser.add_argument('--listings-dir', default="testdata/real-estate-eval-100", type=str, help='The directory with real-estate listings')
    parser.add_argument('--reoptimize', default=False, action='store_true', help='Run reoptimization')
    parser.add_argument('--policy', type=str, help="One of 'user', 'mincost', 'mintime', 'maxquality', 'harmonicmean'")
    parser.add_argument('--dry-run', default=False, action='store_true', help='Just print plans w/out actually running any')
    parser.add_argument('--parallel', default=False, action='store_true', help='DOES NOT TURN ON PARALLELISM; simply a way for user to route output files to diff. name')

    args = parser.parse_args()

    # register real-estate workload if necessary
    if args.workload == "real-estate":
        print("Registering Datasource")
        pz.DataDirectory().registerUserSource(RealEstateListingSource(args.workload, args.listings_dir), args.workload)

    # re-optimization is unique enough to warrant its own code path
    if args.reoptimize:
        os.makedirs(f"final-eval-results/reoptimization/{args.workload}", exist_ok=True)
        run_reoptimize_eval(args.workload, args.policy, args.parallel)
        exit(1)

    # create directory for intermediate results
    # os.makedirs(f"final-eval-results/{args.opt}/{args.workload}", exist_ok=True)
    os.makedirs(f"final-eval-results/{args.workload}", exist_ok=True)

    # The user has to indicate the evaluation to be run
    if args.workload is None:
        print("Please provide a workload (--workload)")
        exit(1)

    # get PZ plan metrics
    print("Running PZ Plans")
    print("----------------")
    num_plans = evaluate_pz_plans(args.workload, args.dry_run)

    if args.dry_run:
        exit(1)

    # # read results file(s) generated by evaluate_pz_plans
    # results = []
    # for plan_idx in range(num_plans):
    #     if args.workload == "real-estate" and args.opt in ["model", "token-reduction"] and plan_idx == 9:
    #         continue

    #     if args.workload == "biofabric" and args.opt == "codegen" and plan_idx == 3:
    #         continue

    #     with open(f"final-eval-results/{args.opt}/{args.workload}/results-{plan_idx}.json", 'r') as f:
    #         result = json.load(f)
    #         results.append((plan_idx, result))

    # plot_runtime_cost_vs_quality(results, args.opt, args.workload)
