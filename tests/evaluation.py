#!/usr/bin/env python3
from palimpzest.profiler import Profiler, StatsProcessor
import palimpzest as pz

from tabulate import tabulate
from PIL import Image

from palimpzest.constants import Model
from palimpzest.execution import Execution
from palimpzest.elements import DataRecord, GroupBySig

import matplotlib.pyplot as plt
import pandas as pd

import argparse
import json
import shutil
import subprocess
import time
import os


class Email(pz.TextFile):
    """Represents an email, which in practice is usually from a text file"""
    sender = pz.Field(desc="The email address of the sender", required=True)
    subject = pz.Field(desc="The subject of the email", required=True)
    # to = pz.ListField(element_type=pz.StringField, desc="The email address(es) of the recipient(s)", required=True)
    # cced = pz.ListField(element_type=pz.StringField, desc="The email address(es) CC'ed on the email", required=True)

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

class CodeGenEasyTextRealEstateListing(RealEstateListingFiles):
    """Represents a real estate listing with specific fields extracted from its text."""
    address = pz.StringField(desc="The address of the property")
    price = pz.NumericField(desc="The listed price of the property")
    sq_ft = pz.NumericField(desc="The square footage (sq. ft.) of the property")
    bedrooms = pz.NumericField(desc="The number of bedrooms")
    bathrooms = pz.NumericField(desc="The number of bathrooms")

class CodeGenHardTextRealEstateListing(RealEstateListingFiles):
    """Represents a real estate listing with specific fields extracted from its text."""
    has_walk_in_closet = pz.BooleanField(desc="True if the property has a walk-in closet and False otherwise")
    garage_spaces = pz.NumericField(desc="The number of garage spaces the property has")
    has_city_view = pz.BooleanField(desc="True if the propery has a view of the city and False otherwise")

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

    return models


def score_plan(datasetid, records, idx, size) -> float:
    """
    Computes the F1 score of the plan
    """
    # parse records
    records = [
        {
            key: record.__dict__[key]
            for key in record.__dict__
            if not key.startswith('_')
        }
        for record in records
    ]
    records_df = pd.DataFrame(records)
    if records_df.empty:
        return 0.0, 0.0, 0.0

    records_df.to_csv(f'eval-results/{datasetid}-preds-{idx}.csv', index=False)

    preds = None
    if "enron" in datasetid:
        preds = records_df.filename.apply(lambda fn: os.path.basename(fn)).tolist()
    elif "real-estate" in datasetid:
        preds = list(records_df.listing)
    elif "token-reduction" in datasetid:
        preds = list(records_df.listing)
    elif "codegen" in datasetid:
        preds = list(records_df.listing)

    # get groundtruth
    gt_df = None
    if datasetid == "enron-eval":
        gt_df = pd.read_csv("testdata/groundtruth/enron-eval.csv")
    elif datasetid == "enron-eval-tiny":
        gt_df = pd.read_csv("testdata/groundtruth/enron-eval-tiny.csv")
    elif "real-estate" in datasetid:
        gt_df = pd.read_csv("testdata/groundtruth/real-estate-eval.csv")
    elif "token-reduction" in datasetid:
        gt_df = pd.read_csv("testdata/groundtruth/real-estate-eval.csv")
    elif "codegen-easy" in datasetid:
        gt_df = pd.read_csv(f"testdata/groundtruth/codegen-easy-eval-{size}.csv")
    elif "codegen-hard" in datasetid:
        gt_df = pd.read_csv("testdata/groundtruth/codegen-hard-eval.csv")

    targets = None
    if "enron" in datasetid:
        targets = list(gt_df[gt_df.label == 1].filename)
    elif "real-estate" in datasetid:
        targets = list(gt_df[gt_df.label == 1].listing)
    elif "token-reduction" in datasetid:
        targets = list(gt_df[gt_df.label == 1].listing)
    elif "codegen-easy" in datasetid:
        targets = list(gt_df[gt_df.label == 1].listing)
    elif "codegen-hard" in datasetid:
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

    return precision, recall, f1_score


def run_pz_plan(datasetid, plan, idx, size=None):
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
    _, _, f1_score = score_plan(datasetid, records, idx, size)

    plan_info = {"models": [], "op_names": [], "generated_fields": [], "query_strategies": []}
    cost = 0.0
    stats = sp.profiling_data
    while stats is not None:
        cost += stats.total_usd
        plan_info["models"].append(stats.model_name)
        plan_info["op_names"].append(stats.op_name)
        plan_info["generated_fields"].append(stats.generated_fields)
        plan_info["query_strategies"].append(stats.query_strategy)
        stats = stats.source_op_stats

    # compute label
    print(f"PLAN {idx}: {buildNestedStr(plan.dumpPhysicalTree())}")

    return runtime, cost, f1_score, plan_info


def evaluate_pz_plans(dataset_ids, limit=None):
    """
    This creates the PZ set of plans for the Enron email evaluation.

    Make sure to pre-register the dataset(s) with:

    $ pz reg --path testdata/enron-eval --name enron-eval

    (Note that the real-estate dataset is registered dynamically.)
    """
    # turn off DSPy cache
    os.environ["DSP_CACHEBOOL"] = "FALSE"

    # initialize list of results to return
    all_results = []

    # TODO: we can expand these datasets, but they're good enough for now
    for datasetid in dataset_ids:
        size = None if "codegen" not in datasetid else int(datasetid.split("-")[-1])

        logicalTree = None
        if "enron" in datasetid:
            emails = pz.Dataset(datasetid, schema=Email)
            emails = emails.filterByStr("The email refers to a fraudulent scheme (i.e., \"Raptor\", \"Deathstar\", \"Chewco\", and/or \"Fat Boy\")")
            # emails = emails.filterByStr("The email is sent by Jeffrey Skilling (jeff.skilling@enron.com), or Andy Fastow (andy.fastow@enron.com), or refers to either one of them by name")
            emails = emails.filterByStr("The email is not quoting from a news article or an article written by someone outside of Enron")
            logicalTree = emails.getLogicalTree()

        elif "real-estate" in datasetid:
            def within_two_miles_of_mit(record):
                # NOTE: I'm using this hard-coded function so that folks w/out a
                #       Geocoding API key from google can still run this example
                try:
                    far_away_addrs = ["Melcher St", "Sleeper St", "437 D St", "Seaport", "Liberty"]
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

            listings = pz.Dataset(datasetid, schema=RealEstateListingFiles)
            listings = listings.convert(TextRealEstateListing, depends_on="text_content")
            listings = listings.convert(ImageRealEstateListing, image_conversion=True, depends_on="image_contents")
            listings = listings.filterByStr(
                "The interior is modern and attractive, and has lots of natural sunlight",
                depends_on=["is_modern_and_attractive", "has_natural_sunlight"]
            )
            listings = listings.filterByFn(within_two_miles_of_mit, depends_on="address")
            listings = listings.filterByFn(in_price_range, depends_on="price")
            logicalTree = listings.getLogicalTree()

        elif "token-reduction" in datasetid:
            def within_two_miles_of_mit(record):
                # NOTE: I'm using this hard-coded function so that folks w/out a
                #       Geocoding API key from google can still run this example
                try:
                    far_away_addrs = ["Melcher St", "Sleeper St", "437 D St", "Seaport", "Liberty"]
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

            listings = pz.Dataset(datasetid, schema=RealEstateListingFiles)
            listings = listings.convert(TextRealEstateListing, depends_on="text_content")
            listings = listings.convert(ImageRealEstateListing, image_conversion=True, depends_on="image_contents")
            listings = listings.filterByStr(
                "The interior is modern and attractive, and has lots of natural sunlight",
                depends_on=["is_modern_and_attractive", "has_natural_sunlight"]
            )
            listings = listings.filterByFn(within_two_miles_of_mit, depends_on="address")
            listings = listings.filterByFn(in_price_range, depends_on="price")
            logicalTree = listings.getLogicalTree()

        elif "codegen-easy" in datasetid:
            # address
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

            # price
            def in_price_range(record):
                try:
                    price = record.price
                    if type(price) == str:
                        price = price.strip()
                        price = int(price.replace("$","").replace(",",""))
                    return price <= 3e6
                except:
                    return False

            # sq_ft
            def big_enough(record):
                try:
                    sq_ft = record.sq_ft
                    if type(sq_ft) == str:
                        sq_ft = sq_ft.strip()
                        sq_ft = int(sq_ft.replace(",",""))
                    return sq_ft > 550
                except:
                    return False

            # bedrooms
            def one_bed(record):
                try:
                    bedrooms = record.bedrooms
                    if type(bedrooms) == str:
                        bedrooms = float(bedrooms.strip())
                    return bedrooms == 1
                except:
                    return False

            listings = pz.Dataset(datasetid, schema=RealEstateListingFiles, nocache=True)
            listings = listings.convert(CodeGenEasyTextRealEstateListing, depends_on="text_content")
            listings = listings.filterByFn(within_two_miles_of_mit)
            listings = listings.filterByFn(in_price_range)
            listings = listings.filterByFn(big_enough)
            listings = listings.filterByFn(one_bed)
            logicalTree = listings.getLogicalTree()

        elif "codegen-hard" in datasetid:
            def has_walk_in_closet(record):
                try:
                    has_walk_in_closet = record.has_walk_in_closet
                    if type(has_walk_in_closet) == str:
                        has_walk_in_closet = float(has_walk_in_closet.strip())
                    return has_walk_in_closet >= 0.01
                except:
                    return False

            # garage space
            def one_garage_space(record):
                try:
                    garage_spaces = record.garage_spaces
                    if type(garage_spaces) == str:
                        garage_spaces = int(garage_spaces.strip())
                    return garage_spaces == 1
                except:
                    return False

            # has deck or not
            def has_city_view(record):
                try:
                    has_city_view = record.has_city_view
                    if type(has_city_view) == str:
                        has_city_view = True if has_city_view.lower() == "true" else False
                    return has_city_view
                except:
                    return False

            listings = pz.Dataset(datasetid, schema=RealEstateListingFiles, nocache=True)
            listings = listings.convert(CodeGenHardTextRealEstateListing, depends_on="text_content")
            listings = listings.filterByFn(has_walk_in_closet)
            listings = listings.filterByFn(one_garage_space)
            listings = listings.filterByFn(has_city_view)
            logicalTree = listings.getLogicalTree()

        # NOTE: the following weird iteration over physical plans by idx is intentional and necessary
        #       at the moment in order for stats collection to work properly. For some yet-to-be-discovered
        #       reason, `createPhysicalPlanCandidates` is creating physical plans which share the same
        #       copy of some operators. This means that if we naively iterate over the plans and execute them
        #       some plans' profilers will count 2x (or 3x or 4x etc.) the number of records processed,
        #       dollars spent, time spent, etc. This workaround recreates the physical plans on each
        #       iteration to ensure that they are new.
    
        # get total number of plans
        allow_codegen = "codegen" in datasetid
        allow_token_reduction = "token-reduction" in datasetid
        num_plans = len(logicalTree.createPhysicalPlanCandidates(max=limit, allow_codegen=allow_codegen, allow_token_reduction=allow_token_reduction, shouldProfile=True))

        # remove codegen samples from previous dataset from cache
        cache = pz.DataDirectory().getCacheService()
        cache.rmCachedData("codeEnsemble")
        cache.rmCachedData("codeSamples")

        results = []
        for idx in range(num_plans):
        # for idx, (totalTimeInitEst, totalCostInitEst, qualityInitEst, plan) in enumerate(candidatePlans):
            # skip all-Gemini plan which opens too many files
            # if "enron" in datasetid and idx == 17:
            #     continue

            # TODO: for now, re-create candidate plans until we debug duplicate profiler issue
            candidatePlans = logicalTree.createPhysicalPlanCandidates(max=limit, allow_codegen=allow_codegen, shouldProfile=True)
            _, _, _, plan, _ = candidatePlans[idx]

            # workaround to disabling cache: delete all cached generations after each plan
            bad_files = ["testdata/enron-eval/assertion.log", "testdata/enron-eval/azure_openai_usage.log", "testdata/enron-eval/openai_usage.log"]
            for file in bad_files:
                if os.path.exists(file):
                    os.remove(file)

            print("----------------------")
            print(f"Plan: {buildNestedStr(plan.dumpPhysicalTree())}")
            print("---")
            runtime, cost, f1_score, plan_info = run_pz_plan(datasetid, plan, idx, size)

            # add to results
            result_dict = {"runtime": runtime, "cost": cost, "f1_score": f1_score, "plan_info": plan_info}
            results.append(result_dict)
            with open(f'eval-results/{datasetid}-results-{idx}.json', 'w') as f:
                json.dump(result_dict, f)

            # workaround to disabling cache: delete all cached generations after each plan
            dspy_cache_dir = os.path.join(os.path.expanduser("~"), "cachedir_joblib/joblib/dsp/")
            if os.path.exists(dspy_cache_dir):
                shutil.rmtree(dspy_cache_dir)

        # add results for this dataset
        all_results.append(results)

    return num_plans, all_results


def plot_runtime_cost_vs_quality(all_results, datasetid):
    # create figure
    fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True)

    # NOTE: for now we only use this fcn. for enron and real-estate evals which have one dataset
    # parse results into fields
    results = all_results[0]
    for idx, result_dict in enumerate(results):
        runtime = result_dict["runtime"]
        cost = result_dict["cost"]
        f1_score = result_dict["f1_score"]
        models = (
            result_dict["models"]
            if "models" in result_dict
            else result_dict["plan_info"]["models"]
        )
        op_names = (
            result_dict["plan_info"]["op_names"]
            if "plan_info" in result_dict
            else None
        )
        generated_fields = (
            result_dict["plan_info"]["generated_fields"]
            if "plan_info" in result_dict
            else None
        )

        if "enron" in datasetid:
            # hack adjustment: original enron evaluation script only took models for LLM operators;
            # we do this by taking the first three elements in the list
            models = models[:3]
 
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

        elif "real-estate" in datasetid:
            text = ""
            if all([model is None or "gpt-4" in model for model in models]):
                # add text for ALL-GPT4
                text = "ALL-GPT4"
            elif any([model is not None and "mistralai" in model for model in models]):
                text = "MIXTRAL-GPT4"
            elif any([model is not None and "gemini" in model for model in models]):
                text = "GEMINI-GPT4"

            all_convert_then_filter, text_then_image, image_then_text = True, False, False
            num_converts = 0
            for op_name, gen_fields in zip(list(reversed(op_names)), list(reversed(generated_fields))):
                if "induce" not in op_name and "filter" not in op_name:
                    continue

                if "induce" in op_name:
                    num_converts += 1

                    if num_converts == 1 and "address" in gen_fields:
                        text_then_image = True
                    elif num_converts == 1 and "has_natural_sunlight" in gen_fields:
                        image_then_text = True

                if "filter" in op_name and num_converts < 2:
                    all_convert_then_filter = False

            # add text depending on whether all converts happen before filters
            # and whether images or text are processed first
            if all_convert_then_filter:
                text += "-CONVERT-BOTH"
            elif text_then_image:
                text += "-TEXT-FIRST"
            elif image_then_text:
                text += "-IMAGE-FIRST"

        # set label and color
        color = None
        marker = None

        # plot runtime vs. f1_score
        axs[0].scatter(f1_score, runtime, alpha=0.4, color=color, marker=marker) 

        # plot cost vs. f1_score
        axs[1].scatter(f1_score, cost, alpha=0.4, color=color, marker=marker)

        # add annotations
        if text is not None:
            if "enron" in datasetid:
                ha, va = 'right', None
                if text == "ALL-GPT4":
                    va = 'top'
                elif text == "MIXTRAL-GPT4":
                    va = 'bottom'
                elif text == "GEMINI-GPT4":
                    va = 'top'
                elif text == "ALL-MIXTRAL":
                    va = 'bottom'
                axs[0].annotate(text, (f1_score, runtime), ha=ha, va=va)
                axs[1].annotate(text, (f1_score, cost), ha=ha, va=va)

            elif "real-estate" in datasetid and (f1_score > 0.8 or runtime < 100):
                ha = 'left' if f1_score < 0.8 else 'right'
                runtime_va = 'bottom' if runtime < 100 else 'top'
                cost_va = 'bottom' if f1_score < 0.8 else 'top'
                axs[0].annotate(text, (f1_score, runtime), ha=ha, va=runtime_va)
                axs[1].annotate(text, (f1_score, cost), ha=ha, va=cost_va)

    # savefig
    axs[0].set_title("Runtime and Cost vs. F1 Score")
    axs[0].set_ylabel("Runtime (seconds)")
    axs[1].set_ylabel("Cost (USD)")
    axs[1].set_xlabel("F1 Score")
    # axs[0].legend(bbox_to_anchor=(1.03, 1.0))
    fig_name = f"eval-results/{datasetid}.png"
    fig.savefig(fig_name, bbox_inches="tight")


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
    fig.savefig(f"eval-results/{plot_filename}.png", bbox_inches="tight")


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
    parser.add_argument('--datasetid', type=str, help='The dataset id')
    parser.add_argument('--eval' , type=str, help='The evaluation to run')
    parser.add_argument('--limit' , type=int, help='The number of plans to consider')
    parser.add_argument('--size' , type=int, help='The dataset size to run the evaluation for (only for codegen)')
    parser.add_argument('--listings-dir', type=str, help='The directory with real-estate listings')

    args = parser.parse_args()

    # create directory for intermediate results
    os.makedirs("eval-results", exist_ok=True)

    # The user has to indicate the evaluation to be run
    if args.eval is None:
        print("Please provide an evaluation")
        exit(1)

    # re-optimization is unique enough to warrant its own code path
    if args.eval == "reoptimize":
        run_reoptimize_eval(args.datasetid)
        exit(1)

    dataset_ids = []
    if args.eval == "enron":
        dataset_ids.append(args.datasetid)

    elif args.eval == "real-estate":
        # register user data source
        print("Registering Datasource")
        pz.DataDirectory().registerUserSource(RealEstateListingSource(args.datasetid, args.listings_dir), args.datasetid)
        dataset_ids.append(args.datasetid)

    elif args.eval == "token-reduction":
        # register user data source
        print("Registering Datasource")
        pz.DataDirectory().registerUserSource(RealEstateListingSource(args.datasetid, args.listings_dir), args.datasetid)
        dataset_ids.append(args.datasetid)

    elif args.eval == "codegen-easy":
        # register user data sources
        print("Registering Datasources")
        # for size in [5, 10, 15, 20, 25, 30]:
        size = args.size
        datasetid = f"{args.datasetid}-{size}"
        listings_dir = f"testdata/real-estate-eval-{size}"
        pz.DataDirectory().registerUserSource(RealEstateListingSource(datasetid, listings_dir), datasetid)
        dataset_ids.append(datasetid)

    # get PZ plan metrics
    print("Running PZ Plans")
    print("----------------")
    num_plans, _ = evaluate_pz_plans(dataset_ids, limit=args.limit)

    if args.eval == "codegen-easy":
        dataset_ids = []
        for size in [5, 10, 15, 20, 25, 30]:
            datasetid = f"{args.datasetid}-{size}"
            dataset_ids.append(datasetid)

    all_results = []
    for datasetid in dataset_ids:
        results = []
        for plan_idx in range(num_plans):
            # skip gemini plan for codegen plot b/c we don't have cost for it
            if args.eval == "codegen-easy" and plan_idx == 3:
                continue

            with open(f"eval-results/{datasetid}-results-{plan_idx}.json", 'r') as f:
                result = json.load(f)
                results.append(result)

        all_results.append(results)

    if args.eval in ["enron", "real-estate"]:
        plot_runtime_cost_vs_quality(all_results, args.datasetid)

    elif "codegen" in args.eval:
        plot_runtime_vs_dataset_size(all_results, plot_filename=f"{args.eval}-eval")
