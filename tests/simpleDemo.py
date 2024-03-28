#!/usr/bin/env python3
from palimpzest.tools.profiler import Profiler
from palimpzest.tools.runner import Runner

import palimpzest as pz
from PIL import Image

import numpy as np
import gradio as gr

import argparse
import requests
import json
import time
import os
import csv

class ScientificPaper(pz.PDFFile):
   """Represents a scientific research paper, which in practice is usually from a PDF file"""
   title = pz.Field(desc="The title of the paper. This is a natural language title, not a number or letter.", required=True)
   publicationYear = pz.Field(desc="The year the paper was published. This is a number.", required=False)
   author = pz.Field(desc="The name of the first author of the paper", required=True)
   institution = pz.Field(desc="The institution of the first author of the paper", required=True)
   journal = pz.Field(desc="The name of the journal the paper was published in", required=True)
   fundingAgency = pz.Field(desc="The name of the funding agency that supported the research", required=False)

def buildSciPaperPlan(datasetId):
    """A dataset-independent declarative description of authors of good papers"""
    return pz.Dataset(datasetId, schema=ScientificPaper)


def buildMITBatteryPaperPlan(datasetId):
    """A dataset-independent declarative description of authors of good papers"""
    sciPapers = pz.Dataset(datasetId, schema=ScientificPaper)
    batteryPapers = sciPapers.filterByStr("The paper is about batteries")
    mitPapers = batteryPapers.filterByStr("The paper is from MIT")

    return mitPapers

class VLDBPaperListing(pz.Schema):
    """VLDBPaperListing represents a single paper from the VLDB conference"""
    title = pz.Field(desc="The title of the paper", required=True)
    authors = pz.Field(desc="The authors of the paper", required=True)
    pdfLink = pz.Field(desc="The link to the PDF of the paper", required=True)

def downloadVLDBPapers(vldbListingPageURLsId, outputDir):
    """ This function downloads a bunch of VLDB papers from an online listing and saves them to disk.  It also saves a CSV file of the paper listings."""
    runner = Runner(pz.MaxQuality(), verbose=True)

    # 1. Grab the input VLDB listing page(s) and scrape them for paper metadata
    tfs = pz.Dataset(vldbListingPageURLsId, schema=pz.TextFile, desc="A file full of URLs of VLDB journal pages")
    urls = tfs.convert(pz.URL, desc="The actual URLs of the VLDB pages", cardinality="oneToMany")   
    htmlContent = urls.map(pz.DownloadHTMLFunction())
    vldbPaperListings = htmlContent.convert(VLDBPaperListing, desc="The actual listings for each VLDB paper", cardinality="oneToMany")

    # 2. Get the PDF URL for each paper that's listed and download it
    vldbPaperURLs = vldbPaperListings.convert(pz.URL, desc="The URLs of the PDFs of the VLDB papers")
    pdfContent = vldbPaperURLs.map(pz.DownloadBinaryFunction())

    # 3. Save the paper listings to a CSV file and the PDFs to disk
    if not os.path.exists(outputDir):
        os.makedirs(outputDir)
    outputPath = os.path.join(outputDir, "vldbPaperListings.csv")

    with open(outputPath, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=vldbPaperListings.schema().fieldNames())
        writer.writeheader()
        for record in runner.execute(vldbPaperListings):
            writer.writerow(record.asDict())

    for idx, r in enumerate(runner.execute(pdfContent)):
        with open(os.path.join(outputDir, str(idx) + ".pdf"), "wb") as f:
            f.write(r.content)

    runner.printTable(runner.execute(vldbPaperListings), gradio=True)


class GitHubUpdate(pz.Schema):
    """GitHubUpdate represents a single commit message from a GitHub repo"""
    commitId = pz.Field(desc="The unique identifier for the commit", required=True)
    reponame = pz.Field(desc="The name of the repository", required=True)
    commit_message = pz.Field(desc="The message associated with the commit", required=True)
    commit_date = pz.Field(desc="The date the commit was made", required=True)
    committer_name = pz.Field(desc="The name of the person who made the commit", required=True)
    file_names = pz.Field(desc="The list of files changed in the commit", required=False)

def testStreaming(datasetId: str):
    return pz.Dataset(datasetId, schema=GitHubUpdate)

def testCount(datasetId):
    files = pz.Dataset(datasetId)
    fileCount = files.aggregate("COUNT")
    return fileCount

def testAverage(datasetId):
    data = pz.Dataset(datasetId)
    average = data.aggregate("AVERAGE")
    return average

def testLimit(datasetId, n):
    data = pz.Dataset(datasetId)
    limitData = data.limit(n)
    return limitData

class Email(pz.TextFile):
    """Represents an email, which in practice is usually from a text file"""
    sender = pz.Field(desc="The email address of the sender", required=True)
    subject = pz.Field(desc="The subject of the email", required=True)

def buildEnronPlan(datasetId):
    emails = pz.Dataset(datasetId, schema=Email)
    return emails

class DogImage(pz.ImageFile):
    breed = pz.Field(desc="The breed of the dog", required = True)

def buildImagePlan(datasetId):
    images = pz.Dataset(datasetId, schema=pz.ImageFile)
    filteredImages = images.filterByStr("The image contains one or more dogs")
    dogImages = filteredImages.convert(DogImage, desc = "Images of dogs")
    return dogImages

#
# Get battery papers and emit!
#
if __name__ == "__main__":
    # parse arguments
    startTime = time.time()
    parser = argparse.ArgumentParser(description='Run a simple demo')
    parser.add_argument('--verbose', default=False, action='store_true', help='Print verbose output')
    parser.add_argument('--datasetid', type=str, help='The dataset id')
    parser.add_argument('--task' , type=str, help='The task to run')
    parser.add_argument('--policy', type=str, help="One of 'user', 'mincost', 'mintime', 'maxquality', 'harmonicmean'")

    args = parser.parse_args()

    # The user has to indicate the dataset id and the task
    if args.datasetid is None:
        print("Please provide a dataset id")
        exit(1)
    if args.task is None:
        print("Please provide a task")
        exit(1)

    datasetid = args.datasetid
    task = args.task
    policy = pz.MaxHarmonicMean()
    if args.policy is not None:
        if args.policy == "user":
            policy = pz.UserChoice()
        elif args.policy == "mincost":
            policy = pz.MinCost()
        elif args.policy == "mintime":
            policy = pz.MinTime()
        elif args.policy == "maxquality":
            policy = pz.MaxQuality()
        elif args.policy == "harmonicmean":
            policy = pz.MaxHarmonicMean()

    if os.getenv('OPENAI_API_KEY') is None and os.getenv('TOGETHER_API_KEY') is None:
        print("WARNING: Both OPENAI_API_KEY and TOGETHER_API_KEY are unset")

    runner = Runner(policy, verbose=args.verbose)

    if task == "paper":
        rootSet = buildMITBatteryPaperPlan(datasetid)
        runner.printTable(runner.execute(rootSet), gradio=True, cols=["title", "publicationYear", "author", "institution", "journal", "fundingAgency"])

    elif task == "enron":
        rootSet = buildEnronPlan(datasetid)
        runner.printTable(runner.execute(rootSet), gradio=True, cols=["sender", "subject"])

    elif task == "enronoptimize":
        rootSet = buildEnronPlan(datasetid)
        execution = pz.Execution(rootSet, policy)
        physicalTree = execution.executeAndOptimize()
        records = [r for r in physicalTree]
        runner.printTable(records, cols=["sender", "subject"], gradio=True, plan=physicalTree)

    elif task == "scitest":
        rootSet = buildSciPaperPlan(datasetid)
        runner.printTable(runner.execute(rootSet), gradio=True, cols=["title", "author", "institution", "journal", "fundingAgency"])

    elif task == "streaming":
        # register the ephemeral dataset
        datasetid = "ephemeral:githubtest"
        owner = "mikecafarella"
        repo = "palimpzest"
        url = f"https://api.github.com/repos/{owner}/{repo}/commits"
        blockTime = 5

        class GitHubCommitSource(pz.UserSource):
            def __init__(self, datasetId):
                super().__init__(pz.RawJSONObject, datasetId)

            def userImplementedIterator(self):
                per_page = 100
                params = {
                    'per_page': per_page,
                    'page': 1
                }
                while True:
                    response = requests.get(url, params=params)
                    commits = response.json()

                    if not commits or response.status_code != 200:
                        break

                    for commit in commits:
                        # Process each commit here
                        commitStr = json.dumps(commit)
                        dr = pz.DataRecord(self.schema)
                        dr.json = commitStr
                        yield dr

                    if len(commits) < per_page:
                        break

                    params['page'] += 1
                    time.sleep(1)

        pz.DataDirectory().registerUserSource(GitHubCommitSource(datasetid), datasetid)

        rootSet = testStreaming(datasetid)
        runner.printTable(runner.execute(rootSet), gradio=True, title="Streaming items")

    elif task == "image":
        def buildNestedStr(node, indent=0, buildStr=""):
            elt, child = node
            indentation = " " * indent
            buildStr =  f"{indentation}{elt}" if indent == 0 else buildStr + f"\n{indentation}{elt}"
            if child is not None:
                return buildNestedStr(child, indent=indent+2, buildStr=buildStr)
            else:
                return buildStr

        print("Starting image task")
        rootSet = buildImagePlan(datasetid)
        physicalTree = runner.execute(rootSet, title="Dogs")
        records = [r for r in physicalTree]

        print("Obtained records", records)
        imgs, breeds = [], []
        for record in records:
            print("Trying to open ", record.filename)
            img = Image.open(record.filename).resize((128,128))
            img_arr = np.asarray(img)
            imgs.append(img_arr)
            breeds.append(record.breed)

        with gr.Blocks() as demo:
            img_blocks, breed_blocks = [], []
            for img, breed in zip(imgs, breeds):
                with gr.Row():
                    with gr.Column():
                        img_blocks.append(gr.Image(value=img))
                    with gr.Column():
                        breed_blocks.append(gr.Textbox(value=breed))

            plan_str = buildNestedStr(physicalTree.dumpPhysicalTree())
            gr.Textbox(value=plan_str, info="Query Plan")

        demo.launch()

        # if profiling was turned on; capture statistics
        if Profiler.profiling_on():
            profiling_data = physicalTree.getProfilingData()

            with open('profiling.json', 'w') as f:
                json.dump(profiling_data, f)

    elif task == "vldb":
        downloadVLDBPapers(datasetid, "vldbPapers")

    elif task == "count":
        rootSet = testCount(datasetid)
        runner.printTable(runner.execute(rootSet), gradio=True)

    elif task == "average":
        rootSet = testAverage(datasetid)
        runner.printTable(runner.execute(rootSet), gradio=True)

    elif task == "limit":
        rootSet = testLimit(datasetid, 5)
        runner.printTable(runner.execute(rootSet), gradio=True)

    else:
        print("Unknown task")
        exit(1)

    endTime = time.time()
    print("Elapsed time:", endTime - startTime)
