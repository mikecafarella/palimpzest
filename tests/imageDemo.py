#!/usr/bin/env python3
""" This scripts is a demo for image processing, it is simply an abridged version of simpleDemo.py """
import context
from palimpzest.constants import PZ_DIR
import palimpzest as pz

from tabulate import tabulate
from PIL import Image

import gradio as gr
import numpy as np
import pandas as pd

import argparse
import requests
import json
import time
import os

class DogImage(pz.ImageFile):
    breed = pz.Field(desc="The breed of the dog", required = True)

def buildImagePlan(datasetId):
    images = pz.Dataset(datasetId, schema=pz.ImageFile)
    filteredImages = images.filterByStr("The image contains one or more dogs")
    dogImages = filteredImages.convert(DogImage, desc = "Images of dogs")
    return dogImages


def buildNestedStr(node, indent=0, buildStr=""):
    elt, child = node
    indentation = " " * indent
    buildStr =  f"{indentation}{elt}" if indent == 0 else buildStr + f"\n{indentation}{elt}"
    if child is not None:
        return buildNestedStr(child, indent=indent+2, buildStr=buildStr)
    else:
        return buildStr

def emitDataset(rootSet, policy, title="Dataset", verbose=False):
    def emitNestedTuple(node, indent=0):
        elt, child = node
        print(" " * indent, elt)
        if child is not None:
            emitNestedTuple(child, indent=indent+2)

    syntacticElements = rootSet.dumpSyntacticTree()
    # print("Syntactic operator tree")
    # emitNestedTuple(syntacticElements)

    # Print the (possibly optimized) logical tree
    logicalTree = rootSet.getLogicalTree()
    logicalElements = logicalTree.dumpLogicalTree()
    #print("Logical operator tree")
    #emitNestedTuple(logicalElements)

    # Generate candidate physical plans
    candidatePlans = logicalTree.createPhysicalPlanCandidates()

    for idx, cp in enumerate(candidatePlans):
        print("----------")
        print(f"Plan {idx}: Time est: {cp[0]:.3f} -- Cost est: {cp[1]:.3f} -- Quality est: {cp[2]:.3f}")
        print("Physical operator tree")
        physicalOps = cp[3].dumpPhysicalTree()
        emitNestedTuple(physicalOps)

    # have policy select the candidate plan to execute
    planTime, planCost, quality, physicalTree = policy.choose(candidatePlans)
    # planTime, planCost, quality, physicalTree =  candidatePlans[-1]
    print("----------")
    print(f"Policy is: {str(policy)}")
    print(f"Chose plan: Time est: {planTime:.3f} -- Cost est: {planCost:.3f} -- Quality est: {quality:.3f}")
    emitNestedTuple(physicalTree.dumpPhysicalTree())
    return physicalTree

#
# Get battery papers and emit!
#
if __name__ == "__main__":
    # parse arguments
    startTime = time.time()
    parser = argparse.ArgumentParser(description='Run a simple demo')
    parser.add_argument('--no-cache', action='store_true', help='Do not use cached results')

    args = parser.parse_args()
    no_cache = args.no_cache
    if no_cache:
        print("WARNING: Removing cache, this will result in API calls")
        cachedir = PZ_DIR + "/data/cache/"
        for f in os.listdir(cachedir):
            if f.endswith(".cached"):
                os.remove(os.path.join(cachedir, f))

    verbose = True
    datasetid = "images-tiny"
    policy = pz.UserChoice()

    if os.getenv('OPENAI_API_KEY') is None and os.getenv('TOGETHER_API_KEY') is None:
        print("WARNING: Both OPENAI_API_KEY and TOGETHER_API_KEY are unset")

    print("Starting image task")
    rootSet = buildImagePlan(datasetid)
    physicalTree = emitDataset(rootSet, policy, title="Dogs", verbose=verbose)
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

    endTime = time.time()
    print("Elapsed time:", endTime - startTime)
    demo.launch()
