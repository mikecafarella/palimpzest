import gradio as gr
import pandas as pd

from tabulate import tabulate
from palimpzest.policy import Policy, UserChoice
from palimpzest.sets import Set

class Runner:
    """Convenience class for running and visualizing PZ programs"""
    def __init__(self, policy: Policy = None, verbose: bool = False):
        self.policy = policy
        self.verbose = verbose

    def printTable(self, records, cols=None, gradio=False, plan=None):
        def buildNestedStr(node, indent=0, buildStr=""):
            elt, child = node
            indentation = " " * indent
            buildStr =  f"{indentation}{elt}" if indent == 0 else buildStr + f"\n{indentation}{elt}"
            if child is not None:
                return buildNestedStr(child, indent=indent+2, buildStr=buildStr)
            else:
                return buildStr

        records = [
            {
                key: record.__dict__[key]
                for key in record.__dict__
                if not key.startswith('_')
            }
            for record in records
        ]
        records_df = pd.DataFrame(records)
        print_cols = records_df.columns if cols is None else cols

        if not gradio:
            print(tabulate(records_df[print_cols], headers="keys", tablefmt='psql'))

        else:
            with gr.Blocks() as demo:
                gr.Dataframe(records_df[print_cols])

                if plan is not None:
                    plan_str = buildNestedStr(plan.dumpPhysicalTree())
                    gr.Textbox(value=plan_str, info="Query Plan")

            demo.launch()

    def execute(self, s: Set, title="Dataset"):
        def emitNestedTuple(node, indent=0):
            elt, child = node
            print(" " * indent, elt)
            if child is not None:
                emitNestedTuple(child, indent=indent+2)

        # Print the syntactic tree
        # syntacticElements = rootSet.dumpSyntacticTree()
        # emitNestedTuple(syntacticElements)
        logicalTree = s.getLogicalTree()

        # Print the (possibly optimized) logical tree
        #logicalElements = logicalTree.dumpLogicalTree()
        #emitNestedTuple(logicalElements)

        # Generate candidate physical plans
        candidatePlans = logicalTree.createPhysicalPlanCandidates()    

        # print out plans to the user if it is their choice and it's verbose
        if self.verbose:
            if self.policy is not None and isinstance(self.policy, UserChoice):
                print("----------")
                for idx, cp in enumerate(candidatePlans):
                    print(f"Plan {idx}: Time est: {cp[0]:.3f} -- Cost est: {cp[1]:.3f} -- Quality est: {cp[2]:.3f}")
                    print("Physical operator tree")
                    physicalOps = cp[3].dumpPhysicalTree()
                    emitNestedTuple(physicalOps)
                    print("----------")

        # have policy select the candidate plan to execute
        planTime, planCost, quality, physicalTree = self.policy.choose(candidatePlans)
        if self.verbose:
            print("----------")
            print(f"Policy is: {str(self.policy)}")
            print(f"Chose plan: Time est: {planTime:.3f} -- Cost est: {planCost:.3f} -- Quality est: {quality:.3f}")
            emitNestedTuple(physicalTree.dumpPhysicalTree())
        return physicalTree
    