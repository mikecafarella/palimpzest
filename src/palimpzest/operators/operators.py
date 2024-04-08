from __future__ import annotations

from palimpzest.constants import Model
from palimpzest.datamanager import DataDirectory
from palimpzest.elements import *
from palimpzest.operators import (
    ApplyCountAggregateOp,
    ApplyAverageAggregateOp,
    ApplyUserFunctionOp,
    CacheScanDataOp,
    FilterCandidateOp,
    InduceFromCandidateOp,
    LimitScanOp,
    MarshalAndScanDataOp,
    ParallelFilterCandidateOp,
    ParallelInduceFromCandidateOp,
    PhysicalOp,
    ApplyGroupByOp
)

from copy import deepcopy
from itertools import permutations
from typing import List, Tuple

import os
import random

# DEFINITIONS
PhysicalPlan = Tuple[float, float, float, PhysicalOp]


class LogicalOperator:
    """
    A logical operator is an operator that operates on Sets. Right now it can be one of:
    - BaseScan (scans data from DataSource)
    - CacheScan (scans cached Set)
    - FilteredScan (scans input Set and applies filter)
    - ConvertScan (scans input Set and converts it to new Schema)
    - LimitScan (scans up to N records from a Set)
    - ApplyAggregateFunction (applies an aggregation on the Set)
    """
    def __init__(self, outputSchema: Schema, inputSchema: Schema, inputOp: LogicalOperator=None):
        self.outputSchema = outputSchema
        self.inputSchema = inputSchema
        self.inputOp = inputOp

    def dumpLogicalTree(self) -> Tuple[LogicalOperator, LogicalOperator]:
        raise NotImplementedError("Abstract method")

    def _getPhysicalTree(self, strategy: str=None, model: Model=None) -> PhysicalOp:
        raise NotImplementedError("Abstract method")

    @staticmethod
    def _computeFilterReorderings(rootOp: LogicalOperator) -> List[LogicalOperator]:
        """
        Given the logicalOperator, compute all possible equivalent plans with the filters re-ordered.

        Right now I only consider re-ordering filters in plans w/two or more filters back-to-back.
        In theory we might be able to re-order filters w/a convert sandwiched in-between them
        if the second filter is unrelated to the conversion, but I will defer that to future work.
        """
        # base case, if this operator is a BaseScan or CacheScan, return operator
        if isinstance(rootOp, BaseScan) or isinstance(rootOp, CacheScan):
            return [rootOp]

        # if this operator is not a FilteredScan: compute the re-orderings for its inputOp,
        # point rootOp to each of the re-orderings, and return
        if not isinstance(rootOp, FilteredScan):
            subTrees = LogicalOperator._computeFilterReorderings(rootOp.inputOp)

            all_plans = []
            for tree in subTrees:
                rootOpCopy = deepcopy(rootOp)
                rootOpCopy.inputOp = tree
                all_plans.append(rootOpCopy)

            return all_plans

        # otherwise, if this operator is a FilteredScan, make one plan per permutation of filters
        # in this (potential) chain of filters and recurse
        else:
            # use while loop to get all consecutive filtered scans
            filterOps = [rootOp]
            nextOp = rootOp.inputOp
            while isinstance(nextOp, FilteredScan):
                filterOps.append(nextOp)
                nextOp = nextOp.inputOp

            # get the final nextOp
            finalInputOp = nextOp

            # compute all permutations of operators and make new copies of Ops
            opPermutations = permutations(filterOps)
            opPermutations = [[deepcopy(op) for op in ops] for ops in opPermutations]

            # iterate over each permutation
            for ops in opPermutations:
                # manually link up filter operations for this permutation
                for idx, op in enumerate(ops):
                    op.inputOp = ops[idx + 1] if idx + 1 < len(filterOps) else None

            # compute filter reorderings for rest of tree
            subTrees = LogicalOperator._computeFilterReorderings(finalInputOp)

            # compute cross-product of opPermutations and subTrees by linking final op w/first op in subTree
            for ops in opPermutations:
                for tree in subTrees:
                    ops[-1].inputOp = tree

            # return roots of opPermutations
            return list(map(lambda ops: ops[0], opPermutations))

    def _createLogicalPlans(self, max: int=10) -> List[LogicalOperator]:
        """
        Given the logical plan implied by this LogicalOperator, enumerate up to `max`
        other logical plans (including this one) and return the list.
        """
        logicalPlans = []

        # enumerate filter orderings
        filterReorderedPlans = LogicalOperator._computeFilterReorderings(self)
        logicalPlans.extend(filterReorderedPlans)

        # sample up to max logicalPlans (or all plans if there are fewer than max) 
        logicalPlans = random.sample(logicalPlans, min(max, len(logicalPlans)))

        return logicalPlans

    def _createPhysicalPlans(self, max: int=10, shouldProfile: bool=False) -> PhysicalOp:
        """
        Given the logical plan implied by this LogicalOperator, enumerate up to `max`
        possible physical plans and return them as a list.
        """
        # TODO:
        # 1. do something better than random search to create candidate plans
        # 2. we want to design our soln. to try these w/sampling to help determine
        #    the most effective plan(s) and better est. the accuracy tradeoff(s)

        # brute force/random: for each FilteredScan & ConvertScan try:
        # 1. swapping different models
        #    a. different model hyperparams? TODO
        #
        # TODO
        # 2. different prompt strategies
        #    a. Zero-Shot vs. Few-Shot vs. COT vs. DSPy
        # 3. input sub-selection
        #    a. vector DB, LLM attention, ask-the-LLM

        # choose set of acceptable models based on possible llmservices
        models = []
        if os.getenv('OPENAI_API_KEY') is not None:
            models.extend([Model.GPT_3_5, Model.GPT_4])

        if os.getenv('TOGETHER_API_KEY') is not None:
            models.extend([Model.MIXTRAL])

        # TODO: uncomment once dspy pushes v2.4.1 to PyPI
        # if os.getenv('GOOGLE_API_KEY') is not None:
        #     models.extend([Model.GEMINI_1])

        assert len(models) > 0, "No models available to create physical plans! You must set at least one of the following environment variables: [OPENAI_API_KEY, TOGETHER_API_KEY, GOOGLE_API_KEY]"

        physicalPlans = []
        for model in models:
            physicalPlan = self._getPhysicalTree(strategy=PhysicalOp.LOCAL_PLAN, model=model, shouldProfile=shouldProfile)
            physicalPlans.append(physicalPlan)

        return physicalPlans

    def createPhysicalPlanCandidates(self, cost_estimate_sample_data: List[Dict[str, Any]]=None, shouldProfile: bool=False) -> List[PhysicalPlan]:
        """Return a set of physical trees of operators."""
        # create set of logical plans (e.g. consider different filter/join orderings)
        logicalPlans = self._createLogicalPlans()

        # iterate through logical plans and evaluate multiple physical plans
        physicalPlans = [
            physicalPlan
            for logicalPlan in logicalPlans
            for physicalPlan in logicalPlan._createPhysicalPlans(shouldProfile=shouldProfile)
        ]

        # estimate the cost (in terms of USD, latency, throughput, etc.) for each plan
        plans = []
        for physicalPlan in physicalPlans:
            planCost = physicalPlan.estimateCost(cost_estimate_sample_data=cost_estimate_sample_data)

            totalTime = planCost["totalTime"]
            totalCost = planCost["totalUSD"]  # for now, cost == USD
            quality = planCost["quality"]

            plans.append((totalTime, totalCost, quality, physicalPlan))

        # drop duplicate plans in terms of time, cost, and quality, as these can cause
        # plans on the pareto frontier to be dropped if they are "dominated" by a duplicate
        dedup_plans, dedup_desc_set = [], set()
        for plan in plans:
            planDesc = (plan[0], plan[1], plan[2])
            if planDesc not in dedup_desc_set:
                dedup_desc_set.add(planDesc)
                dedup_plans.append(plan)

        # compute the pareto frontier of candidate physical plans and return the list of such plans
        # - brute force: O(d*n^2);
        #   - for every tuple, check if it is dominated by any other tuple;
        #   - if it is, throw it out; otherwise, add it to pareto frontier
        #
        # more efficient algo.'s exist, but they are non-trivial to implement, so for now I'm using
        # brute force; it may ultimately be best to compute a cheap approx. of the pareto front:
        # - e.g.: https://link.springer.com/chapter/10.1007/978-3-642-12002-2_6
        paretoFrontierPlans = []
        for i, (totalTime_i, totalCost_i, quality_i, plan) in enumerate(dedup_plans):
            paretoFrontier = True

            # check if any other plan dominates plan i
            for j, (totalTime_j, totalCost_j, quality_j, _) in enumerate(dedup_plans):
                if i == j:
                    continue

                # if plan i is dominated by plan j, set paretoFrontier = False and break
                if totalTime_j <= totalTime_i and totalCost_j <= totalCost_i and quality_j >= quality_i:
                    paretoFrontier = False
                    break

            # add plan i to pareto frontier if it's not dominated
            if paretoFrontier:
                paretoFrontierPlans.append((totalTime_i, totalCost_i, quality_i, plan))

        return paretoFrontierPlans


class ConvertScan(LogicalOperator):
    """A ConvertScan is a logical operator that represents a scan of a particular data source, with conversion applied."""
    def __init__(self, outputSchema: Schema, inputOp: LogicalOperator, cardinality: str=None, desc: str=None, targetCacheId: str=None):
        super().__init__(outputSchema, inputOp.outputSchema)
        self.inputOp = inputOp
        self.cardinality = cardinality
        self.desc = desc
        self.targetCacheId = targetCacheId

    def __str__(self):
        return "ConvertScan(" + str(self.inputSchema) + ", " + str(self.outputSchema) + ", " + str(self.desc) + ")"

    def dumpLogicalTree(self):
        """Return the logical tree of this LogicalOperator."""
        return (self, self.inputOp.dumpLogicalTree())

    def _getPhysicalTree(self, strategy: str=None, model: Model=None, shouldProfile: bool=False):
        # If the input is in core, and the output is NOT in core but its superclass is, then we should do a
        # 2-stage conversion. This will maximize chances that there is a pre-existing conversion to the superclass
        # in the known set of functions
        intermediateSchema = self.outputSchema
        while not intermediateSchema == Schema and not PhysicalOp.solver.easyConversionAvailable(intermediateSchema, self.inputSchema):
            intermediateSchema = intermediateSchema.__bases__[0]

        if intermediateSchema == Schema or intermediateSchema == self.outputSchema:
            if DataDirectory().current_config.get("parallel") == True:
                return ParallelInduceFromCandidateOp(self.outputSchema,
                                                     self.inputOp._getPhysicalTree(strategy=strategy, 
                                                                                   model=model, 
                                                                                   shouldProfile=shouldProfile), 
                                                    model, 
                                                    self.cardinality,
                                                    desc=self.desc, 
                                                    targetCacheId=self.targetCacheId, 
                                                    shouldProfile=shouldProfile)
            else:
                return InduceFromCandidateOp(self.outputSchema, 
                                             self.inputOp._getPhysicalTree(strategy=strategy, 
                                                                           model=model, 
                                                                           shouldProfile=shouldProfile), 
                                            model, 
                                            self.cardinality,
                                            desc=self.desc, 
                                            targetCacheId=self.targetCacheId, 
                                            shouldProfile=shouldProfile)
        else:
            if DataDirectory().current_config.get("parallel") == True:
                return ParallelInduceFromCandidateOp(self.outputSchema,
                                                     ParallelInduceFromCandidateOp(
                                                         intermediateSchema,
                                                         self.inputOp._getPhysicalTree(strategy=strategy, 
                                                                                       model=model, 
                                                                                       shouldProfile=shouldProfile),       
                                                         model,
                                                         self.cardinality,
                                                         shouldProfile=shouldProfile),
                                                     model,
                                                     "oneToOne",
                                                     desc=self.desc,
                                                     targetCacheId=self.targetCacheId, 
                                                     shouldProfile=shouldProfile)
            else:
                return InduceFromCandidateOp(self.outputSchema,
                                             InduceFromCandidateOp(
                                                 intermediateSchema,
                                                 self.inputOp._getPhysicalTree(strategy=strategy, 
                                                                               model=model, 
                                                                               shouldProfile=shouldProfile),
                                                 model,
                                                 self.cardinality,                              
                                                 shouldProfile=shouldProfile),
                                             model,
                                             "oneToOne",    
                                             desc=self.desc,
                                             targetCacheId=self.targetCacheId, 
                                             shouldProfile=shouldProfile)


class CacheScan(LogicalOperator):
    """A CacheScan is a logical operator that represents a scan of a cached Set."""
    def __init__(self, outputSchema: Schema, cachedDataIdentifier: str):
        super().__init__(outputSchema, None)
        self.cachedDataIdentifier = cachedDataIdentifier

    def __str__(self):
        return "CacheScan(" + str(self.outputSchema) + ", " + str(self.cachedDataIdentifier) + ")"

    def dumpLogicalTree(self):
        """Return the logical tree of this LogicalOperator."""
        return (self, None)

    def _getPhysicalTree(self, strategy: str=None, model: Model=None, shouldProfile: bool=False):
        return CacheScanDataOp(self.outputSchema, self.cachedDataIdentifier, shouldProfile=shouldProfile)


class BaseScan(LogicalOperator):
    """A BaseScan is a logical operator that represents a scan of a particular data source."""
    def __init__(self, outputSchema: Schema, datasetIdentifier: str):
        super().__init__(outputSchema, None)
        self.datasetIdentifier = datasetIdentifier

    def __str__(self):
        return "BaseScan(" + str(self.outputSchema) + ", " + self.datasetIdentifier + ")"

    def dumpLogicalTree(self):
        """Return the logical tree of this LogicalOperator."""
        return (self, None)

    def _getPhysicalTree(self, strategy: str=None, model: Model=None, shouldProfile: bool=False):
        return MarshalAndScanDataOp(self.outputSchema, self.datasetIdentifier, shouldProfile=shouldProfile)


class LimitScan(LogicalOperator):
    def __init__(self, outputSchema: Schema, inputOp: LogicalOperator, limit: int, targetCacheId: str=None):
        super().__init__(outputSchema, inputOp.outputSchema)
        self.inputOp = inputOp
        self.targetCacheId = targetCacheId
        self.limit = limit

    def __str__(self):
        return "LimitScan(" + str(self.inputSchema) + ", " + str(self.outputSchema) + ")"

    def dumpLogicalTree(self):
        """Return the logical tree of this LogicalOperator."""
        return (self, self.inputOp.dumpLogicalTree())

    def _getPhysicalTree(self, strategy: str=None, model: Model=None, shouldProfile: bool=False):
        return LimitScanOp(self.outputSchema, self.inputOp._getPhysicalTree(strategy=strategy, model=model, shouldProfile=shouldProfile), self.limit, targetCacheId=self.targetCacheId, shouldProfile=shouldProfile)


class FilteredScan(LogicalOperator):
    """A FilteredScan is a logical operator that represents a scan of a particular data source, with filters applied."""
    def __init__(self, outputSchema: Schema, inputOp: LogicalOperator, filter: Filter, targetCacheId: str=None):
        super().__init__(outputSchema, inputOp.outputSchema)
        self.inputOp = inputOp
        self.filter = filter
        self.targetCacheId = targetCacheId

    def __str__(self):
        return "FilteredScan(" + str(self.outputSchema) + ", " + "Filters: " + str(self.filter) + ")"

    def dumpLogicalTree(self):
        """Return the logical tree of this LogicalOperator."""
        return (self, self.inputOp.dumpLogicalTree())

    def _getPhysicalTree(self, strategy: str=None, model: Model=None, shouldProfile: bool=False):
        if DataDirectory().current_config.get("parallel") == True:
            return ParallelFilterCandidateOp(self.outputSchema, self.inputOp._getPhysicalTree(strategy=strategy, model=model, shouldProfile=shouldProfile), self.filter, model=model, targetCacheId=self.targetCacheId, shouldProfile=shouldProfile)
        else:
            return FilterCandidateOp(self.outputSchema, self.inputOp._getPhysicalTree(strategy=strategy, model=model, shouldProfile=shouldProfile), self.filter, model=model, targetCacheId=self.targetCacheId, shouldProfile=shouldProfile)


class GroupByAggregate(LogicalOperator):
    def __init__(self, outputSchema: Schema, inputOp: LogicalOperator, gbySig: elements.GroupBySig, targetCacheId: str=None):
        super().__init__(outputSchema, inputOp.outputSchema)
        (valid, error) = gbySig.validateSchema(inputOp.outputSchema)
        if (not valid):
            raise TypeError(error)
        self.inputOp = inputOp 
        self.gbySig = gbySig
        self.targetCacheId = targetCacheId
    def __str__(self):
        descStr = "Grouping Fields:" 
        return (f"GroupBy({elements.GroupBySig.serialize(self.gbySig)})")
    
    def dumpLogicalTree(self):
        """Return the logical subtree rooted at this operator"""
        return (self, self.inputOp.dumpLogicalTree())

    def _getPhysicalTree(self, strategy: str=None, model: Model=None, shouldProfile: bool=False):
        return ApplyGroupByOp(self.inputOp._getPhysicalTree(strategy=strategy, model=model, shouldProfile=shouldProfile), self.gbySig, targetCacheId=self.targetCacheId, shouldProfile=shouldProfile)


class ApplyAggregateFunction(LogicalOperator):
    """ApplyAggregateFunction is a logical operator that applies a function to the input set and yields a single result."""
    def __init__(self, outputSchema: Schema, inputOp: LogicalOperator, aggregationFunction: AggregateFunction, targetCacheId: str=None):
        super().__init__(outputSchema, inputOp.outputSchema)
        self.inputOp = inputOp
        self.aggregationFunction = aggregationFunction
        self.targetCacheId=targetCacheId

    def __str__(self):
        return "ApplyAggregateFunction(function: " + str(self.aggregationFunction) + ")"

    def dumpLogicalTree(self):
        """Return the logical subtree rooted at this operator"""
        return (self, self.inputOp.dumpLogicalTree())

    def _getPhysicalTree(self, strategy: str=None, model: Model=None, shouldProfile: bool=False):
        if self.aggregationFunction.funcDesc == "COUNT":
            return ApplyCountAggregateOp(self.inputOp._getPhysicalTree(strategy=strategy, model=model, shouldProfile=shouldProfile), self.aggregationFunction, targetCacheId=self.targetCacheId, shouldProfile=shouldProfile)
        elif self.aggregationFunction.funcDesc == "AVERAGE":
            return ApplyAverageAggregateOp(self.inputOp._getPhysicalTree(strategy=strategy, model=model, shouldProfile=shouldProfile), self.aggregationFunction, targetCacheId=self.targetCacheId, shouldProfile=shouldProfile)
        else:
            raise Exception(f"Cannot find implementation for {self.aggregationFunction}")


class ApplyUserFunction(LogicalOperator):
    """ApplyUserFunction is a logical operator that applies a user-provided function to the input set and yields a result."""
    def __init__(self, outputSchema: Schema, inputOp: LogicalOperator, fnid:str, targetCacheId: str=None):
        super().__init__(outputSchema, inputOp.outputSchema)
        self.inputOp = inputOp
        self.fnid = fnid
        self.fn = DataDirectory().getUserFunction(fnid)
        self.targetCacheId=targetCacheId

    def __str__(self):
        return "ApplyUserFunction(function: " + str(self.fnid) + ")"

    def dumpLogicalTree(self):
        """Return the logical subtree rooted at this operator"""
        return (self, self.inputOp.dumpLogicalTree())

    def _getPhysicalTree(self, strategy: str=None, model: Model=None, shouldProfile: bool=False):
        return ApplyUserFunctionOp(self.inputOp._getPhysicalTree(strategy=strategy, model=model, shouldProfile=shouldProfile), self.fn, targetCacheId=self.targetCacheId, shouldProfile=shouldProfile)
