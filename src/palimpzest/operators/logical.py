"""This module required refactoring the name because there were conflicts when importing pz.operators about what is the content"""

from __future__ import annotations

from palimpzest.constants import Model, QueryStrategy
from palimpzest.datamanager import DataDirectory
from palimpzest.elements import *

from copy import deepcopy
from itertools import permutations
from typing import List, Tuple

import numpy as np
import pandas as pd

import os
import random

from typing import List, Tuple

# DEFINITIONS
# PhysicalPlan = Tuple[float, float, float, PhysicalOp]


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

    def __init__(
        self,
        outputSchema: Schema,
        inputSchema: Schema,
    ):
        self.outputSchema = outputSchema
        self.inputSchema = inputSchema

        # def dumpLogicalTree(self) -> Tuple[LogicalOperator, LogicalOperator]:
        # """Return the logical subtree rooted at this operator"""
        # return (self, self.inputOp.dumpLogicalTree())
        # return (self, None)
        # raise NotImplementedError("Abstract method")

    def _getModels(self, include_vision: bool = False):
        models = []
        if os.getenv("OPENAI_API_KEY") is not None:
            models.extend([Model.GPT_3_5, Model.GPT_4])

        if os.getenv("TOGETHER_API_KEY") is not None:
            models.extend([Model.MIXTRAL])

        if os.getenv("GOOGLE_API_KEY") is not None:
            models.extend([Model.GEMINI_1])

        if include_vision:
            models.append(Model.GPT_4V)

        return models

    # TODO check what happens in refactoring out the inputOp pointer
    @staticmethod
    def _compute_legal_permutations(
        filterAndConvertOps: List[LogicalOperator],
    ) -> List[List[LogicalOperator]]:
        # There are a few rules surrounding which permutation(s) of logical operators are legal:
        # 1. if a filter depends on a field in a convert's outputSchema, it must be executed after the convert
        # 2. if a convert depends on another operation's outputSchema, it must be executed after that operation
        # 3. if depends_on is not specified for a convert operator, it cannot be swapped with another convert
        # 4. if depends_on is not specified for a filter, it can not be swapped with a convert (but it can be swapped w/adjacent filters)

        # compute implicit depends_on relationships, keep in mind that operations closer to the end of the list are executed first;
        # if depends_on is not specified for a convert or filter, it implicitly depends_on all preceding converts
        for idx, op in enumerate(filterAndConvertOps):
            if op.depends_on is None:
                all_prior_generated_fields = []
                for upstreamOp in filterAndConvertOps[idx + 1 :]:
                    if isinstance(upstreamOp, ConvertScan):
                        all_prior_generated_fields.extend(upstreamOp.generated_fields)
                op.depends_on = all_prior_generated_fields

        # compute all permutations of operators
        opPermutations = permutations(filterAndConvertOps)

        # iterate over permutations and determine if they are legal;
        # keep in mind that operations closer to the end of the list are executed first
        legalOpPermutations = []
        for opPermutation in opPermutations:
            is_valid = True
            for idx, op in enumerate(opPermutation):
                # if this op is a filter, we can skip because no upstream ops will conflict with this
                if isinstance(op, FilteredScan):
                    continue

                # invalid if upstream op depends on field generated by this op
                for upstreamOp in opPermutation[idx + 1 :]:
                    for col in upstreamOp.depends_on:
                        if col in op.generated_fields:
                            is_valid = False
                            break
                    if is_valid is False:
                        break
                if is_valid is False:
                    break

            # if permutation order is valid, then:
            # 1. make unique copy of each logical op
            # 2. update inputOp's
            # 3. update inputSchema's
            if is_valid:
                opCopyPermutation = [deepcopy(op) for op in opPermutation]
                for idx, op in enumerate(opCopyPermutation):
                    op.inputOp = (
                        opCopyPermutation[idx + 1]
                        if idx + 1 < len(opCopyPermutation)
                        else None
                    )

                # set schemas in reverse order
                for idx in range(len(opCopyPermutation) - 1, 0, -1):
                    op = opCopyPermutation[idx - 1]
                    op.inputSchema = opCopyPermutation[idx].outputSchema

                    if isinstance(op, FilteredScan):
                        op.outputSchema = op.inputSchema

                legalOpPermutations.append(opCopyPermutation)

        return legalOpPermutations

    # TODO: debug if deepcopy is not making valid copies to resolve duplicate profiler issue
    @staticmethod
    def _createLogicalPlans(rootOp: LogicalOperator) -> List[LogicalOperator]:
        """
        Given the logicalOperator, compute all possible equivalent plans with filter
        and convert operations re-ordered.
        """
        # base case, if this operator is a BaseScan or CacheScan, return operator
        if isinstance(rootOp, BaseScan) or isinstance(rootOp, CacheScan):
            return [rootOp]

        # if this operator is not a FilteredScan: compute the re-orderings for its inputOp,
        # point rootOp to each of the re-orderings, and return
        if not isinstance(rootOp, FilteredScan) and not isinstance(rootOp, ConvertScan):
            subTrees = LogicalOperator._createLogicalPlans(rootOp.inputOp)

            all_plans = []
            for tree in subTrees:
                rootOpCopy = deepcopy(rootOp)
                rootOpCopy.inputOp = tree
                all_plans.append(rootOpCopy)

            return all_plans

        # otherwise, if this operator is a FilteredScan or ConvertScan, make one plan per (legal)
        # permutation of consecutive converts and filters and recurse
        else:
            # get list of consecutive converts and filters
            filterAndConvertOps = []
            nextOp = rootOp
            while isinstance(nextOp, FilteredScan) or isinstance(nextOp, ConvertScan):
                filterAndConvertOps.append(nextOp)
                nextOp = nextOp.inputOp

            # compute set of legal permutations
            opPermutations = LogicalOperator._compute_legal_permutations(
                filterAndConvertOps
            )

            # compute filter reorderings for rest of tree
            subTrees = LogicalOperator._createLogicalPlans(nextOp)

            # compute cross-product of opPermutations and subTrees by linking final op w/first op in subTree
            for ops in opPermutations:
                for tree in subTrees:
                    ops[-1].inputOp = tree
                    ops[-1].inputSchema = tree.outputSchema

            # return roots of opPermutations
            return list(map(lambda ops: ops[0], opPermutations))


class ConvertScan(LogicalOperator):
    """A ConvertScan is a logical operator that represents a scan of a particular data source, with conversion applied."""

    def __init__(
        self,
        cardinality: str = None,
        image_conversion: bool = False,
        depends_on: List[str] = None,
        desc: str = None,
        targetCacheId: str = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.cardinality = cardinality
        self.image_conversion = image_conversion
        self.depends_on = depends_on
        self.desc = desc
        self.targetCacheId = targetCacheId

        # compute generated fields as set of fields in outputSchema that are not in inputSchema
        self.generated_fields = [
            field
            for field in self.outputSchema.fieldNames()
            if field not in self.inputSchema.fieldNames()
        ]

    def __str__(self):
        return f"ConvertScan({self.inputSchema} -> {str(self.outputSchema)},{str(self.desc)})"

    def dumpLogicalTree(self):
        """Return the logical tree of this LogicalOperator."""
        return (self, self.inputOp.dumpLogicalTree())

        # def _getPhysicalTree(
        self,
        strategy: str = (None,)
        source: PhysicalOp = (None,)
        model: Model = (None,)
        query_strategy: QueryStrategy = (None,)
        token_budget: float = (None,)
        shouldProfile: bool = (False,)

    # ):
    # TODO: dont set input op here
    # If the input is in core, and the output is NOT in core but its superclass is, then we should do a
    # 2-stage conversion. This will maximize chances that there is a pre-existing conversion to the superclass
    # in the known set of functions
    # intermediateSchema = self.outputSchema
    # while (
    #     not intermediateSchema == Schema
    #     and not PhysicalOp.solver.easyConversionAvailable(
    #         intermediateSchema, self.inputSchema
    #     )
    # ):
    #     intermediateSchema = intermediateSchema.__bases__[0]

    # if intermediateSchema == Schema or intermediateSchema == self.outputSchema:
    #     if DataDirectory().current_config.get("parallel") == True:
    #         return ParallelInduceFromCandidateOp(
    #             self.outputSchema,
    #             source,
    #             model,
    #             self.cardinality,
    #             self.image_conversion,
    #             query_strategy=query_strategy,
    #             token_budget=token_budget,
    #             desc=self.desc,
    #             targetCacheId=self.targetCacheId,
    #             shouldProfile=shouldProfile,
    #         )
    #     else:
    #         return InduceFromCandidateOp(
    #             self.outputSchema,
    #             source,
    #             model,
    #             self.cardinality,
    #             self.image_conversion,
    #             query_strategy=query_strategy,
    #             token_budget=token_budget,
    #             desc=self.desc,
    #             targetCacheId=self.targetCacheId,
    #             shouldProfile=shouldProfile,
    #         )
    # else:
    #     if DataDirectory().current_config.get("parallel") == True:
    #         return ParallelInduceFromCandidateOp(
    #             self.outputSchema,
    #             ParallelInduceFromCandidateOp(
    #                 intermediateSchema,
    #                 source,
    #                 model,
    #                 self.cardinality,
    #                 self.image_conversion,  # TODO: only one of these should have image_conversion
    #                 query_strategy=query_strategy,
    #                 token_budget=token_budget,
    #                 shouldProfile=shouldProfile,
    #             ),
    #             model,
    #             "oneToOne",
    #             image_conversion=self.image_conversion,  # TODO: only one of these should have image_conversion
    #             query_strategy=query_strategy,
    #             token_budget=token_budget,
    #             desc=self.desc,
    #             targetCacheId=self.targetCacheId,
    #             shouldProfile=shouldProfile,
    #         )
    #     else:
    #         return InduceFromCandidateOp(
    #             self.outputSchema,
    #             InduceFromCandidateOp(
    #                 intermediateSchema,
    #                 source,
    #                 model,
    #                 self.cardinality,
    #                 self.image_conversion,  # TODO: only one of these should have image_conversion
    #                 query_strategy=query_strategy,
    #                 token_budget=token_budget,
    #                 shouldProfile=shouldProfile,
    #             ),
    #             model,
    #             "oneToOne",
    #             image_conversion=self.image_conversion,  # TODO: only one of these should have image_conversion
    #             query_strategy=query_strategy,
    #             token_budget=token_budget,
    #             desc=self.desc,
    #             targetCacheId=self.targetCacheId,
    #             shouldProfile=shouldProfile,
    #         )


class CacheScan(LogicalOperator):
    """A CacheScan is a logical operator that represents a scan of a cached Set."""

    def __init__(self, cachedDataIdentifier: str, *args, **kwargs):
        kwargs["inputSchema"] = None
        super().__init__(None, *args, **kwargs)
        self.cachedDataIdentifier = cachedDataIdentifier

    def dumpLogicalTree(self):
        return (self, None)

    def __str__(self):
        return f"CacheScan({str(self.outputSchema)},{str(self.cachedDataIdentifier)})"


class BaseScan(LogicalOperator):
    """A BaseScan is a logical operator that represents a scan of a particular data source."""

    def __init__(self, datasetIdentifier: str, *args, **kwargs):
        kwargs["inputSchema"] = None

        super().__init__(*args, **kwargs)
        self.datasetIdentifier = datasetIdentifier

    def __str__(self):
        return f"BaseScan({str(self.outputSchema)},{self.datasetIdentifier})"

    def dumpLogicalTree(self):
        return (self, None)


class LimitScan(LogicalOperator):
    def __init__(self, limit: int, targetCacheId: str = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.targetCacheId = targetCacheId
        self.limit = limit

    def __str__(self):
        return f"LimitScan({str(self.inputSchema)}, {str(self.outputSchema)})"


class FilteredScan(LogicalOperator):
    """A FilteredScan is a logical operator that represents a scan of a particular data source, with filters applied."""

    def __init__(
        self,
        filter: Filter,
        depends_on: List[str] = None,
        targetCacheId: str = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.filter = filter
        self.depends_on = depends_on
        self.targetCacheId = targetCacheId

    def __str__(self):
        return f"FilteredScan({str(self.outputSchema)}, {str(self.filter)})"


class GroupByAggregate(LogicalOperator):
    def __init__(
        self,
        gbySig: elements.GroupBySig,
        targetCacheId: str = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        (valid, error) = gbySig.validateSchema(self.inputSchema)
        if not valid:
            raise TypeError(error)
        self.gbySig = gbySig
        self.targetCacheId = targetCacheId

    def __str__(self):
        descStr = "Grouping Fields:"
        return f"GroupBy({elements.GroupBySig.serialize(self.gbySig)})"


class ApplyAggregateFunction(LogicalOperator):
    """ApplyAggregateFunction is a logical operator that applies a function to the input set and yields a single result."""

    def __init__(
        self,
        aggregationFunction: AggregateFunction,
        targetCacheId: str = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.aggregationFunction = aggregationFunction
        self.targetCacheId = targetCacheId

    def __str__(self):
        return f"ApplyAggregateFunction(function: {str(self.aggregationFunction)})"


class ApplyUserFunction(LogicalOperator):
    """ApplyUserFunction is a logical operator that applies a user-provided function to the input set and yields a result."""

    def __init__(
        self,
        fnid: str,
        targetCacheId: str = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.fnid = fnid
        self.fn = DataDirectory().getUserFunction(fnid)
        self.targetCacheId = targetCacheId

    def __str__(self):
        return f"ApplyUserFunction(function: {str(self.fnid)})"