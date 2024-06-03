from __future__ import annotations
from .physical import PhysicalOp, MAX_ID_CHARS

from palimpzest.constants import *
from palimpzest.elements import *
from palimpzest.solver.task_descriptors import TaskDescriptor
from palimpzest.profiler import Profiler

from typing import Any, Dict


import concurrent
import hashlib


class FilterOp(PhysicalOp):

    def __init__(
        self,
        outputSchema: Schema,
        source: PhysicalOp,
        filter: Filter,
        model: Model,
        prompt_strategy: PromptStrategy = PromptStrategy.DSPY_COT_BOOL,
        targetCacheId: str = None,
        shouldProfile=False,
        max_workers=1,
        *args,
        **kwargs,
    ):
        super().__init__(outputSchema=outputSchema, shouldProfile=shouldProfile)
        self.source = source
        self.filter = filter
        self.model = model if filter.filterFn is None else None
        self.prompt_strategy = prompt_strategy
        self.targetCacheId = targetCacheId
        self.max_workers = max_workers

        # NOTE: need to construct profiler after all fields used by self.opId() are set
        self.profiler = Profiler(op_id=self.opId())
        self.profile = self.profiler.iter_profiler

        # # construct TaskDescriptor
        # taskDescriptor = self._makeTaskDescriptor()

        # # synthesize task function
        # if not taskDescriptor.op_id in PhysicalOp.synthesizedFns:
        #     PhysicalOp.synthesizedFns[taskDescriptor.op_id] = PhysicalOp.solver.synthesize(taskDescriptor, shouldProfile=self.shouldProfile)

    def __eq__(self, other: PhysicalOp):
        return (
            isinstance(other, self.__class__)
            and self.model == other.model
            and self.filter == other.filter
            and self.prompt_strategy == other.prompt_strategy
            and self.outputSchema == other.outputSchema
            and self.source == other.source
        )

    def __str__(self):
        model_str = self.model.value if self.model is not None else str(None)
        return f"{self.__class__.__name__}({str(self.outputSchema)}, Filter: {str(self.filter)}, Model: {model_str}, Prompt Strategy: {str(self.prompt_strategy.value)})"

    def _makeTaskDescriptor(self):
        return TaskDescriptor(
            physical_op=self.__class__.__name__,
            inputSchema=self.source.outputSchema,
            op_id=self.opId(),
            filter=self.filter,
            model=self.model,
            prompt_strategy=self.prompt_strategy,
            plan_idx=self.plan_idx,
        )

    def copy(self, *args, **kwargs):
        return self.__class__(
            outputSchema=self.outputSchema,
            source=self.source,
            model=self.model,
            filter=self.filter,
            prompt_strategy=self.prompt_strategy,
            targetCacheId=self.targetCacheId,
            shouldProfile=self.shouldProfile,
            max_workers=self.max_workers,
            *args,
            **kwargs,
        )

    def opId(self):
        d = {
            "operator": self.__class__.__name__,
            "outputSchema": str(self.outputSchema),
            "source": self.source.opId(),
            "filter": str(self.filter),
            "model": self.model.value if self.model is not None else None,
            "prompt_strategy": self.prompt_strategy.value,
            "targetCacheId": self.targetCacheId,
        }
        ordered = json.dumps(d, sort_keys=True)
        return hashlib.sha256(ordered.encode()).hexdigest()[:MAX_ID_CHARS]

    def estimateCost(self, cost_est_data: Dict[str, Any] = None):
        """
        See InduceFromCandidateOp.estimateCost() for NOTEs and TODOs on how to improve this method.
        """
        # fetch cost estimates from source operation
        inputEstimates, subPlanCostEst = self.source.estimateCost(cost_est_data)

        filter_str = (
            self.filter.filterCondition
            if self.filter.filterCondition is not None
            else str(self.filter.filterFn)
        )
        op_filter = f"(filter == '{str(filter_str)}') & (op_name == 'filter' | op_name == 'p_filter')"
        if cost_est_data is not None and cost_est_data[op_filter] is not None:
            # get estimate data for this physical op's model
            model_name = None if self.model is None else self.model.value
            time_per_record = cost_est_data[op_filter][model_name]["time_per_record"]
            usd_per_record = cost_est_data[op_filter][model_name]["cost_per_record"]
            est_num_input_tokens = cost_est_data[op_filter][model_name][
                "est_num_input_tokens"
            ]
            est_num_output_tokens = cost_est_data[op_filter][model_name][
                "est_num_output_tokens"
            ]
            selectivity = cost_est_data[op_filter][model_name]["selectivity"]
            quality = cost_est_data[op_filter][model_name]["quality"]

            # estimate cardinality using sample selectivity and input cardinality est.
            cardinality = inputEstimates["cardinality"] * selectivity

            # apply quality for this filter to overall quality est.
            quality = (
                inputEstimates["quality"]
                if self.model is None
                else inputEstimates["quality"] * quality
            )

            thisCostEst = {
                "time_per_record": time_per_record,
                "usd_per_record": usd_per_record,
                "est_num_output_tokens": est_num_output_tokens,
                "selectivity": selectivity,
                "quality": quality,
            }

            costEst = {
                "cardinality": cardinality,
                "timePerElement": time_per_record,
                "usdPerElement": usd_per_record,
                "cumulativeTimePerElement": inputEstimates["cumulativeTimePerElement"]
                + time_per_record,
                "cumulativeUSDPerElement": inputEstimates["cumulativeUSDPerElement"]
                + usd_per_record,
                "totalTime": cardinality * time_per_record
                + inputEstimates["totalTime"],
                "totalUSD": cardinality * usd_per_record + inputEstimates["totalUSD"],
                "estOutputTokensPerElement": est_num_output_tokens,
                "quality": quality,
            }

            return costEst, {
                "cumulative": costEst,
                "thisPlan": thisCostEst,
                "subPlan": subPlanCostEst,
            }

        # otherwise, if this filter is a function call (not an LLM call) estimate accordingly
        if self.filter.filterFn is not None:
            # estimate output cardinality using a constant assumption of the filter selectivity
            selectivity = EST_FILTER_SELECTIVITY
            cardinality = selectivity * inputEstimates["cardinality"]

            # estimate 1 ms execution for filter function
            time_per_record = 0.001
            # (divide non-parallel est. by 10x for parallelism speed-up)
            if self.max_workers > 1:
                time_per_record /= 10

            # assume filter fn has perfect quality
            quality = inputEstimates["quality"]

            thisCostEst = {
                "time_per_record": time_per_record,
                "usd_per_record": 0.0,
                "est_num_output_tokens": inputEstimates["estOutputTokensPerElement"],
                "selectivity": selectivity,
                "quality": quality,
            }

            costEst = {
                "cardinality": cardinality,
                "timePerElement": time_per_record,
                "usdPerElement": 0.0,
                "cumulativeTimePerElement": inputEstimates["cumulativeTimePerElement"]
                + time_per_record,
                "cumulativeUSDPerElement": inputEstimates["cumulativeUSDPerElement"],
                "totalTime": cardinality * time_per_record
                + inputEstimates["totalTime"],
                "totalUSD": inputEstimates["totalUSD"],
                # next operator processes input based on contents, not T/F output by this operator
                "estOutputTokensPerElement": inputEstimates[
                    "estOutputTokensPerElement"
                ],
                "quality": quality,
            }

            return costEst, {
                "cumulative": costEst,
                "thisPlan": thisCostEst,
                "subPlan": subPlanCostEst,
            }

        # estimate number of input tokens from source
        est_num_input_tokens = inputEstimates["estOutputTokensPerElement"]

        # the filter operation's LLM call should only output TRUE or FALSE, thus we expect its
        # number of output tokens to be ~1.25
        est_num_output_tokens = 1.25

        # if we're using a few-shot prompt strategy, the est_num_input_tokens will increase
        # by a small factor due to the added examples; we multiply after computing the
        # est_num_output_tokens b/c the few-shot examples likely won't affect the output length
        if self.prompt_strategy == PromptStrategy.FEW_SHOT:
            est_num_input_tokens *= FEW_SHOT_PROMPT_INFLATION

        # get est. of conversion time per record from model card;
        model_conversion_time_per_record = (
            MODEL_CARDS[self.model.value]["seconds_per_output_token"]
            * est_num_output_tokens
        )

        # get est. of conversion cost (in USD) per record from model card
        model_conversion_usd_per_record = (
            MODEL_CARDS[self.model.value]["usd_per_input_token"] * est_num_input_tokens
            + MODEL_CARDS[self.model.value]["usd_per_output_token"]
            * est_num_output_tokens
        )

        # If we're using DSPy, use a crude estimate of the inflation caused by DSPy's extra API calls
        if self.prompt_strategy == PromptStrategy.DSPY_COT_BOOL:
            model_conversion_time_per_record *= DSPY_TIME_INFLATION
            model_conversion_usd_per_record *= DSPY_COST_INFLATION

        # estimate output cardinality using a constant assumption of the filter selectivity
        selectivity = EST_FILTER_SELECTIVITY
        cardinality = selectivity * inputEstimates["cardinality"]
        cumulativeTimePerElement = (
            model_conversion_time_per_record
            + inputEstimates["cumulativeTimePerElement"]
        )
        cumulativeUSDPerElement = (
            model_conversion_usd_per_record + inputEstimates["cumulativeUSDPerElement"]
        )

        # compute total time and cost for preceding operations + this operation;
        # make sure to use input cardinality (not output cardinality)
        totalTime = (
            model_conversion_time_per_record
            * (inputEstimates["cardinality"] / self.max_workers)
            + inputEstimates["totalTime"]
        )
        totalUSD = (
            model_conversion_usd_per_record * inputEstimates["cardinality"]
            + inputEstimates["totalUSD"]
        )

        # estimate quality of output based on the strength of the model being used
        quality = (MODEL_CARDS[self.model.value]["reasoning"] / 100.0) * inputEstimates[
            "quality"
        ]

        costEst = {
            "cardinality": cardinality,
            "timePerElement": model_conversion_time_per_record,
            "usdPerElement": model_conversion_usd_per_record,
            "cumulativeTimePerElement": cumulativeTimePerElement,
            "cumulativeUSDPerElement": cumulativeUSDPerElement,
            "totalTime": totalTime,
            "totalUSD": totalUSD,
            # next operator processes input based on contents, not T/F output by this operator
            "estOutputTokensPerElement": inputEstimates["estOutputTokensPerElement"],
            "quality": quality,
        }

        return costEst, {"cumulative": costEst, "thisPlan": costEst, "subPlan": None}

    def _passesFilter(self, candidate):
        """Return True if the candidate passes all filters, False otherwise."""
        taskDescriptor = self._makeTaskDescriptor()
        taskFn = PhysicalOp.solver.synthesize(
            taskDescriptor, shouldProfile=self.shouldProfile
        )
        # if not taskDescriptor.op_id in PhysicalOp.synthesizedFns:
        #     raise Exception("This function should have been synthesized during init():", taskDescriptor.op_id)
        # return PhysicalOp.synthesizedFns[taskDescriptor.op_id](candidate)
        return taskFn(candidate)

    def __iter__(self):
        raise NotImplementedError("You are calling a method from the abstract class!")


class FilterCandidateOp(FilterOp):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __iter__(self):
        shouldCache = self.datadir.openCache(self.targetCacheId)

        @self.profile(name="filter", shouldProfile=self.shouldProfile)
        def iteratorFn():
            for nextCandidate in self.source:
                resultRecord = self._passesFilter(nextCandidate)
                if resultRecord._passed_filter:
                    if shouldCache:
                        self.datadir.appendCache(self.targetCacheId, resultRecord)
                    yield resultRecord

                # if we're profiling, then we still need to yield candidate for the profiler to compute its stats;
                # the profiler will check the resultRecord._passed_filter field to see if it needs to be dropped
                elif self.shouldProfile:
                    yield resultRecord

            if shouldCache:
                self.datadir.closeCache(self.targetCacheId)

        return iteratorFn()


class ParallelFilterCandidateOp(FilterOp):

    def __init__(self, streaming=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_workers = 32  # TODO this is hardcoded?
        self.streaming = streaming

    def copy(self):
        return super().copy(streaming=self.streaming)

    def __iter__(self):
        shouldCache = self.datadir.openCache(self.targetCacheId)

        @self.profile(name="p_filter", shouldProfile=self.shouldProfile)
        def iteratorFn():
            inputs = []
            results = []

            for nextCandidate in self.source:
                inputs.append(nextCandidate)

            if self.streaming:
                chunksize = self.max_workers
            else:
                chunksize = len(inputs)

            # Grab items from the list of inputs in chunks using self.max_workers
            for i in range(0, len(inputs), chunksize):
                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=self.max_workers
                ) as executor:
                    results = list(
                        executor.map(self._passesFilter, inputs[i : i + chunksize])
                    )

                    for resultRecord in results:
                        if resultRecord._passed_filter:
                            if shouldCache:
                                self.datadir.appendCache(
                                    self.targetCacheId, resultRecord
                                )
                            yield resultRecord

                        # if we're profiling, then we still need to yield candidate for the profiler to compute its stats;
                        # the profiler will check the resultRecord._passed_filter field to see if it needs to be dropped
                        elif self.shouldProfile:
                            yield resultRecord
            if shouldCache:
                self.datadir.closeCache(self.targetCacheId)

        return iteratorFn()