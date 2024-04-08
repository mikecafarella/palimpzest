from __future__ import annotations

from palimpzest.constants import *
from palimpzest.corelib.schemas import ImageFile
from palimpzest.datamanager import DataDirectory
from palimpzest.elements import *
from palimpzest.solver.solver import Solver
from palimpzest.solver.task_descriptors import TaskDescriptor
from palimpzest.profiler import OperatorStats, Profiler, StatsProcessor

from typing import Any, Callable, Dict, Tuple, Union

import concurrent
import hashlib
import sys

# DEFINITIONS
MAX_ID_CHARS = 10
IteratorFn = Callable[[], DataRecord]


class PhysicalOp:
    LOCAL_PLAN = "LOCAL"
    REMOTE_PLAN = "REMOTE"

    synthesizedFns = {}
    solver = Solver(verbose=LOG_LLM_OUTPUT)

    def __init__(self, outputSchema: Schema, source: PhysicalOp=None, shouldProfile=False) -> None:
        self.outputSchema = outputSchema
        self.source = source
        self.datadir = DataDirectory()
        self.shouldProfile = shouldProfile

        # NOTE: this must be overridden in each physical operator's __init__ method;
        #       we have to do it their b/c the opId() (which is an argument to the
        #       profiler's constructor) may not be valid until the physical operator
        #       has initialized all of its member fields
        self.profiler = None

    def opId(self) -> str:
        raise NotImplementedError("Abstract method")

    def dumpPhysicalTree(self) -> Tuple[PhysicalOp, Union[PhysicalOp, None]]:
        """Return the physical tree of operators."""
        if self.source is None:
            return (self, None)

        return (self, self.source.dumpPhysicalTree())

    def getProfilingData(self) -> OperatorStats:
        # simply return stats for this operator if there is no source
        if self.shouldProfile and self.source is None:
            return self.profiler.get_data()

        # otherwise, fetch the source operator's stats first, and then return
        # the current operator's stats w/a copy of its sources' stats
        elif self.shouldProfile:
            source_operator_stats = self.source.getProfilingData()
            operator_stats = self.profiler.get_data()
            operator_stats.source_op_stats = source_operator_stats
            return operator_stats

        # raise an exception if this method is called w/out profiling turned on
        else:
            raise Exception("Profiling was not turned on; please ensure shouldProfile=True when executing plan.")

    def estimateCost(self, cost_estimate_sample_data: List[Dict[str, Any]]=None) -> Dict[str, Any]:
        """Returns dict of time, cost, and quality metrics."""
        raise NotImplementedError("Abstract method")

class MarshalAndScanDataOp(PhysicalOp):
    def __init__(self, outputSchema: Schema, datasetIdentifier: str, shouldProfile=False):
        super().__init__(outputSchema=outputSchema, shouldProfile=shouldProfile)
        self.datasetIdentifier = datasetIdentifier

        # NOTE: need to construct profiler after all fields used by self.opId() are set
        self.profiler = Profiler(op_id=self.opId())
        self.profile = self.profiler.iter_profiler

    def __str__(self):
        return "MarshalAndScanDataOp(" + str(self.outputSchema) + ", " + self.datasetIdentifier + ")"

    def opId(self):
        d = {
            "operator": "MarshalAndScanDataOp",
            "outputSchema": str(self.outputSchema),
            "datasetIdentifier": self.datasetIdentifier,
        }
        ordered = json.dumps(d, sort_keys=True)
        return hashlib.sha256(ordered.encode()).hexdigest()[:MAX_ID_CHARS]

    def estimateCost(self, cost_estimate_sample_data: List[Dict[str, Any]]=None):
        cardinality = self.datadir.getCardinality(self.datasetIdentifier) + 1
        size = self.datadir.getSize(self.datasetIdentifier)
        perElementSizeInKb = (size / float(cardinality)) / 1024.0

        # if we have sample data, use it to get a better estimate of the timePerElement
        # and the output tokens per element
        timePerElement = None
        if cost_estimate_sample_data is not None:
            timePerElement = StatsProcessor._est_time_per_record(cost_estimate_sample_data, filter="op_name == 'base_scan'")
        else:
            # estimate time spent reading each record
            datasetType = self.datadir.getRegisteredDatasetType(self.datasetIdentifier)
            timePerElement = (
                LOCAL_SCAN_TIME_PER_KB * perElementSizeInKb
                if datasetType in ["dir", "file"]
                else MEMORY_SCAN_TIME_PER_KB * perElementSizeInKb
            )

        # NOTE: downstream operators will ignore this estimate if they have a cost_estimate dict.
        # estimate per-element number of tokens output by this operator
        estOutputTokensPerElement = (
            (size / float(cardinality)) # per-element size in bytes
            * ELEMENT_FRAC_IN_CONTEXT   # fraction of the element which is provided in context
            * BYTES_TO_TOKENS           # convert bytes to tokens
        )

        # assume no cost for reading data
        usdPerElement = 0

        return {
            "cardinality": cardinality,
            "timePerElement": timePerElement,
            "usdPerElement": usdPerElement,
            "cumulativeTimePerElement": timePerElement,
            "cumulativeUSDPerElement": usdPerElement,
            "totalTime": timePerElement * cardinality,
            "totalUSD": usdPerElement * cardinality,
            "estOutputTokensPerElement": estOutputTokensPerElement,
            "quality": 1.0,
        }

    def __iter__(self) -> IteratorFn:
        @self.profile(name="base_scan", shouldProfile=self.shouldProfile)
        def iteratorFn():
            for nextCandidate in self.datadir.getRegisteredDataset(self.datasetIdentifier):
                yield nextCandidate

        return iteratorFn()

class CacheScanDataOp(PhysicalOp):
    def __init__(self, outputSchema: Schema, cacheIdentifier: str, shouldProfile=False):
        super().__init__(outputSchema=outputSchema, shouldProfile=shouldProfile)
        self.cacheIdentifier = cacheIdentifier

        # NOTE: need to construct profiler after all fields used by self.opId() are set
        self.profiler = Profiler(op_id=self.opId())
        self.profile = self.profiler.iter_profiler

    def __str__(self):
        return "CacheScanDataOp(" + str(self.outputSchema) + ", " + self.cacheIdentifier + ")"

    def opId(self):
        d = {
            "operator": "CacheScanDataOp",
            "outputSchema": str(self.outputSchema),
            "cacheIdentifier": self.cacheIdentifier,
        }
        ordered = json.dumps(d, sort_keys=True)
        return hashlib.sha256(ordered.encode()).hexdigest()[:MAX_ID_CHARS]

    def estimateCost(self, cost_estimate_sample_data: List[Dict[str, Any]]=None):
        # TODO: at the moment, getCachedResult() looks up a pickled file that stores
        #       the cached data specified by self.cacheIdentifier, opens the file,
        #       and then returns an iterator over records in the pickled file.
        #
        #       I'm guessing that in the future we may want to load the cached data into
        #       the DataDirectory._cache object on __init__ (or in the background) so
        #       that this operation doesn't require a read from disk. If that happens, be
        #       sure to switch LOCAL_SCAN_TIME_PER_KB --> MEMORY_SCAN_TIME_PER_KB; and store
        #       metadata about the cardinality and size of cached data upfront so that we
        #       can access it in constant time.
        #
        #       At a minimum, we could use this function call to load the data into DataManager._cache
        #       since we have to iterate over it anyways; which would cache the data before the __iter__
        #       method below gets called.
        cached_data_info = [(1, sys.getsizeof(data)) for data in self.datadir.getCachedResult(self.cacheIdentifier)]
        cardinality = sum(list(map(lambda tup: tup[0], cached_data_info))) + 1
        size = sum(list(map(lambda tup: tup[1], cached_data_info)))
        perElementSizeInKb = (size / float(cardinality)) / 1024.0

        # if we have sample data, use it to get a better estimate of the timePerElement
        # and the output tokens per element
        timePerElement = None
        if cost_estimate_sample_data is not None:
            timePerElement = StatsProcessor._est_time_per_record(cost_estimate_sample_data, filter="op_name == 'cache_scan'")
        else:
            # estimate time spent reading each record
            timePerElement = LOCAL_SCAN_TIME_PER_KB * perElementSizeInKb

        # assume no cost for reading data
        usdPerElement = 0

        # NOTE: downstream operators will ignore this estimate if they have a cost_estimate dict.
        # estimate per-element number of tokens output by this operator
        estOutputTokensPerElement = (
            (size / float(cardinality)) # per-element size in bytes
            * ELEMENT_FRAC_IN_CONTEXT   # fraction of the element which is provided in context
            * BYTES_TO_TOKENS           # convert bytes to tokens
        )

        return {
            "cardinality": cardinality,
            "timePerElement": timePerElement,
            "usdPerElement": usdPerElement,
            "cumulativeTimePerElement": timePerElement,
            "cumulativeUSDPerElement": usdPerElement,
            "totalTime": timePerElement * cardinality,
            "totalUSD": usdPerElement * cardinality,
            "estOutputTokensPerElement": estOutputTokensPerElement,
            "quality": 1.0,
        }

    def __iter__(self) -> IteratorFn:
        @self.profile(name="cache_scan", shouldProfile=self.shouldProfile)
        def iteratorFn():
            # NOTE: see comment in `estimateCost()` 
            for nextCandidate in self.datadir.getCachedResult(self.cacheIdentifier):
                yield nextCandidate
        return iteratorFn()


class InduceFromCandidateOp(PhysicalOp):
    def __init__(self, outputSchema: Schema, source: PhysicalOp, model: Model, cardinality: str, prompt_strategy: PromptStrategy=PromptStrategy.DSPY_COT_QA, query_strategy: QueryStrategy=QueryStrategy.BONDED_WITH_FALLBACK, desc: str=None, targetCacheId: str=None, shouldProfile=False):
        super().__init__(outputSchema=outputSchema, source=source, shouldProfile=shouldProfile)
        self.model = model
        self.cardinality = cardinality
        self.prompt_strategy = prompt_strategy
        self.query_strategy = query_strategy
        self.desc = desc
        self.targetCacheId = targetCacheId

        if outputSchema == ImageFile and source.outputSchema == File:
            # TODO : find a more general way by llm provider 
            # TODO : which module is responsible of setting PromptStrategy.IMAGE_TO_TEXT? 
            if self.model in [Model.GPT_3_5, Model.GPT_4]:
                self.model = Model.GPT_4V
            if self.model == Model.GEMINI_1:
                self.model = Model.GEMINI_1V
            if self.model in [Model.MIXTRAL, Model.LLAMA2]:
                import random
                self.model = random.choice([Model.GPT_4V, Model.GEMINI_1V])
            self.prompt_strategy = PromptStrategy.IMAGE_TO_TEXT

        # TODO: remove once dspy pushes v2.4.1 to PyPI
        if self.model == Model.GEMINI_1:
            self.model = Model.GPT_4
        elif self.model == Model.GEMINI_1V:
            self.model = Model.GPT_4V

        # NOTE: need to construct profiler after all fields used by self.opId() are set
        self.profiler = Profiler(op_id=self.opId())
        self.profile = self.profiler.iter_profiler

        # construct TaskDescriptor
        taskDescriptor = self._makeTaskDescriptor()

        # synthesize task function
        if not str(taskDescriptor) in PhysicalOp.synthesizedFns:
            PhysicalOp.synthesizedFns[str(taskDescriptor)] = PhysicalOp.solver.synthesize(taskDescriptor, shouldProfile=self.shouldProfile)

    def __str__(self):
        return "InduceFromCandidateOp(" + str(self.outputSchema) + ", Model: " + str(self.model.value) + ", Prompt Strategy: " + str(self.prompt_strategy.value) + ")"

    def _makeTaskDescriptor(self):
        return TaskDescriptor(
            physical_op="InduceFromCandidateOp",
            inputSchema=self.source.outputSchema,
            outputSchema=self.outputSchema,
            op_id=self.opId(),
            model=self.model,
            cardinality=self.cardinality,
            prompt_strategy=self.prompt_strategy,
            query_strategy=self.query_strategy,
            conversionDesc=self.desc,
            pdfprocessor=self.datadir.current_config.get("pdfprocessing"),
        )

    def _is_quick_conversion(self):
        td = self._makeTaskDescriptor()
        is_file_to_text_file = (td.outputSchema == TextFile and td.inputSchema == File)

        return PhysicalOp.solver.isSimpleConversion(td) or is_file_to_text_file

    def opId(self):
        d = {
            "operator": "InduceFromCandidateOp",
            "outputSchema": str(self.outputSchema),
            "source": self.source.opId(),
            "model": self.model.value,
            "prompt_strategy": self.prompt_strategy.value,
            "desc": self.desc,
            "targetCacheId": self.targetCacheId,
        }
        ordered = json.dumps(d, sort_keys=True)
        return hashlib.sha256(ordered.encode()).hexdigest()[:MAX_ID_CHARS]

    def estimateCost(self, cost_estimate_sample_data: List[Dict[str, Any]]=None):
        # fetch cost estimates from source operation
        inputEstimates = self.source.estimateCost(cost_estimate_sample_data)

        # if induce has a quick conversion; set "no-op" cost estimates
        if self._is_quick_conversion():
            # we assume time cost of these conversions is negligible
            return inputEstimates

        # if we have sample estimates, let's use those instead of our prescriptive estimates
        if cost_estimate_sample_data is not None:
            # get state variables
            input_fields = self.source.outputSchema.fieldNames()
            generated_fields = [field for field in self.outputSchema.fieldNames() if field not in input_fields]
            input_fields_str = "-".join(sorted(input_fields))
            generated_fields_str = "-".join(sorted(generated_fields))

            # compute estimates
            filter = f"(input_fields == '{input_fields_str}') & (generated_fields == '{generated_fields_str}') & (op_name == 'induce')"
            time_per_record = StatsProcessor._est_time_per_record(cost_estimate_sample_data, filter=filter)
            usd_per_record = StatsProcessor._est_usd_per_record(cost_estimate_sample_data, filter=filter)
            _, est_num_output_tokens = StatsProcessor._est_num_input_output_tokens(cost_estimate_sample_data, filter=filter)
            selectivity = StatsProcessor._est_selectivity(cost_estimate_sample_data, filter=filter)
            quality = StatsProcessor._est_quality(cost_estimate_sample_data, filter=filter)

            # estimate cardinality using sample selectivity and input cardinality est.
            cardinality = inputEstimates['cardinality'] * selectivity

            return {
                "cardinality": cardinality,
                "timePerElement": time_per_record,
                "usdPerElement": usd_per_record,
                "cumulativeTimePerElement": inputEstimates['cumulativeTimePerElement'] + time_per_record,
                "cumulativeUSDPerElement": inputEstimates['cumulativeUSDPerElement'] + usd_per_record,
                "totalTime": cardinality * time_per_record + inputEstimates['totalTime'],
                "totalUSD": cardinality * usd_per_record + inputEstimates['totalUSD'],
                "estOutputTokensPerElement": est_num_output_tokens,
                "quality": quality,
            }

        # estimate number of input tokens from source
        est_num_input_tokens = inputEstimates["estOutputTokensPerElement"]

        # estimate number of output tokens as constant multiple of input tokens (for now)
        est_num_output_tokens = OUTPUT_TOKENS_MULTIPLE * est_num_input_tokens

        # if we're using a few-shot prompt strategy, the est_num_input_tokens will increase
        # by a small factor due to the added examples; we multiply after computing the
        # est_num_output_tokens b/c the few-shot examples likely won't affect the output length
        if self.prompt_strategy == PromptStrategy.FEW_SHOT:
            est_num_input_tokens *= FEW_SHOT_PROMPT_INFLATION

        # get est. of conversion time per record from model card;
        model_conversion_time_per_record = MODEL_CARDS[self.model.value]["seconds_per_output_token"] * est_num_output_tokens

        # get est. of conversion cost (in USD) per record from model card
        model_conversion_usd_per_record = (
            MODEL_CARDS[self.model.value]["usd_per_input_token"] * est_num_input_tokens
            + MODEL_CARDS[self.model.value]["usd_per_output_token"] * est_num_output_tokens
        )

        # If we're using DSPy, use a crude estimate of the inflation caused by DSPy's extra API calls
        if self.prompt_strategy == PromptStrategy.DSPY_COT_QA:
            model_conversion_time_per_record *= DSPY_TIME_INFLATION
            model_conversion_usd_per_record *= DSPY_COST_INFLATION

        # estimate cardinality and selectivity given the "cardinality" set by the user
        selectivity = 1.0 if self.cardinality != "oneToMany" else 2.0
        cardinality = selectivity * inputEstimates["cardinality"]

        # estimate cumulative time per element
        cumulativeTimePerElement = model_conversion_time_per_record + inputEstimates["cumulativeTimePerElement"]
        cumulativeUSDPerElement = model_conversion_usd_per_record + inputEstimates["cumulativeUSDPerElement"]

        # NOTE: the following estimate assumes that nested Python generators effectively
        #       execute a single record at a time in sequence. I.e., there is no time
        #       overlap for execution in two different stages of the chain of generators.
        #
        # compute total time and cost for preceding operations + this operation;
        # make sure to use input cardinality (not output cardinality)
        totalTime = model_conversion_time_per_record * inputEstimates["cardinality"] + inputEstimates["totalTime"]
        totalUSD = model_conversion_usd_per_record * inputEstimates["cardinality"] + inputEstimates["totalUSD"]

        # estimate quality of output based on the strength of the model being used
        quality = (MODEL_CARDS[self.model.value]["MMLU"] / 100.0) * inputEstimates["quality"]

        return {
            "cardinality": cardinality,
            "timePerElement": model_conversion_time_per_record,
            "usdPerElement": model_conversion_usd_per_record,
            "cumulativeTimePerElement": cumulativeTimePerElement,
            "cumulativeUSDPerElement": cumulativeUSDPerElement,
            "totalTime": totalTime,
            "totalUSD": totalUSD,
            "estOutputTokensPerElement": est_num_output_tokens,
            "quality": quality,
        }

    def __iter__(self) -> IteratorFn:
        shouldCache = self.datadir.openCache(self.targetCacheId)

        @self.profile(name="induce", shouldProfile=self.shouldProfile)
        def iteratorFn():    
            for nextCandidate in self.source:
                resultRecordList = self._attemptMapping(nextCandidate)
                if resultRecordList is not None:
                    for resultRecord in resultRecordList:
                        if resultRecord is not None:
                            if shouldCache:
                                self.datadir.appendCache(self.targetCacheId, resultRecord)
                            yield resultRecord
            if shouldCache:
                self.datadir.closeCache(self.targetCacheId)

        return iteratorFn()

    def _attemptMapping(self, candidate: DataRecord):
        """Attempt to map the candidate to the outputSchema. Return None if it fails."""
        taskDescriptor = self._makeTaskDescriptor()
        if not str(taskDescriptor) in PhysicalOp.synthesizedFns:
            raise Exception("This function should have been synthesized during init():", str(taskDescriptor))
        return PhysicalOp.synthesizedFns[str(taskDescriptor)](candidate)


class ParallelInduceFromCandidateOp(PhysicalOp):
    def __init__(self, outputSchema: Schema, source: PhysicalOp, model: Model, cardinality: str, prompt_strategy: PromptStrategy=PromptStrategy.DSPY_COT_QA, query_strategy: QueryStrategy=QueryStrategy.BONDED_WITH_FALLBACK, desc: str=None, targetCacheId: str=None, streaming=False, shouldProfile=False):
        super().__init__(outputSchema=outputSchema, shouldProfile=shouldProfile)
        self.source = source
        self.model = model
        self.cardinality = cardinality
        self.prompt_strategy = prompt_strategy
        self.query_strategy = query_strategy
        self.desc = desc
        self.targetCacheId = targetCacheId
        self.max_workers = 20
        self.streaming = streaming

        if outputSchema == ImageFile and source.outputSchema == File:
            # TODO : find a more general way by llm provider 
            # TODO : which module is responsible of setting PromptStrategy.IMAGE_TO_TEXT? 
            if self.model in [Model.GPT_3_5, Model.GPT_4]:
                self.model = Model.GPT_4V
            if self.model == Model.GEMINI_1:
                self.model = Model.GEMINI_1V
            if self.model in [Model.MIXTRAL, Model.LLAMA2]:
                import random
                self.model = random.choice([Model.GPT_4V, Model.GEMINI_1V])
            self.prompt_strategy = PromptStrategy.IMAGE_TO_TEXT

        # TODO: remove once dspy pushes v2.4.1 to PyPI
        if self.model == Model.GEMINI_1:
            self.model = Model.GPT_4
        elif self.model == Model.GEMINI_1V:
            self.model = Model.GPT_4V

        # NOTE: need to construct profiler after all fields used by self.opId() are set
        self.profiler = Profiler(op_id=self.opId())
        self.profile = self.profiler.iter_profiler

        # construct TaskDescriptor
        taskDescriptor = self._makeTaskDescriptor()

        # synthesize task function
        if not str(taskDescriptor) in PhysicalOp.synthesizedFns:
            PhysicalOp.synthesizedFns[str(taskDescriptor)] = PhysicalOp.solver.synthesize(taskDescriptor, shouldProfile=self.shouldProfile)

    def __str__(self):
        return "ParallelInduceFromCandidateOp(" + str(self.outputSchema) + ", Model: " + str(self.model.value) + ", Prompt Strategy: " + str(self.prompt_strategy.value) + ")"

    def _makeTaskDescriptor(self):
        return TaskDescriptor(
            physical_op="ParallelInduceFromCandidateOp",
            inputSchema=self.source.outputSchema,
            outputSchema=self.outputSchema,
            op_id=self.opId(),
            model=self.model,
            cardinality=self.cardinality,
            prompt_strategy=self.prompt_strategy,
            query_strategy=self.query_strategy,
            conversionDesc=self.desc,
            pdfprocessor=self.datadir.current_config.get("pdfprocessing"),
        )

    def _is_quick_conversion(self):
        td = self._makeTaskDescriptor()
        is_file_to_text_file = td.outputSchema == TextFile and td.inputSchema == File

        return PhysicalOp.solver.isSimpleConversion(td) or is_file_to_text_file

    def opId(self):
        d = {
            "operator": "ParallelInduceFromCandidateOp",
            "outputSchema": str(self.outputSchema),
            "source": self.source.opId(),
            "model": self.model.value,
            "prompt_strategy": self.prompt_strategy.value,
            "desc": self.desc,
            "targetCacheId": self.targetCacheId,
        }
        ordered = json.dumps(d, sort_keys=True)
        return hashlib.sha256(ordered.encode()).hexdigest()[:MAX_ID_CHARS]

    def estimateCost(self, cost_estimate_sample_data: List[Dict[str, Any]]=None):
        """
        See InduceFromCandidateOp.estimateCost() for NOTEs and TODOs on how to improve this method.
        """
        # fetch cost estimates from source operation
        inputEstimates = self.source.estimateCost(cost_estimate_sample_data)

        # if induce has a quick conversion; set "no-op" cost estimates
        if self._is_quick_conversion():
            # we assume time cost of these conversions is negligible
            return inputEstimates

        # if we have sample estimates, let's use those instead of our prescriptive estimates
        if cost_estimate_sample_data is not None:
            # get state variables
            input_fields = json.dumps(sorted(self.source.outputSchema.fieldNames()))
            generated_fields = json.dumps(sorted([field for field in self.outputSchema.fieldNames() if field not in input_fields]))

            # compute estimates
            input_fields_str = "-".join(sorted(input_fields))
            generated_fields_str = "-".join(sorted(generated_fields))
            filter = f"(input_fields == '{input_fields_str}') & (generated_fields == '{generated_fields_str}') & (op_name == 'p_induce')"
            time_per_record = StatsProcessor._est_time_per_record(cost_estimate_sample_data, filter=filter)
            usd_per_record = StatsProcessor._est_usd_per_record(cost_estimate_sample_data, filter=filter)
            _, est_num_output_tokens = StatsProcessor._est_num_input_output_tokens(cost_estimate_sample_data, filter=filter)
            selectivity = StatsProcessor._est_selectivity(cost_estimate_sample_data, filter=filter)
            quality = StatsProcessor._est_quality(cost_estimate_sample_data, filter=filter)

            # estimate cardinality using sample selectivity and input cardinality est.
            cardinality = inputEstimates['cardinality'] * selectivity

            return {
                "cardinality": cardinality,
                "timePerElement": time_per_record,
                "usdPerElement": usd_per_record,
                "cumulativeTimePerElement": inputEstimates['cumulativeTimePerElement'] + time_per_record,
                "cumulativeUSDPerElement": inputEstimates['cumulativeUSDPerElement'] + usd_per_record,
                "totalTime": cardinality * time_per_record + inputEstimates['totalTime'],
                "totalUSD": cardinality * usd_per_record + inputEstimates['totalUSD'],
                "estOutputTokensPerElement": est_num_output_tokens,
                "quality": quality,
            }

        # estimate number of input tokens from source
        est_num_input_tokens = inputEstimates["estOutputTokensPerElement"]

        # estimate number of output tokens as constant multiple of input tokens (for now)
        est_num_output_tokens = OUTPUT_TOKENS_MULTIPLE * est_num_input_tokens

        # if we're using a few-shot prompt strategy, the est_num_input_tokens will increase
        # by a small factor due to the added examples; we multiply after computing the
        # est_num_output_tokens b/c the few-shot examples likely won't affect the output length
        if self.prompt_strategy == PromptStrategy.FEW_SHOT:
            est_num_input_tokens *= FEW_SHOT_PROMPT_INFLATION

        # get est. of conversion time per record from model card;
        model_conversion_time_per_record = MODEL_CARDS[self.model.value]["seconds_per_output_token"] * est_num_output_tokens

        # get est. of conversion cost (in USD) per record from model card
        model_conversion_usd_per_record = (
            MODEL_CARDS[self.model.value]["usd_per_input_token"] * est_num_input_tokens
            + MODEL_CARDS[self.model.value]["usd_per_output_token"] * est_num_output_tokens
        )

        # If we're using DSPy, use a crude estimate of the inflation caused by DSPy's extra API calls
        if self.prompt_strategy == PromptStrategy.DSPY_COT_QA:
            model_conversion_time_per_record *= DSPY_TIME_INFLATION
            model_conversion_usd_per_record *= DSPY_COST_INFLATION

        # estimate cardinality and selectivity given the "cardinality" set by the user
        selectivity = 1.0 if self.cardinality != "oneToMany" else 2.0
        cardinality = selectivity * inputEstimates["cardinality"]

        # estimate cumulative time per element
        cumulativeTimePerElement = model_conversion_time_per_record + inputEstimates["cumulativeTimePerElement"]
        cumulativeUSDPerElement = model_conversion_usd_per_record + inputEstimates["cumulativeUSDPerElement"]

        # compute total time and cost for preceding operations + this operation;
        # make sure to use input cardinality (not output cardinality)
        totalTime = model_conversion_time_per_record * (inputEstimates["cardinality"] / self.max_workers) + inputEstimates["totalTime"]
        totalUSD = model_conversion_usd_per_record * inputEstimates["cardinality"] + inputEstimates["totalUSD"]

        # estimate quality of output based on the strength of the model being used
        quality = (MODEL_CARDS[self.model.value]["MMLU"] / 100.0) * inputEstimates["quality"]

        return {
            "cardinality": cardinality,
            "timePerElement": model_conversion_time_per_record,
            "usdPerElement": model_conversion_usd_per_record,
            "cumulativeTimePerElement": cumulativeTimePerElement,
            "cumulativeUSDPerElement": cumulativeUSDPerElement,
            "totalTime": totalTime,
            "totalUSD": totalUSD,
            "estOutputTokensPerElement": est_num_output_tokens,
            "quality": quality,
        }

    def __iter__(self):
        # This is very crudely implemented right now, since we materialize everything
        shouldCache = self.datadir.openCache(self.targetCacheId)

        @self.profile(name="p_induce", shouldProfile=self.shouldProfile)
        def iteratorFn():
            inputs = []
            results = []

            for nextCandidate in self.source:
                inputs.append(nextCandidate)

            # Grab items from the list inputs in chunks using self.max_workers
            if self.streaming:                
                chunksize =self.max_workers
            else:
                chunksize = len(inputs)

            if chunksize == 0:
                return
            
            for i in range(0, len(inputs), chunksize):
                with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    results = list(executor.map(self._attemptMapping, inputs[i:i+chunksize]))

                    for resultRecordList in results:
                        if resultRecordList is not None:
                            for resultRecord in resultRecordList:
                                if resultRecord is not None:
                                    if shouldCache:
                                        self.datadir.appendCache(self.targetCacheId, resultRecord)
                                    yield resultRecord
            if shouldCache:
                self.datadir.closeCache(self.targetCacheId)

        return iteratorFn()

    def _attemptMapping(self, candidate: DataRecord):
        """Attempt to map the candidate to the outputSchema. Return None if it fails."""
        taskDescriptor = self._makeTaskDescriptor()
        if not str(taskDescriptor) in PhysicalOp.synthesizedFns:
            raise Exception("This function should have been synthesized during init():", str(taskDescriptor))
        return PhysicalOp.synthesizedFns[str(taskDescriptor)](candidate)


class FilterCandidateOp(PhysicalOp):
    def __init__(self, outputSchema: Schema, source: PhysicalOp, filter: Filter, model: Model, prompt_strategy: PromptStrategy=PromptStrategy.DSPY_COT_BOOL, targetCacheId: str=None, shouldProfile=False):
        super().__init__(outputSchema=outputSchema, shouldProfile=shouldProfile)
        self.source = source
        self.filter = filter
        self.model = model
        self.prompt_strategy = prompt_strategy
        self.targetCacheId = targetCacheId

        # NOTE: need to construct profiler after all fields used by self.opId() are set
        self.profiler = Profiler(op_id=self.opId())
        self.profile = self.profiler.iter_profiler

        # construct TaskDescriptor
        taskDescriptor = self._makeTaskDescriptor()

        # synthesize task function
        if not str(taskDescriptor) in PhysicalOp.synthesizedFns:
            PhysicalOp.synthesizedFns[str(taskDescriptor)] = PhysicalOp.solver.synthesize(taskDescriptor, shouldProfile=self.shouldProfile)

    def __str__(self):
        return "FilterCandidateOp(" + str(self.outputSchema) + ", " + "Filter: " + str(self.filter) + ", Model: " + str(self.model.value) + ", Prompt Strategy: " + str(self.prompt_strategy.value) + ")"

    def _makeTaskDescriptor(self):
        return TaskDescriptor(
            physical_op="FilterCandidateOp",
            inputSchema=self.source.outputSchema,
            op_id=self.opId(),
            filter=self.filter,
            model=self.model,
            prompt_strategy=self.prompt_strategy,
        )

    def opId(self):
        d = {
            "operator": "FilterCandidateOp",
            "outputSchema": str(self.outputSchema),
            "source": self.source.opId(),
            "filter": str(self.filter),
            "model": self.model.value,
            "prompt_strategy": self.prompt_strategy.value,
            "targetCacheId": self.targetCacheId,
        }
        ordered = json.dumps(d, sort_keys=True)
        return hashlib.sha256(ordered.encode()).hexdigest()[:MAX_ID_CHARS]

    def estimateCost(self, cost_estimate_sample_data: List[Dict[str, Any]]=None):
        """
        See InduceFromCandidateOp.estimateCost() for NOTEs and TODOs on how to improve this method.
        """
        # fetch cost estimates from source operation
        inputEstimates = self.source.estimateCost(cost_estimate_sample_data)

        if cost_estimate_sample_data is not None:
            # compute estimates
            filter = f"(filter == '{str(self.filter)}') & (op_name == 'filter')"
            time_per_record = StatsProcessor._est_time_per_record(cost_estimate_sample_data, filter=filter)
            usd_per_record = StatsProcessor._est_usd_per_record(cost_estimate_sample_data, filter=filter)
            _, est_num_output_tokens = StatsProcessor._est_num_input_output_tokens(cost_estimate_sample_data, filter=filter)
            selectivity = StatsProcessor._est_selectivity(cost_estimate_sample_data, filter=filter)
            quality = StatsProcessor._est_quality(cost_estimate_sample_data, filter=filter)

            # estimate cardinality using sample selectivity and input cardinality est.
            cardinality = inputEstimates['cardinality'] * selectivity

            return {
                "cardinality": cardinality,
                "timePerElement": time_per_record,
                "usdPerElement": usd_per_record,
                "cumulativeTimePerElement": inputEstimates['cumulativeTimePerElement'] + time_per_record,
                "cumulativeUSDPerElement": inputEstimates['cumulativeUSDPerElement'] + usd_per_record,
                "totalTime": cardinality * time_per_record + inputEstimates['totalTime'],
                "totalUSD": cardinality * usd_per_record + inputEstimates['totalUSD'],
                "estOutputTokensPerElement": est_num_output_tokens,
                "quality": quality,
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
        model_conversion_time_per_record = MODEL_CARDS[self.model.value]["seconds_per_output_token"] * est_num_output_tokens

        # get est. of conversion cost (in USD) per record from model card
        model_conversion_usd_per_record = (
            MODEL_CARDS[self.model.value]["usd_per_input_token"] * est_num_input_tokens
            + MODEL_CARDS[self.model.value]["usd_per_output_token"] * est_num_output_tokens
        )

        # If we're using DSPy, use a crude estimate of the inflation caused by DSPy's extra API calls
        if self.prompt_strategy == PromptStrategy.DSPY_COT_BOOL:
            model_conversion_time_per_record *= DSPY_TIME_INFLATION
            model_conversion_usd_per_record *= DSPY_COST_INFLATION

        # estimate output cardinality using a constant assumption of the filter selectivity
        selectivity = EST_FILTER_SELECTIVITY
        cardinality = selectivity * inputEstimates["cardinality"]
        cumulativeTimePerElement = model_conversion_time_per_record + inputEstimates["cumulativeTimePerElement"]
        cumulativeUSDPerElement = model_conversion_usd_per_record + inputEstimates["cumulativeUSDPerElement"]

        # compute total time and cost for preceding operations + this operation;
        # make sure to use input cardinality (not output cardinality)
        totalTime = model_conversion_time_per_record * inputEstimates["cardinality"] + inputEstimates["totalTime"]
        totalUSD = model_conversion_usd_per_record * inputEstimates["cardinality"] + inputEstimates["totalUSD"]

        # estimate quality of output based on the strength of the model being used
        quality = (MODEL_CARDS[self.model.value]["reasoning"] / 100.0) * inputEstimates["quality"]

        return {
            "cardinality": cardinality,
            "timePerElement": model_conversion_time_per_record,
            "usdPerElement": model_conversion_usd_per_record,
            "cumulativeTimePerElement": cumulativeTimePerElement,
            "cumulativeUSDPerElement": cumulativeUSDPerElement,
            "totalTime": totalTime,
            "totalUSD": totalUSD,
            "estOutputTokensPerElement": est_num_output_tokens,
            "quality": quality,
        }

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

    def _passesFilter(self, candidate):
        """Return True if the candidate passes all filters, False otherwise."""
        taskDescriptor = self._makeTaskDescriptor()
        if not str(taskDescriptor) in PhysicalOp.synthesizedFns:
            raise Exception("This function should have been synthesized during init():", str(taskDescriptor))
        return PhysicalOp.synthesizedFns[str(taskDescriptor)](candidate)


class ParallelFilterCandidateOp(PhysicalOp):
    def __init__(self, outputSchema: Schema, source: PhysicalOp, filter: Filter, model: Model, prompt_strategy: PromptStrategy=PromptStrategy.DSPY_COT_BOOL, targetCacheId: str=None, streaming=False, shouldProfile=False):
        super().__init__(outputSchema=outputSchema, shouldProfile=shouldProfile)
        self.source = source
        self.filter = filter
        self.model = model
        self.prompt_strategy = prompt_strategy
        self.targetCacheId = targetCacheId
        self.max_workers = 20
        self.streaming = streaming

        # NOTE: need to construct profiler after all fields used by self.opId() are set
        self.profiler = Profiler(op_id=self.opId())
        self.profile = self.profiler.iter_profiler

        # construct TaskDescriptor
        taskDescriptor = self._makeTaskDescriptor()

        # synthesize task function
        if not str(taskDescriptor) in PhysicalOp.synthesizedFns:
            PhysicalOp.synthesizedFns[str(taskDescriptor)] = PhysicalOp.solver.synthesize(taskDescriptor, shouldProfile=self.shouldProfile)

    def __str__(self):
        return "ParallelFilterCandidateOp(" + str(self.outputSchema) + ", " + "Filter: " + str(self.filter) + ", Model: " + str(self.model.value) + ", Prompt Strategy: " + str(self.prompt_strategy.value) + ")"

    def _makeTaskDescriptor(self):
        return TaskDescriptor(
            physical_op="ParallelFilterCandidateOp",
            inputSchema=self.source.outputSchema,
            op_id=self.opId(),
            filter=self.filter,
            model=self.model,
            prompt_strategy=self.prompt_strategy,
        )

    def opId(self):
        d = {
            "operator": "ParallelFilterCandidateOp",
            "outputSchema": str(self.outputSchema),
            "source": self.source.opId(),
            "filter": str(self.filter),
            "model": self.model.value,
            "prompt_strategy": self.prompt_strategy.value,
            "targetCacheId": self.targetCacheId,
        }
        ordered = json.dumps(d, sort_keys=True)
        return hashlib.sha256(ordered.encode()).hexdigest()[:MAX_ID_CHARS]

    def estimateCost(self, cost_estimate_sample_data: List[Dict[str, Any]]=None):
        # fetch cost estimates from source operation
        inputEstimates = self.source.estimateCost(cost_estimate_sample_data)

        if cost_estimate_sample_data is not None:
            # compute estimates
            filter = f"(filter == '{str(self.filter)}') & (op_name == 'p_filter')"
            time_per_record = StatsProcessor._est_time_per_record(cost_estimate_sample_data, filter=filter)
            usd_per_record = StatsProcessor._est_usd_per_record(cost_estimate_sample_data, filter=filter)
            _, est_num_output_tokens = StatsProcessor._est_num_input_output_tokens(cost_estimate_sample_data, filter=filter)
            selectivity = StatsProcessor._est_selectivity(cost_estimate_sample_data, filter=filter)
            quality = StatsProcessor._est_quality(cost_estimate_sample_data, filter=filter)

            # estimate cardinality using sample selectivity and input cardinality est.
            cardinality = inputEstimates['cardinality'] * selectivity

            return {
                "cardinality": cardinality,
                "timePerElement": time_per_record,
                "usdPerElement": usd_per_record,
                "cumulativeTimePerElement": inputEstimates['cumulativeTimePerElement'] + time_per_record,
                "cumulativeUSDPerElement": inputEstimates['cumulativeUSDPerElement'] + usd_per_record,
                "totalTime": cardinality * time_per_record + inputEstimates['totalTime'],
                "totalUSD": cardinality * usd_per_record + inputEstimates['totalUSD'],
                "estOutputTokensPerElement": est_num_output_tokens,
                "quality": quality,
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
        model_conversion_time_per_record = MODEL_CARDS[self.model.value]["seconds_per_output_token"] * est_num_output_tokens

        # get est. of conversion cost (in USD) per record from model card
        model_conversion_usd_per_record = (
            MODEL_CARDS[self.model.value]["usd_per_input_token"] * est_num_input_tokens
            + MODEL_CARDS[self.model.value]["usd_per_output_token"] * est_num_output_tokens
        )

        # If we're using DSPy, use a crude estimate of the inflation caused by DSPy's extra API calls
        if self.prompt_strategy == PromptStrategy.DSPY_COT_BOOL:
            model_conversion_time_per_record *= DSPY_TIME_INFLATION
            model_conversion_usd_per_record *= DSPY_COST_INFLATION

        # estimate output cardinality using a constant assumption of the filter selectivity
        selectivity = EST_FILTER_SELECTIVITY
        cardinality = selectivity * inputEstimates["cardinality"]
        cumulativeTimePerElement = model_conversion_time_per_record + inputEstimates["cumulativeTimePerElement"]
        cumulativeUSDPerElement = model_conversion_usd_per_record + inputEstimates["cumulativeUSDPerElement"]

        # compute total time and cost for preceding operations + this operation;
        # make sure to use input cardinality (not output cardinality)
        totalTime = model_conversion_time_per_record * (inputEstimates["cardinality"] / self.max_workers) + inputEstimates["totalTime"]
        totalUSD = model_conversion_usd_per_record * inputEstimates["cardinality"] + inputEstimates["totalUSD"]

        # estimate quality of output based on the strength of the model being used
        quality = (MODEL_CARDS[self.model.value]["reasoning"] / 100.0) * inputEstimates["quality"]

        return {
            "cardinality": cardinality,
            "timePerElement": model_conversion_time_per_record,
            "usdPerElement": model_conversion_usd_per_record,
            "cumulativeTimePerElement": cumulativeTimePerElement,
            "cumulativeUSDPerElement": cumulativeUSDPerElement,
            "totalTime": totalTime,
            "totalUSD": totalUSD,
            "estOutputTokensPerElement": est_num_output_tokens,
            "quality": quality,
        }

    def __iter__(self):
        shouldCache = self.datadir.openCache(self.targetCacheId)

        @self.profile(name="p_filter", shouldProfile=self.shouldProfile)
        def iteratorFn():
            inputs = []
            results = []

            for nextCandidate in self.source: 
                inputs.append(nextCandidate)

            if self.streaming:                
                chunksize =self.max_workers
            else:
                chunksize = len(inputs)

            # Grab items from the list of inputs in chunks using self.max_workers
            for i in range(0, len(inputs), chunksize):                
                with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:                    
                    results = list(executor.map(self._passesFilter, inputs[i:i+chunksize]))

                    for resultRecord in results:
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

    def _passesFilter(self, candidate):
        """Return True if the candidate passes all filters, False otherwise."""
        taskDescriptor = self._makeTaskDescriptor()
        if not str(taskDescriptor) in PhysicalOp.synthesizedFns:
            raise Exception("This function should have been synthesized during init():", str(taskDescriptor))

        return PhysicalOp.synthesizedFns[str(taskDescriptor)](candidate)

def agg_init(func):
    if (func.lower() == 'count'):
        return 0
    elif (func.lower() == 'average'):
        return (0,0)
    else:
        raise Exception("Unknown agg function " + func)

def agg_merge(func, state, val):
    if (func.lower() == 'count'):
        return state + 1
    elif (func.lower() == 'average'):
        sum, cnt = state
        return (sum + val, cnt + 1)
    else:
        raise Exception("Unknown agg function " + func)

def agg_final(func, state):
    if (func.lower() == 'count'):
        return state
    elif (func.lower() == 'average'):
        sum, cnt = state
        return float(sum)/cnt
    else:
        raise Exception("Unknown agg function " + func)


class ApplyGroupByOp(PhysicalOp):
        def __init__(self, source: PhysicalOp, gbySig: GroupBySig,  targetCacheId: str=None, shouldProfile=False):
            super().__init__(outputSchema=gbySig.outputSchema(), shouldProfile=shouldProfile)
            self.source = source
            self.gbySig = gbySig
            self.targetCacheId=targetCacheId
            self.shouldProfile=shouldProfile

        def __str__(self):
            return str(self.gbySig)
        def opId(self):
            d = {
                "operator": "ApplyGroupByOp",
                "source": self.source.opId(),
                "gbySig": str(GroupBySig.serialize(self.gbySig)),
                "targetCacheId": self.targetCacheId,
            }
            ordered = json.dumps(d, sort_keys=True)
            return hashlib.sha256(ordered.encode()).hexdigest()[:MAX_ID_CHARS]
        def dumpPhysicalTree(self):
            """Return the physical tree of operators."""
            return (self, self.source.dumpPhysicalTree())
        
        def estimateCost(self):
            inputEstimates = self.source.estimateCost()

            outputEstimates = {**inputEstimates}
            outputEstimates['cardinality'] = 1

            # for now, assume applying the aggregate takes negligible additional time (and no cost in USD)
            outputEstimates['timePerElement'] = 0
            outputEstimates['usdPerElement'] = 0
            outputEstimates['estOutputTokensPerElement'] = 0

            return outputEstimates



        def __iter__(self):
            datadir = DataDirectory()
            shouldCache = datadir.openCache(self.targetCacheId)
            aggState = {}

            @self.profile(name="gby", op_id=self.opId(), shouldProfile=self.shouldProfile)
            def iteratorFn():
                for r in self.source:
                    #build group array
                    group = ()
                    for f in self.gbySig.gbyFields:
                        if (not hasattr(r, f)):
                            raise TypeError(f"ApplyGroupOp record missing expected field {f}")
                        group = group + (getattr(r,f),)
                    if group in aggState:
                        state = aggState[group]
                    else:
                        state = []
                        for fun in self.gbySig.aggFuncs:
                            state.append(agg_init(fun))
                    for i in range(0,len(self.gbySig.aggFuncs)):
                        fun = self.gbySig.aggFuncs[i]
                        if (not hasattr(r, self.gbySig.aggFields[i])):
                            raise TypeError(f"ApplyGroupOp record missing expected field {self.gbySig.aggFields[i]}")
                        field = getattr(r, self.gbySig.aggFields[i])
                        state[i] = agg_merge(fun, state[i], field)
                    aggState[group] = state

                gbyFields = self.gbySig.gbyFields
                aggFields = self.gbySig.getAggFieldNames()
                for g in aggState.keys():
                    dr = DataRecord(self.gbySig.outputSchema())
                    for i in range(0, len(g)):
                        k = g[i]
                        setattr(dr, gbyFields[i], k)
                    vals = aggState[g]
                    for i in range(0, len(vals)):
                        v = agg_final(self.gbySig.aggFuncs[i], vals[i])
                        setattr(dr, aggFields[i], v)
                    if shouldCache:
                        datadir.appendCache(self.targetCacheId, dr)
                    yield dr

                if shouldCache:
                    datadir.closeCache(self.targetCacheId)

            return iteratorFn()



class ApplyCountAggregateOp(PhysicalOp):
    def __init__(self, source: PhysicalOp, aggFunction: AggregateFunction, targetCacheId: str=None, shouldProfile=False):
        super().__init__(outputSchema=Number, shouldProfile=shouldProfile)
        self.source = source
        self.aggFunction = aggFunction
        self.targetCacheId = targetCacheId

        # NOTE: need to construct profiler after all fields used by self.opId() are set
        self.profiler = Profiler(op_id=self.opId())
        self.profile = self.profiler.iter_profiler

    def __str__(self):
        return "ApplyCountAggregateOp(" + str(self.outputSchema) + ", " + "Function: " + str(self.aggFunction) + ")"

    def opId(self):
        d = {
            "operator": "ApplyCountAggregateOp",
            "source": self.source.opId(),
            "aggFunction": str(self.aggFunction),
            "targetCacheId": self.targetCacheId,
        }
        ordered = json.dumps(d, sort_keys=True)
        return hashlib.sha256(ordered.encode()).hexdigest()[:MAX_ID_CHARS]

    def estimateCost(self, cost_estimate_sample_data: List[Dict[str, Any]]=None):
        # get input estimates and pass through to output
        inputEstimates = self.source.estimateCost(cost_estimate_sample_data)
        outputEstimates = {**inputEstimates}

        # the profiler will record timing info for this operator, which can be used
        # to improve timing related estimates
        if cost_estimate_sample_data is not None:
            # compute estimates
            filter = f"(op_name == 'count')"
            time_per_record = StatsProcessor._est_time_per_record(cost_estimate_sample_data, filter=filter)

            # output cardinality for an aggregate will be 1
            cardinality = 1

            # update cardinality, timePerElement and related stats
            outputEstimates['cardinality'] = cardinality
            outputEstimates['timePerElement'] = time_per_record
            outputEstimates['cumulativeTimePerElement'] = inputEstimates['cumulativeTimePerElement'] + time_per_record
            outputEstimates['totalTime'] = cardinality * time_per_record + inputEstimates['totalTime']

            return outputEstimates

        # output cardinality for an aggregate will be 1
        outputEstimates['cardinality'] = 1

        # for now, assume applying the aggregate takes negligible additional time (and no cost in USD)
        outputEstimates['timePerElement'] = 0
        outputEstimates['usdPerElement'] = 0
        outputEstimates['estOutputTokensPerElement'] = 0

        return outputEstimates

    def __iter__(self):
        datadir = DataDirectory()
        shouldCache = datadir.openCache(self.targetCacheId)

        @self.profile(name="count", shouldProfile=self.shouldProfile)
        def iteratorFn():
            counter = 0
            for record in self.source:
                counter += 1

            # NOTE: this will set the parent_uuid to be the uuid of the final source record;
            #       this is ideal for computing the op_time of the count operation, but maybe
            #       we should set this DataRecord as having multiple parents in the future
            dr = DataRecord(Number, parent_uuid=record._uuid)
            dr.value = counter
            if shouldCache:
                datadir.appendCache(self.targetCacheId, dr)
            yield dr

            if shouldCache:
                datadir.closeCache(self.targetCacheId)

        return iteratorFn()


class ApplyUserFunctionOp(PhysicalOp):
    def __init__(self, source: PhysicalOp, fn:UserFunction, targetCacheId: str=None, shouldProfile=False):
        super().__init__(outputSchema=fn.outputSchema, shouldProfile=shouldProfile)
        self.source = source
        self.fn = fn
        self.targetCacheId = targetCacheId
        if not source.outputSchema == fn.inputSchema:
            raise Exception("Supplied UserFunction input schema does not match output schema of input source")

        # NOTE: need to construct profiler after all fields used by self.opId() are set
        self.profiler = Profiler(op_id=self.opId())
        self.profile = self.profiler.iter_profiler

    def __str__(self):
        return "ApplyUserFunctionOp(" + str(self.outputSchema) + ", " + "Function: " + str(self.fn.udfid) + ")"

    def opId(self):
        d = {
            "operator": "ApplyUserFunctionOp",
            "source": self.source.opId(),
            "fn": str(self.fn.udfid),
            "targetCacheId": self.targetCacheId,
        }
        ordered = json.dumps(d, sort_keys=True)
        return hashlib.sha256(ordered.encode()).hexdigest()[:MAX_ID_CHARS]

    def estimateCost(self, cost_estimate_sample_data: List[Dict[str, Any]]=None):
        # get input estimates and pass through to output
        inputEstimates = self.source.estimateCost(cost_estimate_sample_data)
        outputEstimates = {**inputEstimates}

        # the profiler will record selectivity and timing info for this operator,
        # which can be used to improve timing related estimates
        if cost_estimate_sample_data is not None:
            # compute estimates
            filter = f"(filter == '{str(self.filter)}') & (op_name == 'p_filter')"
            time_per_record = StatsProcessor._est_time_per_record(cost_estimate_sample_data, filter=filter)
            selectivity = StatsProcessor._est_selectivity(cost_estimate_sample_data, filter=filter)

            # estimate cardinality using sample selectivity and input cardinality est.
            cardinality = inputEstimates['cardinality'] * selectivity

            # update cardinality, timePerElement and related stats
            outputEstimates['cardinality'] = cardinality
            outputEstimates['timePerElement'] = time_per_record
            outputEstimates['cumulativeTimePerElement'] = inputEstimates['cumulativeTimePerElement'] + time_per_record
            outputEstimates['totalTime'] = cardinality * time_per_record + inputEstimates['totalTime']

            return outputEstimates

        # for now, assume applying the user function takes negligible additional time (and no cost in USD)
        outputEstimates["timePerElement"] = 0
        outputEstimates["usdPerElement"] = 0
        outputEstimates["estOutputTokensPerElement"] = 0

        return outputEstimates

    def __iter__(self):
        datadir = DataDirectory()
        shouldCache = datadir.openCache(self.targetCacheId)

        @self.profile(name="applyfn", shouldProfile=self.shouldProfile)
        def iteratorFn():
            for nextCandidate in self.source:
                try:
                    dr = self.fn.map(nextCandidate)
                    if shouldCache:
                        datadir.appendCache(self.targetCacheId, dr)
                    yield dr
                except Exception as e:
                    print("Error in applying function", e)
                    pass

            if shouldCache:
                datadir.closeCache(self.targetCacheId)

        return iteratorFn()


class ApplyAverageAggregateOp(PhysicalOp):
    def __init__(self, source: PhysicalOp, aggFunction: AggregateFunction, targetCacheId: str=None, shouldProfile=False):
        super().__init__(outputSchema=Number, shouldProfile=shouldProfile)
        self.source = source
        self.aggFunction = aggFunction
        self.targetCacheId = targetCacheId

        if not source.outputSchema == Number:
            raise Exception("Aggregate function AVERAGE is only defined over Numbers")

        # NOTE: need to construct profiler after all fields used by self.opId() are set
        self.profiler = Profiler(op_id=self.opId())
        self.profile = self.profiler.iter_profiler

    def __str__(self):
        return "ApplyAverageAggregateOp(" + str(self.outputSchema) + ", " + "Function: " + str(self.aggFunction) + ")"

    def opId(self):
        d = {
            "operator": "ApplyAverageAggregateOp",
            "source": self.source.opId(),
            "aggFunction": str(self.aggFunction),
            "targetCacheId": self.targetCacheId,
        }
        ordered = json.dumps(d, sort_keys=True)
        return hashlib.sha256(ordered.encode()).hexdigest()[:MAX_ID_CHARS]

    def estimateCost(self, cost_estimate_sample_data: List[Dict[str, Any]]=None):
        # get input estimates and pass through to output
        inputEstimates = self.source.estimateCost(cost_estimate_sample_data)
        outputEstimates = {**inputEstimates}

        # the profiler will record timing info for this operator, which can be used
        # to improve timing related estimates
        if cost_estimate_sample_data is not None:
            # compute estimates
            filter = f"(op_name == 'average')"
            time_per_record = StatsProcessor._est_time_per_record(cost_estimate_sample_data, filter=filter)

            # output cardinality for an aggregate will be 1
            cardinality = 1

            # update cardinality, timePerElement and related stats
            outputEstimates['cardinality'] = cardinality
            outputEstimates['timePerElement'] = time_per_record
            outputEstimates['cumulativeTimePerElement'] = inputEstimates['cumulativeTimePerElement'] + time_per_record
            outputEstimates['totalTime'] = cardinality * time_per_record + inputEstimates['totalTime']

            return outputEstimates

        # output cardinality for an aggregate will be 1
        outputEstimates["cardinality"] = 1

        # for now, assume applying the aggregate takes negligible additional time (and no cost in USD)
        outputEstimates["timePerElement"] = 0
        outputEstimates["usdPerElement"] = 0
        outputEstimates["estOutputTokensPerElement"] = 0

        return outputEstimates

    def __iter__(self):
        datadir = DataDirectory()
        shouldCache = datadir.openCache(self.targetCacheId)

        @self.profile(name="average", shouldProfile=self.shouldProfile)
        def iteratorFn():
            sum = 0
            counter = 0
            for nextCandidate in self.source:
                try:
                    sum += int(nextCandidate.value)
                    counter += 1
                except:
                    pass

            # NOTE: this will set the parent_uuid to be the uuid of the final source record;
            #       this is ideal for computing the op_time of the count operation, but maybe
            #       we should set this DataRecord as having multiple parents in the future
            dr = DataRecord(Number, parent_uuid=nextCandidate._uuid)
            dr.value = sum / float(counter)
            if shouldCache:
                datadir.appendCache(self.targetCacheId, dr)
            yield dr

            if shouldCache:
                datadir.closeCache(self.targetCacheId)

        return iteratorFn()


class LimitScanOp(PhysicalOp):
    def __init__(self, outputSchema: Schema, source: PhysicalOp, limit: int, targetCacheId: str=None, shouldProfile=False):
        super().__init__(outputSchema=outputSchema, shouldProfile=shouldProfile)
        self.source = source
        self.limit = limit
        self.targetCacheId = targetCacheId

        # NOTE: need to construct profiler after all fields used by self.opId() are set
        self.profiler = Profiler(op_id=self.opId())
        self.profile = self.profiler.iter_profiler

    def __str__(self):
        return "LimitScanOp(" + str(self.outputSchema) + ", " + "Limit: " + str(self.limit) + ")"

    def opId(self):
        d = {
            "operator": "LimitScanOp",
            "outputSchema": str(self.outputSchema),
            "source": self.source.opId(),
            "limit": self.limit,
            "targetCacheId": self.targetCacheId,
        }
        ordered = json.dumps(d, sort_keys=True)
        return hashlib.sha256(ordered.encode()).hexdigest()[:MAX_ID_CHARS]

    def estimateCost(self, cost_estimate_sample_data: List[Dict[str, Any]]=None):
        # get input estimates and pass through to output
        inputEstimates = self.source.estimateCost(cost_estimate_sample_data)
        outputEstimates = {**inputEstimates}
        
        # the profiler will record selectivity and timing info for this operator,
        # which can be used to improve timing related estimates
        if cost_estimate_sample_data is not None:
            # compute estimates
            filter = f"(filter == '{str(self.filter)}') & (op_name == 'p_filter')"
            time_per_record = StatsProcessor._est_time_per_record(cost_estimate_sample_data, filter=filter)

            # output cardinality for limit can be at most self.limit
            cardinality = min(self.limit, inputEstimates["cardinality"])

            # update cardinality, timePerElement and related stats
            outputEstimates['cardinality'] = cardinality
            outputEstimates['timePerElement'] = time_per_record
            outputEstimates['cumulativeTimePerElement'] = inputEstimates['cumulativeTimePerElement'] + time_per_record
            outputEstimates['totalTime'] = cardinality * time_per_record + inputEstimates['totalTime']

            return outputEstimates

        # output cardinality for limit can be at most self.limit
        outputEstimates["cardinality"] = min(self.limit, inputEstimates["cardinality"])

        return outputEstimates

    def __iter__(self):
        datadir = DataDirectory()
        shouldCache = datadir.openCache(self.targetCacheId)

        @self.profile(name="limit", shouldProfile=self.shouldProfile)
        def iteratorFn():
            counter = 0
            for nextCandidate in self.source: 
                if counter >= self.limit:
                    break
                if shouldCache:
                    datadir.appendCache(self.targetCacheId, nextCandidate)
                yield nextCandidate
                counter += 1

            if shouldCache:
                datadir.closeCache(self.targetCacheId)

        return iteratorFn()
