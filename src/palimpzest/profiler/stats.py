from __future__ import annotations

from palimpzest.constants import MODEL_CARDS

from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pandas as pd

import json

@dataclass
class OperatorStats:
    """
    Dataclass for storing statistics captured within a given operator.

    Note that not every field will be computed for a given operation. This class
    represents the union of all statistics computed across all types of operators.

    The other Stats classes (below) are concerned with the statistics computed for a
    single induce / filter operation executed on a single input record. This dataclass
    aggregates across all of those statistics (e.g. by capturing the total number
    of output tokens generated across all generations, the total number of records
    processed, etc.)
    """
    ############################
    ##### Universal Fields #####
    ############################
    # [set in Profiler.__init__]
    # the ID of the operation in which these stats were collected
    op_id: str=None
    # [set in Profiler.iter_profiler]
    # the name of the operation in which these stats were collected
    op_name: str=None
    # [set in PhysicalOp.getProfilingData]
    # the name of the operation in which these stats were collected
    source_op_stats: OperatorStats=None
    # [computed in Profiler.iter_profiler]
    # a list of record dictionaries processed by the operation; each record dictionary
    # has the format: {"op_id": str, "uuid": str "parent_uuid": str, "stats": Stats, **record.asDict()}
    records: List[Dict[str, Any]]=field(default_factory=list)
    # [computed in Profiler.iter_profiler]
    # total number of records returned by the iterator for this operator
    total_records: int=0
    # [computed in Profiler.iter_profiler]
    # total time spent in this iterator; this will include time spent in input operators
    total_cumulative_iter_time: float=0.0
    # [computed in Profiler.iter_profiler]
    # keep track of the total time spent inside of the profiler
    total_time_in_profiler: float=0.0
    # [computed in StatsProcessor]
    # total time spent in this operation; does not include time spent waiting on parent/source operators
    total_op_time: float=0.0

    ##############################################
    ##### Universal Induce and Filter Fields #####
    #####    [computed in StatsProcessor]    #####
    ##############################################
    # usage statistics computed for induce and filter operations
    total_input_tokens: int=0
    total_output_tokens: int=0
    # dollar cost associated with usage statistics
    total_input_usd: float=0.0
    total_output_usd: float=0.0
    total_usd: float=0.0
    # time spent waiting for LLM calls to return (in seconds)
    total_llm_call_duration: float=0.0
    # name of the model used for generation
    model_name: str=None
    # keep track of finish reasons
    finish_reasons: defaultdict[int]=field(default_factory=lambda: defaultdict(int))
    # record input fields and the output fields generated in an induce operation
    input_fields: List[str]=field(default_factory=list)
    generated_fields: List[str]=field(default_factory=list)
    # list of answers
    answers: List[str]=field(default_factory=list)
    # list of lists of token log probabilities for the subset of tokens that comprise the answer
    answer_log_probs: List[List[float]]=field(default_factory=list)

    #################################################
    ##### Field for Induce w/Conventional Query #####
    #####     [computed in StatsProcessor]      #####
    #################################################
    # record aggregated stats on a per-field basis for conventional induce queries
    per_field_op_stats: defaultdict[OperatorStats]=field(default_factory=lambda: defaultdict(lambda: OperatorStats()))

    #########################################
    #####  Fields for Induce w/CodeGen  #####
    #####  [computed in StatsProcessor] #####
    #########################################
    # record aggregated stats on a per-step basis for codegen queries
    # (here the steps are: initial code generation, advice generation, advised code generation)
    code_gen_op_stats: defaultdict[OperatorStats]=field(default_factory=lambda: defaultdict(lambda: OperatorStats()))

    ##########################################
    ##### Optional Non-LLM Induce Fields #####
    #####  [computed in StatsProcessor]  #####
    ##########################################
    # time spent waiting for API calls to return (in seconds)
    total_api_call_duration: float=0.0

    def to_dict(self):
        # create copy of self
        self_copy = deepcopy(self)

        # convert defaultdict -> dict before calling asdict()
        self_copy.finish_reasons = dict(self_copy.finish_reasons)
        self_copy.per_field_op_stats = dict(self_copy.per_field_op_stats)
        self_copy.code_gen_op_stats = dict(self_copy.code_gen_op_stats)

        # call to_dict() on source_op_stats
        if self_copy.source_op_stats is not None:
            self_copy.source_op_stats = self_copy.source_op_stats.to_dict()

        return asdict(self_copy)


@dataclass
class Stats:
    """
    Base dataclass for storing statistics captured during the execution of an induce / filter
    operation on a single input record.
    """
    # (set in profiler.py) this is the total time spent waiting for the iterator
    # to yield the data record associated with this Stats object; note that this
    # will capture the total time spent in this operation and all source operations
    # due to the way in which we set timers in profiler.py rather than in the
    # physical operator code
    cumulative_iter_time: float=None
    # the time spent by the data record just in this operation; this is computed
    # in the StatsProcessor as (the cumulative_iter_time of this data record in
    # this operation) minus (the cumulative_iter_time of this data record in the
    # source operation)
    op_time: float=None

    def to_dict(self):
        return asdict(self)

@dataclass
class ApiStats(Stats):
    """Staistics captured from a (non-LLM) API call."""
    # total time spent waiting for the API to return a response
    api_call_duration_secs: float=0.0

@dataclass
class GenerationStats(Stats):
    """Statistics captured from LLM calls (i.e. the Generator's .generate() functions)."""
    # name of the LLM used to generate the output; should be a key in the MODEL_CARDS
    model_name: str=None
    # total time spent from sending input to LLM to receiving output string
    llm_call_duration_secs: float=0.0
    # the prompt used to generate the output
    prompt: str=None
    # the usage dictionary for the LLM call
    usage: Dict[str, int]=field(default_factory=dict)
    # the reason the LLM stopped generating tokens
    finish_reason: str=None
    # the answer extracted from the generated output
    answer: str=None
    # the log probabilities for the subset of tokens that comprise the answer
    answer_log_probs: List[float]=None

@dataclass
class BondedQueryStats(Stats):
    """Statistics captured from bonded queries."""
    # the generation statistics from the call to a generator
    gen_stats: GenerationStats=None
    # the set of input fields passed into the generation
    input_fields: List[str]=field(default_factory=list)
    # the set of fields the bonded query was supposed to generate values for
    generated_fields: List[str]=field(default_factory=list)

@dataclass
class FieldQueryStats(Stats):
    """Statistics captured from generating the value for a single field in a conventional query."""
    # the generation statistics from the call to a generator
    gen_stats: GenerationStats=None
    # the field the statistics were generated for
    field_name: str=None

@dataclass
class ConventionalQueryStats(Stats):
    """Statistics captured from conventional queries."""
    # the list of FieldQueryStats for each field in the generated_fields
    field_query_stats_lst: List[FieldQueryStats]=field(default_factory=list)
    # the set of input fields passed into the generation
    input_fields: List[str]=field(default_factory=list)
    # the set of fields the conventional query was supposed to generate values for
    generated_fields: List[str]=field(default_factory=list)

@dataclass
class CodeGenStepStats(Stats):
    """
    Statistics captured from running the initial code generation step.
    In this step, the CODEGEN_PROMPT has formatting applied and then an initial
    piece of code is generated with a call to `llmCodeGen`, which returns the `code`
    and a GenerationStats object.
    """
    # CODEGEN_PROMPT after formatting is applied
    codegen_prompt: str=None
    # initial code generated by call to `llmCodeGen` with formatted CODEGEN_PROMPT
    code: str=None
    # the generation statistics from the call to `llmCodeGen`
    gen_stats: GenerationStats=None
    
@dataclass
class AdviceGenStepStats(Stats):
    """
    Statistics captured from running the advice generation step.
    In this step, the ADVICEGEN_PROMPT has formatting applied and then a set of
    pieces of advice are generated with a call to `llmAdviceGen`, which returns the
    `advices` and a GenerationStats object.
    """
    # ADVICEGEN_PROMPT after formatting is applied
    advicegen_prompt: str=None
    # list of advice strings generated by call to `llmAdviceGen` with formatted ADVICEGEN_PROMPT
    advices: List[str]=field(default_factory=list)
    # the generation statistics from the call to `llmAdviceGen`
    gen_stats: GenerationStats=None

@dataclass
class AdvisedCodeGenStepStats(Stats):
    """
    Statistics captured from running the and advised code generation step.
    In this step, the ADVICED_CODEGEN_PROMPT has formatting applied and then a
    piece of code is generated with a call to `llmCodeGen` for some given piece
    of `advice`. The call to `llmCodeGen` returns the `code` and a GenerationStats object.
    """
    # ADVICED_CODEGEN_PROMPT after formatting is applied
    adviced_codegen_prompt: str=None
    # the piece of advice used to in this advised code generation
    advice: str=None
    # the code generated with this piece of advice
    code: str=None
    # the generation statistics from the call to `llmCodeGen`
    gen_stats: GenerationStats=None

@dataclass
class FullCodeGenStats(Stats):
    """
    The entire set of statistics that can be captured from a call to the `codeGen` function.

    This includes:
    - stats for the initial code generation step
    - stats for the advice generation step
    - a list of stats for each code generation step with a given piece of advice
    """
    # stats from the initial code generation step
    init_code_gen_stats: CodeGenStepStats=None
    # stats from the advice generation step (this is an optional step)
    advice_gen_stats: AdviceGenStepStats=None
    # list of stats from the advised code generation step (one set of stats per-piece of advice)
    advised_code_gen_stats: List[AdvisedCodeGenStepStats]=field(default_factory=list)

@dataclass
class InduceLLMStats(Stats):
    """Dataclass containing all possible statistics which could be returned from an induce w/LLM operation."""
    # stats from bonded query
    bonded_query_stats: BondedQueryStats=None
    # stats from conventional query
    conventional_query_stats: ConventionalQueryStats=None
    # stats from code generation
    full_code_gen_stats: FullCodeGenStats=None

@dataclass
class InduceNonLLMStats(Stats):
    """Dataclass containing all possible statistics which could be returned from a hard-coded induce operation."""
    # stats containing time spent calling some external API
    api_stats: ApiStats=None

@dataclass
class FilterLLMStats(Stats):
    """Dataclass containing all possible statistics which could be returned from a filter operation."""
    # the generation statistics from the call to the filter LLM
    gen_stats: GenerationStats=None
    # the filter condition for this filter
    filter: str=None


class StatsProcessor:
    """
    This class implements a set of standardized functions for processing profiling statistics
    collected by PZ.

    TODO: implement other methods here to help with understanding profile data
    """
    def __init__(self, profiling_data: OperatorStats) -> None:
        """
        The profiling data is an OperatorStats object, which, when converted into a dict,
        has the following format:

        {
            #### unique identifier for this instance of this operator
            "op_id": str,
            #### name of this operator
            "op_name": str,
            #### total records processed by op
            "total_records": int,
            #### sum of cumulative_iter_time for records in op (in seconds)
            "total_cumulative_iter_time": float,
            #### sum of op_time for records in op (in seconds) -- this is computed in StatsProcessor.__init__()
            "total_op_time": float,
            #### total time spent inside of profiler code (in seconds)
            "total_time_in_profiler": float,
            #### total input and output tokens processed in op
            "total_input_tokens": int,
            "total_output_tokens": int,
            #### total dollars spent in op
            "total_input_usd": float,
            "total_output_usd": float,
            "total_usd": float,
            #### total time spent executing LLM calls in this op (in seconds)
            "total_llm_call_duration": float,
            #### distribution of finish reasons for LLM calls in this op
            "finish_reasons": Dict[str, int],
            #### total time spent waiting on non-LLM API calls (in seconds)
            "total_api_call_duration": float,
            #### name of the model used to perform generation (if operation used LLM)
            "model_name": str,
            #### the input fields for records coming into this op, and the fields this op generated
            "input_fields": List[str],
            "generated_fields": List[str],
            #### the list of answers
            "answers": List[str],
            #### list of lists of token log probabilities for the subset of tokens that comprise the answer
            "answer_log_probs": List[List[float]],
            #### ONLY for induce ops with conventional queries -- per-field breakdown of these operator_stats
            "per_field_op_stats": Dict[str, Dict[str, Any]],
            #### ONLY for induce ops with code gen -- per-state breakdown of these operator_stats
            "code_gen_op_stats": Dict[str, Dict[str, Any]],
            "records": [
                {
                    #### unique identifier for this instance of this operator
                    "op_id": str,
                    #### unique identifier for this record
                    "uuid": str,
                    #### unique identifier for the parent/source of this record
                    "parent_uuid": str,
                    #### 
                    "stats": {
                        #### Total time in seconds spent waiting for operator to yield this record; includes time spent in source operators
                        "cumulative_iter_time": float,
                        #### Total time in seconds spent by record in this operator -- this is computed in StatsProcessor.__init__()
                        "op_time": 0.0,
                        #### per-record stats; can include zero-or-more of the following fields:
                        ## for induce operations with an LLM
                        "bonded_query_stats": Dict[str, Any],
                        "conventional_query_stats": Dict[str, Any],
                        "full_code_gen_stats": Dict[str, Any],
                        ## for induce operations w/out an LLM
                        "api_stats": Dict[str, Any],
                        ## for filter operations
                        "gen_stats: Dict[str, Any],
                    },
                    #### dictionary representation of the record after being processed by this operator
                    "<field-name-1>": value1,
                    "<field-name-2>": value2,
                },
                ...
            ],
            #### the data structure recurses until the original source operation (e.g. a scan) is reached
            "source": {
                "op_id": str,
                "op_name": str,
                ...
                "records": [...],
                "source": {
                    ...
                }
            },
        }
        """
        # compute aggregate stats for the operator
        self.profiling_data = self._compute_agg_op_stats(profiling_data)

        # compute op_time for each record and the total_op_time for each operator
        self.profiling_data = self._compute_op_time(self.profiling_data)


    def _update_gen_stats(self, profiling_data: OperatorStats, gen_stats: GenerationStats, field_name: str=None, code_gen_step: str=None) -> OperatorStats:
        """
        This helper function takes in profiling data and a GenerationStats object and updates
        the profiling data's aggregate stats fields with the data from gen_stats.

        One important detail is that for conventional queries, we create a GenerationStats
        object per-record and per-field. Thus, for these queries we not only want to keep
        track of the aggregate stats across all generated fields, but we also want to keep
        track of the aggregate stats on a per-field basis. (This is worth tracking because
        certain fields can be significantly more expensive to generate than others).

        Because of this, the OperatorStats object has a recursively defined field
        called `per_field_op_stats`, which is a dictionary mapping field names
        to an OperatorStats object which tracks the aggregate stats for generating
        that specific field.

        So to summarize:
        - OperatorStats has a set of top-level fields which contain statistics aggregated
            across all fields AND all records for the given operation.
            - For all but one operation, these are the only aggregates we compute

        - OperatorStats also has a top-level field called `per_field_op_stats` which
            stores aggregates for each generated field, where the aggregation is only across all records.
            - `per_field_op_stats` will only ever be filled for induce operations with a conventional query

        Finally, we have a similar recursive structure for our Code Generation statistics.
        OperatorStats has a top-level field called `code_gen_op_stats` which tracks
        aggregate statistics across: initial code generation, advice generation, and advised
        code generation. This is conceptually identical to what we just described with
        per-field aggregates in `per_field_op_stats`, except instead of breaking down
        the statistics per-field we break them down per code generation step.
        """
        def _update_aggregates(agg_op_stats: OperatorStats, gen_stats: GenerationStats) -> OperatorStats:
            # update timing and token stats
            agg_op_stats.total_llm_call_duration += gen_stats.llm_call_duration_secs
            agg_op_stats.total_input_tokens += gen_stats.usage["prompt_tokens"]
            agg_op_stats.total_output_tokens += gen_stats.usage["completion_tokens"]

            # compute and update USD cost of generation
            usd_per_input_token = MODEL_CARDS[gen_stats.model_name]["usd_per_input_token"]
            usd_per_output_token = MODEL_CARDS[gen_stats.model_name]["usd_per_output_token"]
            agg_op_stats.total_input_usd += gen_stats.usage["prompt_tokens"] * usd_per_input_token
            agg_op_stats.total_output_usd += gen_stats.usage["prompt_tokens"] * usd_per_output_token
            agg_op_stats.total_usd = agg_op_stats.total_input_usd + agg_op_stats.total_output_usd

            # update distribution of finish reasons
            agg_op_stats.finish_reasons[gen_stats.finish_reason] += 1

            # update list of answer logprobs
            agg_op_stats.answer_log_probs.append(gen_stats.answer_log_probs)

            # update list of answers
            agg_op_stats.answers.append(gen_stats.answer)

            # NOTE: this assumes a single model is used w/in an operation, which is currently true
            # update model name
            agg_op_stats.model_name = gen_stats.model_name

            return agg_op_stats

        # If this method is invoked without a field_name or code_gen_step (i.e. field_name=None
        # and code_gen_step=None), then we simply update the top-level aggregate stats
        # (i.e. the aggregates across all LLM generations)
        if field_name is None and code_gen_step is None:
            profiling_data = _update_aggregates(profiling_data, gen_stats)

        # If the method is invoked with a field_name, then we update the `per_field_op_stats` for that field
        if field_name is not None:
            profiling_data.per_field_op_stats[field_name] = _update_aggregates(profiling_data.per_field_op_stats[field_name], gen_stats)

        # If the method is invoked with a code_gen_step, then we update the `code_gen_op_stats` for that step
        if code_gen_step is not None:
            profiling_data.code_gen_op_stats[code_gen_step] = _update_aggregates(profiling_data.code_gen_op_stats[code_gen_step], gen_stats)

        return profiling_data


    def _update_code_gen_stats(self, profiling_data: OperatorStats, full_code_gen_stats: FullCodeGenStats) -> OperatorStats:
        """
        This helper function takes in profiling data and a FullCodeGenStats object and updates
        the profiling data's aggregate stats fields with the data from full_code_gen_stats.

        A FullCodeGenStats object is a bit of a beast. It currently contains the following:
        - A CodeGenStepStats object for the initial code generation
        - An AdviceGenStepStats object for the initial advice generation
        - A list of AdvisedCodeGenStepStats objects -- one per piece of advice -- which is used
          to perform a new (and ideally better) code generation

        Each of these stats objects within FullCodeGenStats is a wrapper around a GenerationStats
        object, which we already know how to process (see _update_gen_stats above).

        Thus, similar to how we handle conventional queries, this function will update
        the aggregate operator stats (aggregated across all generation calls) AND it will
        update aggregate stats for:
        - initial code generations
        - advice generations
        - advised code generations
        """
        # get stats for initial code generation step
        gen_stats = full_code_gen_stats.init_code_gen_stats.gen_stats

        # update aggregate operator stats and the stats for the initial code generation step
        profiling_data = self._update_gen_stats(profiling_data, gen_stats)
        profiling_data = self._update_gen_stats(profiling_data, gen_stats, code_gen_step="init_code_gen")

        # get stats for advice generation step
        gen_stats = full_code_gen_stats.advice_gen_stats.gen_stats

        # update aggregate operator stats and the stats for the advice generation step
        profiling_data = self._update_gen_stats(profiling_data, gen_stats)
        profiling_data = self._update_gen_stats(profiling_data, gen_stats, code_gen_step="advice_gen")

        # get stats for each advised code generation (one per-piece of generated advice)
        for advised_code_gen in full_code_gen_stats.advised_code_gen_stats:
            gen_stats = advised_code_gen.gen_stats

            # update aggregate operator stats and the stats for the advised code generation step
            profiling_data = self._update_gen_stats(profiling_data, gen_stats)
            profiling_data = self._update_gen_stats(profiling_data, gen_stats, code_gen_step="advised_code_gen")

        return profiling_data


    def _compute_agg_op_stats(self, profiling_data: OperatorStats) -> OperatorStats:
        """
        This function computes the aggregate fields for the given OperatorStats object (`profiling_data`).
        The OperatorStats object has a `records` field which is a a list of record dictionaries processed
        by the operation. Each record dictionary has the format:
        
        {"op_id": str, "uuid": str "parent_uuid": str, "stats": Stats, **record.asDict()}

        A record dictionary's "stats" field must be one of:
        - Stats
        - InduceNonLLMStats
        - InduceLLMStats
        - FilterLLMStats

        Stats is only present for non-induce/filter operations, and its only field will
        be the cumulative_iter_time for the record.

        InduceNonLLMStats is either empty or has a single field (api_stats).

        InduceLLMStats is the most complex Stats object, it can contain one or more of
        the following sub-fields:
        - bonded_query_stats
        - conventional_query_stats
        - full_code_gen_stats

        FilterLLMStats has a single field gen_stats which is guaranteed to be filled.
        """
        for record_dict in profiling_data.records:
            # retrieve stats for this operation
            stats = record_dict["stats"]

            # non-LLM induce objects will have no stats or a single ApiStats object
            if isinstance(stats, InduceNonLLMStats):
                api_stats = stats.api_stats
                if api_stats is not None:
                    profiling_data.total_api_call_duration += api_stats.api_call_duration_secs

            # LLM induce objects are the most complex; they may have one or more of:
            # - BondedQueryStats
            # - ConventionalQueryStats
            # - FullCodeGenStats
            elif isinstance(stats, InduceLLMStats):
                # process bonded query stats
                bonded_query_stats = stats.bonded_query_stats
                if bonded_query_stats is not None:
                    # set input fields and output fields generated by induce operation
                    profiling_data.input_fields = bonded_query_stats.input_fields
                    profiling_data.generated_fields = bonded_query_stats.generated_fields

                    # update the aggregate operator stats associated with LLM generation
                    profiling_data = self._update_gen_stats(profiling_data, bonded_query_stats.gen_stats)

                # process conventional query stats
                conventional_query_stats = stats.conventional_query_stats
                if conventional_query_stats is not None:
                    # set input fields and output fields generated by induce operation
                    profiling_data.input_fields = conventional_query_stats.input_fields
                    profiling_data.generated_fields = conventional_query_stats.generated_fields

                    # update the aggregate (and per-field aggregate) operator stats associated with LLM generation
                    for field_query_stats in conventional_query_stats.field_query_stats_lst:
                        field_name = field_query_stats.field_name
                        field_gen_stats = field_query_stats.gen_stats
                        profiling_data = self._update_gen_stats(profiling_data, field_gen_stats)
                        profiling_data = self._update_gen_stats(profiling_data, field_gen_stats, field_name=field_name)

                # process codegen stats
                full_code_gen_stats = stats.full_code_gen_stats
                if full_code_gen_stats is not None:
                    profiling_data = self._update_code_gen_stats(profiling_data, full_code_gen_stats)

            # filter llm objects will have a single GenerationStats object
            elif isinstance(stats, FilterLLMStats):
                # update aggregate statistics with filter generation stats
                profiling_data = self._update_gen_stats(profiling_data, stats.gen_stats)

        return profiling_data


    def _compute_op_time(self, profiling_data: OperatorStats) -> OperatorStats:
        """
        This helper function computes the time spent by each record in each operation
        (i.e. the record's op_time). It then aggregates the op_times for every record
        in each operation to get a total_op_time.

        Inside the profiler we are only able to track the time it takes for a record
        to be yielded by the operator's iterator method. This time (stored in
        "cumulative_iter_time") is cumulative in the sense that it also captures time
        spent waiting for source/parent operators to yield this record.

        In this function, for each record we effectively compute:

        op_time = (cumulative_iter_time) - (the cumulative_iter_time of this record's parent)

        Once we've computed each record's op_time we finally compute the total_op_time
        for each operator.
        """
        # base case: this is the source operation
        if profiling_data.source_op_stats is None:
            # in this case: op_time == cumulative_iter_time
            for record_dict in profiling_data.records:
                record_dict['stats'].op_time = record_dict['stats'].cumulative_iter_time

            # compute total_op_time
            profiling_data.total_op_time = sum(list(map(lambda record_dict: record_dict['stats'].op_time, profiling_data.records)))

            return profiling_data

        # TODO: this is N^2 in # of records; we may want to use a dictionary to speed this up
        # for each record we need to identify its parent to compute the op_time
        # NOTE: source_op_stats will be a dictionary b/c profiling_data
        for record_dict in profiling_data.records:
            uuid = record_dict['uuid']
            parent_uuid = record_dict['parent_uuid']
            for source_record_dict in profiling_data.source_op_stats.records:
                # NOTE: right now, because some operations create new DataRecord objects (e.g. induce, agg.)
                #       while other operations pass through the same record (e.g. filter, limit), there are
                #       two possible scenarios:
                #         1. the record's parent_uuid will equal the source_record's uuid (in the induce/agg case)
                #         2. the record's uuid will equal the source_record's uuid (in the filter/limit case)
                if parent_uuid == source_record_dict['uuid'] or uuid == source_record_dict['uuid']:
                    record_dict['stats'].op_time = record_dict['stats'].cumulative_iter_time - source_record_dict['stats'].cumulative_iter_time

        # compute total_op_time
        profiling_data.total_op_time = sum(list(map(lambda record_dict: record_dict['stats'].op_time, profiling_data.records)))

        # recurse
        profiling_data.source_op_stats = self._compute_op_time(profiling_data.source_op_stats)

        return profiling_data

    @staticmethod
    def _est_time_per_record(cost_est_sample_data: List[Dict[str, Any]], filter: str=None, agg: str="mean") -> float:
        """
        Given sample cost data observations, and potentially a filter to identify a unique operator,
        compute the aggregate over the `op_time` column.
        """
        # return self.profiling_data['total_op_time'] / self.profiling_data['total_records']

        # convert data to dataframe and filter if applicable
        df = pd.DataFrame(cost_est_sample_data)
        if filter is not None:
            df = df.query(filter)

        # compute aggregate
        return df['op_time'].agg(agg=agg).iloc[0]

    @staticmethod
    def _est_num_input_output_tokens(cost_est_sample_data: List[Dict[str, Any]], filter: str=None, agg: str="mean") -> Tuple[float, float]:
        """
        Given sample cost data observations, and potentially a filter to identify a unique operator,
        compute the aggregate over the `num_input_tokens` and `num_output_tokens` columns.
        """
        # avg_num_input_tokens = self.profiling_data['total_input_tokens'] / self.profiling_data['total_records']
        # avg_num_output_tokens = self.profiling_data['total_output_tokens'] / self.profiling_data['total_records']
        # return avg_num_input_tokens, avg_num_output_tokens

        # convert data to dataframe and filter if applicable
        df = pd.DataFrame(cost_est_sample_data)
        if filter is not None:
            df = df.query(filter)

        # compute aggregate
        return df['num_input_tokens'].agg(agg=agg).iloc[0], df['num_output_tokens'].agg(agg=agg).iloc[0]

    @staticmethod
    def _est_usd_per_record(cost_est_sample_data: List[Dict[str, Any]], filter: str=None, agg: str="mean") -> float:
        """
        Given sample cost data observations, and potentially a filter to identify a unique operator,
        compute the aggregate over the sum of the `input_usd` and `output_usd` columns.
        """
        # return op_agg_stats['total_usd'] / op_agg_stats['total_records']

        # convert data to dataframe and filter if applicable
        df = pd.DataFrame(cost_est_sample_data)
        if filter is not None:
            df = df.query(filter)

        # compute average combined input/output usd spent
        return (df['input_usd'] + df['output_usd']).agg(agg=agg).iloc[0]

    @staticmethod
    def _est_selectivity(cost_est_sample_data: List[Dict[str, Any]], filter: str) -> float:
        """
        Given sample cost data observations and a filter to identify a unique operator,
        compute the ratio of records between this operator and its source operator.
        """
        # return op_agg_stats['total_records'] / source_op_agg_stats['total_records']

        # convert data to dataframe and filter if applicable
        df = pd.DataFrame(cost_est_sample_data)

        # get subset of records matching filter (this should uniquely identify an operator)
        op_df = df.query(filter)

        # get subset of records that were the source to this operator
        source_op_id = op_df.source_op_id.iloc[0]
        source_op_df = df.query(f"op_id=='{source_op_id}'")

        return len(op_df) / len(source_op_df)

    @staticmethod
    def _est_quality(cost_est_sample_data: List[Dict[str, Any]], filter: str=None) -> float:
        """
        Given an operator's aggregate stats, estimate the quality of its answers as
        an average of its model quality as measured by is MMLU score and the average of its
        answer token log probabilities.
        """
        # # we assume perfect quality for non-LLM operations
        # est_quality = 1.0

        # # if we generated outputs with an LLM, estimate quality as the average of
        # # the general model quality and the avg. ouput token log probability
        # if len(op_agg_stats['answers']) > 0:
        #     all_answer_log_probs = np.array(op_agg_stats['answer_log_probs'])
        #     avg_token_log_probability = np.mean(all_answer_log_probs)
        #     model_quality = (MODEL_CARDS[op_agg_stats['model_name']]['MMLU'] / 100.0)
        #     est_quality = np.mean([model_quality, avg_token_log_probability]) 

        # return est_quality

        # convert data to dataframe and filter if applicable
        df = pd.DataFrame(cost_est_sample_data)
        if filter is not None:
            df = df.query(filter)

        # get all answer token log probabilities and compute the mean
        all_answer_log_probs = np.array([
            log_prob
            for log_probs in df.answer_log_probs.tolist()
            for log_prob in log_probs
        ])
        avg_token_log_probability = np.mean(all_answer_log_probs)

        # get prior believe of model quality
        model_name = df.model_name.iloc[0]
        model_quality = (MODEL_CARDS[model_name]['MMLU'] / 100.0)

        # compute true mean of model's prior quality and our measurement of avg. log probability
        # NOTE: recall that avg_token_log_probability will be a small neg. number
        est_quality = np.mean([model_quality, 1 + avg_token_log_probability])

        return est_quality

    def _parse_record_llm_stats(self, record_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract gen_stats fields for get_cost_estimate_sample_data.
        """
        # create OperatorStats object with a single record
        op_stats = OperatorStats()
        op_stats.records = [record_dict]

        # re-use _compute_agg_op_stats to compute statistics across all possible stats objects
        op_stats = self._compute_agg_op_stats(op_stats)

        # get values needed to compute observation metrics
        additional_fields_dict = {
            "model_name": op_stats.model_name,
            "input_fields": "-".join(sorted(op_stats.input_fields)),
            "generated_fields": "-".join(sorted(op_stats.generated_fields)),
            "num_input_tokens": op_stats.total_input_tokens,
            "num_output_tokens": op_stats.total_output_tokens,
            "input_usd": op_stats.total_input_usd,
            "output_usd": op_stats.total_output_usd,
            "answer": op_stats.answers[0] if len(op_stats.answers) > 0 else None,
            "answer_log_probs": op_stats.answer_log_probs[0] if len(op_stats.answer_log_probs) > 0 else None,
        }

        return additional_fields_dict


    def get_cost_estimate_sample_data(self) -> List[Dict[str, Any]]:
        """
        This function returns a dataset of observations of key statistics which
        can be used to improve our physical operators cost estimates.
        """
        # initialize operator data variable
        op_data = self.profiling_data

        # construct table of observation data from sample batch of processed records
        cost_est_sample_data = []
        while op_data is not None:
            # append observation data for each record
            for record_dict in op_data.records:
                # compute minimal observation which is supported by all operators
                # TODO: one issue with this setup is that cache_scans of previously computed queries
                #       may not match w/these observations due to the diff. op_name
                observation = {
                    "op_id": op_data.op_id,
                    "op_name": op_data.op_name,
                    "source_op_id": op_data.source_op_stats.op_id if op_data.source_op_stats is not None else None,
                    "op_time": record_dict["stats"].op_time,
                }

                # add additional fields for induce or filter w/LLM
                additional_fields_dict = self._parse_record_llm_stats(record_dict)
                observation = dict(observation, **additional_fields_dict)

                # add observation to list of observations
                cost_est_sample_data.append(observation)

            # update op_data
            op_data = op_data.source_op_stats

        return cost_est_sample_data


    def get_avg_record_stats(self):
        """
        Return a representation of an average trace for a record. E.g., it
        starts in such and such operation and takes blah seconds to load
        on avg., on median, p95, p99, max, etc. Then it goes to induce...
        """
        pass

    def get_operator_aggregate_stats(self):
        """
        Return mapping op_id -> agg. stats. Also include computation tree.
        Also compute mean, median, p95, p99, max stats.
        """
        pass

    def get_output_record_lineages(self):
        """
        Get the lineage of transformations for each record in the final output
        result set. The output is a list of lists, where each element in the outer
        list represents a lineage of computation, and each element in the inner list
        (i.e. the lineage of computation) is the state of each record after each
        physical operation.
        """
        pass

    def get_input_record_lineages(self):
        """
        Get the lineage of transformations for each record in the input set.
        The output is a list of lists, where each element in the outer list
        represents a lineage of computation, and each element in the inner list
        (i.e. the lineage of computation) is the state of each record after each
        physical operation.
        """
        pass
