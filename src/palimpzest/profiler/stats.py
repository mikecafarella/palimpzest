from __future__ import annotations

from palimpzest.constants import Model, MODEL_CARDS

from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pandas as pd

import ast
import json
import re

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
    # has the format: {"op_id": str, "uuid": str "parent_uuid": str, "stats": Stats, **record._asDict()}
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
    # the string associated with the filter
    filter: str=None
    # keep track of finish reasons
    finish_reasons: defaultdict[int]=field(default_factory=lambda: defaultdict(int))
    # record input fields and the output fields generated in an induce operation
    input_fields: List[str]=field(default_factory=list)
    generated_fields: List[str]=field(default_factory=list)
    # list of answers
    answers: List[str]=field(default_factory=list)
    # list of lists of token log probabilities for the subset of tokens that comprise the answer
    answer_log_probs: List[List[float]]=field(default_factory=list)
    # the query strategy used
    query_strategy: str=None
    # the token budget used during generation
    token_budget: float = None

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
    # # (here the steps are: initial code generation, advice generation, advised code generation)
    # # code_gen_op_stats: defaultdict[OperatorStats]=field(default_factory=lambda: defaultdict(lambda: OperatorStats()))
    total_code_exec_duration_secs: float=0.0

    ##########################################
    ##### Optional Non-LLM Induce Fields #####
    #####  [computed in StatsProcessor]  #####
    ##########################################
    # time spent waiting for API calls to return (in seconds)
    total_api_call_duration: float=0.0

    ##########################################
    ##### Optional Non-LLM Filter Fields #####
    #####  [computed in StatsProcessor]  #####
    ##########################################
    # time spent waiting for API calls to return (in seconds)
    total_fn_call_duration: float=0.0

    def to_dict(self):
        # create copy of self
        self_copy = deepcopy(self)

        # convert defaultdict -> dict before calling asdict()
        self_copy.finish_reasons = dict(self_copy.finish_reasons)
        self_copy.per_field_op_stats = dict(self_copy.per_field_op_stats)

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
    input_fields: List[str]=None
    # the set of fields the bonded query was supposed to generate values for
    generated_fields: List[str]=None

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
class CodeGenSingleStats(Stats):
    prompt_template: str=None
    context: Dict[str, Any]=field(default_factory=dict)
    code: str=None
    gen_stats: GenerationStats=field(default_factory=GenerationStats)

@dataclass
class CodeGenEnsembleStats(Stats):
    advice_gen_stats: GenerationStats=None
    code_versions_stats: Dict[str, CodeGenSingleStats]=field(default_factory=dict)

@dataclass
class CodeExecutionSingleStats(Stats):
    code_response: Any=None
    code_exec_duration_secs: float=0.0

@dataclass
class CodeExecutionEnsembleStats(Stats):
    majority_response: Any=None
    code_versions_stats: Dict[str, CodeExecutionSingleStats]=field(default_factory=dict)

@dataclass
class FullCodeGenStats(Stats):
    code_gen_stats: Dict[str, CodeGenEnsembleStats]=field(default_factory=dict)
    code_exec_stats: Dict[str, CodeExecutionEnsembleStats]=field(default_factory=dict)
# TODO: aggregate other information about codegen (majority_response, code, etc.)

@dataclass
class InduceLLMStats(Stats):
    """Dataclass containing all possible statistics which could be returned from an induce w/LLM operation."""
    # query strategy used
    query_strategy: str=None
    # the token budget used during generation
    token_budget: float = None
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

@dataclass
class FilterNonLLMStats(Stats):
    """Dataclass containing all possible statistics which could be returned from a filter operation."""
    # the time it takes to execute the filter function
    fn_call_duration_secs: float=None
    # the filter condition for this filter
    filter: str=None

# TODO: move to utils
def _get_JSON_from_answer(answer: str) -> Dict[str, Any]:
    """
    This function parses an LLM response which is supposed to output a JSON object
    and optimistically searches for the substring containing the JSON object.
    """
    if not answer.strip().startswith('{'):
        # Find the start index of the actual JSON string
        # assuming the prefix is followed by the JSON object/array
        start_index = answer.find('{') if '{' in answer else answer.find('[')
        if start_index != -1:
            # Remove the prefix and any leading characters before the JSON starts
            answer = answer[start_index:]

    if not answer.strip().endswith('}'):
        # Find the end index of the actual JSON string
        # assuming the suffix is preceded by the JSON object/array
        end_index = answer.rfind('}') if '}' in answer else answer.rfind(']')
        if end_index != -1:
            # Remove the suffix and any trailing characters after the JSON ends
            answer = answer[:end_index + 1]

    # Handle weird escaped values. I am not sure why the model
    # is returning these, but the JSON parser can't take them
    answer = answer.replace("\_", "_")

    # Handle comments in the JSON response. Use regex from // until end of line 
    answer = re.sub(r'\/\/.*$', '', answer, flags=re.MULTILINE)
    return json.loads(answer)


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
            #### total time spent waiting on non-LLM filter function calls (in seconds)
            "total_fn_call_duration": float,
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


    def _update_gen_stats(self, profiling_data: OperatorStats, gen_stats: GenerationStats, field_name: str=None) -> OperatorStats:
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
            if gen_stats is not None:
                # update timing and token stats
                agg_op_stats.total_llm_call_duration += gen_stats.llm_call_duration_secs
                agg_op_stats.total_input_tokens += gen_stats.usage["prompt_tokens"]
                agg_op_stats.total_output_tokens += gen_stats.usage["completion_tokens"]

                # compute and update USD cost of generation
                usd_per_input_token = MODEL_CARDS[gen_stats.model_name]["usd_per_input_token"]
                usd_per_output_token = MODEL_CARDS[gen_stats.model_name]["usd_per_output_token"]
                agg_op_stats.total_input_usd += gen_stats.usage["prompt_tokens"] * usd_per_input_token
                agg_op_stats.total_output_usd += gen_stats.usage["completion_tokens"] * usd_per_output_token
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
        if field_name is None:
            profiling_data = _update_aggregates(profiling_data, gen_stats)

        # If the method is invoked with a field_name, then we update the `per_field_op_stats` for that field
        if field_name is not None:
            profiling_data.per_field_op_stats[field_name] = _update_aggregates(profiling_data.per_field_op_stats[field_name], gen_stats)

        return profiling_data


    def _update_code_gen_stats(self, profiling_data: OperatorStats, full_code_gen_stats: FullCodeGenStats) -> OperatorStats:
        
        code_gen_stats = [ens_stats for field, ens_stats in full_code_gen_stats.code_gen_stats.items()]
        advice_gen_stats = [ens_stats.advice_gen_stats for ens_stats in code_gen_stats]
        code_ensemble_gen_stats = [[s.gen_stats for _, s in ens_stats.code_versions_stats.items()] for ens_stats in code_gen_stats]
        
        for s in advice_gen_stats:
            profiling_data = self._update_gen_stats(profiling_data, s)
        for s in code_ensemble_gen_stats:
            for s_ in s:
                profiling_data = self._update_gen_stats(profiling_data, s_)
        
        exec_stats = [[s.code_exec_duration_secs for _, s in ens_stats.code_versions_stats.items()] for field, ens_stats in full_code_gen_stats.code_exec_stats.items()]
        profiling_data.total_code_exec_duration_secs += sum([sum(t, 0.0) for t in exec_stats], 0.0)

        return profiling_data

    def _aggregate_record_stats(self, profiling_data: OperatorStats) -> OperatorStats:
        """
        Implements the aggregation functionality of _compute_agg_op_stats.
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
                # set query strategy
                profiling_data.query_strategy = stats.query_strategy
                profiling_data.token_budget = stats.token_budget

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

            # non-LLM induce objects will have no stats or a single ApiStats object
            elif isinstance(stats, FilterNonLLMStats):
                profiling_data.filter = stats.filter
                profiling_data.total_fn_call_duration += stats.fn_call_duration_secs

            # filter llm objects will have a single GenerationStats object
            elif isinstance(stats, FilterLLMStats):
                # update aggregate statistics with filter generation stats
                profiling_data.filter = stats.filter
                profiling_data = self._update_gen_stats(profiling_data, stats.gen_stats)

        return profiling_data

    def _compute_agg_op_stats(self, profiling_data: OperatorStats) -> OperatorStats:
        """
        This function computes the aggregate fields for the given OperatorStats object (`profiling_data`).
        The OperatorStats object has a `records` field which is a a list of record dictionaries processed
        by the operation. Each record dictionary has the format:
        
        {"op_id": str, "uuid": str "parent_uuid": str, "stats": Stats, **record._asDict()}

        A record dictionary's "stats" field must be one of:
        - Stats
        - InduceNonLLMStats
        - InduceLLMStats
        - FilterNonLLMStats
        - FilterLLMStats

        Stats is only present for non-induce/filter operations, and its only field will
        be the cumulative_iter_time for the record.

        InduceNonLLMStats is either empty or has a single field (api_stats).

        InduceLLMStats is the most complex Stats object, it can contain one or more of
        the following sub-fields:
        - bonded_query_stats
        - conventional_query_stats
        - full_code_gen_stats

        FilterNonLLMStats has a single field (fn_call_duration_secs)

        FilterLLMStats has a single field gen_stats which is guaranteed to be filled.
        """
        # base case: this is the source operation
        if profiling_data.source_op_stats is None:
            return self._aggregate_record_stats(profiling_data)

        # compute aggregates for this set of profiling data
        profiling_data = self._aggregate_record_stats(profiling_data)

        # recurse
        profiling_data.source_op_stats = self._compute_agg_op_stats(profiling_data.source_op_stats)

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
    def _est_time_per_record(op_df: pd.DataFrame, model_name: str=None, agg: str="mean") -> float:
        """
        Given sample cost data observations for a specific operation, compute the aggregate over
        the `op_time` column.
        """
        # use model-specific estimate if possible
        if model_name is not None:
            model_df = op_df[op_df.model_name == model_name]
            if not model_df.empty:
                return model_df['op_time'].agg(agg=agg).iloc[0]

        # compute aggregate
        return op_df['op_time'].agg(agg=agg).iloc[0]

    @staticmethod
    def _est_num_input_output_tokens(op_df: pd.DataFrame, model_name: str=None, agg: str="mean") -> Tuple[float, float]:
        """
        Given sample cost data observations for a specific operation, compute the aggregate over
        the `num_input_tokens` and `num_output_tokens` columns.
        """
        # use model-specific estimate if possible
        if model_name is not None:
            model_df = op_df[op_df.model_name == model_name]
            if not model_df.empty:
                return model_df['num_input_tokens'].agg(agg=agg).iloc[0], model_df['num_output_tokens'].agg(agg=agg).iloc[0]

        # compute aggregate
        return op_df['num_input_tokens'].agg(agg=agg).iloc[0], op_df['num_output_tokens'].agg(agg=agg).iloc[0]

    @staticmethod
    def _est_usd_per_record(op_df: pd.DataFrame, model_name: str=None, agg: str="mean") -> float:
        """
        Given sample cost data observations for a specific operation, compute the aggregate over
        the sum of the `input_usd` and `output_usd` columns.
        """
        # use model-specific estimate if possible
        if model_name is not None:
            model_df = op_df[op_df.model_name == model_name]
            if not model_df.empty:
                return (model_df['input_usd'] + model_df['output_usd']).agg(agg=agg).iloc[0]

        # # adjust cost from sample model to this model
        # this_model_usd_per_input_token = MODEL_CARDS[model_name]['usd_per_input_token']
        # model_to_usd_per_input_token = {
        #     model_name: model_card['usd_per_input_token']
        #     for model_name, model_card in MODEL_CARDS.items()
        # }
        # this_model_usd_per_output_token = MODEL_CARDS[model_name]['usd_per_output_token']
        # model_to_usd_per_output_token = {
        #     model_name: model_card['usd_per_output_token']
        #     for model_name, model_card in MODEL_CARDS.items()
        # }
        # df.loc[:, 'adj_input_usd'] = df.apply(lambda row: row['input_usd'] * (this_model_usd_per_input_token / model_to_usd_per_input_token[row['model_name']]), axis=1)
        # df.loc[:, 'adj_output_usd'] = df.apply(lambda row: row['output_usd'] * (this_model_usd_per_output_token / model_to_usd_per_output_token[row['model_name']]), axis=1)

        # # compute average combined input/output usd spent
        # return (df['adj_input_usd'] + df['adj_output_usd']).agg(agg=agg).iloc[0]
        return (op_df['input_usd'] + op_df['output_usd']).agg(agg=agg).iloc[0]

    @staticmethod
    def _est_selectivity(df: pd.DataFrame, op_df: pd.DataFrame, model_name: str=None) -> float:
        """
        Given sample cost data observations for the plan and a specific operation, compute
        the ratio of records between this operator and its source operator.
        """
        # use model-specific estimate if possible
        if model_name is not None:
            model_op_df = op_df[op_df.model_name == model_name]
            if not model_op_df.empty:
                num_input_records = model_op_df.shape[0]

                # get subset of records that were the source to this operator
                op_name = str(model_op_df.op_name.iloc[0])
                num_output_records = None
                if "filter" in op_name:
                    num_output_records = model_op_df.passed_filter.sum()
                else:
                    op_ids = model_op_df.op_id.unique().tolist()
                    num_output_records = df[df.source_op_id.isin(op_ids)].shape[0]

                return num_output_records / num_input_records

        # otherwise average selectivity across all ops
        num_input_records = op_df.shape[0]

        # get subset of records that were the source to this operator
        op_name = str(op_df.op_name.iloc[0])
        num_output_records = None
        if "filter" in op_name:
            num_output_records = op_df.passed_filter.sum()
        else:
            op_ids = op_df.op_id.unique().tolist()
            num_output_records = df[df.source_op_id.isin(op_ids)].shape[0]

        return num_output_records / num_input_records

    @staticmethod
    def _est_quality(op_df: pd.DataFrame, model_name: str=None) -> float:
        """
        Given sample cost data observations for a specific operation, compute the an estimate
        of the quality of its outputs by using GPT-4 as a champion model.
        """
        # get unique set of records
        record_uuids = op_df.record_uuid.unique()

        # compute GPT-4's answer (per-record) across all models; fall-back to most common answer if GPT-4 is not present
        record_uuid_to_answer = {}
        for record_uuid in record_uuids:
            record_df = op_df[op_df.record_uuid == record_uuid]
            gpt4_most_common_answer = record_df[record_df.model_name == Model.GPT_4.value].answer.mode()

            if not gpt4_most_common_answer.empty:
                record_uuid_to_answer[record_uuid] = gpt4_most_common_answer.iloc[0]
            else:
                record_uuid_to_answer[record_uuid] = record_df.answer.mode().iloc[0]

        def _is_correct(row):
            # simple equality check suffices for filter
            if "filter" in row["op_name"]:
                return row['answer'] == row['accepted_answer']

            # otherwise, check equality on a per-key basis
            try:
                # we'll measure recal on accepted_answer, as extraneous info is often not an issue
                answer = row['answer']
                accepted_answer = row['accepted_answer']
                
                tp = 0
                for key, value in accepted_answer.items():
                    if key in answer and answer[key] == value:
                        tp += 1
                
                return tp / len(accepted_answer)

            except Exception as e:
                print(f"WARNING: error decoding answer or accepted_answer: {str(e)}")
                return False

        # compute accepted answers and clean all answers
        op_df.loc[:, 'accepted_answer'] = op_df.record_uuid.apply(lambda uuid: record_uuid_to_answer[uuid])
        op_df.loc[:, 'correct'] = op_df.apply(lambda row: _is_correct(row), axis=1)

        # get subset of observations for model_name and estimate quality w/fraction of answers that match accepted answer
        model_df = (
            op_df[op_df.model_name == model_name]
            if model_name is not None
            else op_df[op_df.model_name.isna()]
        )

        est_quality = (
            model_df.correct.sum() / model_df.shape[0]
            if not model_df.empty
            else (
                op_df.correct.sum() / op_df.shape[0]
                if not op_df.empty
                else MODEL_CARDS[model_name]["MMLU"] / 100.0
            )
        )

        return est_quality

    def _parse_record_llm_stats(self, record_dict: Dict[str, Any], op_name: str) -> Dict[str, Any]:
        """
        Extract gen_stats fields for get_cost_estimate_sample_data.
        """
        # create OperatorStats object with a single record
        op_stats = OperatorStats()
        op_stats.records = [record_dict]

        # re-use _compute_agg_op_stats to compute statistics across all possible stats objects
        op_stats = self._compute_agg_op_stats(op_stats)

        def _clean_answer(answer, op_name):
            # extract JSON for induce
            if "induce" in op_name:
                return _get_JSON_from_answer(answer)

            # extract T/F for filter
            elif "filter" in op_name:
                return "true" in answer.lower()

            else:
                return answer

        def _get_answer(op_name, record_dict, generated_fields):
            # return T/F for filter
            if "filter" in op_name:
                return record_dict["_passed_filter"]

            # return key->value mapping for generated fields for induce
            answer = {}
            for field in generated_fields:
                answer[field] = record_dict[field]
            
            return answer

        # get values needed to compute observation metrics
        additional_fields_dict = {
            "model_name": op_stats.model_name,
            "filter": op_stats.filter,
            "input_fields": "-".join(sorted(op_stats.input_fields)),
            "generated_fields": "-".join(sorted(op_stats.generated_fields)),
            "num_input_tokens": op_stats.total_input_tokens,
            "num_output_tokens": op_stats.total_output_tokens,
            "input_usd": op_stats.total_input_usd,
            "output_usd": op_stats.total_output_usd,
            # "answer": _clean_answer(op_stats.answers[0], op_name) if len(op_stats.answers) > 0 else None,
            "answer": _get_answer(op_name, record_dict, op_stats.generated_fields),
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
                    "record_uuid": record_dict["uuid"],
                    "record_parent_uuid": record_dict["parent_uuid"],
                    "op_id": op_data.op_id,
                    "op_name": op_data.op_name,
                    "source_op_id": op_data.source_op_stats.op_id if op_data.source_op_stats is not None else None,
                    "op_time": record_dict["stats"].op_time,
                    "passed_filter": record_dict["_passed_filter"] if "_passed_filter" in record_dict else None,
                }

                # add additional fields for induce or filter w/LLM
                additional_fields_dict = self._parse_record_llm_stats(record_dict, op_data.op_name)
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
