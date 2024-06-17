from __future__ import annotations

from palimpzest.elements import DataRecord
from dataclasses import dataclass, asdict, field

from typing import Any, Dict, List

@dataclass
class RecordOpStats:
    """
    Dataclass for storing statistics about the execution of an operator on a single record.
    """
    # record id; a unique identifier for this record
    record_uuid: str

    # unique identifier for the parent of this record
    record_parent_uuid: str

    # operation id; a unique identifier for this operation
    op_id: str

    # operation name
    op_name: str

    # the time spent by the data record just in this operation
    op_time: float

    # the cost (in dollars) to generate this record at this operation
    op_cost: float

    # a dictionary with the record state after being processed by the operator
    record_state: Dict[str, Any]

    # an OPTIONAL dictionary with more detailed information about the processing of this record
    record_details: Dict[str, Any] = None

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_record_and_kwargs(record: DataRecord, **kwargs: Dict[str, Any]) -> RecordOpStats:
        return RecordOpStats(
            record_uuid=record._uuid,
            record_parent_uuid=record._parent_uuid,
            op_id=kwargs['op_id'],
            op_name=kwargs['op_name'],
            op_time=kwargs['op_time'],
            op_cost=kwargs['op_cost'],
            record_state=record._asDict(include_bytes=False),
            record_details=kwargs.get('record_details', None),
        )

@dataclass
class OperatorStats:
    """
    Dataclass for storing statistics captured within a given operator.
    """
    # the index of the operator in the plan
    op_idx: int

    # the ID of the physical operation in which these stats were collected
    op_id: str

    # the name of the physical operation in which these stats were collected
    op_name: str

    # the total time spent in this operation
    total_op_time: float = 0.0

    # the total cost of this operation
    total_op_cost: float = 0.0

    # a list of RecordOpStats processed by the operation
    record_op_stats_lst: List[RecordOpStats] = field(default_factory=list)

    # an OPTIONAL dictionary with more detailed information about this operation;
    op_details: Dict[str, Any] = field(default_factory=dict)

    def __iadd__(self, record_op_stats: RecordOpStats):
        self.total_op_time += record_op_stats.op_time
        self.total_op_cost += record_op_stats.op_cost
        self.record_op_stats_lst.append(record_op_stats)
    # ##############################################
    # ##### Universal Convert and Filter Fields #####
    # #####    [computed in StatsProcessor]    #####
    # ##############################################
    # # usage statistics computed for induce and filter operations
    # total_input_tokens: int = 0
    # total_output_tokens: int = 0
    # # dollar cost associated with usage statistics
    # total_input_usd: float = 0.0
    # total_output_usd: float = 0.0
    # total_usd: float = 0.0
    # # time spent waiting for LLM calls to return (in seconds)
    # total_llm_call_duration: float = 0.0
    # # name of the model used for generation
    # model_name: str = None
    # # the string associated with the filter
    # filter: str = None
    # # keep track of finish reasons
    # finish_reasons: defaultdict[int] = field(default_factory=lambda: defaultdict(int))
    # # record input fields and the output fields generated in an induce operation
    # input_fields: List[str] = field(default_factory=list)
    # generated_fields: List[str] = field(default_factory=list)
    # # list of answers
    # answers: List[str] = field(default_factory=list)
    # # list of lists of token log probabilities for the subset of tokens that comprise the answer
    # answer_log_probs: List[List[float]] = field(default_factory=list)
    # # the query strategy used
    # query_strategy: str = None
    # # the token budget used during generation
    # token_budget: float = None

    def to_dict(self):
        return asdict(self)


@dataclass
class PlanStats:
    """
    Dataclass for storing statistics captured for an entire plan.
    """
    # string for identifying the physical plan
    plan_id: str

    # dictionary of OperatorStats objects (one for each operator)
    operator_stats: Dict[str, OperatorStats] = field(default_factory=dict)

    # total runtime for the plan measured from the start to the end of PhysicalPlan.execute()
    total_plan_time: float = 0.0

    # total time as computed by summing up the total_op_time for each operator
    total_plan_op_time: float = 0.0

    # total cost for plan
    total_plan_cost: float = 0.0

    def finalize(self, total_plan_time: float):
        self.total_plan_time = total_plan_time
        self.total_plan_op_time = sum([op_stats.total_op_time for _, op_stats in self.operator_stats.items()])
        self.total_plan_cost = sum([op_stats.total_op_cost for _, op_stats in self.operator_stats.items()])


@dataclass
class ExecutionStats:
    """
    Dataclass for storing statistics captured for the entire execution of a workload.
    """
    # string for identifying this workload execution
    execution_id: str = None

    # dictionary of PlanStats objects (one for each plan run during execution)
    plan_stats: Dict[str, PlanStats] = field(default_factory=dict)

    # total runtime for a call to pz.Execute
    total_execution_time: float = 0.0

    # total cost for a call to pz.Execute
    total_execution_cost: float = 0.0


@dataclass
class OperatorCostEstimates:
    """
    Dataclass for storing estimates of key metrics of interest for each operator.
    """
    # (estimated) number of records output by this operator
    cardinality: float

    # (estimated) avg. time spent in this operator per-record
    time_per_record: float

    # (estimated) dollars spent per-record by this operator
    cost_per_record: float

    # (estimated) quality of the output from this operator
    quality: float


@dataclass
class SampleExecutionData:
    """
    Dataclass for storing observations from sample execution data.
    """
    # record id; a unique identifier for this record
    record_uuid: str

    # unique identifier for the parent of this record
    record_parent_uuid: str

    # the ID of the physical operation in which these stats were collected
    op_id: str

    # the name of the physical operation in which these stats were collected
    op_name: str

    # the time spent by the data record just in this operation
    op_time: float

    # the cost (in dollars) to generate this record at this operation
    op_cost: float

    # the ID of the physical operation which produced the input record for this record and operation
    source_op_id: str = None

    # boolean indicating whether the record was filtered by the filter operation; None if this is not a filter operation
    passed_filter: bool = None

    # the name of the model used to process this record; None if this operation did not use an LLM
    model_name: str = None

    # string description of the filter; None if this operation is not a filter operation
    filter_str: str = None

    # the input fields to the convert; None if this operation is not a convert operation
    input_fields_str: str = None

    # the fields generated by the convert; None if this operation is not a convert operation
    generated_fields_str: str = None

    # the total number of input tokens processed by this operator; None if this operation did not use an LLM
    total_input_tokens: int = None

    # the total number of output tokens processed by this operator; None if this operation did not use an LLM
    total_output_tokens: int = None

    # the total cost of processing the input tokens; None if this operation did not use an LLM
    total_input_cost: float = None

    # the total cost of processing the output tokens; None if this operation did not use an LLM
    total_output_cost: float = None

    # the answer generated by this operator; None if the operation is not a convert or a filter
    answer: str = None
