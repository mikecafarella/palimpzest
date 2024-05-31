from palimpzest.constants import Model, PromptStrategy, QueryStrategy
from palimpzest.elements import Filter, Schema

from dataclasses import dataclass

@dataclass
class TaskDescriptor:
    """Dataclass for describing tasks sent to the Solver."""
    # the name of the physical operation in need of a function from the solver
    physical_op: str
    # the input schema
    inputSchema: Schema
    # the output schema
    outputSchema: Schema = None
    # the operation id
    op_id: str = None
    # the model to use in the task
    model: Model = None
    # the cardinality ("oneToOne" or "oneToMany") of the operation
    cardinality: str = None
    # whether or not the task is an image conversion task
    image_conversion: bool = None
    # the prompt strategy
    prompt_strategy: PromptStrategy = None
    # the query strategy
    query_strategy: QueryStrategy = None
    # the token budget
    token_budget: float = None
    # the filter for filter operations
    filter: Filter = None
    # the optional description of the conversion being applied (if task is a conversion)
    conversionDesc: str = None
    # name of the pdfprocessing tool to use (if applicable)
    pdfprocessor: str = None
    # TODO: remove
    plan_idx: int = None
    # use heatmap from solver
    heatmap_json_obj: dict = None

    def __str__(self) -> str:
        """Use the __repr__() function which is automagically implemented by @dataclass"""
        return self.__repr__()
