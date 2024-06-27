from __future__ import annotations

from palimpzest.constants import *
from palimpzest.corelib import Schema
from palimpzest.dataclasses import RecordOpStats, OperatorCostEstimates
from palimpzest.elements import DataRecord
from palimpzest.operators import logical, DataRecordsWithStats, PhysicalOperator

from typing import List, Union

import time


class DataSourcePhysicalOp(PhysicalOperator):
    """
    Physical operators which implement DataSources require slightly more information
    in order to accurately compute naive cost estimates. Thus, we use a slightly
    modified abstract base class for these operators.
    """
    def naiveCostEstimates(
        self,
        source_op_cost_estimates: OperatorCostEstimates,
        input_cardinality: Union[int, float],
        input_record_size_in_bytes: Union[int, float],
    ) -> OperatorCostEstimates:
        """
        In addition to 
        This function returns a naive estimate of this operator's:
        - cardinality
        - time_per_record
        - cost_per_record
        - quality
    
        For the implemented operator. These will be used by the CostEstimator
        when PZ does not have sample execution data -- and it will be necessary
        in some cases even when sample execution data is present. (For example,
        the cardinality of each operator cannot be estimated based on sample
        execution data alone -- thus DataSourcePhysicalOps need to give
        at least ballpark correct estimates of this quantity).
        """
        raise NotImplementedError("Abstract method")


class MarshalAndScanDataOp(DataSourcePhysicalOp):

    implemented_op = logical.BaseScan

    def __init__(
        self,
        outputSchema: Schema,
        dataset_type: str,
        shouldProfile=False,
        *args, **kwargs
    ):
        super().__init__(outputSchema=outputSchema, shouldProfile=shouldProfile)
        self.dataset_type = dataset_type

    def __eq__(self, other: PhysicalOperator):
        return (
            isinstance(other, self.__class__)
            and self.outputSchema == other.outputSchema
            and self.dataset_type == other.dataset_type
        )

    def __str__(self):
        return (
            f"{self.op_name()}("
            + str(self.outputSchema) 
            + ", "
            + self.dataset_type
            + ")"
        )

    def copy(self):
        return MarshalAndScanDataOp(
            self.outputSchema,
            self.dataset_type,
            self.shouldProfile,
        )

    def get_op_dict(self):
        return {
            "operator": self.op_name(),
            "outputSchema": str(self.outputSchema),
        }

    def naiveCostEstimates(
        self,
        source_op_cost_estimates: OperatorCostEstimates,
        input_cardinality: Union[int, float],
        input_record_size_in_bytes: Union[int, float],
    ) -> OperatorCostEstimates:
        # get inputs needed for naive cost estimation
        # TODO: we should rename cardinality --> "multiplier" or "selectivity" one-to-one / one-to-many

        # estimate time spent reading each record
        perRecordSizeInKb = input_record_size_in_bytes / 1024.0
        timePerRecord = (
            LOCAL_SCAN_TIME_PER_KB * perRecordSizeInKb
            if self.dataset_type in ["dir", "file"]
            else MEMORY_SCAN_TIME_PER_KB * perRecordSizeInKb
        )

        # estimate output cardinality
        cardinality = (
            source_op_cost_estimates.cardinality
            if input_cardinality == Cardinality.ONE_TO_ONE
            else source_op_cost_estimates.cardinality * NAIVE_EST_ONE_TO_MANY_SELECTIVITY
        )

        # for now, assume no cost per record for reading data
        return OperatorCostEstimates(
            cardinality=cardinality,
            time_per_record=timePerRecord,
            cost_per_record=0,
            quality=1.0,
        )

    def __call__(self, candidate: DataRecord) -> List[DataRecordsWithStats]:
        """
        This function takes the candidate -- which is a DataRecord with a SourceRecord schema --
        and invokes its get_item_fn on the given idx to return the next DataRecord from the DataSource.
        """
        start_time = time.time()
        output = candidate.get_item_fn(candidate.idx)
        records = [output] if candidate.cardinality == Cardinality.ONE_TO_ONE else output
        end_time = time.time()

        # create RecordOpStats objects
        record_op_stats_lst = []
        for record in records:
            record_op_stats = RecordOpStats(
                record_uuid=record._uuid,
                record_parent_uuid=record._parent_uuid,
                record_state=record._asDict(include_bytes=False),
                op_id=self.get_op_id(),
                op_name=self.op_name(),
                time_per_record=(end_time - start_time) / len(records),
                cost_per_record=0.0,
            )
            record_op_stats_lst.append(record_op_stats)

        return records, record_op_stats_lst

class CacheScanDataOp(DataSourcePhysicalOp):
    implemented_op = logical.CacheScan

    def __init__(
        self,
        outputSchema: Schema,
        cachedDataIdentifier: str,
        shouldProfile=False,
    ):
        super().__init__(outputSchema=outputSchema, shouldProfile=shouldProfile)
        self.cachedDataIdentifier = cachedDataIdentifier

    def __eq__(self, other: PhysicalOperator):
        return (
            isinstance(other, self.__class__)
            and self.cachedDataIdentifier == other.cachedDataIdentifier
            and self.outputSchema == other.outputSchema
        )

    def __str__(self):
        return (
            f"{self.op_name()}("
            + str(self.outputSchema)
            + ", "
            + self.cachedDataIdentifier
            + ")"
        )

    def copy(self):
        return CacheScanDataOp(
            self.outputSchema,
            self.cachedDataIdentifier,
            self.shouldProfile,
        )

    def get_op_dict(self):
        return {
            "operator": self.op_name(),
            "outputSchema": str(self.outputSchema),
            "cachedDataIdentifier": self.cachedDataIdentifier,
        }

    def naiveCostEstimates(
        self, 
        source_op_cost_estimates: OperatorCostEstimates,
        input_cardinality: Union[int, float],
        input_record_size_in_bytes: Union[int, float],
    ):
        # get inputs needed for naive cost estimation
        # TODO: we should rename cardinality --> "multiplier" or "selectivity" one-to-one / one-to-many

        # estimate time spent reading each record
        perRecordSizeInKb = input_record_size_in_bytes / 1024.0
        timePerRecord = LOCAL_SCAN_TIME_PER_KB * perRecordSizeInKb

        # estimate output cardinality
        cardinality = (
            source_op_cost_estimates.cardinality
            if input_cardinality == Cardinality.ONE_TO_ONE
            else source_op_cost_estimates.cardinality * NAIVE_EST_ONE_TO_MANY_SELECTIVITY
        )

        # for now, assume no cost per record for reading from cache
        return OperatorCostEstimates(
            cardinality=cardinality,
            time_per_record=timePerRecord,
            cost_per_record=0,
            quality=1.0,
        )

    def __call__(self, candidate: DataRecord) -> List[DataRecordsWithStats]:
        start_time = time.time()
        output = candidate.get_item_fn(candidate.idx)
        records = [output] if candidate.cardinality == Cardinality.ONE_TO_ONE else output
        end_time = time.time()

        # create RecordOpStats objects
        record_op_stats_lst = []
        for record in records:
            record_op_stats = RecordOpStats(
                record_uuid=record._uuid,
                record_parent_uuid=record._parent_uuid,
                record_state=record._asDict(include_bytes=False),
                op_id=self.get_op_id(),
                op_name=self.op_name(),
                time_per_record=(end_time - start_time) / len(records),
                cost_per_record=0.0,
            )
            record_op_stats_lst.append(record_op_stats)

        return records, record_op_stats_lst