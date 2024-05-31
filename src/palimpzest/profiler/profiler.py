from __future__ import annotations

from palimpzest.constants import MODEL_CARDS
from palimpzest.profiler.stats import *

from functools import wraps
from typing import Any, Dict

import os
import time

# DEFINITIONS
PZ_PROFILING_ENV_VAR = "PZ_PROFILING"


class Profiler:
    """
    This class maintains a set of utility tools and functions to help with profiling PZ programs.
    """

    def __init__(self, op_id: str):
        # store op_id for the operation associated with this profiler
        self.op_id = op_id

        # object which maintains the set of records and stats processed by this operator
        self.operator_stats = OperatorStats(op_id=self.op_id)


    @staticmethod
    def profiling_on() -> bool:
        """
        Returns a boolean indicating whether user has turned on profiling.
        """
        return os.getenv(PZ_PROFILING_ENV_VAR) is not None and os.getenv(PZ_PROFILING_ENV_VAR).lower() == "true"


    def get_data(self) -> Dict[str, Any]:
        """
        Return the operator statistics.
        """
        return self.operator_stats


    def iter_profiler(self, name: str, shouldProfile: bool = False):
        """
        iter_profiler is a decorator factory. This function takes in a `name` argument
        and returns a decorator which will decorate an iterator. In practice, this
        looks almost identical to how you would normally use a decorator. The only
        difference is that now we can use the `name` inside of our decorated function
        to identify the profiling information on a per-iterator basis.

        To use the profiler, simply apply it to an iterator as follows:

        @profiler(name="foo", shouldProfile=True)
        def someIterator():
            # do normal iterator things
            yield dr
        """
        def profile_decorator(iterator):
            # return iterator if profiling is not set to True
            if not shouldProfile:
                return iterator

            @wraps(iterator)
            def timed_iterator():
                # set operator name in aggregate stats
                self.operator_stats.op_name = name

                # iterate through records and update profiling stats as we go
                t_record_start = time.time()
                for record in iterator():
                    t_record_end = time.time()
                    self.operator_stats.total_records += 1

                    # for non-induce/filter operators, we need to create an empty Stats object for the op_id
                    if self.op_id not in record._stats:
                        record._stats[self.op_id] = Stats()

                    # add time spent waiting for iterator to yield record to record's Stats object
                    # and the total maintained by the OperatorStats object; this measures the time
                    # spent by the record in this operator and all source operators
                    record._stats[self.op_id].cumulative_iter_time = t_record_end - t_record_start
                    self.operator_stats.total_cumulative_iter_time += t_record_end - t_record_start

                    # update state of record for complete history of computation
                    record_state = record._asDict(include_bytes=False)
                    record_state["op_id"] = self.op_id
                    record_state["uuid"] = record._uuid
                    record_state["parent_uuid"] = record._parent_uuid
                    record_state["stats"] = record._stats[self.op_id]
                    if hasattr(record, '_passed_filter') and "filter" in name:
                        record_state["_passed_filter"] = record._passed_filter

                    # add record state to set of records computed by this operator
                    self.operator_stats.records.append(record_state)

                    # # update aggregate stats for operators with LLM workloads
                    # self._update_agg_stats(record._stats[self.op_id])

                    # track time spent in profiler
                    self.operator_stats.total_time_in_profiler += time.time() - t_record_end

                    # if this is a filter operation and the record did not pass the filter,
                    # then this record is meant to be filtered out, so do not yield it
                    if "filter" in name and record._passed_filter is False:
                        t_record_start = time.time()
                        continue

                    yield record

                    # start timer for next iteration
                    t_record_start = time.time()

            return timed_iterator
        return profile_decorator
