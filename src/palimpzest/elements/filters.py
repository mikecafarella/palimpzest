from __future__ import annotations
from typing import Any, Dict

#############################
# Filters that can be applied against a particular Schema
#############################
class Filter():
    """A filter that can be applied to a Set"""
    def __init__(self, filterCondition: str=None, filterFn: callable=None) -> None:
        self.filterCondition = filterCondition
        self.filterFn = filterFn

    def serialize(self) -> Dict[str: Any]:
        return {"filterCondition": self.filterCondition, "filterFn": str(self.filterFn)}

    def deserialize(d: Dict[str, Any]) -> Filter:
        # TODO: won't work with function; we currently don't use this anywhere though
        return Filter(d["filterCondition"])

    def __str__(self) -> str:
        return (
            "Filter(" + self.filterCondition + ")"
            if self.filterCondition is not None
            else "Filter(" + str(self.filterFn) + ")"
        )

    def __hash__(self) -> int:
        # custom hash function
        return (
            hash(self.filterCondition)
            if self.filterCondition is not None
            else hash(str(self.filterFn))
        )

    def __eq__(self, other: Filter) -> bool:
        # __eq__ should be defined for consistency with __hash__
        return (
            isinstance(other, Filter)
            and self.filterCondition == other.filterCondition
            and self.filterFn == other.filterFn
        )
