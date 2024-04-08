from __future__ import annotations
from typing import Any, Dict
from palimpzest.elements import Field, OperatorDerivedSchema, Schema

#signature for a group by aggregate that applies
# group and aggregation to an input tuple
class GroupBySig:
    def __init__(self, gbyFields: list[str], aggFuncs:list[str], aggFields:list[str]):
        self.gbyFields = gbyFields 
        self.aggFields = aggFields
        self.aggFuncs = aggFuncs

    def validateSchema(self, inputSchema: Schema) -> tuple[bool, str]:
        for f in self.gbyFields:
            if not hasattr(inputSchema, f):
                return (False, "Supplied schema has no field " + f)
        for f in self.aggFields:
            if not hasattr(inputSchema, f):
                return (False, "Supplied schema has no field " + f)
        return (True, None)
    
    def serialize(a) -> Dict[str, Any]:
        out = {"groupByFields":a.gbyFields, "aggFuncs":a.aggFuncs, "aggFields": a.aggFields}
        return out
    
    def deserialize(d) -> GroupBySig:
        return GroupBySig(d["groupByFields"], d["aggFuncs"], d["aggFields"])
    

    def __str__(self) -> str:
        return "GroupBy(" + repr(GroupBySig.serialize(self)) + ")"

    def __hash__(self) -> int:
        # custom hash function
        return hash(repr(GroupBySig.serialize(self)))

    def __eq__(self, other: GroupBySig) -> bool:
        # __eq__ should be defined for consistency with __hash__
        return isinstance(other, GroupBySig) and GroupBySig.serialize(self) == GroupBySig.serialize(other)
    
    def getAggFieldNames(self) -> list[str]:
        ops = []
        for i in range(0, len(self.aggFields)):
            ops.append(self.aggFuncs[i] + "(" + self.aggFields[i] + ")")
        return ops
    
    def outputSchema(self) -> OperatorDerivedSchema: 
        #the output class varies depending on the group by, so here 
        # we dynamically construct this output
        s = type('CustomGroupBy', (OperatorDerivedSchema,), {})

        for g in self.gbyFields:
            f = Field(desc = g, required=True)
            setattr(s, g, f)
        ops = self.getAggFieldNames()
        for op in ops:
            f = Field(desc = op, required=True)
            setattr(s, op, f)
        return s