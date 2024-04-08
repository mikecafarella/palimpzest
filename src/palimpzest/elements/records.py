from palimpzest.elements import Schema

import json
import uuid

# DEFINITIONS
MAX_UUID_CHARS = 10

class DataRecord:
    """A DataRecord is a single record of data matching some Schema."""
    def __init__(self, schema: Schema, parent_uuid=None):
        # schema for the data record
        self._schema = schema

        # unique identifier for the record
        self._uuid = str(uuid.uuid4())[:MAX_UUID_CHARS]
        self._parent_uuid = parent_uuid

        # attribute which may collect profiling stats pertaining to a record;
        # keys will the the ID of the operation which generated the stats and the
        # values will be Stats objects
        self._stats = {}

    def __setattr__(self, key, value):
        if not key.startswith("_") and not hasattr(self._schema, key):
            raise Exception(f"Schema {self._schema} does not have a field named {key}")

        super().__setattr__(key, value)



    def __getitem__(self, key):
        return super().__getattr__(key)

    @property
    def schema(self):
        return self._schema

    def asTextJSON(self):
        """Return a JSON representation of this DataRecord"""
        keys = sorted(self.__dict__)
        # Make a dictionary out of the key/value pairs
        d = {k: str(self.__dict__[k]) for k in keys if not k.startswith("_") and not isinstance(self.__dict__[k] , bytes)}
        d["data type"] = str(self._schema.__name__)
        d["data type description"]  = str(self._schema.__doc__)

        return json.dumps(d, indent=2)
    
    def asDict(self, include_bytes: bool=True):
        """Return a dictionary representation of this DataRecord"""
        keys = sorted(self.__dict__)
        # Make a dictionary out of the key/value pairs
        d = (
            {k: self.__dict__[k] for k in keys if not k.startswith("_")}
            if include_bytes
            else {k: self.__dict__[k] if not isinstance(self.__dict__[k] , bytes) else "<bytes>" for k in keys if not k.startswith("_")}
        )
        return d

    def asJSON(self):
        """Return a JSON representation of this DataRecord"""
        keys = sorted(self.__dict__)
        # Make a dictionary out of the key/value pairs
        d = {k: self.__dict__[k] for k in keys if not k.startswith("_")}
        d["data type"] = str(self._schema.__name__)
        d["data type description"]  = str(self._schema.__doc__)
        return json.dumps(d, indent=2)

    def __str__(self):
        keys = sorted(self.__dict__)
        items = ("{}={!r}".format(k, str(self.__dict__[k])[:15]) for k in keys)
        return "{}({})".format(type(self).__name__, ", ".join(items))

    def __eq__(self, other):
        return self.__dict__ == other.__dict__
