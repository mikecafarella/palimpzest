from palimpzest.elements import DataRecord, File, Number, Schema, RawJSONObject
from typing import Any, Callable, Dict, List, Union

import os
import time
import requests
import json


class AbstractDataSource:
    """
    An AbstractDataSource is an Iterable which yields DataRecords adhering to a given schema.
    
    This base class must have its `__iter__` method implemented by a subclass, with each
    subclass reading data files from some real-world source (i.e. a directory, an S3 prefix,
    etc.).

    Many (if not all) DataSources should use Schemas from `palimpzest.elements.core`.
    In the future, programmers can implement their own DataSources using custom Schemas.
    """
    def __init__(self, schema: Schema) -> None:
        self.schema = schema

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(schema={self.schema})"

    def __eq__(self, __value: object) -> bool:
        return self.__dict__ == __value.__dict__

    def serialize(self) -> Dict[str, Any]:
        return {"schema": self.schema.jsonSchema()}

class DataSource(AbstractDataSource):
    def __init__(self, schema: Schema, dataset_id: str) -> None:
        super().__init__(schema)
        self.dataset_id = dataset_id

    def universalIdentifier(self):
        """Return a unique identifier for this Set."""
        return self.dataset_id

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(schema={self.schema}, dataset_id={self.dataset_id})"


class MemorySource(DataSource):
    """MemorySource returns multiple objects that reflect contents of an in-memory Python list"""
    def __init__(self, vals: List[Union[int, float]], dataset_id: str):
        # For the moment we assume that we are given a list of floats or ints, but someday it could be strings or something else
        super().__init__(Number, dataset_id)
        self.vals = vals

    def __iter__(self) -> Callable[[], DataRecord]:
        def valIterator():
            for idx, v in enumerate(self.vals):
                dr = DataRecord(self.schema, scan_idx=idx)
                dr.value = v
                yield dr

        return valIterator()


class DirectorySource(DataSource):
    """DirectorySource returns multiple File objects from a real-world source (a directory on disk)"""
    def __init__(self, path: str, dataset_id: str) -> None:
        super().__init__(File, dataset_id)
        self.path = path
        self.idx = 0

    def __iter__(self) -> Callable[[], DataRecord]:
        def filteredIterator():
            for filename in os.listdir(self.path):
                file_path = os.path.join(self.path, filename)
                if os.path.isfile(file_path):
                    dr = DataRecord(self.schema, scan_idx=self.idx)
                    dr.filename = file_path
                    with open(file_path, "rb") as f:
                        dr.contents = f.read()
                    yield dr

                    self.idx += 1

        return filteredIterator()

    def serialize(self) -> Dict[str, Any]:
        return {
            "schema": self.schema.jsonSchema(),
            "path": self.path,
            "source_type": "directory",
        }


class FileSource(DataSource):
    """FileSource returns a single File object from a single real-world local file"""
    def __init__(self, path: str, dataset_id: str) -> None:
        super().__init__(File, dataset_id)
        self.path = path
        self.idx = 0

    def __iter__(self) -> Callable[[], DataRecord]:
        def filteredIterator():
            for path in [self.path]:
                dr = DataRecord(self.schema, scan_idx=self.idx)
                dr.filename = path
                with open(path, "rb") as f:
                    dr.contents = f.read()

                yield dr

                self.idx += 1

        return filteredIterator()

    def serialize(self) -> Dict[str, Any]:
        return {
            "schema": self.schema.jsonSchema(),
            "path": self.path,
            "source_type": "file",
        }

class UserSource(DataSource):
    """UserSource is a DataSource that is created by the user and not loaded from a file"""
    def __init__(self, schema: Schema, dataset_id: str) -> None:
        super().__init__(schema, dataset_id)

    def __iter__(self) -> Callable[[], DataRecord]:
        def userIterator():
            return self.userImplementedIterator()

        return userIterator()
    
    def userImplementedIterator(self) -> Callable[[], DataRecord]:
        raise Exception("User sources must implement their own iterator.")
    
    def serialize(self) -> Dict[str, Any]:
        return {
            "schema": self.schema.jsonSchema(),
            "source_type": "user-defined:" + self.__class__.__name__,
        }
    
    def getSize(self):
        return 100 # this should be overridden
    
    def getCardinality(self):
        return 100 # this should be overridden


