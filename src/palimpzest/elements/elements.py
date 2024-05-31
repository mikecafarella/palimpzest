from __future__ import annotations
from typing import Any, List, Dict

import json

#####################################################
# Base classes for building schemas
# - A Field is defined by the tuple: (description, required)
# - A Schema is a set of Fields, but more generally can define a (sub)document
#   - E.g., a schema can define a text file, a paragraph, a figure, a title,
#     a record, an image, a plot, an equation, etc.
#   - A Schema might define a pretty precise object, e.g., "A histogram that measures coulombic efficiency"
# - A DataRecord (see records.py) will be defined by a schema and will have a value
#   for each field in the Schema
#####################################################

class Field:
    """
    A Field is defined by its description and a boolean flag indicating if it is required (for a given Schema).
    The Field class can be subclassed to specify that values of the subclass should belong to a specific type.

    For example, if you wanted to define Fields relevant to indexing research papers, you could define a field
    representing the title of a paper, the year it was published, and the journal it was published in:
    
    ```python
    paper_title = Field(desc="The title of a scientific paper", required=True)
    paper_year = Field(desc="The year the paper was published", required=True)
    paper_journal = Field(desc="The name of the journal that published the paper", required=False)
    ```

    Note that because not all papers are published in journals, this field might be optional (`required=False`).
    """
    def __init__(self, desc: str, required: bool=False) -> None:
        self._desc = desc
        self.required = required

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(desc={self._desc})"
    
    def __hash__(self) -> int:
        return hash(self._desc + str(self.required) + self.__class__.__name__)
    
    def __eq__(self, other: Field) -> bool:
        return self._desc == other._desc and self.required == other.required and self.__class__ == other.__class__
    
    @property
    def desc(self) -> str:
        return self._desc
    
    def jsonSchema(self) -> Dict[str, str]:
        return {"description": self._desc, "type": "undefined"}


class BooleanField(Field):
    """A BooleanField is a Field that is True or False."""
    def __init__(self, desc: str, required: bool=False):
        super().__init__(desc=desc, required=required)

    def jsonSchema(self) -> Dict[str, str]:
        return {"description": self._desc, "type": "boolean"}


class StringField(Field):
    """A StringField is a Field that is definitely a string of text."""
    def __init__(self, desc: str, required: bool=False):
        super().__init__(desc=desc, required=required)

    def jsonSchema(self) -> Dict[str, str]:
        return {"description": self._desc, "type": "string"}


class NumericField(Field):
    """A NumericField is a Field that is definitely an integer or a float."""
    def __init__(self, desc: str, required: bool=False):
        super().__init__(desc=desc, required=required)

    def jsonSchema(self) -> Dict[str, str]:
        return {"description": self._desc, "type": "numeric"}


class BytesField(Field):
    """A BytesField is a Field that is definitely an array of bytes."""
    def __init__(self, desc: str, required: bool=False):
        super().__init__(desc=desc, required=required)

    def jsonSchema(self) -> Dict[str, str]:
        return {"description": self._desc, 
                "type": "string",
                "contentEncoding": "base64",
                "contentMediaType": "application/octet-stream"}


class SchemaMetaclass(type):
    """
    This is a metaclass for our Schema class.
    """
    def __str__(cls) -> str:
        """
        Emit a string that contains the names of all the class members that are Fields.
        """
        # get attributes that are Fields
        fields = SchemaMetaclass.fieldNames(cls)

        return f"{cls.__name__}({', '.join(fields)})"

    def __eq__(cls, other) -> bool:
        """
        Equality function for the Schema which checks that the ordered fields and class names are the same.
        """
        cls_schema = SchemaMetaclass.getDesc(cls)
        other_schema = SchemaMetaclass.getDesc(other)

        return cls_schema == other_schema

    def __hash__(cls) -> int:
        """Hash function for the Schema which is a simple hash of its ordered Fields and class name."""
        ordered = SchemaMetaclass.getDesc(cls)

        return hash(ordered.encode())

    def fieldNames(cls) -> List[Any]:
        """Return a list of the fields in this Schema"""
        attributes = dir(cls)
        attributes = [attr for attr in attributes if not attr.startswith('__')]
        fields = [attr for attr in attributes if isinstance(getattr(cls, attr), Field)]

        return fields
    
    def getDesc(cls) -> str:
        """Return a description of the schema"""
        fields = SchemaMetaclass.fieldNames(cls)
        d = {k: hash(getattr(cls, k)) for k in fields}

        # TODO: this causes an exception why trying to use Schema in a type definition
        # e.g. TaskDescriptor = Tuple[str, Union[tuple, None], Schema, Schema]
        # will throw the following exception:
        #
        # File "/Users/matthewrusso/palimpzest/src/palimpzest/elements/elements.py", line 168, in getDesc
        #     d["__class__"] = o.__name__
        # AttributeError: '_SpecialForm' object has no attribute '__name__'
        #
        d["__class__"] = cls.__name__

        return json.dumps(d, sort_keys=True)

    def className(cls) -> str:
        """Return the name of this class"""
        return cls.__name__

    def jsonSchema(cls) -> Dict[str, Any]:
        """The JSON representation of the Schema"""
        fields = SchemaMetaclass.fieldNames(cls)

        schema = {"properties": {}, "required": [], "type": "object", "description": cls.__doc__}
        for k in fields:
            if k.startswith("_"):
                continue
            v = getattr(cls, k)
            if v is None:
                continue

            schema["properties"][k] = v.jsonSchema()

            if v.required:
                schema["required"].append(k)
        return schema

# TODO: how does deserialize actually work with Schema (formerly Element)
# TODO: should we put the SchemaMetaclass functionality into Schema and make it a @dataclass?
class Schema(metaclass=SchemaMetaclass):
    """
    A Schema is defined by a set of named Fields. Much of the class is implemented
    in the SchemaMetaclass (which I need to ask Mike more about). Because Schema is a MetaClass,
    its fields are defined similar to how they are defined in a Python dataclass.

    For example, if you wanted to define a schema for research papers, you could define a schema
    with fields representing the paper's title, publication year, and publishing journal:

    ```python
    class ResearchPaper(Schema):
        paper_title = Field(desc="The title of a scientific paper", required=True)
        paper_year = Field(desc="The year the paper was published", required=True)
        paper_journal = Field(desc="The name of the journal that published the paper", required=False)
    ```

    Note that because not all papers are published in journals, this field might be optional (`required=False`).
    """
    def __str__(self) -> str:
        return f"{self.__class__.__name__}(desc={self._desc})"

    def asJSON(self, value_dict: Dict[str, Any], include_data_cols: bool=True) -> str:
        """Return a JSON representation of an instantiated object of this Schema"""
        # TODO: rethink relationship between records and schemas
        # fields = self.__class__.fieldNames()

        # TODO: rethink relationship between records and schemas
        # Make a dictionary out of the key/value pairs
        d = value_dict
        # d = {k: value_dict[k] for k in fields}
        if include_data_cols:
            d["data type"] = str(self.__class__.__name__)
            d["data type description"]  = str(self.__class__.__doc__)

        return json.dumps(d, indent=2)


# TODO: Under the definition of __eq__ in SchemaMetaclass, I think that an equality check like
#       Any([TextFile, PDFFile]) == TextFile will return `False`. I believe this is the behavior
#       that we want (these schemas are not the same), but I need to make sure that an equality
#       check between two instances of these schemas will not return `False` if they are both the
#       the same object (e.g. the same file of text), but one has Schema == Any([TextFile, PDFFile])
#       and the other has Schema = TextFile
#
#####################################################
# A Schema that can be one of multiple other kinds of Schemas.
# For example, I might want to process Any([PDF, WordDoc, TextFile])
#####################################################
class Any(Schema):
    """
    This represents ANY of the specified Schemas. For example, you may not know if a document
    is a PDF or a Word document, but you know it's one of those two.
    """
    def __init__(self, possibleSchemas: List[Schema], desc: str):
        super().__init__(desc=desc)
        self._possibleSchemas = possibleSchemas

    @property
    def children(self) -> List[Schema]:
        return self._possibleSchemas

class OperatorDerivedSchema(Schema):
    """Schema defined by an operator, e.g., a join or a group by"""


class File(Schema):
    """
    A File is defined by two Fields:
    - the filename (string)
    - the contents of the file (bytes)
    """
    filename = StringField(desc="The UNIX-style name of the file", required=True)
    contents = BytesField(desc="The contents of the file", required=True)

class Number(Schema):
    """Just a number. Often used for aggregates"""
    value = NumericField(desc="A single number", required=True)

class TextFile(File):
    """A text file is a File that contains only text. No binary data."""

class RawJSONObject(Schema):
    """A JSON object, which is a dictionary of key-value pairs."""
    json = StringField(desc="String representation of a JSON object", required=True)
