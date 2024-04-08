from palimpzest.constants import PromptStrategy
from palimpzest.elements import DataRecord
from palimpzest.generators import DSPyGenerator, ImageTextGenerator
from palimpzest.profiler import Stats, BondedQueryStats, ConventionalQueryStats, FieldQueryStats
from palimpzest.solver.task_descriptors import TaskDescriptor

from typing import Any, Dict, List, Tuple

import base64
import json


def _construct_query_prompt(td: TaskDescriptor, doc_type: str, generate_field_names: List[str]) -> str:
    """
    This function constructs the prompt for a bonded query.
    """
    # build string of input fields and their descriptions
    multilineInputFieldDescription = ""
    for field_name in td.inputSchema.fieldNames():
        f = getattr(td.inputSchema, field_name)
        multilineInputFieldDescription += f"INPUT FIELD {field_name}: {f.desc}\n"

    # build string of output fields and their descriptions
    multilineOutputFieldDescription = ""
    for field_name in generate_field_names:
        f = getattr(td.outputSchema, field_name)
        multilineOutputFieldDescription += f"OUTPUT FIELD {field_name}: {f.desc}\n"

    # add input/output schema descriptions (if they have a docstring)
    optionalInputDesc = "" if td.inputSchema.__doc__ is None else f"Here is a description of the input object: {td.inputSchema.__doc__}."
    optionalOutputDesc = "" if td.outputSchema.__doc__ is None else f"Here is a description of the output object: {td.outputSchema.__doc__}."

    # construct sentence fragments which depend on cardinality of conversion ("oneToOne" or "oneToMany")
    targetOutputDescriptor = "an output JSON object that describes an object of type"
    outputSingleOrPlural = "the output object"
    appendixInstruction = "Be sure to emit a JSON object only"
    if td.cardinality == "oneToMany":
        targetOutputDescriptor = "an output array of zero or more JSON objects that describe objects of type"
        outputSingleOrPlural = "the output objects"
        appendixInstruction = "Be sure to emit a JSON object only. The root-level JSON object should have a single field, called 'items' that is a list of the output objects. Every output object in this list should be a dictionary with the output fields described above. You must decide the correct number of output objects."

    # construct promptQuestion
    promptQuestion = None
    if td.prompt_strategy != PromptStrategy.IMAGE_TO_TEXT:
        promptQuestion = f"""I would like you to create {targetOutputDescriptor} {doc_type}. 
        You will use the information in an input JSON object that I will provide. The input object has type {td.inputSchema.className()}.
        All of the fields in {outputSingleOrPlural} can be derived using information from the input object.
        {optionalInputDesc}
        {optionalOutputDesc}
        Here is every input field name and a description: 
        {multilineInputFieldDescription}
        Here is every output field name and a description:
        {multilineOutputFieldDescription}.
        {appendixInstruction}
        """ + "" if td.conversionDesc is None else f" Keep in mind that this process is described by this text: {td.conversionDesc}."                

    else:
        promptQuestion = f"""You are an image analysis bot. Analyze the supplied image and create {targetOutputDescriptor} {doc_type}.
        You will use the information in the image that I will provide. The input image has type {td.inputSchema.className()}.
        All of the fields in {outputSingleOrPlural} can be derived using information from the input image.
        {optionalInputDesc}
        {optionalOutputDesc}
        Here is every output field name and a description:
        {multilineOutputFieldDescription}.
        {appendixInstruction}
        """ + "" if td.conversionDesc is None else f" Keep in mind that this process is described by this text: {td.conversionDesc}." 

    return promptQuestion


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

    return json.loads(answer)


def _create_data_record_from_json(jsonObj: Any, td: TaskDescriptor, candidate: DataRecord) -> DataRecord:
    # initialize data record
    dr = DataRecord(td.outputSchema, parent_uuid=candidate._uuid)

    # copy fields from the candidate (input) record if they already exist,
    # otherwise parse them from the generated jsonObj
    for field_name in td.outputSchema.fieldNames():
        if field_name in td.inputSchema.fieldNames():
            setattr(dr, field_name, getattr(candidate, field_name))
        else:
            # parse the json object and set the DataRecord's fields with their generated values 
            setattr(dr, field_name, jsonObj[field_name])

    return dr


def runBondedQuery(candidate: DataRecord, td: TaskDescriptor, verbose: bool=False) -> Tuple[List[DataRecord], Stats, str]:
    """
    Run a bonded query, in which all new fields in the outputSchema are generated simultaneously
    in a single LLM call. This is in contrast to a conventional query, in which each output field
    is generated using its own LLM call.

    At the moment, tasks with cardinality == "oneToMany" can only be executed using bonded queries.

    This is not a theoretical limitation of conventional queries, but there are some practical
    difficulties with guaranteeing that each field has the same number of outputs generated
    in each separate LLM invocation.
    """
    # initialize list of output data records
    drs = []

    # construct list of fields in outputSchema which will need to be generated
    generate_field_names = []
    for field_name in td.outputSchema.fieldNames():
        if field_name not in td.inputSchema.fieldNames():
            generate_field_names.append(field_name)

    # fetch input information
    text_content = candidate.asTextJSON()
    doc_schema = str(td.outputSchema)
    doc_type = td.outputSchema.className()

    # construct prompt question
    promptQuestion = _construct_query_prompt(td, doc_type, generate_field_names)

    # generate LLM response and capture statistics
    answer, bonded_query_stats = None, None
    try:
        if td.prompt_strategy == PromptStrategy.DSPY_COT_QA:
            # invoke LLM to generate output JSON
            generator = DSPyGenerator(td.model.value, td.prompt_strategy, doc_schema, doc_type, verbose)
            answer, gen_stats = generator.generate(text_content, promptQuestion)

            # construct BondedQueryStats object
            bonded_query_stats = BondedQueryStats(
                gen_stats=gen_stats,
                input_fields=td.inputSchema.fieldNames(),
                generated_fields=generate_field_names,
            )

        elif td.prompt_strategy == PromptStrategy.IMAGE_TO_TEXT:
            # b64 decode of candidate.contents
            image_b64 = base64.b64encode(candidate.contents).decode('utf-8')

            # invoke LLM to generate output JSON
            generator = ImageTextGenerator(td.model.value)
            answer, gen_stats = generator.generate(image_b64, promptQuestion)

            # construct BondedQueryStats object
            bonded_query_stats = BondedQueryStats(
                gen_stats=gen_stats,
                input_fields=td.inputSchema.fieldNames(),
                generated_fields=generate_field_names,
            )

        # TODO
        elif td.prompt_strategy == PromptStrategy.ZERO_SHOT:
            raise Exception("not implemented yet")

        # TODO
        elif td.prompt_strategy == PromptStrategy.FEW_SHOT:
            raise Exception("not implemented yet")

        # parse JSON object from the answer
        jsonObj = _get_JSON_from_answer(answer)

        # parse JSON output and construct data records
        if td.cardinality == "oneToMany":
            for elt in jsonObj["items"]:
                dr = _create_data_record_from_json(elt, td, candidate)
                drs.append(dr)
        else:
            dr = _create_data_record_from_json(jsonObj, td, candidate)
            drs = [dr]

    except Exception as e:
        print(f"Bonded query processing error: {str(e)}")
        return None, bonded_query_stats, str(e)

    return drs, bonded_query_stats, None


def runConventionalQuery(candidate: DataRecord, td: TaskDescriptor, verbose: bool=False) -> Tuple[DataRecord, Stats]:
    """
    Run a conventional query, in which each output field is generated using its own LLM call.

    At the moment, conventional queries cannot execute tasks with cardinality == "oneToMany".
    """
    # initialize output data record
    dr = DataRecord(td.outputSchema, parent_uuid=candidate._uuid)

    # copy fields from the candidate (input) record if they already exist
    # and construct list of fields in outputSchema which will need to be generated
    generate_field_names = []
    for field_name in td.outputSchema.fieldNames():
        if field_name in td.inputSchema.fieldNames():
            setattr(dr, field_name, getattr(candidate, field_name))
        else:
            generate_field_names.append(field_name)

    # fetch input information
    text_content = candidate.asTextJSON()
    doc_schema = str(td.outputSchema)
    doc_type = td.outputSchema.className()

    # iterate over fields and generate their values using an LLM
    query_stats = {}
    for field_name in generate_field_names:
        # construct prompt question
        promptQuestion = _construct_query_prompt(td, doc_type, [field_name])
        try:
            field_stats = None
            if td.prompt_strategy == PromptStrategy.DSPY_COT_QA:
                # invoke LLM to generate output JSON
                generator = DSPyGenerator(td.model.value, td.prompt_strategy, doc_schema, doc_type, verbose)
                answer, field_stats = generator.generate(text_content, promptQuestion)

            elif td.prompt_strategy == PromptStrategy.IMAGE_TO_TEXT:                               
                # b64 decode of candidate.contents
                image_b64 = base64.b64encode(candidate.contents).decode('utf-8')

                # invoke LLM to generate output JSON
                generator = ImageTextGenerator(td.model.value)
                answer, field_stats = generator.generate(image_b64, promptQuestion)

            # TODO
            elif td.prompt_strategy == PromptStrategy.ZERO_SHOT:
                raise Exception("not implemented yet")

            # TODO
            elif td.prompt_strategy == PromptStrategy.FEW_SHOT:
                raise Exception("not implemented yet")

            # set the DataRecord's field with its generated value
            setattr(dr, field_name, answer)

            # update query_stats
            query_stats[f"{field_name}"] = field_stats

        except Exception as e:
            print(f"Conventional field processing error: {e}")
            setattr(dr, field_name, None)
            query_stats[f"{field_name}"] = None

    # construct ConventionalQueryStats object
    field_query_stats_lst = [FieldQueryStats(gen_stats=gen_stats, field_name=field_name) for field_name, gen_stats in query_stats.items()]
    conventional_query_stats = ConventionalQueryStats(
        field_query_stats_lst=field_query_stats_lst,
        input_fields=td.inputSchema.fieldNames(),
        generated_fields=generate_field_names,
    )

    return dr, conventional_query_stats


# TODO: @Zui
def runCodeGenQuery(candidate: DataRecord, td: TaskDescriptor, verbose: bool=False) -> Tuple[DataRecord, Stats]:
    """
    I think this would roughly map to the internals of _makeCodeGenTypeConversionFn() in your branch.
    Similar to the functions above, I moved most of the details of generating responses
    """
    # dr = None
    # full_code_gen_stats = None
    
    # return dr, full_code_gen_stats
    raise Exception("not implemented yet")