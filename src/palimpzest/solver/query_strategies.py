from palimpzest.datamanager import DataDirectory
from palimpzest.constants import PromptStrategy, CodeGenStrategy
from palimpzest.elements import DataRecord
from palimpzest.generators import DSPyGenerator, ImageTextGenerator, RegularGeneratorDSPyText
from palimpzest.profiler import Stats, BondedQueryStats, ConventionalQueryStats, FieldQueryStats, CodeGenEnsembleStats, FullCodeGenStats, GenerationStats
from palimpzest.solver.task_descriptors import TaskDescriptor
from palimpzest.utils import API, codeEnsembleGeneration, codeEnsembleExecution, reGenerationCondition, CodeGenSingleStats

from typing import Any, Dict, List, Tuple

import base64
import json
import re

DSPY_PREFIX = """
Answer question(s) about a Email(contents, filename, sender, subject).

---

Follow the following format.

Context: contains full text of the Email

Question: one or more question about the Email

Reasoning: Let's think step by step in order to ${produce the answer}. We ...

Answer: print the answer only, separated by a newline character

---

Context:
"""


def _construct_query_prompt(td: TaskDescriptor, doc_schema: str, doc_type: str, generate_field_names: List[str], is_conventional: bool=False, context: str=None, wrap_prompt_in_dspy_text: bool=False, filterQuestion: str=None) -> str:
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
    targetOutputDescriptor = f"an output JSON object that describes an object of type {doc_type}."
    outputSingleOrPlural = "the output object"
    appendixInstruction = "Be sure to emit a JSON object only"
    if td.cardinality == "oneToMany":
        targetOutputDescriptor = f"an output array of zero or more JSON objects that describe objects of type {doc_type}."
        outputSingleOrPlural = "the output objects"
        appendixInstruction = "Be sure to emit a JSON object only. The root-level JSON object should have a single field, called 'items' that is a list of the output objects. Every output object in this list should be a dictionary with the output fields described above. You must decide the correct number of output objects."

    # if this is a conventional query, focus only on generating output field
    if is_conventional:
        targetOutputDescriptor = f"an output JSON object with a single key \"{generate_field_names[0]}\" whose value is specified in the input object."

    # construct promptQuestion
    promptQuestion = None
    if td.prompt_strategy != PromptStrategy.IMAGE_TO_TEXT:
        promptQuestion = f"""I would like you to create {targetOutputDescriptor}. 
        You will use the information in an input JSON object that I will provide. The input object has type {td.inputSchema.className()}.
        All of the fields in {outputSingleOrPlural} can be derived using information from the input object.
        {optionalInputDesc}
        {optionalOutputDesc}
        Here is every input field name and a description: 
        {multilineInputFieldDescription}
        Here is every output field name and a description:
        {multilineOutputFieldDescription}
        {appendixInstruction}
        """ + "" if td.conversionDesc is None else f" Keep in mind that this process is described by this text: {td.conversionDesc}."                

    else:
        promptQuestion = f"""You are an image analysis bot. Analyze the supplied image(s) and create {targetOutputDescriptor} {doc_type}.
        You will use the information in the image that I will provide. The input image(s) has type {td.inputSchema.className()}.
        All of the fields in {outputSingleOrPlural} can be derived using information from the input image(s).
        {optionalInputDesc}
        {optionalOutputDesc}
        Here is every output field name and a description:
        {multilineOutputFieldDescription}
        {appendixInstruction}
        """ + "" if td.conversionDesc is None else f" Keep in mind that this process is described by this text: {td.conversionDesc}."

    # TODO: implement w/its own PromptStrategy
    # for the evaluation -- to ensure we aren't caching any responses -- simulate DSPY signature
    # by wrapping promptQuestion in DSPy context
    if wrap_prompt_in_dspy_text and td.prompt_strategy != PromptStrategy.DSPY_COT_BOOL:
        produce_the_answer = "${produce the answer}"
        promptQuestion = f"""Answer question(s) about a {doc_schema}.

        ---

        Follow the following format.

        Context: contains full text of the {doc_type}

        Question: one or more question about the {doc_type}

        Reasoning: Let's think step by step in order to {produce_the_answer}. We ...

        Answer: print the answer only, separated by a newline character

        ---

        Context:\n{context}

        Question: {promptQuestion}

        Reasoning: Let's think step by step in order to
        """

    elif wrap_prompt_in_dspy_text and td.prompt_strategy == PromptStrategy.DSPY_COT_BOOL:
        produce_the_answer = "${produce the answer}"
        promptQuestion = f"""Answer condition questions about a {doc_schema}.

        ---

        Follow the following format.

        Context: contains full text of the {doc_type}

        Question: one or more conditions about the Email {doc_type}

        Reasoning: Let's think step by step in order to {produce_the_answer}. We ...

        Answer: often a TRUE/FALSE answer to the condition question(s) about the Email

        ---

        Context:\n{context}

        Question: {filterQuestion}

        Reasoning: Let's think step by step in order to
        """

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

    # Handle comments in the JSON response. Use regex from // until end of line 
    answer = re.sub(r'\/\/.*$', '', answer, flags=re.MULTILINE)
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
            setattr(dr, field_name, jsonObj.get(field_name, None)) # the use of get prevents a KeyError if an individual field is missing. TODO: is this behavior desired?

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
    text_content = candidate.asJSON(include_bytes=False)
    doc_schema = str(td.outputSchema)
    doc_type = td.outputSchema.className()

    # construct prompt question
    wrap_prompt_in_dspy_text = (td.prompt_strategy == PromptStrategy.DSPY_COT_QA)
    promptQuestion = _construct_query_prompt(td, doc_schema, doc_type, generate_field_names, context=text_content, wrap_prompt_in_dspy_text=wrap_prompt_in_dspy_text)

    # generate LLM response and capture statistics
    answer, bonded_query_stats = None, None
    try:
        if td.prompt_strategy == PromptStrategy.DSPY_COT_QA:
            # invoke LLM to generate output JSON
            # generator = DSPyGenerator(td.model.value, td.prompt_strategy, doc_schema, doc_type, verbose)
            generator = RegularGeneratorDSPyText(td.model.value, td.prompt_strategy, doc_schema, doc_type, verbose)
            # answer, gen_stats = generator.generate(text_content, promptQuestion, budget=td.token_budget)
            answer, gen_stats = generator.generate(promptQuestion, budget=td.token_budget)

            # construct BondedQueryStats object
            bonded_query_stats = BondedQueryStats(
                gen_stats=gen_stats,
                input_fields=td.inputSchema.fieldNames(),
                generated_fields=generate_field_names,
            )

        elif td.prompt_strategy == PromptStrategy.IMAGE_TO_TEXT:
            # TODO: this is very hacky; need to come up w/more general solution for multimodal schemas
            # b64 decode of candidate.contents or candidate.image_contents
            base64_images = []
            if hasattr(candidate, "contents"):
                base64_images = [base64.b64encode(candidate.contents).decode('utf-8')]
            else:
                base64_images = [base64.b64encode(image).decode('utf-8') for image in candidate.image_contents]

            # invoke LLM to generate output JSON
            generator = ImageTextGenerator(td.model.value)
            answer, gen_stats = generator.generate(base64_images, promptQuestion)

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
            if len(jsonObj["items"]) == 0:
                raise Exception("No output objects were generated with bonded query - trying with conventional query...")
            for elt in jsonObj["items"]:
                dr = _create_data_record_from_json(elt, td, candidate)
                drs.append(dr)
        else:
            dr = _create_data_record_from_json(jsonObj, td, candidate)
            drs = [dr]

    except Exception as e:
        print(f"Bonded query processing error: {e}")
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
        elif hasattr(candidate, field_name) and (getattr(candidate, field_name) is not None):
            setattr(dr, field_name, getattr(candidate, field_name))
        else:
            generate_field_names.append(field_name)

    # fetch input information
    text_content = candidate.asJSON(include_bytes=False)
    doc_schema = str(td.outputSchema)
    doc_type = td.outputSchema.className()

    if td.cardinality == "oneToMany":
        # TODO here the problem is: which is the 1:N field that we are splitting the output into?
        # do we need to know this to construct the prompt question ?
        # for now, we will just assume there is only one list in the JSON.
        dct = json.loads(text_content)
        split_attribute = [att for att in dct.keys() if type(dct[att]) == list][0]
        n_splits = len(dct[split_attribute])

        if td.prompt_strategy == PromptStrategy.DSPY_COT_QA:
            # TODO Hacky to nest return and not disrupt the rest of method!!!
            # NOTE: this is a bonded query, but we are treating it as a conventional query
            query_stats = {}
            drs = []

            # iterate over the length of the split attribute, and generate a new JSON for each split
            for idx in range(n_splits):
                if verbose: 
                    print(f"Processing {split_attribute} with index {idx}")
                new_json = {k:v for k,v in dct.items() if k != split_attribute}
                new_json[split_attribute] = dct[split_attribute][idx]

                text_content = json.dumps(new_json)
                promptQuestion = _construct_query_prompt(td, doc_schema, doc_type, generate_field_names, context=text_content, wrap_prompt_in_dspy_text=True)
                # generator = DSPyGenerator(td.model.value, td.prompt_strategy, doc_schema, doc_type, verbose)
                generator = RegularGeneratorDSPyText(td.model.value, td.prompt_strategy, doc_schema, doc_type, verbose)
                answer, record_stats = None, None
                try:
                    # answer, record_stats = generator.generate(text_content, promptQuestion)
                    answer, record_stats = generator.generate(promptQuestion)
                    jsonObj = _get_JSON_from_answer(answer)["items"][0]
                    query_stats[f"all_fields_one_to_many_conventional_{idx}"] = record_stats
                except IndexError as e:
                    query_stats[f"all_fields_one_to_many_conventional_{idx}"] = record_stats
                    print("Could not find any items in the JSON response")
                    continue
                except json.JSONDecodeError as e:
                    query_stats[f"all_fields_one_to_many_conventional_{idx}"] = record_stats
                    print(f"Could not decode JSON response: {e}")
                    print(answer)
                    continue
                except Exception as e:
                    query_stats[f"all_fields_one_to_many_conventional_{idx}"] = record_stats
                    print(f"Could not decode JSON response: {e}")
                    print(answer)
                    continue

                dr = _create_data_record_from_json(jsonObj, td, candidate)
                drs.append(dr)

            # TODO how to stat this? I feel that we need a new Stats class for this type of query
            # construct ConventionalQueryStats object
            field_query_stats_lst = [FieldQueryStats(gen_stats=gen_stats, field_name=field_name) for
                                     field_name, gen_stats in query_stats.items()]
            conventional_query_stats = ConventionalQueryStats(
                field_query_stats_lst=field_query_stats_lst,
                input_fields=td.inputSchema.fieldNames(),
                generated_fields=generate_field_names,
            )

            return drs, conventional_query_stats

        else:
            raise Exception("Conventional queries cannot execute tasks with cardinality == 'oneToMany'")

    # iterate over fields and generate their values using an LLM
    query_stats = {}
    for field_name in generate_field_names:
        # construct prompt question
        wrap_prompt_in_dspy_text = (td.prompt_strategy == PromptStrategy.DSPY_COT_QA)
        promptQuestion = _construct_query_prompt(td, doc_schema, doc_type, [field_name], context=text_content, wrap_prompt_in_dspy_text=wrap_prompt_in_dspy_text)
        field_stats = None
        try:
            field_stats = None
            if td.prompt_strategy == PromptStrategy.DSPY_COT_QA:
                # print(f"FALL BACK FIELD: {field_name}")
                # print("---------------")
                # invoke LLM to generate output JSON
                # generator = DSPyGenerator(td.model.value, td.prompt_strategy, doc_schema, doc_type, verbose)
                generator = RegularGeneratorDSPyText(td.model.value, td.prompt_strategy, doc_schema, doc_type, verbose)
                # answer, field_stats = generator.generate(text_content, promptQuestion, budget=td.token_budget)
                answer, field_stats = generator.generate(promptQuestion, budget=td.token_budget)

            elif td.prompt_strategy == PromptStrategy.IMAGE_TO_TEXT:                               
                # TODO: this is very hacky; need to come up w/more general solution for multimodal schemas
                # b64 decode of candidate.contents or candidate.image_contents
                base64_images = []
                if hasattr(candidate, "contents"):
                    base64_images = [base64.b64encode(candidate.contents).decode('utf-8')]
                else:
                    base64_images = [base64.b64encode(image).decode('utf-8') for image in candidate.image_contents]

                # invoke LLM to generate output JSON
                generator = ImageTextGenerator(td.model.value)
                answer, field_stats = generator.generate(base64_images, promptQuestion)

            # TODO
            elif td.prompt_strategy == PromptStrategy.ZERO_SHOT:
                raise Exception("not implemented yet")

            # TODO
            elif td.prompt_strategy == PromptStrategy.FEW_SHOT:
                raise Exception("not implemented yet")

            # update query_stats
            query_stats[f"{field_name}"] = field_stats

            # extract result from JSON and set the DataRecord's field with its generated value
            jsonObj = _get_JSON_from_answer(answer)
            setattr(dr, field_name, jsonObj[field_name])

        except Exception as e:
            print(f"Conventional field processing error: {e}")
            setattr(dr, field_name, None)
            query_stats[f"{field_name}"] = field_stats

    # construct ConventionalQueryStats object
    field_query_stats_lst = [FieldQueryStats(gen_stats=gen_stats, field_name=field_name) for field_name, gen_stats in query_stats.items()]
    conventional_query_stats = ConventionalQueryStats(
        field_query_stats_lst=field_query_stats_lst,
        input_fields=td.inputSchema.fieldNames(),
        generated_fields=generate_field_names,
    )

    return dr, conventional_query_stats


def runCodeGenQuery(candidate: DataRecord, td: TaskDescriptor, verbose: bool=False) -> Tuple[DataRecord, Stats]:
    """
    I think this would roughly map to the internals of _makeCodeGenTypeConversionFn() in your branch.
    Similar to the functions above, I moved most of the details of generating responses
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
    
    if td.cardinality == "oneToMany":
        # TODO here the problem is: which is the 1:N field that we are splitting the output into?
        # do we need to know this to construct the prompt question ?
        # for now, we will just assume there is only one list in the JSON.
        dct = candidate.asJSON(include_bytes=False, include_data_cols=False)
        dct = json.loads(dct)
        split_attribute = [att for att in dct.keys() if type(dct[att]) == list][0]
        n_splits = len(dct[split_attribute])

        # TODO Hacky to nest return and not disrupt the rest of method!!!
        # NOTE: this is a bonded query, but we are treating it as a conventional query
        drs = []
        full_code_gen_stats, conv_query_stats = FullCodeGenStats(), {}
        for idx in range(n_splits):
            # initialize output data record
            dr = DataRecord(td.outputSchema, parent_uuid=candidate._uuid)

            cache = DataDirectory().getCacheService()
            for field_name in generate_field_names:
                code_ensemble_id = "_".join([td.op_id, field_name])
                cached_code_ensemble_info = cache.getCachedData("codeEnsemble", code_ensemble_id)
                if cached_code_ensemble_info is not None:
                    code_ensemble, _ = cached_code_ensemble_info
                    gen_stats = CodeGenEnsembleStats()
                    examples = cache.getCachedData("codeSamples", code_ensemble_id)
                else:
                    code_ensemble, gen_stats, examples = dict(), None, list()

                if verbose: 
                    print(f"Processing {split_attribute} with index {idx}")
                new_json = {k:v for k,v in dct.items() if k != split_attribute}
                new_json[split_attribute] = dct[split_attribute][idx]

                # examples.append(new_json)
                candidate_dicts = []
                for _idx in range(n_splits):
                    candidate_dict = dict(new_json)
                    candidate_dict[split_attribute] = dct[split_attribute][_idx]
                    candidate_dicts.append(candidate_dict)

                # print(type(candidate_dicts))
                # print(candidate_dicts)
                examples.extend(candidate_dicts)
                cache.putCachedData("codeSamples", code_ensemble_id, examples)
                api = API.from_task_descriptor(td, field_name, input_fields=new_json.keys())
                if len(code_ensemble)==0 or reGenerationCondition(api, examples=examples):
                    code_ensemble, gen_stats = codeEnsembleGeneration(api, examples=examples, code_num_examples=n_splits)
                    cache.putCachedData("codeEnsemble", code_ensemble_id, (code_ensemble, gen_stats))

                for code_name, code in code_ensemble.items():
                    print(f"CODE NAME: {code_name}")
                    print("-----------------------")
                    print(code)
                
                answer, exec_stats = codeEnsembleExecution(api, code_ensemble, new_json)
                full_code_gen_stats.code_gen_stats[f"{field_name}_{idx}"] = gen_stats
                full_code_gen_stats.code_exec_stats[f"{field_name}_{idx}"] = exec_stats
                
                if answer is None:
                    print(f"CODEGEN FALLING BACK TO CONVENTIONAL FOR FIELD {field_name}")
                    # construct prompt question
                    doc_schema = str(td.outputSchema)
                    doc_type = td.outputSchema.className()
                    field_stats = None
                    try:
                        # print(f"FALL BACK FIELD: {field_name}")
                        # print("---------------")
                        # invoke LLM to generate output JSON
                        text_content = json.dumps(new_json)
                        promptQuestion = _construct_query_prompt(td, doc_schema, doc_type, [field_name], context=text_content, wrap_prompt_in_dspy_text=True)
                        # generator = DSPyGenerator(td.model.value, td.prompt_strategy, doc_schema, doc_type, verbose)
                        generator = RegularGeneratorDSPyText(td.model.value, td.prompt_strategy, doc_schema, doc_type, verbose)
                        # answer, field_stats = generator.generate(text_content, promptQuestion, budget=td.token_budget)
                        answer, field_stats = generator.generate(promptQuestion, budget=td.token_budget)

                        # update conv_query_stats
                        conv_query_stats[f"{field_name}_{idx}_fallback"] = field_stats

                        # extract result from JSON and set the DataRecord's field with its generated value
                        jsonObj = _get_JSON_from_answer(answer)
                        answer = jsonObj[field_name]
                    except:
                        # update conv_query_stats
                        conv_query_stats[f"{field_name}_{idx}_fallback"] = field_stats
                        answer = None

                print(f'SETTING {field_name} to be {answer}')
                while type(answer) == type([]):
                    answer = answer[0]
                setattr(dr, field_name, answer)
                
            # TODO: last minute hack; for some reason some records are not setting a filename
            # I will need to debug this more thoroughly in the future, but for now this is an easy fix
            if not hasattr(dr, 'filename'):
                setattr(dr, 'filename', candidate.filename)

            drs.append(dr)
        
        # construct ConventionalQueryStats object
        field_query_stats_lst = [
            FieldQueryStats(gen_stats=gen_stats, field_name=field_name)
            for field_name, gen_stats in conv_query_stats.items()
        ]
        conventional_query_stats = ConventionalQueryStats(
            field_query_stats_lst=field_query_stats_lst,
            input_fields=td.inputSchema.fieldNames(),
            generated_fields=generate_field_names,
        )

        return drs, full_code_gen_stats, conventional_query_stats

    else:
        full_code_gen_stats, conv_query_stats = FullCodeGenStats(), {}
        cache = DataDirectory().getCacheService()
        for field_name in generate_field_names:
            code_ensemble_id = "_".join([td.op_id, field_name])
            cached_code_ensemble_info = cache.getCachedData("codeEnsemble", code_ensemble_id)
            if cached_code_ensemble_info is not None:
                code_ensemble, _ = cached_code_ensemble_info
                gen_stats = CodeGenEnsembleStats()
                examples = cache.getCachedData("codeSamples", code_ensemble_id)
            else:
                code_ensemble, gen_stats, examples = dict(), None, list()

            # remove bytes data from candidate
            candidate_dict = candidate.asJSON(include_bytes=False, include_data_cols=False)
            candidate_dict = {k: v for k, v in candidate_dict.items() if v != "<bytes>"}

            examples.append(candidate_dict)
            cache.putCachedData("codeSamples", code_ensemble_id, examples)
            api = API.from_task_descriptor(td, field_name, input_fields=candidate_dict.keys())
            if len(code_ensemble)==0 or reGenerationCondition(api, examples=examples):
                code_ensemble, gen_stats = codeEnsembleGeneration(api, examples=examples)
                cache.putCachedData("codeEnsemble", code_ensemble_id, (code_ensemble, gen_stats))

            for code_name, code in code_ensemble.items():
                print(f"CODE NAME: {code_name}")
                print("-----------------------")
                print(code)
        
            answer, exec_stats = codeEnsembleExecution(api, code_ensemble, candidate_dict)
            full_code_gen_stats.code_gen_stats[field_name] = gen_stats
            full_code_gen_stats.code_exec_stats[field_name] = exec_stats

            if answer is None:
                print(f"CODEGEN FALLING BACK TO CONVENTIONAL FOR FIELD {field_name}")
                # construct prompt question
                doc_schema = str(td.outputSchema)
                doc_type = td.outputSchema.className()
                field_stats = None
                try:
                    # print(f"FALL BACK FIELD: {field_name}")
                    # print("---------------")
                    # invoke LLM to generate output JSON
                    text_content = json.loads(candidate_dict)
                    promptQuestion = _construct_query_prompt(td, doc_schema, doc_type, [field_name], context=text_content, wrap_prompt_in_dspy_text=True)
                    # generator = DSPyGenerator(td.model.value, td.prompt_strategy, doc_schema, doc_type, verbose)
                    generator = RegularGeneratorDSPyText(td.model.value, td.prompt_strategy, doc_schema, doc_type, verbose)
                    # answer, field_stats = generator.generate(text_content, promptQuestion, budget=td.token_budget)
                    answer, field_stats = generator.generate(promptQuestion, budget=td.token_budget)

                    # update stats
                    conv_query_stats[f"{field_name}_fallback"] = field_stats

                    # extract result from JSON and set the DataRecord's field with its generated value
                    jsonObj = _get_JSON_from_answer(answer)
                    answer = jsonObj[field_name]
                except:
                    # update stats
                    conv_query_stats[f"{field_name}_fallback"] = field_stats
                    answer = None

            print(f'SETTING {field_name} to be {answer}')
            setattr(dr, field_name, answer)

        # TODO: last minute hack; for some reason some records are not setting a filename
        # I will need to debug this more thoroughly in the future, but for now this is an easy fix
        if not hasattr(dr, 'filename'):
            setattr(dr, 'filename', candidate.filename)
        
        # construct ConventionalQueryStats object
        field_query_stats_lst = [
            FieldQueryStats(gen_stats=gen_stats, field_name=field_name)
            for field_name, gen_stats in conv_query_stats.items()
        ]
        conventional_query_stats = ConventionalQueryStats(
            field_query_stats_lst=field_query_stats_lst,
            input_fields=td.inputSchema.fieldNames(),
            generated_fields=generate_field_names,
        )

        return dr, full_code_gen_stats, conventional_query_stats
