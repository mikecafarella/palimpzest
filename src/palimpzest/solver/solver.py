from palimpzest.constants import Model, PromptStrategy
from palimpzest.elements import DataRecord, EquationImage, File, Filter, ImageFile, PDFFile, Schema, TextFile, BytesField
from palimpzest.tools.dspysearch import run_cot_bool, run_cot_qa, exec_codegen, gen_filter_signature_class, gen_qa_signature_class
from palimpzest.tools.openai_image_converter import do_image_analysis
from palimpzest.tools.pdfparser import get_text_from_pdf
from palimpzest.tools.profiler import Profiler
from palimpzest.tools.skema_tools import equations_to_latex_base64, equations_to_latex
from palimpzest.solver.sandbox import *
from palimpzest.solver.codegen import getConversionCodes

from collections import defaultdict
from copy import deepcopy
from papermage import Document
from typing import Any, Dict, Tuple, Union

import json
import base64
import modal
import os

# DEFINITIONS
# TaskDescriptor = Tuple[str, Union[tuple, None], Schema, Schema]


class Solver:
    """This solves for needed operator implementations"""
    def __init__(self, verbose: bool=False):
        self._hardcodedFns = {}
        self._simpleTypeConversions = set()
        self._hardcodedFns = set()
        self._hardcodedFns.add((PDFFile, File))
        self._hardcodedFns.add((PDFFile, File))
        self._hardcodedFns.add((ImageFile, File))
        self._hardcodedFns.add((TextFile, File))
        # self._hardcodedFns.add((EquationImage, ImageFile))
        self._verbose = verbose

    def easyConversionAvailable(self, outputSchema: Schema, inputSchema: Schema):
        return (outputSchema, inputSchema) in self._simpleTypeConversions or (outputSchema, inputSchema) in self._hardcodedFns

    def _makeSimpleTypeConversionFn(self, outputSchema, inputSchema):
        """This is a very simple function that converts a DataRecord from one Schema to another, when we know they have identical fields."""
        def _simpleTypeConversionFn(candidate: DataRecord):
            if not candidate.schema == inputSchema:
                return None

            dr = DataRecord(outputSchema)
            for field in outputSchema.fieldNames():
                if hasattr(candidate, field):
                    setattr(dr, field, getattr(candidate, field))
                elif field.required:
                    return None
            return dr
        return _simpleTypeConversionFn

    def _makeHardCodedTypeConversionFn(self, outputSchema: Schema, inputSchema: Schema, config: Dict[str, Any], op_id: str):
        """This converts from one type to another when we have a hard-coded method for doing so."""
        if outputSchema == PDFFile and inputSchema == File:
            if config.get("pdfprocessing") == "modal":
                print("handling PDF processing remotely")
                remoteFunc = modal.Function.lookup("palimpzest.tools.allenpdf", "processPapermagePdf")
            else:
                remoteFunc = None
                
            def _fileToPDF(candidate: DataRecord):
                pdf_bytes = candidate.contents
                pdf_filename = candidate.filename
                if remoteFunc is not None:
                    docJsonStr = remoteFunc.remote([pdf_bytes])
                    docdict = json.loads(docJsonStr[0])
                    doc = Document.from_json(docdict)
                    text_content = ""
                    for p in doc.pages:
                        text_content += p.text
                else:
                    text_content = get_text_from_pdf(candidate.filename, candidate.contents)
                dr = DataRecord(outputSchema)
                dr.filename = pdf_filename
                dr.contents = pdf_bytes
                dr.text_contents = text_content
                return dr
            return _fileToPDF
        elif outputSchema == TextFile and inputSchema == File:
            def _fileToText(candidate: DataRecord):
                if not candidate.schema == inputSchema:
                    return None
                text_content = str(candidate.contents, 'utf-8')
                dr = DataRecord(outputSchema)
                dr.filename = candidate.filename
                dr.contents = text_content
                return dr
            return _fileToText
        elif outputSchema == EquationImage and inputSchema == ImageFile:
            print("handling image to equation through skema")
            def _imageToEquation(candidate: DataRecord):
                if not candidate.element == inputSchema:
                    return None

                dr = DataRecord(outputSchema)
                dr.filename = candidate.filename
                dr.contents = candidate.contents
                dr.equation_text = equations_to_latex(candidate.contents)
                print("Running equations_to_latex_base64: ", dr.equation_text)
                return dr
            return _imageToEquation

        # TODO: maybe move this to _makeLLMTypeConversionFn?
        elif outputSchema == ImageFile and inputSchema == File:
            def _fileToImage(candidate: DataRecord):
                if not candidate.schema == inputSchema:
                    return None
                # b64 decode of candidate.contents
                image_bytes = base64.b64encode(candidate.contents).decode('utf-8')
                dr = DataRecord(outputSchema)
                dr.filename = candidate.filename
                if 'OPENAI_API_KEY' not in os.environ:
                    raise ValueError("OPENAI_API_KEY not found in environment variables")
                # get openai key from environment
                openai_key = os.environ['OPENAI_API_KEY']
                # TODO: consider multiple image models
                dr.contents = candidate.contents
                dr.text_description, stats = do_image_analysis(openai_key, image_bytes)

                # if profiling, set record's stats for the given op_id
                if Profiler.profiling_on():
                    dr._stats[op_id] = {"fields": {"text_description": stats}}

                return dr
            return _fileToImage

        else:
            raise Exception(f"Cannot hard-code conversion from {inputSchema} to {outputSchema}")

    def _makeLLMTypeConversionFn(self, outputSchema: Schema, inputSchema: Schema, config: Dict[str, Any], model: Model, prompt_strategy: PromptStrategy, op_id: str, conversionDesc: str):
            def fn(candidate: DataRecord):
                # iterate through all empty fields in the outputSchema and ask questions to fill them
                # for field in inputSchema.__dict__:
                dr = DataRecord(outputSchema)
                text_content = candidate.asTextJSON()
                doc_schema = str(outputSchema)
                doc_type = outputSchema.className()
                stats = {}

                field_stats = None

                fieldDescriptions = ""
                for field_name in outputSchema.fieldNames():
                    f = getattr(outputSchema, field_name)
                    fieldDescriptions += f"{field_name}: {f.desc}\n"

                promptQuestion = f"""I am trying to create an output object of type {doc_type}. I have an input object of type {str(inputSchema)}.
                I must somehow use the information in the input object to create the correct fields in the output object. 
                The input object is described as follows: {inputSchema.__doc__}.
                The output object is described as follows: {outputSchema.__doc__}.
                The output object should have a set of fields that are described in the following JSON schema: {doc_schema}.
                Here are detailed descriptions of each of the fields in the desired output object: \n{fieldDescriptions}.
                "Please return a correct parsable JSON object as an answer. Return ONLY the JSON object.""" + "" if conversionDesc is None else f" Keep in mind that this output is described by this text: {conversionDesc}."                

                try:
                    if prompt_strategy == PromptStrategy.DSPY_COT:
                        answer, field_stats = run_cot_qa(text_content, promptQuestion,
                                                                 model_name=model.value, 
                                                                 verbose=self._verbose, 
                                                                 promptSignature=gen_qa_signature_class(doc_schema, doc_type))
                    try:
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

                        jsonObj = json.loads(answer)
                        for field_name in outputSchema.fieldNames():
                            # parse the json object and set the fields of the record
                            setattr(dr, field_name, jsonObj[field_name])
                            stats[f"{field_name}"] = field_stats
                    except Exception as e:
                        print(f"Error: {e}")
                        for field_name in outputSchema.fieldNames():
                            setattr(dr, field_name, None)
                except Exception as e:
                    print(f"Error: {e}")
                    for field_name in outputSchema.fieldNames():
                        setattr(dr, field_name, None)

                # if profiling, set record's stats for the given op_id
                if Profiler.profiling_on():
                    dr._stats[op_id] = {"fields": stats}

                return dr

            def fnOrig(candidate: DataRecord):
                # iterate through all empty fields in the outputSchema and ask questions to fill them
                # for field in inputSchema.__dict__:
                dr = DataRecord(outputSchema)
                text_content = candidate.asTextJSON()
                doc_schema = str(outputSchema)
                doc_type = outputSchema.className()
                stats = {}
                for field_name in outputSchema.fieldNames():
                    f = getattr(outputSchema, field_name)
                    try:
                        # TODO: allow for mult. fcns
                        field_stats = None
                        if prompt_strategy == PromptStrategy.DSPY_COT:
                            answer, field_stats = run_cot_qa(text_content,
                                                             f"What is the {field_name} of the {doc_type}? ({f.desc})" + "" if conversionDesc is None else f" Keep in mind that this output is described by this text: {conversionDesc}.",
                                                             model_name=model.value, verbose=self._verbose, promptSignature=gen_qa_signature_class(doc_schema, doc_type))
                        # TODO
                        elif prompt_strategy == PromptStrategy.ZERO_SHOT:
                            raise Exception("not implemented yet")
                        # TODO
                        elif prompt_strategy == PromptStrategy.FEW_SHOT:
                            raise Exception("not implemented yet")

                        setattr(dr, field_name, answer)
                        stats[f"{field_name}"] = field_stats

                    except Exception as e:
                        print(f"Error: {e}")
                        setattr(dr, field_name, None)
                
                # if profiling, set record's stats for the given op_id
                if Profiler.profiling_on():
                    dr._stats[op_id] = {"fields": stats}

                return dr

            def fnWithBypass(candidate: DataRecord, drInit: DataRecord=None):
                # iterate through all empty fields in the outputSchema and ask questions to fill them
                # for field in inputSchema.__dict__:
                dr = drInit if drInit is not None else DataRecord(outputSchema)
                text_content = candidate.asTextJSON()
                dict_content = candidate.asDict()
                doc_schema = str(outputSchema)
                doc_type = outputSchema.className()
                inputs = {k:v for k,v in dict_content.items() if k in inputSchema.fieldNames()}

                stats = {}
                for field_name in outputSchema.fieldNames():
                    if hasattr(dr, field_name) and getattr(dr, field_name) is not None:
                        continue
                    f = getattr(outputSchema, field_name)
                    try:
                        # TODO: allow for mult. fcns
                        field_stats = None
                        if prompt_strategy == PromptStrategy.DSPY_COT:
                            answer, field_stats = run_cot_qa(text_content,
                                                             f"What is the {field_name} of the {doc_type}? ({f.desc})" + "" if conversionDesc is None else f" Keep in mind that this output is described by this text: {conversionDesc}.",
                                                             model_name=model.value, verbose=self._verbose, promptSignature=gen_qa_signature_class(doc_schema, doc_type))
                        # TODO
                        elif prompt_strategy == PromptStrategy.ZERO_SHOT:
                            raise Exception("not implemented yet")
                        # TODO
                        elif prompt_strategy == PromptStrategy.FEW_SHOT:
                            raise Exception("not implemented yet")

                        setattr(dr, field_name, answer)
                        stats[f"{field_name}"] = field_stats
                        
                    except Exception as e:
                        print(f"Error: {e}")
                        setattr(dr, field_name, None)

                # if profiling, set record's stats for the given op_id
                if Profiler.profiling_on():
                    dr._stats[op_id] = {"fields": stats}
                return dr
            
            return fnWithBypass

    def _makeCodeGenTypeConversionFn(self, outputSchema: Schema, inputSchema: Schema, config: Dict[str, Any], model: Model, op_id: str, conversionDesc: str):
            def fn(candidate: DataRecord):
                # generate a program for each field in the outputSchema to fill them
                # generated program will be reused if the (inputSchema, getattr(outputSchema, field_name)) is the same
                # for field in inputSchema.__dict__:
                dr = DataRecord(outputSchema)
                dict_content = candidate.asDict()
                inputs = {k:v for k,v in dict_content.items() if k in inputSchema.fieldNames()}

                stats = {}
                for field_name in outputSchema.fieldNames():
                    f = getattr(outputSchema, field_name)
                    if isinstance(f, BytesField):
                        continue
                    if field_name in inputSchema.fieldNames():
                        setattr(dr, field_name, inputs[field_name])
                        continue
                    api = API(name = "extract", inputs = [
                        {'name': input_field_name, 'type': 'str', 'desc': getattr(inputSchema,input_field_name).desc} for input_field_name in inputSchema.fieldNames()
                    ], outputs=[
                        {'name': field_name, 'type': 'str', 'desc': f.desc}
                    ])
                    codes, field_codegen_stats = getConversionCodes(inputSchema, {field_name: f}, conversionDesc, config, model, api, example_inputs=inputs)
                    answers, field_stats = list(), defaultdict(float)
                    for code in codes:
                        answer, code_stats = exec_codegen(api, code, inputs)
                        answers.append(answer)
                        for k,v in code_stats.items(): field_stats[k] += v
                        if config.get('codegen_logging', default=False):
                            print(f"Code:\n====================\n{code}\n====================\nInputs:\n{DumpsJson(inputs,indent=4)}\nOutput:\n{answer}\n\n")
                    majority_answer = max(set(answers), key = answers.count)
                    
                    # For logging purpose only, set the field to the answer + " (code extracted)"
                    if config.get('codegen_logging', default=False):
                        majority_answer += " (code extracted)"
                    setattr(dr, field_name, majority_answer)
                    stats[f"{field_name}"] = {"conversion_stats": field_stats, "codegen_stats": field_codegen_stats}

                # if profiling, set record's stats for the given op_id
                if Profiler.profiling_on():
                    dr._stats[op_id] = {"fields": stats}
                return dr
            return fn

    def _makeHybridTypeConversionFn(self, outputSchema: Schema, inputSchema: Schema, config: Dict[str, Any], model: Model, prompt_strategy: PromptStrategy, op_id: str, conversionDesc: str):
            def fn(candidate: DataRecord):
                dr = self._makeCodeGenTypeConversionFn(outputSchema, inputSchema, config, model, op_id, conversionDesc)(candidate)
                return self._makeLLMTypeConversionFn(outputSchema, inputSchema, config, model, prompt_strategy, op_id, conversionDesc)(candidate, drInit=dr)
            return fn

    def _makeLLMFilterFn(self, inputSchema: Schema, filter: Filter, config: Dict[str, Any], model: Model, prompt_strategy: PromptStrategy, op_id: str):
            # parse inputs
            doc_schema = str(inputSchema)
            doc_type = inputSchema.className()

            # By default, a filter requires an LLM invocation to run
            # Someday maybe we will offer the user the chance to run a hard-coded function.
            # Or maybe we will ask the LLM to synthesize traditional code here.
            def createLLMFilter(filterCondition: str):
                def llmFilter(candidate: DataRecord):
                    if not candidate.schema == inputSchema:
                        return False
                    text_content = candidate.asTextJSON()
                    # TODO: allow for mult. fcns
                    response, stats = None, {}
                    if prompt_strategy == PromptStrategy.DSPY_BOOL:
                        response, stats = run_cot_bool(text_content, filterCondition, model_name=model.value,
                                                       verbose=self._verbose, promptSignature=gen_filter_signature_class(doc_schema, doc_type))
                    # TODO
                    elif prompt_strategy == PromptStrategy.ZERO_SHOT:
                        raise Exception("not implemented yet")
                    # TODO
                    elif prompt_strategy == PromptStrategy.FEW_SHOT:
                        raise Exception("not implemented yet")

                    # if profiling, set record's stats for the given op_id and clear any state from the previous operation
                    if Profiler.profiling_on():
                        candidate = deepcopy(candidate)
                        candidate._state = {}
                        candidate._stats[op_id] = stats

                    # set _passed_filter attribute and return record
                    setattr(candidate, "_passed_filter", response.lower() == "true")

                    return candidate

                return llmFilter
            return createLLMFilter(str(filter))

    def synthesize(self, taskDescriptor: Any, config: Dict[str, Any]):
        """Return a function that maps from inputType to outputType."""
        functionName, functionParams, outputSchema, inputSchema = taskDescriptor

        if functionName == "InduceFromCandidateOp" or functionName == "ParallelInduceFromCandidateOp":
            model, prompt_strategy, op_id, conversionDesc = functionParams
            typeConversionDescriptor = (outputSchema, inputSchema)
            if typeConversionDescriptor in self._simpleTypeConversions:
                return self._makeSimpleTypeConversionFn(outputSchema, inputSchema)
            elif typeConversionDescriptor in self._hardcodedFns:
                return self._makeHardCodedTypeConversionFn(outputSchema, inputSchema, config, op_id) # TODO: add another model for image processing (e.g., Claude)?
            elif config.get("codegen", default=False):
                return self._makeHybridTypeConversionFn(outputSchema, inputSchema, config, model, prompt_strategy, op_id, conversionDesc)
            else:
                return self._makeLLMTypeConversionFn(outputSchema, inputSchema, config, model, prompt_strategy, op_id, conversionDesc)
        elif functionName == "FilterCandidateOp" or functionName == "ParallelFilterCandidateOp":
            filter, model, prompt_strategy, op_id = functionParams
            return self._makeLLMFilterFn(inputSchema, filter, config, model, prompt_strategy, op_id)
        else:
            raise Exception("Cannot synthesize function for task descriptor: " + str(taskDescriptor))
