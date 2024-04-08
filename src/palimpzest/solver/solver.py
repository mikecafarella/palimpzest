from palimpzest.constants import PromptStrategy, QueryStrategy
from palimpzest.elements import DataRecord, File, TextFile, Schema
from palimpzest.corelib import EquationImage, ImageFile, PDFFile
from palimpzest.generators import DSPyGenerator
from palimpzest.profiler import ApiStats, FilterLLMStats, InduceLLMStats, InduceNonLLMStats
from palimpzest.solver.query_strategies import runBondedQuery, runConventionalQuery, runCodeGenQuery
from palimpzest.solver.task_descriptors import TaskDescriptor
from palimpzest.tools.pdfparser import get_text_from_pdf
from palimpzest.tools.skema_tools import equations_to_latex

from papermage import Document

import json
import modal
import time


class Solver:
    """
    This class exposes a synthesize() method, which takes in a physical operator's
    high-level description of a task to be performed (e.g. to convert a record between
    two schemas, or to filter records adhering to a specific schema) and returns a
    function which executes that task.

    The functions returned by the Solver are responsible for marshalling input records
    and producing valid output records (where "validity" is task-specific).
    
    These functions are NOT responsible for managing the details of LLM output generation.
    That responsibility lies in the Generator class(es).
    """
    def __init__(self, verbose: bool=False):
        self._hardcodedFns = {}
        self._simpleTypeConversions = set()
        self._hardcodedFns = set()
        self._hardcodedFns.add((PDFFile, File))
        self._hardcodedFns.add((PDFFile, File))
        self._hardcodedFns.add((TextFile, File))
        # self._hardcodedFns.add((ImageFile, File))
        # self._hardcodedFns.add((EquationImage, ImageFile))
        self._verbose = verbose

    def isSimpleConversion(self, td: TaskDescriptor) -> bool:
        """
        Returns true if the given task descriptor has a simple type conversion.
        """
        typeConversionDescriptor = (td.outputSchema, td.inputSchema)
        return typeConversionDescriptor in self._simpleTypeConversions


    def easyConversionAvailable(self, outputSchema: Schema, inputSchema: Schema):
        return (outputSchema, inputSchema) in self._simpleTypeConversions or (outputSchema, inputSchema) in self._hardcodedFns


    def _makeSimpleTypeConversionFn(self, td: TaskDescriptor, shouldProfile: bool=False):
        """This is a very simple function that converts a DataRecord from one Schema to another, when we know they have identical fields."""
        def _simpleTypeConversionFn(candidate: DataRecord):
            if not candidate.schema == td.inputSchema:
                return None

            dr = DataRecord(td.outputSchema, parent_uuid=candidate._uuid)
            for field in td.outputSchema.fieldNames():
                if hasattr(candidate, field):
                    setattr(dr, field, getattr(candidate, field))
                elif field.required:
                    return None

            # if profiling, set record's stats for the given op_id to be an empty Stats object
            if shouldProfile:
                candidate._stats[td.op_id] = InduceNonLLMStats()

            return [dr]
        return _simpleTypeConversionFn


    def _makeHardCodedTypeConversionFn(self, td: TaskDescriptor, shouldProfile: bool=False):
        """This converts from one type to another when we have a hard-coded method for doing so."""
        if td.outputSchema == PDFFile and td.inputSchema == File:
            if td.pdfprocessor == "modal":
                print("handling PDF processing remotely")
                remoteFunc = modal.Function.lookup("palimpzest.tools", "processPapermagePdf")
            else:
                remoteFunc = None
                
            def _fileToPDF(candidate: DataRecord):
                # parse PDF variables
                pdf_bytes = candidate.contents
                pdf_filename = candidate.filename

                # generate text_content from PDF
                start_time = time.time()
                if remoteFunc is not None:
                    docJsonStr = remoteFunc.remote([pdf_bytes])
                    docdict = json.loads(docJsonStr[0])
                    doc = Document.from_json(docdict)
                    text_content = ""
                    for p in doc.pages:
                        text_content += p.text
                else:
                    text_content = get_text_from_pdf(candidate.filename, candidate.contents)

                # construct an ApiStats object to reflect time spent waiting
                api_stats = ApiStats(api_call_duration_secs=time.time() - start_time)

                # construct data record
                dr = DataRecord(td.outputSchema, parent_uuid=candidate._uuid)
                dr.filename = pdf_filename
                dr.contents = pdf_bytes
                dr.text_contents = text_content
                # if profiling, set record's stats for the given op_id
                if shouldProfile:
                    dr._stats[td.op_id] = InduceNonLLMStats(api_stats=api_stats)
                return [dr]
            return _fileToPDF

        elif td.outputSchema == TextFile and td.inputSchema == File:
            def _fileToText(candidate: DataRecord):
                if not candidate.schema == td.inputSchema:
                    return None
                text_content = str(candidate.contents, 'utf-8')
                dr = DataRecord(td.outputSchema, parent_uuid=candidate._uuid)
                dr.filename = candidate.filename
                dr.contents = text_content
                # if profiling, set record's stats for the given op_id to be an empty Stats object
                if shouldProfile:
                    candidate._stats[td.op_id] = InduceNonLLMStats()
                return [dr]
            return _fileToText

        elif td.outputSchema == EquationImage and td.inputSchema == ImageFile:
            print("handling image to equation through skema")
            def _imageToEquation(candidate: DataRecord):
                if not candidate.element == td.inputSchema:
                    return None

                dr = DataRecord(td.outputSchema, parent_uuid=candidate._uuid)
                dr.filename = candidate.filename
                dr.contents = candidate.contents
                dr.equation_text, api_stats = equations_to_latex(candidate.contents)
                print("Running equations_to_latex_base64: ", dr.equation_text)
                # if profiling, set record's stats for the given op_id
                if shouldProfile:
                    dr._stats[td.op_id] = InduceNonLLMStats(api_stats=api_stats)
                return [dr]
            return _imageToEquation

        else:
            raise Exception(f"Cannot hard-code conversion from {td.inputSchema} to {td.outputSchema}")


    def _makeLLMTypeConversionFn(self, td: TaskDescriptor, shouldProfile: bool=False):
        def fn(candidate: DataRecord):
            # initialize stats objects
            bonded_query_stats, conventional_query_stats = None, None

            if td.query_strategy == QueryStrategy.CONVENTIONAL:
                # NOTE: runConventionalQuery does exception handling internally
                dr, conventional_query_stats = runConventionalQuery(candidate, td, self._verbose)

                # if profiling, set record's stats for the given op_id
                if shouldProfile:
                    dr._stats[td.op_id] = InduceLLMStats(conventional_query_stats=conventional_query_stats)

                return [dr]

            elif td.query_strategy == QueryStrategy.BONDED:
                drs, bonded_query_stats, err_msg = runBondedQuery(candidate, td, self._verbose)

                # if bonded query failed, manually set fields to None
                if err_msg is not None:
                    print(f"BondedQuery Error: {err_msg}")
                    dr = DataRecord(td.outputSchema, parent_uuid=candidate._uuid)
                    for field_name in td.outputSchema.fieldNames():
                        setattr(dr, field_name, None)
                    drs = [dr]

                # if profiling, set record's stats for the given op_id
                if shouldProfile:
                    for dr in drs:
                        dr._stats[td.op_id] = InduceLLMStats(bonded_query_stats=bonded_query_stats)

                return drs

            elif td.query_strategy == QueryStrategy.BONDED_WITH_FALLBACK:
                drs, bonded_query_stats, err_msg = runBondedQuery(candidate, td, self._verbose)

                # if bonded query failed, run conventional query
                if err_msg is not None:
                    print(f"BondedQuery Error: {err_msg}")
                    print("Falling back to conventional query")
                    dr, conventional_query_stats = runConventionalQuery(candidate, td, self._verbose)
                    drs = [dr]

                # if profiling, set record's stats for the given op_id
                if shouldProfile:
                    for dr in drs:
                        dr._stats[td.op_id] = InduceLLMStats(
                            bonded_query_stats=bonded_query_stats,
                            conventional_query_stats=conventional_query_stats,
                        )

                return drs

            else:
                raise ValueError(f"Unrecognized QueryStrategy: {td.query_strategy.value}")

        return fn


    # TODO: @Zui
    def _makeCodeGenTypeConversionFn(self, td: TaskDescriptor, shouldProfile: bool=False):
        """
        If you look at my implementation of _makeLLMTypeConversionFn above, you'll see that
        I've moved a lot of the core logic around generating outputs and collecting statistics
        into a set of functions which are defined in palimpzest.solver.query_strategies.
        """
        def fn(candidate: DataRecord):
            # initialize stats objects
            full_code_gen_stats = None

            # TODO
            if td.query_strategy == QueryStrategy.CODE_GEN:
                # drs, full_code_gen_stats, err_msg = runCodeGenQuery(candidate, td, self._verbose)

                # # if code gen query failed, manually set fields to None
                # if err_msg is not None:
                #     print(f"CodeGenQuery Error: {err_msg}")
                #     dr = DataRecord(td.outputSchema, parent_uuid=candidate._uuid)
                #     for field_name in td.outputSchema.fieldNames():
                #         setattr(dr, field_name, None)
                #     drs = [dr]

                # # if profiling, set record's stats for the given op_id
                # if shouldProfile:
                #     for dr in drs:
                #         dr._stats[td.op_id] = InduceLLMStats(full_code_gen_stats=full_code_gen_stats)

                # return drs
                raise Exception("not implemented yet")

            # TODO
            elif td.query_strategy == QueryStrategy.CODE_GEN_WITH_FALLBACK:
                # similar to in _makeLLMTypeConversionFn; maybe we can have one strategy in which we try
                # to use code generation, but if it fails then we fall back to a conventional query strategy?
                raise Exception("not implemented yet")

            else:
                raise ValueError(f"Unrecognized QueryStrategy: {td.query_strategy.value}")

        return fn


    def _makeFilterFn(self, td: TaskDescriptor, shouldProfile: bool=False):
            # compute record schema and type
            doc_schema = str(td.inputSchema)
            doc_type = td.inputSchema.className()

            # By default, a filter requires an LLM invocation to run
            # Someday maybe we will offer the user the chance to run a hard-coded function.
            # Or maybe we will ask the LLM to synthesize traditional code here.
            def createLLMFilter(filterCondition: str):
                def llmFilter(candidate: DataRecord):
                    # do not filter candidate if it doesn't match inputSchema
                    if not candidate.schema == td.inputSchema:
                        return False

                    # create generator
                    generator = None
                    if td.prompt_strategy == PromptStrategy.DSPY_COT_BOOL:
                        generator = DSPyGenerator(td.model.value, td.prompt_strategy, doc_schema, doc_type, self._verbose)
                    # TODO
                    elif td.prompt_strategy == PromptStrategy.ZERO_SHOT:
                        raise Exception("not implemented yet")
                    # TODO
                    elif td.prompt_strategy == PromptStrategy.FEW_SHOT:
                        raise Exception("not implemented yet")
                    # TODO
                    elif td.prompt_strategy == PromptStrategy.CODE_GEN_BOOL:
                        raise Exception("not implemented yet")

                    # invoke LLM to generate filter decision (True or False)
                    text_content = candidate.asTextJSON()
                    response, gen_stats = generator.generate(context=text_content, question=filterCondition)

                    # if profiling, set record's stats for the given op_id
                    if shouldProfile:
                        candidate._stats[td.op_id] = FilterLLMStats(gen_stats=gen_stats, filter=filterCondition)

                    # set _passed_filter attribute and return record
                    setattr(candidate, "_passed_filter", response.lower() == "true")

                    return candidate

                return llmFilter
            return createLLMFilter(str(td.filter))


    def synthesize(self, td: TaskDescriptor, shouldProfile: bool=False):
        """
        Return a function that implements the desired task as specified by some PhysicalOp.
        Right now, the two primary tasks that the Solver provides solutions for are:

        1. Induce operations
        2. Filter operations

        The shouldProfile parameter also determines whether or not PZ should compute
        profiling statistics for LLM invocations and attach them to each record.
        """
        # synthesize a function to induce from inputType to outputType
        if "InduceFromCandidateOp" in td.physical_op:
            typeConversionDescriptor = (td.outputSchema, td.inputSchema)
            if typeConversionDescriptor in self._simpleTypeConversions:
                return self._makeSimpleTypeConversionFn(td, shouldProfile)
            elif typeConversionDescriptor in self._hardcodedFns:
                return self._makeHardCodedTypeConversionFn(td, shouldProfile)
            else:
                return self._makeLLMTypeConversionFn(td, shouldProfile)

        # synthesize a function to filter records
        elif "FilterCandidateOp" in td.physical_op:
            return self._makeFilterFn(td, shouldProfile)

        else:
            raise Exception("Cannot synthesize function for task descriptor: " + str(td))
