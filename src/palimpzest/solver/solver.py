from io import BytesIO
import numpy as np
from palimpzest.constants import PromptStrategy, QueryStrategy
from palimpzest.elements import DataRecord, File, TextFile, Schema
from palimpzest.corelib import EquationImage, ImageFile, PDFFile, Download, XLSFile, Table
from palimpzest.generators import DSPyGenerator
from palimpzest.profiler import ApiStats, FilterLLMStats, FilterNonLLMStats, InduceLLMStats, InduceNonLLMStats
from palimpzest.solver.query_strategies import runBondedQuery, runConventionalQuery, runCodeGenQuery
from palimpzest.solver.task_descriptors import TaskDescriptor
from palimpzest.tools.pdfparser import get_text_from_pdf
from palimpzest.tools.skema_tools import equations_to_latex

from papermage import Document

import json
import modal
import time
import pandas as pd

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
        # TODO GV: As we add more hardcoded functions, we should find a more scalable way to manage them,
        # a simple idea could be to have a dictionary of hardcoded functions, where the key is a tuple of the input and output schema
        self._hardcodedFns.add((PDFFile, File))
        self._hardcodedFns.add((PDFFile, File))
        self._hardcodedFns.add((TextFile, File))
        self._hardcodedFns.add((File, Download))
        self._hardcodedFns.add((XLSFile, File))
        self._hardcodedFns.add((Table, XLSFile))
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

            return [dr], None
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
                dr.text_contents = text_content[:10000] # TODO Very hacky
                # if profiling, set record's stats for the given op_id
                if shouldProfile:
                    dr._stats[td.op_id] = InduceNonLLMStats(api_stats=api_stats)
                return [dr], None
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
                return [dr], None
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
                return [dr], None
            return _imageToEquation

        elif td.outputSchema == File and td.inputSchema == Download: # TODO make sure this is also true for children classes of File
            def _downloadToFile(candidate: DataRecord):
                if not candidate.schema == td.inputSchema:
                    return None
                dr = DataRecord(td.outputSchema, parent_uuid=candidate._uuid)
                # Assign a filename that is parsed from the URL
                dr.filename = candidate.url.split("/")[-1]
                dr.contents = candidate.content
                if shouldProfile:
                    candidate._stats[td.op_id] = InduceNonLLMStats()
                return [dr], None
            return _downloadToFile

        elif td.outputSchema == XLSFile and td.inputSchema == File:
            def _fileToXLS(candidate: DataRecord):
                if not candidate.schema == td.inputSchema:
                    return None
                dr = DataRecord(td.outputSchema, parent_uuid=candidate._uuid)
                dr.filename = candidate.filename
                dr.contents = candidate.contents

                start_time = time.time()
                xls = pd.ExcelFile(BytesIO(candidate.contents), engine='openpyxl')
                api_stats = ApiStats(api_call_duration_secs=time.time() - start_time)

                dr.number_sheets = len(xls.sheet_names)
                dr.sheet_names = xls.sheet_names

                if shouldProfile:
                    candidate._stats[td.op_id] = InduceNonLLMStats(api_stats=api_stats)
                return [dr], None
            return _fileToXLS

        elif td.outputSchema == Table and td.inputSchema == XLSFile:
            cardinality = td.cardinality
            def _excelToTable(candidate: DataRecord):
                xls_bytes = candidate.contents
                # dr.sheets = [xls.parse(name) for name in candidate.sheet_names]
                sheet_names = [candidate.sheet_names[0]] if cardinality is None else candidate.sheet_names

                records = []
                for sheet_name in sheet_names:
                    start_time = time.time()
                    dataframe = pd.read_excel(BytesIO(xls_bytes), sheet_name=sheet_name, engine='openpyxl')
                    api_stats = ApiStats(api_call_duration_secs=time.time() - start_time)

                    # construct data record
                    dr = DataRecord(td.outputSchema, parent_uuid=candidate._uuid)
                    rows = []
                    for row in dataframe.values[:100]: # TODO Extend this with dynamic sizing of context length
                        row_record = [str(x) for x in row]
                        rows += [row_record]
                    dr.rows = rows
                    dr.filename = candidate.filename
                    dr.header = dataframe.columns.values.tolist()
                    dr.name = candidate.filename.split("/")[-1] + "_" + sheet_name

                    if shouldProfile:
                        dr._stats[td.op_id] = InduceNonLLMStats(api_stats=api_stats)
                    records.append(dr)
                return records, None
            return _excelToTable

        else:
            raise Exception(f"There is no hard-coded conversion from {td.inputSchema} to {td.outputSchema}")


    def _makeLLMTypeConversionFn(self, td: TaskDescriptor, shouldProfile: bool=False):
        def fn(candidate: DataRecord):
            # initialize stats objects
            bonded_query_stats, conventional_query_stats = None, None

            if td.query_strategy == QueryStrategy.CONVENTIONAL:
                # NOTE: runConventionalQuery does exception handling internally
                dr, conventional_query_stats = runConventionalQuery(candidate, td, self._verbose)

                # if profiling, set record's stats for the given op_id
                if shouldProfile:
                    dr._stats[td.op_id] = InduceLLMStats(
                        query_strategy=td.query_strategy.value,
                        token_budget=td.token_budget,
                        conventional_query_stats=conventional_query_stats,
                    )

                return [dr], None

            elif td.query_strategy == QueryStrategy.BONDED:
                drs, new_heatmap_obj, bonded_query_stats, err_msg = runBondedQuery(candidate, td, self._verbose)

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
                        dr._stats[td.op_id] = InduceLLMStats(
                            query_strategy=td.query_strategy.value,
                            token_budget=td.token_budget,
                            bonded_query_stats=bonded_query_stats,
                        )

                return drs, new_heatmap_obj

            elif td.query_strategy == QueryStrategy.BONDED_WITH_FALLBACK:
                drs, new_heatmap_obj, bonded_query_stats, err_msg = runBondedQuery(candidate, td, self._verbose)

                # if bonded query failed, run conventional query
                if err_msg is not None:
                    print(f"BondedQuery Error: {err_msg}")
                    print("Falling back to conventional query")
                    dr, conventional_query_stats = runConventionalQuery(candidate, td, self._verbose)
                    drs = [dr] if type(dr) is not list else dr
                # if profiling, set record's stats for the given op_id
                if shouldProfile:
                    for dr in drs:
                        # TODO: divide bonded query_stats time, cost, and input/output tokens by len(drs)
                        dr._stats[td.op_id] = InduceLLMStats(
                            query_strategy=td.query_strategy.value,
                            token_budget=td.token_budget,
                            bonded_query_stats=bonded_query_stats,
                            conventional_query_stats=conventional_query_stats,
                        )

                return drs, new_heatmap_obj

            elif td.query_strategy == QueryStrategy.CODE_GEN:
                dr, full_code_gen_stats = runCodeGenQuery(candidate, td, self._verbose)
                drs = [dr]

                # if profiling, set record's stats for the given op_id
                if shouldProfile:
                    for dr in drs:
                        dr._stats[td.op_id] = InduceLLMStats(
                            query_strategy=td.query_strategy.value,
                            token_budget=td.token_budget,
                            full_code_gen_stats=full_code_gen_stats,
                        )

                return drs, None

            elif td.query_strategy == QueryStrategy.CODE_GEN_WITH_FALLBACK:
                # similar to in _makeLLMTypeConversionFn; maybe we can have one strategy in which we try
                # to use code generation, but if it fails then we fall back to a conventional query strategy?
                dr, full_code_gen_stats, conventional_query_stats = runCodeGenQuery(candidate, td, self._verbose)
                drs = [dr] if type(dr) is not list else dr
                # # Deleting all failure fields
                # for field_name in td.outputSchema.fieldNames():
                #     if hasattr(new_candidate, field_name) and (getattr(new_candidate, field_name) is None):
                #         delattr(new_candidate, field_name)
                # if td.cardinality == 'oneToMany':
                #     td.cardinality = 'oneToOne'
                # dr, conventional_query_stats = runConventionalQuery(new_candidate, td, self._verbose)
                # drs = [dr] if type(dr) is not list else dr
                for dr in drs:
                    dr._parent_uuid = candidate._uuid

                # if profiling, set record's stats for the given op_id
                if shouldProfile:
                    for dr in drs:
                        # TODO: divide bonded query_stats time, cost, and input/output tokens by len(drs)
                        dr._stats[td.op_id] = InduceLLMStats(
                            query_strategy=td.query_strategy.value,
                            token_budget=td.token_budget,
                            full_code_gen_stats=full_code_gen_stats,
                            conventional_query_stats=conventional_query_stats,
                        )

                return drs, None

            else:
                raise ValueError(f"Unrecognized QueryStrategy: {td.query_strategy.value}")

        return fn

    def _makeFilterFn(self, td: TaskDescriptor, shouldProfile: bool=False):
        # compute record schema and type
        doc_schema = str(td.inputSchema)
        doc_type = td.inputSchema.className()

        # if filter has a function, simply return a wrapper around that function
        if td.filter.filterFn is not None:
            def nonLLMFilter(candidate: DataRecord):
                start_time = time.time()
                result = td.filter.filterFn(candidate)
                fn_call_duration_secs = time.time() - start_time

                # if profiling, set record's stats for the given op_id
                if shouldProfile:
                    candidate._stats[td.op_id] = FilterNonLLMStats(
                        fn_call_duration_secs=fn_call_duration_secs,
                        filter=str(td.filter.filterFn),
                    )

                # set _passed_filter attribute and return record
                setattr(candidate, "_passed_filter", result)
                print(f"ran filter function on {candidate}")

                return candidate

            return nonLLMFilter

        # otherwise, the filter requires an LLM invocation to run
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
            text_content = candidate._asJSON(include_bytes=False)
            try:
                response, _, gen_stats = generator.generate(context=text_content, question=td.filter.filterCondition, plan_idx=td.plan_idx)

                # if profiling, set record's stats for the given op_id
                if shouldProfile:
                    candidate._stats[td.op_id] = FilterLLMStats(gen_stats=gen_stats, filter=td.filter.filterCondition)

                # set _passed_filter attribute and return record
                setattr(candidate, "_passed_filter", "true" in response.lower())
            except Exception as e:
                # If there is an exception consider the record as not passing the filter
                print(f"Error invoking LLM for filter: {e}")
                setattr(candidate, "_passed_filter", False)

            return candidate

        return llmFilter


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
