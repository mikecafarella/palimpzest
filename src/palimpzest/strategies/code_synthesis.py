from __future__ import annotations
import json
import time
from typing import List, Tuple
from palimpzest.constants import Model
from palimpzest.datamanager.datamanager import DataDirectory
from .strategy import PhysicalOpStrategy

from palimpzest.utils import API, getJsonFromAnswer
from palimpzest.generators import codeEnsembleExecution

from palimpzest.constants import *
from palimpzest.elements import *
from palimpzest.operators import logical, physical, convert

CODEGEN_PROMPT = """You are a helpful programming assistant and an expert {language} programmer. Implement the {language} function `{api}` that extracts `{output}` ({output_desc}) from given inputs:
{inputs_desc}
{examples_desc}
Notice that the evaluation will severely punish incorrect outputs. Thus, when the function is uncertain, it should return `None` to abstain instead of returning an incorrect guess.
{advice}
Return the implementation only."""

# TYPE DEFINITIONS
FieldName = str
CodeName = str
Code = str
DataRecordDict = Dict[str, Any]
Exemplar = Tuple[DataRecordDict, DataRecordDict]
CodeEnsemble = Dict[CodeName, Code]

class LLMConvertCodeSynthesis(convert.LLMConvert):

    strategy: CodeSynthStrategy # Default is CodeSynthStrategy.SINGLE,

    def __init__(self, 
                cache_across_plans: bool = True,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.cache_across_plans = cache_across_plans

        # read the list of exemplars already generated by this operator if present
        if self.cache_across_plans:
            cache = DataDirectory().getCacheService()
            exemplars_cache_id = self.get_op_id()
            exemplars = cache.getCachedData("codeExemplars", exemplars_cache_id)
            # set and return exemplars if it is not empty
            if exemplars is not None and len(exemplars) > 0:
                self.exemplars = exemplars
        else:
            self.exemplars = []

    def _shouldSynthesize(
        self,
        num_exemplars: int=1,               # if strategy == EXAMPLE_ENSEMBLE
        code_regenerate_frequency: int=200, # if strategy == ADVICE_ENSEMBLE_WITH_VALIDATION
    ) -> bool:
        """ This function determines whether code synthesis 
        should be performed based on the strategy and the number of exemplars available. """

        if self.strategy == CodeSynthStrategy.NONE:
            return False
        
        elif self.strategy == CodeSynthStrategy.SINGLE:
            return not self.code_synthesized and len(self.exemplars) >= num_exemplars
        
        elif self.strategy == CodeSynthStrategy.EXAMPLE_ENSEMBLE:
            if len(self.exemplars) <= num_exemplars:
                return False
            return not self.code_synthesized
        
        elif self.strategy == CodeSynthStrategy.ADVICE_ENSEMBLE:
            return False
        
        elif self.strategy == CodeSynthStrategy.ADVICE_ENSEMBLE_WITH_VALIDATION:
            return len(self.exemplars) % code_regenerate_frequency == 0
        
        else:
            raise Exception("not implemented yet")


    def _fetch_code_ensemble(self, generate_field_names: List[str]) -> Tuple[Dict[CodeName, Code]]:
        # if we are allowed to cache synthesized code across plan executions, check the cache
        if not self.no_cache_across_plans:
            field_to_code_ensemble = {}
            cache = DataDirectory().getCacheService()
            for field_name in generate_field_names:
                code_ensemble_cache_id = "_".join([self.get_op_id(), field_name])
                code_ensemble = cache.getCachedData("codeEnsembles", code_ensemble_cache_id)
                if code_ensemble is not None:
                    field_to_code_ensemble[field_name] = code_ensemble

            # set and return field_to_code_ensemble if all fields are present and have code
            if all([field_to_code_ensemble.get(field_name, None) is not None for field_name in generate_field_names]):
                self.field_to_code_ensemble = field_to_code_ensemble
                return self.field_to_code_ensemble, {}

        else:
            # if we're not synthesizing new code ensemble(s) and there is nothing to fetch, return empty dicts
            return {}, {}


    def _synthesize_code_ensemble(
        self,
        api: API,
        output_field_name: str,
        strategy: CodeSynthStrategy=CodeSynthStrategy.SINGLE,
        code_ensemble_num: int=1,       # if strategy != SINGLE
        num_exemplars: int=1,           # if strategy != EXAMPLE_ENSEMBLE
    ) -> Tuple[Dict[CodeName, Code], StatsDict]:

        code_ensemble = dict()
        exemplars = self.exemplars

        if strategy == CodeSynthStrategy.NONE:
            # create an ensemble with one function which returns None
            code, code_synth_stats = self._code_synth_default(api)
            code_name = f"{api.name}_v0"
            code_ensemble[code_name] = code
            return code_ensemble, code_synth_stats

        elif strategy == CodeSynthStrategy.SINGLE:
            # use exemplars to create an ensemble with a single synthesized function
            code, code_synth_stats = self._code_synth_single(api, output_field_name, exemplars=exemplars[:num_exemplars])
            code_name = f"{api.name}_v0"
            code_ensemble[code_name] = code
            return code_ensemble, code_synth_stats

        elif strategy == CodeSynthStrategy.EXAMPLE_ENSEMBLE:
            # creates an ensemble of `code_ensemble_num` synthesized functions; each of
            # which uses a different exemplar (modulo the # of exemplars) for its synthesis
            code_synth_stats = self._create_empty_query_stats()
            for i in range(code_ensemble_num):
                code_name = f"{api.name}_v{i}"
                exemplar = exemplars[i % len(exemplars)]
                code, stats = self._code_synth_single(api, output_field_name, exemplars=[exemplar])
                code_ensemble[code_name] = code
                for key in code_synth_stats.keys():
                    code_synth_stats[key] += stats[key]
            return code_ensemble, code_synth_stats

        elif strategy == CodeSynthStrategy.ADVICE_ENSEMBLE:
            # a more advanced approach in which advice is first solicited, and then
            # provided as context when synthesizing the code ensemble
            code_synth_stats = self._create_empty_query_stats()

            # solicit advice
            advices, adv_stats = self._synthesize_advice(api, output_field_name, exemplars=exemplars[:num_exemplars], n_advices=code_ensemble_num)
            for key in code_synth_stats.keys():
                code_synth_stats[key] += adv_stats[key]

            # synthesize code ensemble
            for i, adv in enumerate(advices):
                code_name = f"{api.name}_v{i}"
                code, stats = self._code_synth_single(api, output_field_name, exemplars=exemplars[:num_exemplars], advice=adv)
                code_ensemble[code_name] = code
                for key in code_synth_stats.keys():
                    code_synth_stats[key] += stats[key]
            return code_ensemble, code_synth_stats

        else:
            raise Exception("not implemented yet")


    def NEW_synthesize_code_ensemble(self,
                                     generate_field_names: List[str],
                                     candidate_dict: DataRecordDict,):
        # initialize stats to be collected for each field's code sythesis
        total_code_synth_stats = self._create_empty_query_stats()

        # synthesize the per-field code ensembles
        field_to_code_ensemble = {}
        for field_name in generate_field_names:
            # create api instance
            api = API.from_input_output_schemas(
                inputSchema=self.inputSchema,
                outputSchema=self.outputSchema,
                field_name=field_name,
                input_fields=candidate_dict.keys()
            )

            # synthesize the code ensemble
            code_ensemble, code_synth_stats = self._synthesize_code_ensemble(api, field_name)

            # update stats
            for key in total_code_synth_stats.keys():
                total_code_synth_stats[key] += code_synth_stats[key]

            # add synthesized code ensemble to field_to_code_ensemble
            field_to_code_ensemble[field_name] = code_ensemble

            # add code ensemble to the cache
            if not self.no_cache_across_plans:
                cache = DataDirectory().getCacheService()
                code_ensemble_cache_id = "_".join([self.get_op_id(), field_name])
                cache.putCachedData("codeEnsembles", code_ensemble_cache_id, code_ensemble)

            # TODO: if verbose
            for code_name, code in code_ensemble.items():
                print(f"CODE NAME: {code_name}")
                print("-----------------------")
                print(code)

        # set field_to_code_ensemble and code_synthesized to True
        self.field_to_code_ensemble = field_to_code_ensemble
        self.code_synthesized = True

        return field_to_code_ensemble, total_code_synth_stats


    def convert(self, candidate_content,
                fields) -> None:
        pass

    def __call__(self,candidate):
        "This code is used for codegen with a fallback to default"
        start_time = time.time()
        fields_to_generate = self._generate_field_names(candidate, self.inputSchema, self.outputSchema)

        # convert the data record to a dictionary of field --> value
        # NOTE: the following is how we used to compute the candidate_dict;
        #       now that I am disallowing code synthesis for one-to-many queries,
        #       I don't think we need to invoke the _asJSONStr() method, which
        #       helped format the tabular data in the "rows" column for Medical Schema Matching.
        #       In the longer term, we should come up with a proper solution to make _asDict()
        #       properly format data which relies on the schema's _asJSONStr method.
        #
        #   candidate_dict_str = candidate._asJSONStr(include_bytes=False, include_data_cols=False)
        #   candidate_dict = json.loads(candidate_dict_str)
        #   candidate_dict = {k: v for k, v in candidate_dict.items() if v != "<bytes>"}
        candidate_dict = candidate._asDict(include_bytes=False)

        # Check if code was already synthesized, or if we have at least one converted sample
        if self._shouldSynthesize():
            field_to_code_ensemble, total_code_synth_stats = self.NEW_synthesize_code_ensemble(fields_to_generate, candidate_dict)
        else:
            # read the dictionary of ensembles already synthesized by this operator if present
            if self.field_to_code_ensemble is not None:
                field_to_code_ensemble, total_code_synth_stats = self.field_to_code_ensemble, {}
            else:
                field_to_code_ensemble, total_code_synth_stats = self._fetch_code_ensemble(fields_to_generate)

        # if we have yet to synthesize code (perhaps b/c we are waiting for more exemplars),
        # use GPT-4 to perform the convert (and generate high-quality exemplars) using a bonded query
        if not len(field_to_code_ensemble):
            text_content = json.loads(candidate_dict)
            final_json_objects, query_stats = self._dspy_generate_fields(
                fields_to_generate,
                text_content=text_content,
                model=Model.GPT_4,  # TODO: assert GPT-4 is available; maybe fall back to another model otherwise
                prompt_strategy=PromptStrategy.DSPY_COT_QA,
            )

            drs = []
            for idx, js in enumerate(final_json_objects):
                # create output data record
                dr = self._create_data_record_from_json(
                    jsonObj=js,
                    candidate=candidate,
                    cardinality_idx=idx
                )
                drs.append(dr)

            # construct the set of output data records and record_op_stats
            query_stats_lst = self._extract_stats(records=drs, start_time=start_time, fields=fields_to_generate, query_stats=query_stats)

            # compute the record_op_stats for each data record and return
            record_op_stats_lst = self._create_record_op_stats_lst(drs, query_stats_lst)

            # NOTE: this now includes bytes input fields which will show up as: `field_name = "<bytes>"`;
            #       keep an eye out for a regression in code synth performance and revert if necessary
            # update operator's set of exemplars
            exemplars = [dr._asDict(include_bytes=False) for dr in drs] # TODO: need to extend candidate to same length and zip
            self.exemplars.extend(exemplars)

            # if we are allowed to cache exemplars across plan executions, add exemplars to cache
            if not self.no_cache_across_plans:
                cache = DataDirectory().getCacheService()
                exemplars_cache_id = self.get_op_id()
                cache.putCachedData(f"codeExemplars", exemplars_cache_id, exemplars)

            return drs, record_op_stats_lst


        # add total_code_synth_stats to query_stats
        query_stats = self._create_empty_query_stats()
        for key in total_code_synth_stats:
            query_stats[key] += total_code_synth_stats

        # if we have synthesized code run it on each field
        field_outputs = {}
        for field_name in fields_to_generate:
            # create api instance for executing python code
            api = API.from_input_output_schemas(
                inputSchema=self.inputSchema,
                outputSchema=self.outputSchema,
                field_name=field_name,
                input_fields=candidate_dict.keys()
            )
            code_ensemble = field_to_code_ensemble[field_name]
            answer, exec_stats = codeEnsembleExecution(api, code_ensemble, candidate_dict)

            if answer is not None:
                field_outputs[field_name] = answer
                for key, value in exec_stats.items():
                    query_stats[key] += value
            else:
                # if there is a failure, run a conventional query
                print(f"CODEGEN FALLING BACK TO CONVENTIONAL FOR FIELD {field_name}")
                text_content = json.loads(candidate_dict)
                final_json_objects, field_stats = self._dspy_generate_fields(
                    [field_name],
                    text_content=text_content,
                    model=Model.GPT_3_5,
                    prompt_strategy=PromptStrategy.DSPY_COT_QA,
                )

                # include code execution time in field_stats
                if "fn_call_duration_secs" not in field_stats:
                    field_stats["fn_call_duration_secs"] = 0.0
                field_stats["fn_call_duration_secs"] += exec_stats["fn_call_duration_secs"]

                # update query_stats
                for key, value in field_stats.items():
                    query_stats[key] += value

                # NOTE: we disallow code synth for one-to-many queries, so there will only be
                #       one element in final_json_objects
                # update field_outputs
                field_outputs[field_name] = final_json_objects[0][field_name]

        # construct the set of output data records and record_op_stats
        drs, query_stats_lst = self._extract_data_records_and_stats(candidate, start_time, fields_to_generate, [field_outputs], query_stats)

        # compute the record_op_stats for each data record and return
        record_op_stats_lst = self._create_record_op_stats_lst(drs, query_stats_lst)

        return drs, record_op_stats_lst

    def _parse_ideas(self, text, limit=3):
        return self._parse_multiple_outputs(text, outputs=[f'Idea {i}' for i in range(1, limit+1)])


    def _generate_advice(self, prompt):
        pred, stats = self.gpt4_llm.generate(prompt=prompt)
        advs = self._parse_ideas(pred)
        return advs, stats

    def _synthesize_code(self, prompt, language='Python'):
        pred, stats = self.gpt4_llm.generate(prompt=prompt)
        ordered_keys = [
            f'```{language}',
            f'```{language.lower()}',
            f'```'
        ]
        code = None
        for key in ordered_keys:
            if key in pred:
                code = pred.split(key)[1].split('```')[0].strip()
                break
        return code, stats

    def _code_synth_default(self, api):
        # returns a function with the correct signature which simply returns None
        code = api.api_def() + "  return None\n"
        stats = self._create_empty_query_stats()
        return code, stats


    def _code_synth_single(self, api: API, output_field_name: str, exemplars: List[Exemplar]=list(), advice: str=None, language='Python'):
        context = {
            'language': language,
            'api': api.args_call(),
            'output': api.output,
            'inputs_desc': "\n".join([f"- {field_name} ({api.input_descs[i]})" for i, field_name in enumerate(api.inputs)]),
            'output_desc': api.output_desc,
            'examples_desc': "\n".join([
                EXAMPLE_PROMPT.format(
                    idx = f" {i}",
                    example_inputs = "\n".join([f"- {field_name} = {repr(example[0][field_name])}" for field_name in api.inputs]),
                    example_output = f"{example[1][output_field_name]}"
                ) for i, example in enumerate(exemplars)
            ]),
            'advice': f"Hint: {advice}" if advice else "",
        }
        prompt = CODEGEN_PROMPT.format(**context)
        print("PROMPT")
        print("-------")
        print(f"{prompt}")
        code, stats = self._synthesize_code(prompt, language=language)
        print("-------")
        print("SYNTHESIZED CODE")
        print("---------------")
        print(f"{code}")

        return code, stats


    def _synthesize_advice(self, api: API, output_field_name: str, exemplars: List[Dict[DataRecord, DataRecord]]=list(), language='Python', n_advices=4):
        context = {
            'language': language,
            'api': api.args_call(),
            'output': api.output,
            'inputs_desc': "\n".join([f"- {field_name} ({api.input_descs[i]})" for i, field_name in enumerate(api.inputs)]),
            'output_desc': api.output_desc,
            'examples_desc': "\n".join([
                EXAMPLE_PROMPT.format(
                    idx = f" {i}",
                    example_inputs = "\n".join([f"- {field_name} = {repr(example[0][field_name])}" for field_name in api.inputs]),
                    example_output = f"{example[1][output_field_name]}"
                ) for i, example in enumerate(exemplars)
            ]),
            'n': n_advices,
        }
        prompt = ADVICEGEN_PROMPT.format(**context)
        advs, stats = self._generate_advice(prompt)
        return advs, stats

    def _parse_multiple_outputs(self, text, outputs=['Thought', 'Action']):
        data = {}
        for key in reversed(outputs):
            if key+':' in text:
                remain, value = text.rsplit(key+':', 1)
                data[key.lower()] = value.strip()
                text = remain
            else:
                data[key.lower()] = None
        return data




class CodeSynthesisConvertStrategy(PhysicalOpStrategy):
    """
    This strategy creates physical operator classes that convert records to one schema to another using code synthesis.

    """

    logical_op_class = logical.ConvertScan
    physical_op_class = LLMConvertCodeSynthesis

    @staticmethod
    def __new__(cls, 
                available_models: List[Model],
                prompt_strategy: PromptStrategy,
                *args, **kwargs) -> List[physical.PhysicalOperator]:

        return_operators = []
        for model in available_models:
            # TODO restrict to only GPT 4 ? 
            if model.value not in MODEL_CARDS:
                raise ValueError(f"Model {model} not found in MODEL_CARDS")
            # physical_op_type = type('LLMConvertCodeSynthesis'+model.name,
            physical_op_type = type('LLMConvertCodeSynthesis',
                                    (cls.physical_op_class,),
                                    {'model': model,
                                     'prompt_strategy': prompt_strategy,
                                     })
            return_operators.append(physical_op_type)

        return return_operators
    