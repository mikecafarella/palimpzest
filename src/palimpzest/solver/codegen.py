from pyheaven import *
from palimpzest.constants import Model
from palimpzest.solver.sandbox import API
from palimpzest.elements import Schema, Field
from palimpzest.datamanager import DataDirectory
from palimpzest.tools.dspysearch import run_codegen, run_basic_request, exec_codegen
from typing import Any, Dict, Tuple, Union

# TODO: support parallel (edit cache)
def Lock():
    lock_dir = pjoin(DataDirectory()._dir, "data", "cache", ".codegen_lock"); CreateFile(lock_dir)

def Unlock():
    lock_dir = pjoin(DataDirectory()._dir, "data", "cache", ".codegen_lock"); Delete(lock_dir)

def IsLocked():
    lock_dir = pjoin(DataDirectory()._dir, "data", "cache", ".codegen_lock"); return ExistFile(lock_dir)

CODEGEN_PROMPT = """You are a helpful programming assistant and an expert Python programmer. Implement the Python function `{api}` that extracts `{output}` ({output_desc}) from given inputs:
{inputs_desc}
Example:
{example_inputs}
{example_output}
Notice that the evaluation will severely punish incorrect outputs. Thus, when the function is uncertain, it should return `None` to abstain instead of returning an incorrect guess.
Return the implementation only."""

NONE_CODE = """
def {api}(contents, filename):
    return None
"""

def llmCodeGen(prompt, model, default=NONE_CODE.format(api='extract')):
    try:
        code, stats = run_codegen(prompt, model_name=model.value); return code, stats
    except Exception as e:
        return default, {}


ADVICEGEN_PROMPT = """You are a helpful programming assistant and an expert Python programmer. Your job is to provide programming ideas to help me write Python programs.
For example, if I want to complete a task: "extract the salary number (in USD) from a given employee's document", you can provide me with {n} different ways to do it like:
Idea 1: Use regular expressions to extract the salary number: a number with a dollar sign in front of it. For example, $100,000.
Idea 2: Find the table entry with the salary number.
Idea 3: Use a pre-trained NLP model to extract the salary number.

Now, consider the following Python programming task that extracts `{output}` ({output_desc}) from given inputs:
{inputs_desc}
Example:
{example_inputs}
{example_output}
Please provide me with {n} different ideas to complete this task. Return the ideas only, following the format above.
"""

ADVICED_CODEGEN_PROMPT = """You are a helpful programming assistant and an expert Python programmer. Implement the Python function `{api}` that extracts `{output}` ({output_desc}) from given inputs:
{inputs_desc}
Example:
{example_inputs}
{example_output}
Hint: {advice}
Notice that the evaluation will severely punish incorrect outputs. Thus, when the function is uncertain, it should return `None` to abstain instead of returning an incorrect guess.
Return the implementation only."""

def parse_multiple_outputs(text, outputs=['Thought', 'Action']):
    data = {}
    for key in reversed(outputs):
        if key+':' in text:
            remain, value = text.rsplit(key+':', 1)
            data[key.lower()] = value.strip()
            text = remain
        else:
            data[key.lower()] = None
    return data

def parse_ideas(text, limit=3):
    return parse_multiple_outputs(text, outputs=[f'Idea {i}' for i in range(1, limit+1)])

def llmAdviceGen(prompt, model, default=list()):
    try:
        response, stats = run_basic_request(prompt, model_name=model.value)
        advices = list(parse_ideas(response).values()); return advices, stats
    except Exception as e:
        return default, {}

def codeGen(inputSchema: Schema, outputFieldDict: Dict[str, Field], conversionDesc: str, config: Dict[str, Any], model: Model, api: API, example_inputs: Dict[str, Any]=None):
    outputName, outputField = list(outputFieldDict.items())[0]
    codegen_stats = {}

    # LLM code generation
    codeGenPrompt = CODEGEN_PROMPT.format(
        api = api.args_call(),
        output = outputName,
        output_desc = f"{outputField.desc}. {conversionDesc}" if conversionDesc else outputField.desc,
        inputs_desc = "\n".join([f"- {k}: {getattr(inputSchema,k).desc}." for k in inputSchema.fieldNames()]),
        example_inputs = "\n".join([f"- {k} = {repr(example_inputs[k])}" for k in inputSchema.fieldNames()]) if example_inputs else "",
        example_output = ""
    )
    code, stats = llmCodeGen(codeGenPrompt, model, default=NONE_CODE.format(api=api.name))
    stats["code"] = code
    codegen_stats["code_generation"] = stats

    # more complex code generation process
    num_ensemble        =  config.get('codegen_num_ensemble',       default=    4)
    validation          =  config.get('codegen_validation',         default=False)
    num_iterations      =  config.get('codegen_num_iterations',     default=    5)
    num_max_examples    =  config.get('codegen_num_max_examples',   default=   20)
    num_advices = max(num_ensemble-1, 0)
    codes = [code]
    if num_advices > 0:
        # Advised Code Generation
        adviceGenPrompt = ADVICEGEN_PROMPT.format(
            api = api.args_call(),
            output = outputName,
            output_desc = f"{outputField.desc}. {conversionDesc}" if conversionDesc else outputField.desc,
            inputs_desc = "\n".join([f"- {k}: {getattr(inputSchema,k).desc}." for k in inputSchema.fieldNames()]),
            example_inputs = "\n".join([f"- {k} = {repr(example_inputs[k])}" for k in inputSchema.fieldNames()]) if example_inputs else "",
            example_output = "",
            n = num_advices,
        )
        advices, stats = llmAdviceGen(adviceGenPrompt, model)
        stats["advices"] = advices
        codegen_stats["advice_generation"] = stats
        for idx, advice in enumerate(advices):
            advisedCodeGenPrompt = ADVICED_CODEGEN_PROMPT.format(
                api = api.args_call(),
                output = outputName,
                output_desc = f"{outputField.desc}. {conversionDesc}" if conversionDesc else outputField.desc,
                inputs_desc = "\n".join([f"- {k}: {getattr(inputSchema,k).desc}." for k in inputSchema.fieldNames()]),
                example_inputs = "\n".join([f"- {k} = {repr(example_inputs[k])}" for k in inputSchema.fieldNames()]) if example_inputs else "",
                example_output = "",
                advice = advice,
            )
            advised_code, stats = llmCodeGen(advisedCodeGenPrompt, model, default=NONE_CODE.format(api=api.name))
            codes.append(advised_code)
            stats["advice"] = advice
            stats["advised_code"] = advised_code
            codegen_stats[f"advised_code_generation_{idx}"] = stats
        # Validation & Iteration (there is currently not a lot of validation examples, and the reGenerate is always False, so this may never be used)
        # if validation and num_iterations > 0:
        #     validation_examples = [(example_inputs, example_output) for example_inputs, example_output in validation_examples if example_output is not None]
        #     validation_examples = validation_examples[:num_max_examples]
        #     profiles = [[(exec_codegen(api, code, example_inputs), example_output) for exp in validation_examples] for code in codes]
        #     for i in range(num_iterations):
        #         # TODO: need validation examples, probably from a forced fallback ratio to LLM (e.g., 1% ?)
        #         pass

    return codes, codegen_stats

def getConversionCodes(inputSchema: Schema, outputFieldDict: Dict[str, Field], conversionDesc: str, config: Dict[str, Any], model: Model, api: API, example_inputs: Dict[str, Any]=None):
    outputName, outputField = list(outputFieldDict.items())[0]
    hash_key = str((conversionDesc, inputSchema, outputName, outputField).__hash__()).replace('-','_')
    codedir = pjoin(DataDirectory()._dir, "data", "cache", "codegen", f"{hash_key}.json")
    if ExistFile(codedir):
        return LoadJson(codedir)
    codes, codegen_stats = codeGen(inputSchema, outputFieldDict, conversionDesc, config, model, api, example_inputs=example_inputs)
    SaveJson(codes, codedir)
    return codes, codegen_stats
