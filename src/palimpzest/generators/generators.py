import hashlib
import json

from palimpzest.constants import *
from palimpzest.generators import dspyCOT, gen_filter_signature_class, gen_qa_signature_class, TogetherHFAdaptor
from palimpzest.profiler import GenerationStats
from palimpzest.datamanager import DataDirectory

from openai import OpenAI
from PIL import Image
from tenacity import retry, stop_after_attempt, wait_exponential
from typing import Any, List, Tuple, Union

import google.generativeai as genai

import base64
import dsp
import dspy
import io
import os
import time

from palimpzest.profiler.attentive_trim import find_best_range, get_trimed, best_substring_match, update_heatmap_json

# DEFINITIONS
GenerationOutput = Tuple[str, GenerationStats]

def get_api_key(key: str) -> str:
    # get API key from environment or throw an exception if it's not set
    if key not in os.environ:
        raise ValueError(f"key not found in environment variables")

    return os.environ[key]


class BaseGenerator:
    """
    Abstract base class for Generators.
    """
    def __init__(self):
        pass

    def generate(self) -> GenerationOutput:
        raise NotImplementedError("Abstract method")


class CustomGenerator(BaseGenerator):
    """
    Class for generating outputs with a given model using a custom prompt.
    """
    def __init__(self, model_name: str, verbose: bool=False):
        super().__init__()
        self.model_name = model_name
        self.verbose = verbose

    def _get_model(self) -> dsp.LM:
        model = None
        if self.model_name in [Model.GPT_3_5.value, Model.GPT_4.value]:
            openai_key = get_api_key('OPENAI_API_KEY')
            max_tokens = 4096
            model = dspy.OpenAI(model=self.model_name, api_key=openai_key, temperature=0.0, max_tokens=max_tokens, logprobs=True)

        elif self.model_name in [Model.MIXTRAL.value, Model.LLAMA2.value]:
            together_key = get_api_key('TOGETHER_API_KEY')
            model = TogetherHFAdaptor(self.model_name, together_key, logprobs=1)

        elif self.model_name in [Model.GEMINI_1.value]:
            google_key = get_api_key('GOOGLE_API_KEY')
            model = dspy.Google(model=self.model_name, api_key=google_key)

        else:
            raise ValueError("Model must be one of the language models specified in palimpzest.constants.Model")

        return model

    def _get_usage_and_finish_reason(self, dspy_lm: dsp.LM):
        """
        Parse and return the usage statistics and finish reason.
        """
        usage, finish_reason = None, None
        if self.model_name in [Model.GPT_3_5.value, Model.GPT_4.value]:
            usage = dspy_lm.history[-1]['response']['usage']
            finish_reason = dspy_lm.history[-1]['response']['choices'][-1]['finish_reason']
        elif self.model_name in [Model.GEMINI_1.value]:
            usage = {"prompt_tokens": 0, "completion_tokens": 0}
            finish_reason = dspy_lm.history[-1]['response'][0]._result.candidates[0].finish_reason
        elif self.model_name in [Model.MIXTRAL.value, Model.LLAMA2.value]:
            usage = dspy_lm.history[-1]['response']['usage']
            finish_reason = dspy_lm.history[-1]['response']['finish_reason']

        return usage, finish_reason

    def _get_attn(self, dspy_lm: dsp.LM):
        """
        TODO
        """
        pass

    def _get_answer_log_probs(self, dspy_lm: dsp.LM, answer: str) -> List[float]:
        """
        For the given DSPy LM object:
        1. fetch the data structure containing its output log probabilities
        2. filter the data structure for the specific tokens which appear in `answer`
        3. return the list of those tokens' log probabilities
        """
        # get log probabilities data structure
        tokens, token_logprobs = None, None

        if self.model_name in [Model.GPT_3_5.value, Model.GPT_4.value]:
            # [{'token': 'some', 'bytes': [12, 34, ...], 'logprob': -0.7198808, 'top_logprobs': []}}]
            log_probs = dspy_lm.history[-1]['response']['choices'][-1]['logprobs']['content']
            tokens = list(map(lambda elt: elt['token'], log_probs))
            token_logprobs = list(map(lambda elt: elt['logprob'], log_probs))
        elif self.model_name in [Model.GEMINI_1.value]:
            return None
            # TODO Google gemini does not provide log probabilities! 
            # https://github.com/google/generative-ai-python/issues/238
            # tok_count = dspy_lm.llm.count_tokens(answer).total_tokens
            # tokens = [""] * tok_count
            # token_logprobs = [0] * len(tokens)
        elif self.model_name in [Model.MIXTRAL.value, Model.LLAMA2.value]:
            # reponse: dict_keys(['prompt', 'choices', 'usage', 'finish_reason', 'tokens', 'token_logprobs'])
            tokens = dspy_lm.history[-1]['response']['tokens']
            token_logprobs = dspy_lm.history[-1]['response']['token_logprobs']
        else:
            raise ValueError("Model must be one of the language models specified in palimpzest.constants.Model")

        # get indices of the start and end token for the answer
        # start_idx, end_idx = 0, 0
        # while not answer.strip() == "".join(tokens[start_idx:end_idx+1]).strip():
            # if answer.startswith(tokens[start_idx]):
                # end_idx += 1
            # else:
                # start_idx += 1
                # end_idx = start_idx
        # filter for log probs of tokens which appear in answer
        # answer_log_probs = token_logprobs[start_idx:end_idx+1]
        answer_log_probs = token_logprobs
        # return those tokens log probabilities
        return answer_log_probs

    @retry(
        wait=wait_exponential(multiplier=RETRY_MULTIPLIER, max=RETRY_MAX_SECS),
        stop=stop_after_attempt(RETRY_MAX_ATTEMPTS),
        after=log_attempt_number,
        reraise=True,
    )
    def generate(self, prompt: str) -> GenerationOutput:
        # fetch model
        dspy_lm = self._get_model()

        start_time = time.time()
        
        response = dspy_lm.request(prompt)
        
        end_time = time.time()
        
        answer = response['choices'][0]['message']['content']
        answer_log_probs = response['choices'][0]['logprobs']
        finish_reason = response['choices'][0]['finish_reason']
        usage = response['usage']

        # collect statistics on prompt, usage, and timing
        stats = GenerationStats(
            model_name=self.model_name,
            llm_call_duration_secs=end_time - start_time,
            prompt=dspy_lm.history[-1]['prompt'],
            usage=usage,
            finish_reason=finish_reason,
            answer_log_probs=answer_log_probs,
            answer=answer,
        )

        if self.verbose:
            print("Prompt history:")
            dspy_lm.inspect_history(n=1)

        return answer, stats


class DSPyGenerator(BaseGenerator):
    """
    Class for generating outputs with a given model using DSPy for prompting optimization(s).
    """
    def __init__(self, model_name: str, prompt_strategy: PromptStrategy, doc_schema: str, doc_type: str, verbose: bool=False):
        super().__init__()
        self.model_name = model_name
        self.prompt_strategy = prompt_strategy
        self.verbose = verbose

        # set prompt signature based on prompt_strategy
        if prompt_strategy == PromptStrategy.DSPY_COT_BOOL:
            self.promptSignature = gen_filter_signature_class(doc_schema, doc_type)
        elif prompt_strategy == PromptStrategy.DSPY_COT_QA:
            self.promptSignature = gen_qa_signature_class(doc_schema, doc_type)
        else:
            raise ValueError(f"DSPyGenerator does not support prompt_strategy: {prompt_strategy.value}")

    def _get_model(self, plan_idx) -> dsp.LM:
        model = None
        if self.model_name in [Model.GPT_3_5.value, Model.GPT_4.value]:
            openai_key = get_api_key('OPENAI_API_KEY')
            max_tokens = 4096 if self.prompt_strategy == PromptStrategy.DSPY_COT_QA else 150
            model = dspy.OpenAI(model=self.model_name, api_key=openai_key, temperature=0.0, max_tokens=max_tokens, logprobs=True)

        elif self.model_name in [Model.MIXTRAL.value, Model.LLAMA2.value]:
            together_key = get_api_key('TOGETHER_API_KEY')
            model = TogetherHFAdaptor(self.model_name, together_key, logprobs=1)

        elif self.model_name in [Model.GEMINI_1.value]:
            google_key = get_api_key(f'GOOGLE_API_KEY_{plan_idx}')
            model = dspy.Google(model=self.model_name, api_key=google_key)

        else:
            raise ValueError("Model must be one of the language models specified in palimpzest.constants.Model")

        return model

    def _get_usage_and_finish_reason(self, dspy_lm: dsp.LM):
        """
        Parse and return the usage statistics and finish reason.
        """
        usage, finish_reason = None, None
        if self.model_name in [Model.GPT_3_5.value, Model.GPT_4.value]:
            usage = dspy_lm.history[-1]['response']['usage']
            finish_reason = dspy_lm.history[-1]['response']['choices'][-1]['finish_reason']
        elif self.model_name in [Model.GEMINI_1.value]:
            usage = {"prompt_tokens": 0, "completion_tokens": 0}
            finish_reason = dspy_lm.history[-1]['response'][0]._result.candidates[0].finish_reason
        elif self.model_name in [Model.MIXTRAL.value, Model.LLAMA2.value]:
            usage = dspy_lm.history[-1]['response']['usage']
            finish_reason = dspy_lm.history[-1]['response']['finish_reason']

        return usage, finish_reason

    def _get_attn(self, dspy_lm: dsp.LM):
        """
        TODO
        """
        pass

    def _get_answer_log_probs(self, dspy_lm: dsp.LM, answer: str) -> List[float]:
        """
        For the given DSPy LM object:
        1. fetch the data structure containing its output log probabilities
        2. filter the data structure for the specific tokens which appear in `answer`
        3. return the list of those tokens' log probabilities
        """
        # get log probabilities data structure
        tokens, token_logprobs = None, None

        if self.model_name in [Model.GPT_3_5.value, Model.GPT_4.value]:
            # [{'token': 'some', 'bytes': [12, 34, ...], 'logprob': -0.7198808, 'top_logprobs': []}}]
            log_probs = dspy_lm.history[-1]['response']['choices'][-1]['logprobs']['content']
            tokens = list(map(lambda elt: elt['token'], log_probs))
            token_logprobs = list(map(lambda elt: elt['logprob'], log_probs))
        elif self.model_name in [Model.GEMINI_1.value]:
            return None
            # TODO Google gemini does not provide log probabilities! 
            # https://github.com/google/generative-ai-python/issues/238
            # tok_count = dspy_lm.llm.count_tokens(answer).total_tokens
            # tokens = [""] * tok_count
            # token_logprobs = [0] * len(tokens)
        elif self.model_name in [Model.MIXTRAL.value, Model.LLAMA2.value]:
            # reponse: dict_keys(['prompt', 'choices', 'usage', 'finish_reason', 'tokens', 'token_logprobs'])
            tokens = dspy_lm.history[-1]['response']['tokens']
            token_logprobs = dspy_lm.history[-1]['response']['token_logprobs']
        else:
            raise ValueError("Model must be one of the language models specified in palimpzest.constants.Model")

        # get indices of the start and end token for the answer
        # start_idx, end_idx = 0, 0
        # while not answer.strip() == "".join(tokens[start_idx:end_idx+1]).strip():
            # if answer.startswith(tokens[start_idx]):
                # end_idx += 1
            # else:
                # start_idx += 1
                # end_idx = start_idx
        # filter for log probs of tokens which appear in answer
        # answer_log_probs = token_logprobs[start_idx:end_idx+1]
        answer_log_probs = token_logprobs
        # return those tokens log probabilities
        return answer_log_probs

    @retry(
        wait=wait_exponential(multiplier=RETRY_MULTIPLIER, max=RETRY_MAX_SECS),
        stop=stop_after_attempt(RETRY_MAX_ATTEMPTS),
        after=log_attempt_number,
        reraise=True,
    )
    # the generate method requires a user-provided budget parameter to specify te token budget. Default is 1.0, meaning the full context will be used.
    def generate(self, context: str, question: str, budget: float = 1.0, plan_idx: int=0, heatmap_json_obj: dict=None) -> GenerationOutput:
        # initialize variables around token reduction
        reduction, full_context = False, context

        # fetch model
        dspy_lm = self._get_model(plan_idx)

        # configure DSPy to use this model; both DSPy prompt strategies currently use COT
        dspy.settings.configure(lm=dspy_lm)
        cot = dspyCOT(self.promptSignature)

        json_object = {}
        heatmap_file = ''
        # check if the promptSignature is a QA signature, so we can match the answer to get heatmap
        if budget < 1.0 and self.prompt_strategy == PromptStrategy.DSPY_COT_QA:
            # file_cache = DataDirectory().getFileCacheDir()
            prompt_schema = self.promptSignature
            # print("Prompt QA Signature: ", prompt_schema)
            # print('Question: ', question)
            # ordered = f'{prompt_schema} {question} {plan_idx}'
            # task_hash = hashlib.sha256(ordered.encode()).hexdigest()
            # heatmap_file = os.path.join(file_cache, f"heatmap-{task_hash}.json")
            # print("Heatmap file: ", heatmap_file)
            # if not os.path.exists(heatmap_file):
            if heatmap_json_obj is None:
                # create the heatmap structure with default resolution of 0.001 and count of 0
                buckets = int(1.0 / TOKEN_REDUCTION_GRANULARITY)
                hist = [0] * buckets
                heatmap_json_obj = {'prompt_schema': f'{prompt_schema}',
                               'question': question,
                               'resolution': TOKEN_REDUCTION_GRANULARITY,
                               'count': 0,
                               'heatmap': hist}

            else:
                # only parse the heatmap file if token reduction is enabled (budget is less than 1.0)
                # with open(heatmap_file, 'r') as f:
                #     json_object = json.load(f)
                #     heatmap = json_object['heatmap']
                #     count = json_object['count']
                #     print("count:", count)
                heatmap = heatmap_json_obj['heatmap']
                count = heatmap_json_obj['count']
                print("count:", count)
                # only refer to the heatmap if the count is greater than a enough sample size
                # TODO: only trim the context if the attention is clustered in a small region
                if count >= TOKEN_REDUCTION_SAMPLE:
                    si, ei = find_best_range(heatmap, int(budget/TOKEN_REDUCTION_GRANULARITY), trim_zeros=False)
                    sr, er = si * TOKEN_REDUCTION_GRANULARITY, ei * TOKEN_REDUCTION_GRANULARITY
                    print("start ratio:", sr, "end ratio:", er)
                    context = get_trimed(context, sr, er)
                    reduction = True


        # execute LLM generation
        start_time = time.time()

        print(f"Generating -- {self.model_name} -- Token budget: {budget}")
        # print(f"FALL BACK question: {question}")
        # print(f"FALL BACK CONTEXT")
        # print("--------------------")
        # print(f"{context}")
        pred = cot(question, context)

        end_time = time.time()

        # extract the log probabilities for the actual result(s) which are returned
        answer_log_probs = self._get_answer_log_probs(dspy_lm, pred.answer)
        usage, finish_reason = self._get_usage_and_finish_reason(dspy_lm)

        # collect statistics on prompt, usage, and timing
        stats = GenerationStats(
            model_name=self.model_name,
            llm_call_duration_secs=end_time - start_time,
            prompt=dspy_lm.history[-1]['prompt'],
            usage=usage,
            finish_reason=finish_reason,
            answer_log_probs=answer_log_probs,
            answer=pred.answer,
        )

        # if reduction is enabled but the answer is None, fallback to the full context
        if reduction and pred.answer==None:
            # run query on full context 
            pred = cot(question, full_context)

            # NOTE: in the future, we should capture each of these^ calls in two separate
            #       GenerationStats objects, but for now we just aggregate them
            end_time = time.time()

            # extract the log probabilities for the actual result(s) which are returned
            answer_log_probs = self._get_answer_log_probs(dspy_lm, pred.answer)
            usage, finish_reason = self._get_usage_and_finish_reason(dspy_lm)

            stats.llm_call_duration_secs = end_time - start_time
            stats.prompt = dspy_lm.history[-1]['prompt']
            for k, _ in stats.usage.items():
                stats.usage[k] += usage[k]
            stats.finish_reason = finish_reason
            stats.answer_log_probs = answer_log_probs
            stats.answer = pred.answer

        # print("----------------")
        # print(f"PROMPT")
        # print("----------------")
        # print(dspy_lm.history[-1]['prompt'])

        # print("----------------")
        # print(f"ANSWER")
        # print("----------------")
        print(pred.answer)

        # taken reduction post processing if enabled
        if budget < 1.0 and self.prompt_strategy == PromptStrategy.DSPY_COT_QA and heatmap_json_obj['count'] < MAX_HEATMAP_UPDATES:
            print("Reduction enabled")
            print("answer:", pred.answer)
            try:
                gsi, gei = best_substring_match(pred.answer, full_context)
            except Exception as e:
                print("Error in substring match:", e)
                gsi, gei = 0, len(full_context)
            context_len = len(full_context)
            gsr, ger = gsi/context_len, gei/context_len
            norm_si, norm_ei = int(gsr/TOKEN_REDUCTION_GRANULARITY), int(ger/TOKEN_REDUCTION_GRANULARITY)
            print("best_start:", gsi, "best_end:", gei)
            heatmap_json_obj = update_heatmap_json(heatmap_json_obj, norm_si, norm_ei)
            # with open(heatmap_file, 'w') as f:
            #     json.dump(json_object, f)

        if self.verbose:
            print("Prompt history:")
            dspy_lm.inspect_history(n=1)

        return pred.answer, heatmap_json_obj, stats


class ImageTextGenerator(BaseGenerator):
    """
    Class for generating field descriptions for an image with a given image model.
    """
    def __init__(self, model_name: str):
        super().__init__()
        self.model_name = model_name
        
    def _decode_image(self, base64_string: str) -> bytes:
        return base64.b64decode(base64_string)

    def _get_model_client(self) -> Union[OpenAI, genai.GenerativeModel]:
        client = None
        if self.model_name == Model.GPT_4V.value:
            api_key = get_api_key("OPENAI_API_KEY")
            client = OpenAI(api_key=api_key)

        elif self.model_name == Model.GEMINI_1V.value:
            api_key = get_api_key("GOOGLE_API_KEY")
            genai.configure(api_key=api_key)
            client = genai.GenerativeModel('gemini-pro-vision')

        else:
            raise ValueError(f"Model must be one of the image models specified in palimpzest.constants.Model")

        return client

    def _make_payloads(self, prompt: str, base64_images: List[str]):
        payloads = []
        if self.model_name == Model.GPT_4V.value:
            # create content list
            content = [{"type": "text", "text": prompt}]
            for base64_image in base64_images:
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    },
                })

            # create payload
            payloads = [{
                "model": "gpt-4-vision-preview",
                "messages": [
                    {
                        "role": "user",
                        "content": content,
                    }
                ],
                "max_tokens": 4000,
                "temperature": 0.0,
                "logprobs": True,
            }]

        elif self.model_name == Model.GEMINI_1V.value:
            payloads = [
                [prompt, Image.open(io.BytesIO(self._decode_image(base64_image)))]
                for base64_image in base64_images
            ]

        else:
            raise ValueError(f"Model must be one of the image models specified in palimpzest.constants.Model")

        return payloads

    def _generate_response(self, client: Union[OpenAI, genai.GenerativeModel], payloads: List[Any]) -> Tuple[str, str, dict]:
        answer, finish_reason, usage = None, None, None

        if self.model_name == Model.GPT_4V.value:
            # GPT-4V will always have a single payload
            completion = client.chat.completions.create(**payloads[0])
            candidate = completion.choices[-1]
            answer = candidate.message.content
            finish_reason = candidate.finish_reason
            usage = dict(completion.usage)
            tokens = list(map(lambda elt: elt.token, completion.choices[-1].logprobs.content))
            token_logprobs = list(map(lambda elt: elt.logprob, completion.choices[-1].logprobs.content))

        elif self.model_name == Model.GEMINI_1V.value:
            # iterate through images to generate multiple responses
            answers, finish_reasons = [], []
            for idx, payload in enumerate(payloads):
                response = client.generate_content(payload)
                candidate = response.candidates[-1]
                answer = f"Image {idx}: " + candidate.content.parts[0].text
                finish_reason = candidate.finish_reason
                answers.append(answer)
                finish_reasons.append(finish_reason)

            # combine answers and compute most frequent finish reason
            answer = "\n".join(answers)
            finish_reason = max(set(finish_reasons), key=finish_reasons.count)

            # TODO: implement when google suppports usage and logprob stats
            usage = {}
            tokens = []
            token_logprobs = []

        else:
            raise ValueError(f"Model must be one of the image models specified in palimpzest.constants.Model")

        return answer, finish_reason, usage, tokens, token_logprobs

    def _get_answer_log_probs(self, tokens: List[str], token_logprobs: List[float], answer: str) -> List[float]:
        """
        Filter and return the list of log probabilities for the tokens which appear in `answer`.
        """
        # get indices of the start and end token for the answer

        # start_idx, end_idx = 0, 0
        # while not answer.strip() == "".join(tokens[start_idx:end_idx+1]).strip():
            # if answer.startswith(tokens[start_idx]):
                # end_idx += 1
            # else:
                # start_idx += 1
                # end_idx = start_idx

        # filter for log probs of tokens which appear in answer
        # answer_log_probs = token_logprobs[start_idx:end_idx+1]
        answer_log_probs = token_logprobs
        
        # return those tokens log probabilities
        return answer_log_probs

    @retry(
        wait=wait_exponential(multiplier=RETRY_MULTIPLIER, max=RETRY_MAX_SECS),
        stop=stop_after_attempt(RETRY_MAX_ATTEMPTS),
        after=log_attempt_number,
    )
    def generate(self, base64_images: str, prompt: str) -> GenerationOutput:
        # fetch model client
        client = self._get_model_client()

        # create payload
        payloads = self._make_payloads(prompt, base64_images)

        # generate response
        print(f"Generating")
        start_time = time.time()
        answer, finish_reason, usage, tokens, token_logprobs = self._generate_response(client, payloads)
        end_time = time.time()
        print(answer)

        # extract the log probabilities for the actual result(s) which are returned
        answer_log_probs = self._get_answer_log_probs(tokens, token_logprobs, answer)

        # TODO: To simplify life for the time being, I am aggregating stats for multiple call(s)
        #       to the Gemini vision model into a single GenerationStats object (when we have
        #       more than one image to process). This has no effect on most of our fields --
        #       especially since many of them are not implemented for the Gemini model -- but
        #       we will likely want a more robust solution in the future.
        # collect statistics on prompt, usage, and timing
        stats = GenerationStats(
            model_name=self.model_name,
            llm_call_duration_secs=end_time - start_time,
            prompt=prompt,
            usage=usage,
            finish_reason=finish_reason,
            answer_log_probs=answer_log_probs,
            answer=answer,
        )

        return answer, stats
