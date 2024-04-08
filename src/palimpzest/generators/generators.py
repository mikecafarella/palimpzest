from palimpzest.constants import *
from palimpzest.generators import dspyCOT, gen_filter_signature_class, gen_qa_signature_class, TogetherHFAdaptor
from palimpzest.profiler import GenerationStats

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

    def _get_model(self) -> dsp.LM:
        model = None
        if self.model_name in [Model.GPT_3_5.value, Model.GPT_4.value]:
            openai_key = get_api_key('OPENAI_API_KEY')
            max_tokens = 4096 if self.prompt_strategy == PromptStrategy.DSPY_COT_QA else 150
            model = dspy.OpenAI(model=self.model_name, api_key=openai_key, temperature=0.0, max_tokens=max_tokens, logprobs=True)

        elif self.model_name in [Model.MIXTRAL.value]:
            together_key = get_api_key('TOGETHER_API_KEY')
            model = TogetherHFAdaptor(self.model_name, together_key, logprobs=1)

        elif self.model_name in [Model.GEMINI_1.value]:
            google_key = get_api_key('GOOGLE_API_KEY')
            model = dspy.Google(model=self.model_name, api_key=google_key, return_dict=True)

        else:
            raise ValueError("Model must be one of the language models specified in palimpzest.constants.Model")

        return model

    def _get_attn(dspy_lm: dsp.LM):
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
        if self.model_name in [Model.GPT_3_5.value, Model.GPT_4.value, Model.GEMINI_1.value]:
            # [{'token': 'some', 'bytes': [12, 34, ...], 'logprob': -0.7198808, 'top_logprobs': []}}]
            log_probs = dspy_lm.history[-1]['response']['choices'][-1]['logprobs']['content']
            tokens = list(map(lambda elt: elt['token'], log_probs))
            token_logprobs = list(map(lambda elt: elt['logprob'], log_probs))
        elif self.model_name in [Model.MIXTRAL.value]:
            # reponse: dict_keys(['prompt', 'choices', 'usage', 'finish_reason', 'tokens', 'token_logprobs'])
            tokens = dspy_lm.history[-1]['response']['tokens']
            token_logprobs = dspy_lm.history[-1]['response']['token_logprobs']
        else:
            raise ValueError("Model must be one of the language models specified in palimpzest.constants.Model")

        # get indices of the start and end token for the answer
        start_idx, end_idx = 0, 0
        while not answer.strip() == "".join(tokens[start_idx:end_idx+1]).strip():
            if answer.startswith(tokens[start_idx]):
                end_idx += 1
            else:
                start_idx += 1
                end_idx = start_idx

        # filter for log probs of tokens which appear in answer
        answer_log_probs = token_logprobs[start_idx:end_idx+1]

        # return those tokens log probabilities
        return answer_log_probs

    @retry(
        wait=wait_exponential(multiplier=RETRY_MULTIPLIER, max=RETRY_MAX_SECS),
        stop=stop_after_attempt(RETRY_MAX_ATTEMPTS),
        after=log_attempt_number,
    )
    def generate(self, context: str, question: str) -> GenerationOutput:
        # fetch model
        dspy_lm = self._get_model()

        # configure DSPy to use this model; both DSPy prompt strategies currently use COT
        dspy.settings.configure(lm=dspy_lm)
        cot = dspyCOT(self.promptSignature)

        # execute LLM generation
        start_time = time.time()
        pred = cot(question, context)
        end_time = time.time()

        # extract the log probabilities for the actual result(s) which are returned
        answer_log_probs = self._get_answer_log_probs(dspy_lm, pred.answer)

        # collect statistics on prompt, usage, and timing
        stats = GenerationStats(
            model_name=self.model_name,
            llm_call_duration_secs=end_time - start_time,
            prompt=dspy_lm.history[-1]['prompt'],
            usage=dspy_lm.history[-1]['response']['usage'],
            finish_reason=(
                dspy_lm.history[-1]['response']['finish_reason']
                if isinstance(dspy_lm, TogetherHFAdaptor)
                else dspy_lm.history[-1]['response']['choices'][-1]['finish_reason']
            ),
            answer_log_probs=answer_log_probs,
            answer=pred.answer,
        )

        if self.verbose:
            print("Prompt history:")
            dspy_lm.inspect_history(n=1)

        return pred.answer, stats


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

    def _make_payload(self, prompt: str, base64_image: str):
        payload = None
        if self.model_name == Model.GPT_4V.value:
            payload = {
                "model": "gpt-4-vision-preview",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt,
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                },
                            }
                        ]
                    }
                ],
                "max_tokens": 4000,
                "logprobs": True,
            }

        elif self.model_name == Model.GEMINI_1V.value:
            payload = [prompt, Image.open(io.BytesIO(self._decode_image(base64_image)))]

        else:
            raise ValueError(f"Model must be one of the image models specified in palimpzest.constants.Model")

        return payload

    def _generate_response(self, client: Union[OpenAI, genai.GenerativeModel], payload: Any) -> Tuple[str, str, dict]:
        answer, finish_reason, usage = None, None, None

        if self.model_name == Model.GPT_4V.value:
            completion = client.chat.completions.create(**payload)
            candidate = completion.choices[-1]
            answer = candidate.message.content
            finish_reason = candidate.finish_reason
            usage = dict(completion.usage)
            tokens = list(map(lambda elt: elt.token, completion.choices[-1].logprobs.content))
            token_logprobs = list(map(lambda elt: elt.logprob, completion.choices[-1].logprobs.content))

        elif self.model_name == Model.GEMINI_1V.value:
            response = client.generate_content(payload)
            candidate = response.candidates[-1]
            answer = candidate.content.parts[0].text
            finish_reason = candidate.finish_reason
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
        start_idx, end_idx = 0, 0
        while not answer.strip() == "".join(tokens[start_idx:end_idx+1]).strip():
            if answer.startswith(tokens[start_idx]):
                end_idx += 1
            else:
                start_idx += 1
                end_idx = start_idx

        # filter for log probs of tokens which appear in answer
        answer_log_probs = token_logprobs[start_idx:end_idx+1]

        # return those tokens log probabilities
        return answer_log_probs

    @retry(
        wait=wait_exponential(multiplier=RETRY_MULTIPLIER, max=RETRY_MAX_SECS),
        stop=stop_after_attempt(RETRY_MAX_ATTEMPTS),
        after=log_attempt_number,
    )
    def generate(self, image_b64: str, prompt: str) -> GenerationOutput:
        # fetch model client
        client = self._get_model_client()

        # create payload
        payload = self._make_payload(prompt, image_b64)

        # generate response
        start_time = time.time()
        answer, finish_reason, usage, tokens, token_logprobs = self._generate_response(client, payload)
        end_time = time.time()

        # extract the log probabilities for the actual result(s) which are returned
        answer_log_probs = self._get_answer_log_probs(tokens, token_logprobs, answer)

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
