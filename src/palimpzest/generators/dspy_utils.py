from palimpzest.constants import log_attempt_number, RETRY_MAX_ATTEMPTS, RETRY_MAX_SECS, RETRY_MULTIPLIER
from dsp.modules.hf import HFModel
from tenacity import retry, stop_after_attempt, wait_exponential

import dspy
import requests

### DSPy Signatures ###
# Given a questionn, we'll feed it with the paper context for answer generation.
class FilterOverPaper(dspy.Signature):
    """Answer condition questions about a scientific paper."""

    context = dspy.InputField(desc="contains full text of the paper, including author, institution, title, and body")
    question = dspy.InputField(desc="one or more conditions about the paper")
    answer = dspy.OutputField(desc="often a TRUE/FALSE answer to the condition question(s) about the paper")

class QuestionOverPaper(dspy.Signature):
    """Answer question(s) about a scientific paper."""

    context = dspy.InputField(desc="contains full text of the paper, including author, institution, title, and body")
    question = dspy.InputField(desc="one or more question about the paper")
    answer = dspy.OutputField(desc="print the answer only, separated by a newline character")

# functions which generate signatures
def gen_signature_class(instruction, context_desc, question_desc, answer_desc):
    class QuestionOverDoc(dspy.Signature):
        __doc__ = instruction
        context = dspy.InputField(desc= context_desc)
        question = dspy.InputField(desc= question_desc)
        answer = dspy.OutputField(desc= answer_desc)
    return QuestionOverDoc

def gen_filter_signature_class(doc_schema, doc_type):
    instruction = f"Answer condition questions about a {doc_schema}."
    context_desc = f"contains full text of the {doc_type}"
    question_desc = f"one or more conditions about the {doc_type}"
    answer_desc = f"often a TRUE/FALSE answer to the condition question(s) about the {doc_type}"
    return gen_signature_class(instruction, context_desc, question_desc, answer_desc)

def gen_qa_signature_class(doc_schema, doc_type):
    instruction = f"Answer question(s) about a {doc_schema}."
    context_desc = f"contains full text of the {doc_type}"
    question_desc = f"one or more question about the {doc_type}"
    answer_desc = f"print the answer only, separated by a newline character"
    return gen_signature_class(instruction, context_desc, question_desc, answer_desc)


### DSPy Modules ###
class dspyCOT(dspy.Module):
    """
    Invoke dspy in chain of thought mode
    """
    def __init__(self, f_signature=FilterOverPaper):
        super().__init__()
        self.generate_answer = dspy.ChainOfThought(f_signature)

    def forward(self, question, context):
        context = context
        answer = self.generate_answer(context=context, question=question)
        return answer


### DSPy wrapped LLM calls ###
class TogetherHFAdaptor(HFModel):
    def __init__(self, model, apiKey, **kwargs):
        super().__init__(model=model, is_client=True)
        self.api_base = "https://api.together.xyz/inference"
        self.token = apiKey
        self.model = model

        self.use_inst_template = False
        if any(keyword in self.model.lower() for keyword in ["inst", "instruct"]):
            self.use_inst_template = True

        stop_default = "\n\n---"

        # print("Stop procedure", stop_default)
        self.kwargs = {
            "temperature": 0.0,
            "max_tokens": 256, # 8192
            "top_p": 1,
            "top_k": 20,
            "repetition_penalty": 1,
            "n": 1,
            # "stop": stop_default if "stop" not in kwargs else kwargs["stop"],
            **kwargs
        }

    @retry(
        wait=wait_exponential(multiplier=RETRY_MULTIPLIER, max=RETRY_MAX_SECS),
        stop=stop_after_attempt(RETRY_MAX_ATTEMPTS),
        after=log_attempt_number,
    )
    def _generate(self, prompt, use_chat_api=False, **kwargs):
        url = f"{self.api_base}"

        kwargs = {**self.kwargs, **kwargs}
        stop = kwargs.get("stop")
        temperature = kwargs.get("temperature")
        max_tokens = kwargs.get("max_tokens", 150)
        top_p = kwargs.get("top_p", 0.7)
        top_k = kwargs.get("top_k", 50)
        repetition_penalty = kwargs.get("repetition_penalty", 1)
        logprobs = kwargs.get("logprobs", 0)
        prompt = f"[INST]{prompt}[/INST]" if self.use_inst_template else prompt

        if use_chat_api:
            messages = [
                {"role": "system", "content": "You are a helpful assistant. You must continue the user text directly without *any* additional interjections."},
                {"role": "user", "content": prompt}
            ]
            body = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "top_p": top_p,
                "top_k": top_k,
                "repetition_penalty": repetition_penalty,
                "stop": stop,
                "logprobs": logprobs,
            }
        else:
            body = {
                "model": self.model,
                "prompt": prompt,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "top_p": top_p,
                "top_k": top_k,
                "repetition_penalty": repetition_penalty,
                "stop": stop,
                "logprobs": logprobs,
            }

        headers = {"Authorization": f"Bearer {self.token}"}
        try:
            with requests.Session().post(url, headers=headers, json=body) as resp:
                resp_json = resp.json()
                if use_chat_api:
                    completions = [resp_json['output'].get('choices', [])[0].get('message', {}).get('content', "")]
                else:
                    completions = [resp_json['output'].get('choices', [])[0].get('text', "")]
                response = {
                    "prompt": resp_json['prompt'][-1],
                    "choices": [{"text": c} for c in completions],
                }
                response['usage'] = resp_json['output']['usage']
                response['finish_reason'] = resp_json['output']['finish_reason']
                if logprobs > 0:
                    response['tokens'] = resp_json['output']['choices'][0]['tokens']
                    response['token_logprobs'] = resp_json['output']['choices'][0]['token_logprobs']

                return response
        except Exception as e:
            if resp_json:
                print(f"resp_json:{resp_json}")
            print(f"Failed to parse JSON response: {e}")
            raise Exception("Received invalid JSON response from server")
