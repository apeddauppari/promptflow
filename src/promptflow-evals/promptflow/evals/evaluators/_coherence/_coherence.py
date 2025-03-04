# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

import os
import re

import numpy as np

from promptflow._utils.async_utils import async_run_allowing_running_loop
from promptflow.core import AsyncPrompty, AzureOpenAIModelConfiguration

try:
    from ..._user_agent import USER_AGENT
except ImportError:
    USER_AGENT = None


class _AsyncCoherenceEvaluator:
    def __init__(self, model_config: AzureOpenAIModelConfiguration):
        if model_config.api_version is None:
            model_config.api_version = "2024-02-15-preview"

        prompty_model_config = {"configuration": model_config}
        if USER_AGENT and isinstance(model_config, AzureOpenAIModelConfiguration):
            prompty_model_config.update({"parameters": {"extra_headers": {"x-ms-useragent": USER_AGENT}}})
        current_dir = os.path.dirname(__file__)
        prompty_path = os.path.join(current_dir, "coherence.prompty")
        self._flow = AsyncPrompty.load(source=prompty_path, model=prompty_model_config)

    async def __call__(self, *, question: str, answer: str, **kwargs):
        # Validate input parameters
        question = str(question or "")
        answer = str(answer or "")

        if not (question.strip() and answer.strip()):
            raise ValueError("Both 'question' and 'answer' must be non-empty strings.")

        # Run the evaluation flow
        llm_output = await self._flow(question=question, answer=answer)

        score = np.nan
        if llm_output:
            match = re.search(r"\d", llm_output)
            if match:
                score = float(match.group())

        return {"gpt_coherence": float(score)}


class CoherenceEvaluator:
    """
    Initialize a coherence evaluator configured for a specific Azure OpenAI model.

    :param model_config: Configuration for the Azure OpenAI model.
    :type model_config: AzureOpenAIModelConfiguration

    **Usage**

    .. code-block:: python

        eval_fn = CoherenceEvaluator(model_config)
        result = eval_fn(
            question="What is the capital of Japan?",
            answer="The capital of Japan is Tokyo.")

    **Output format**

    .. code-block:: python

        {
            "gpt_coherence": 1.0
        }
    """

    def __init__(self, model_config: AzureOpenAIModelConfiguration):
        self._async_evaluator = _AsyncCoherenceEvaluator(model_config)

    def __call__(self, *, question: str, answer: str, **kwargs):
        """
        Evaluate coherence.

        :keyword question: The question to be evaluated.
        :paramtype question: str
        :keyword answer: The answer to be evaluated.
        :paramtype answer: str
        :return: The coherence score.
        :rtype: Dict[str, float]
        """
        return async_run_allowing_running_loop(self._async_evaluator, question=question, answer=answer, **kwargs)

    def _to_async(self):
        return self._async_evaluator
