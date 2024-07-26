# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
from azureml.metrics import compute_grades
from azureml.metrics._graders import GraderType
from typing import Union
from enum import Enum
from abc import ABC, abstractmethod


class EvalGraderType(Enum):
    BLEU_SCORE_GRADER = "bleu-score-grader"


class GraderEvaluator(ABC):
    @abstractmethod
    def __call__(self, *, predicted: str, ground_truth: str, **kwargs):
        pass

    @abstractmethod
    def _validate_inputs(self, predicted: str, ground_truth: str):
        pass

    @abstractmethod
    def _evaluate(self, predicted: str, ground_truth: str, **kwargs):
        pass


class BleuScoreGraderEvaluator(GraderEvaluator):

    def __call__(self, *, predicted: str, ground_truth: str, **kwargs):

        _ = self._validate_inputs(predicted=predicted, ground_truth=ground_truth)

        scores = self._evaluate(predicted=predicted, ground_truth=ground_truth, kwargs=kwargs)
        print(f"\nBleuScoreGraderEvaluator >> CALL >> GRADER_SCORES :: {scores}")

        return scores

    def _validate_inputs(self, predicted: str, ground_truth: str):
        if not all(
            [
                predicted and predicted != "None",
                ground_truth and ground_truth != "None",
            ]
        ):
            raise ValueError("Both 'predicted' and 'ground_truth' must be non-empty strings.")
        return True

    def _evaluate(self, predicted: str, ground_truth: str, **kwargs):

        print(f"\nBleuScoreGraderEvaluator >> evaluate_score :: {kwargs}")

        scores = compute_grades(
            grader_type=GraderType.BLEU_SCORE_GRADER, reference_texts=ground_truth, predicted_texts=predicted
        )
        print(f"\nBleuScoreGraderEvaluator >> evaluate_score >> SCORES :: {scores}")

        return scores


class GraderEvaluatorFactory:
    @staticmethod
    def get_grader_evaluator(grader_type: EvalGraderType) -> GraderEvaluator:
        """
        Return the corresponding GraderEvaluator instance based on the EvalGraderType.

        :param grader_type: The type of grader to return.
        :return: An instance of a subclass of GraderEvaluator.
        """
        if grader_type == EvalGraderType.BLEU_SCORE_GRADER:
            return BleuScoreGraderEvaluator()
        else:
            raise ValueError(f"Unsupported GraderType: {grader_type}")


def evaluate_grade(grader_type: EvalGraderType, predicted: str, ground_truth: str) -> Union[float, str]:
    """
    Compute the grades for the given predicted texts and reference texts.

    :param grader_type: The type of grader to use.
    :type grader_type: EvalGraderType
    :param predicted_texts: The predicted texts.
    :type predicted_texts: List[str]
    :param reference_texts: The reference texts.
    :type reference_texts: List[str]
    :return: The grades.
    :rtype: List[str]
    """
    grader_evaluator = GraderEvaluatorFactory.get_grader_evaluator(grader_type=grader_type)
    return grader_evaluator(predicted=predicted, ground_truth=ground_truth)
