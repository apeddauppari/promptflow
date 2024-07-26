# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
from azureml.metrics import compute_aggregate
from azureml.metrics._aggregators import AggregatorType
from typing import List
from enum import Enum
from abc import ABC, abstractmethod


class EvalAggregatorType(Enum):
    MEAN_AGGREGATOR = "mean"


class AggregationEvaluator(ABC):
    @abstractmethod
    def __call__(self, *, scores: List[float], **kwargs):
        pass

    @abstractmethod
    def _validate_scores(self, scores: list):
        pass

    @abstractmethod
    def _evaluate(self, scores: list) -> float:
        pass


class MeanAggregatorEvaluator(AggregationEvaluator):

    def __call__(self, *, scores: List[float]):
        _ = self._validate_scores(scores)
        return self._evaluate(scores=scores)

    def _validate_scores(self, scores: list):
        if not all(isinstance(score, (int, float)) for score in scores):
            raise ValueError("All scores must be integers or floats.")
        if not scores:
            raise ValueError("Scores list cannot be empty.")
        return True

    def _evaluate(self, scores: List[float]) -> float:
        # Compute the score.
        aggregated_score = compute_aggregate(aggregator_type=AggregatorType.MEAN_AGGREGATOR, scores=scores)
        print(f"\nMeanAggregatorEvaluator >> COMPUTE_SCORE >> aggregated_score :: {aggregated_score}")
        return aggregated_score


class AggregatorFactory:
    @staticmethod
    def get_aggregator(aggregator_type: EvalAggregatorType) -> AggregationEvaluator:

        if aggregator_type == EvalAggregatorType.MEAN_AGGREGATOR:
            return MeanAggregatorEvaluator()
        else:
            raise ValueError(f"Unsupported AggregatorType: {aggregator_type}")


def evaluate_aggregate(aggregator_type: EvalAggregatorType, scores: List[float], **kwargs) -> float:
    aggregator = AggregatorFactory.get_aggregator(aggregator_type=aggregator_type)
    return aggregator(scores=scores)
