from typing import List
from promptflow.core import tool
from promptflow.evals.evaluators import EvalAggregatorType, evaluate_aggregate


@tool
def calculate_accuracy(grades: List[str]) -> str:
    aggregated_score = evaluate_aggregate(EvalAggregatorType.MEAN_AGGREGATOR, grades)

    result_json = {}
    result_json["aggregate"] = round(aggregated_score, 2)
    result_json["grades"] = grades
    print(
        f"\n\nPROMPTFLOW >> AGGREGATER FUNCTION :: aggregate_score : {aggregated_score}"
    )
    return aggregated_score
