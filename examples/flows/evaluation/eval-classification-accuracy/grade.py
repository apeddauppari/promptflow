from promptflow.core import tool
from promptflow.evals.evaluators import evaluate_grade, EvalGraderType
from typing import Union


@tool
def grade(groundtruth: str, prediction: str) -> Union[float, str]:
    print(
        f"\nPROMPTFLOW >> GRADER FUNCTION :: prediction : {prediction}, groundtruth : {groundtruth}"
    )

    grade = evaluate_grade(
        EvalGraderType.BLEU_SCORE_GRADER, predicted=prediction, ground_truth=groundtruth
    )
    print(f"\n\nPROMPTFLOW >> GRADER FUNCTION :: BleuScores : {grade}")
    return grade
