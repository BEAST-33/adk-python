# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from google.adk.evaluation.eval_case import Invocation
from google.adk.evaluation.eval_metrics import EvalMetric
from google.adk.evaluation.eval_metrics import JudgeModelOptions
from google.adk.evaluation.evaluator import EvalStatus
from google.adk.evaluation.evaluator import PerInvocationResult
from google.adk.evaluation.final_response_match_v2 import _parse_critique_with_golden_response
from google.adk.evaluation.final_response_match_v2 import FinalResponseMatchV2Evaluator
from google.adk.evaluation.final_response_match_v2 import Label
from google.adk.models.llm_response import LlmResponse
from google.genai import types as genai_types
import pytest


@pytest.mark.parametrize(
    "response_text",
    [
        """```json
  {
    "is_the_agent_response_valid_or_invalid": "valid",
    "reasoning": "The response is valid."
  }
  ```""",
        """```json
  {
    "is_the_agent_response_valid": "undefined label",
  }
  ```""",
    ],
)
def test_parse_critique_with_golden_response_label_not_found(response_text):
  label = _parse_critique_with_golden_response(response_text)
  assert label == Label.NOT_FOUND


@pytest.mark.parametrize(
    "response_text",
    [
        """```json
  {
    "is_the_agent_response_valid": "valid",
    "reasoning": "The response is valid."
  }
  ```""",
        """```json
  {
    "is_the_agent_response_valid": ["valid"],
    "reasoning": "The response is valid."
  }
  ```""",
        """```json
  {
    "is_the_agent_response_valid":\n    [ "valid\n"],
    "reasoning": "The response is valid."
  }
  ```""",
    ],
)
def test_parse_critique_with_golden_response(response_text):
  label = _parse_critique_with_golden_response(response_text)
  assert label == Label.VALID


@pytest.mark.parametrize(
    "response_text",
    [
        """```json
  {
    "is_the_agent_response_invalid": "invalid",
    "reasoning": "The response is invalid."
  }
  ```""",
        """```json
  {
    "is_the_agent_response_invalid": ["invalid"],
    "reasoning": "The response is invalid."
  }
  ```""",
        """```json
  {
    "is_the_agent_response_invalid":\n    [ "invalid\n"],
    "reasoning": "The response is invalid."
  }
  ```""",
    ],
)
def test_parse_critique_with_golden_response_invalid(response_text):
  label = _parse_critique_with_golden_response(response_text)
  assert label == Label.INVALID


def create_test_template() -> str:
  return """
This is a test template.

{{
  "Function API spec": {function_api_spec},
  "User prompt": {prompt},
  "Agent response": {response},
  "Reference response": {golden_response},
}}

The answer should be a json alone which follows the json structure below:
{{
  "is_the_agent_response_valid": [valid or invalid],
  "reasoning":
}}
"""


def _create_test_evaluator_gemini(
    threshold: float,
) -> FinalResponseMatchV2Evaluator:
  evaluator = FinalResponseMatchV2Evaluator(
      EvalMetric(
          metric_name="final_response_match_v2",
          threshold=threshold,
          judge_model_options=JudgeModelOptions(
              judge_model="gemini-2.5-flash",
              num_samples=3,
          ),
      ),
  )
  evaluator._auto_rater_prompt_template = create_test_template()
  return evaluator


def _create_test_invocations(
    candidate: str, reference: str, invocation_id: str = ""
) -> tuple[Invocation, Invocation]:
  """Returns tuple of (actual_invocation, expected_invocation)."""
  actual_invocation = Invocation(
      invocation_id=invocation_id,
      user_content=genai_types.Content(
          parts=[genai_types.Part(text="This is a test query.")],
          role="user",
      ),
      final_response=genai_types.Content(
          parts=[genai_types.Part(text=candidate)],
          role="model",
      ),
  )
  expected_invocation = Invocation(
      invocation_id=invocation_id,
      user_content=genai_types.Content(
          parts=[genai_types.Part(text="This is a test query.")],
          role="user",
      ),
      final_response=genai_types.Content(
          parts=[genai_types.Part(text=reference)],
          role="model",
      ),
  )
  return actual_invocation, expected_invocation


def test_format_auto_rater_prompt():
  evaluator = _create_test_evaluator_gemini(threshold=0.8)
  actual_invocation, expected_invocation = _create_test_invocations(
      "candidate text", "reference text"
  )
  prompt = evaluator.format_auto_rater_prompt(
      actual_invocation, expected_invocation
  )
  assert prompt == """
This is a test template.

{
  "Function API spec": None,
  "User prompt": This is a test query.,
  "Agent response": candidate text,
  "Reference response": reference text,
}

The answer should be a json alone which follows the json structure below:
{
  "is_the_agent_response_valid": [valid or invalid],
  "reasoning":
}
"""


def test_convert_auto_rater_response_to_score_valid():
  evaluator = _create_test_evaluator_gemini(threshold=0.8)
  auto_rater_response = """```json
{
  "is_the_agent_response_valid": "valid",
  "reasoning": "The response is valid."
}
```"""
  llm_response = LlmResponse(
      content=genai_types.Content(
          parts=[genai_types.Part(text=auto_rater_response)],
          role="model",
      )
  )
  score = evaluator.convert_auto_rater_response_to_score(llm_response)
  assert score == 1.0


def test_convert_auto_rater_response_to_score_invalid():
  evaluator = _create_test_evaluator_gemini(threshold=0.8)
  auto_rater_response = """```json
{
  "is_the_agent_response_valid": "invalid",
  "reasoning": "The response is invalid."
}
```"""
  llm_response = LlmResponse(
      content=genai_types.Content(
          parts=[genai_types.Part(text=auto_rater_response)],
          role="model",
      )
  )
  score = evaluator.convert_auto_rater_response_to_score(llm_response)
  assert score == 0.0


def test_convert_auto_rater_response_to_score_invalid_json():
  evaluator = _create_test_evaluator_gemini(threshold=0.8)
  llm_response = LlmResponse(
      content=genai_types.Content(
          parts=[genai_types.Part(text="invalid json")],
          role="model",
      )
  )
  score = evaluator.convert_auto_rater_response_to_score(llm_response)
  assert score is None


def test_convert_auto_rater_response_to_score_missing_key():
  evaluator = _create_test_evaluator_gemini(threshold=0.8)
  llm_response = LlmResponse(
      content=genai_types.Content(
          parts=[genai_types.Part(text="{}")],
          role="model",
      )
  )
  score = evaluator.convert_auto_rater_response_to_score(llm_response)
  assert score is None


def test_aggregate_invocation_results_flat():
  evaluator = _create_test_evaluator_gemini(threshold=0.5)

  # Create 8 test invocations with different invocation id's.
  actual_invocations = []
  expected_invocations = []
  for i in range(8):
    actual_invocation, expected_invocation = _create_test_invocations(
        "candidate text", "reference text", invocation_id=str(i)
    )
    actual_invocations.append(actual_invocation)
    expected_invocations.append(expected_invocation)

  per_invocation_results = [
      PerInvocationResult(
          actual_invocation=actual_invocations[0],
          expected_invocation=expected_invocations[0],
          score=1.0,
          eval_status=EvalStatus.PASSED,
      ),
      PerInvocationResult(
          actual_invocation=actual_invocations[1],
          expected_invocation=expected_invocations[1],
          score=1.0,
          eval_status=EvalStatus.PASSED,
      ),
      PerInvocationResult(
          actual_invocation=actual_invocations[3],
          expected_invocation=expected_invocations[3],
          score=0.0,
          eval_status=EvalStatus.FAILED,
      ),
      PerInvocationResult(
          actual_invocation=actual_invocations[4],
          expected_invocation=expected_invocations[4],
          score=0.0,
          eval_status=EvalStatus.FAILED,
      ),
      PerInvocationResult(
          actual_invocation=actual_invocations[5],
          expected_invocation=expected_invocations[5],
          score=None,
          eval_status=EvalStatus.PASSED,
      ),
      PerInvocationResult(
          actual_invocation=actual_invocations[6],
          expected_invocation=expected_invocations[6],
          score=100.0,
          eval_status=EvalStatus.NOT_EVALUATED,
      ),
      PerInvocationResult(
          actual_invocation=actual_invocations[7],
          expected_invocation=expected_invocations[7],
          score=None,
          eval_status=EvalStatus.NOT_EVALUATED,
      ),
  ]

  aggregated_result = evaluator.aggregate_invocation_results(
      per_invocation_results
  )

  # Only 4 / 8 invocations are evaluated, and 2 / 4 are valid.
  assert aggregated_result.overall_score == 0.5
  assert aggregated_result.overall_eval_status == EvalStatus.PASSED


def test_aggregate_invocation_results_grouped():
  evaluator = _create_test_evaluator_gemini(threshold=0.5)

  # Create 4 test invocations with different invocation id's.
  actual_invocation0, expected_invocation0 = _create_test_invocations(
      "candidate text", "reference text", invocation_id="id0"
  )
  actual_invocation1, expected_invocation1 = _create_test_invocations(
      "candidate text", "reference text", invocation_id="id1"
  )
  actual_invocation2, expected_invocation2 = _create_test_invocations(
      "candidate text", "reference text", invocation_id="id2"
  )
  actual_invocation3, expected_invocation3 = _create_test_invocations(
      "candidate text", "reference text", invocation_id="id3"
  )

  per_invocation_results = [
      # Invocations 0 -> score = 1.0
      PerInvocationResult(
          actual_invocation=actual_invocation0,
          expected_invocation=expected_invocation0,
          score=1.0,
          eval_status=EvalStatus.PASSED,
      ),
      PerInvocationResult(
          actual_invocation=actual_invocation0,
          expected_invocation=expected_invocation0,
          score=1.0,
          eval_status=EvalStatus.PASSED,
      ),
      PerInvocationResult(
          actual_invocation=actual_invocation0,
          expected_invocation=expected_invocation0,
          score=0.0,
          eval_status=EvalStatus.FAILED,
      ),
      PerInvocationResult(
          actual_invocation=actual_invocation0,
          expected_invocation=expected_invocation0,
          score=0.0,
          eval_status=EvalStatus.FAILED,
      ),
      PerInvocationResult(
          actual_invocation=actual_invocation0,
          expected_invocation=expected_invocation0,
          score=1.0,
          eval_status=EvalStatus.PASSED,
      ),
      # Invocations 1 -> score = 0.0
      PerInvocationResult(
          actual_invocation=actual_invocation1,
          expected_invocation=expected_invocation1,
          score=1.0,
          eval_status=EvalStatus.PASSED,
      ),
      PerInvocationResult(
          actual_invocation=actual_invocation1,
          expected_invocation=expected_invocation1,
          score=0.0,
          eval_status=EvalStatus.FAILED,
      ),
      PerInvocationResult(
          actual_invocation=actual_invocation1,
          expected_invocation=expected_invocation1,
          score=None,
          eval_status=EvalStatus.FAILED,
      ),
      PerInvocationResult(
          actual_invocation=actual_invocation1,
          expected_invocation=expected_invocation1,
          score=100.0,
          eval_status=EvalStatus.NOT_EVALUATED,
      ),
      PerInvocationResult(
          actual_invocation=actual_invocation1,
          expected_invocation=expected_invocation1,
          score=None,
          eval_status=EvalStatus.NOT_EVALUATED,
      ),
      # Invocations 2 -> Not evaluated.
      PerInvocationResult(
          actual_invocation=actual_invocation2,
          expected_invocation=expected_invocation2,
          score=1.0,
          eval_status=EvalStatus.NOT_EVALUATED,
      ),
      PerInvocationResult(
          actual_invocation=actual_invocation2,
          expected_invocation=expected_invocation2,
          score=None,
          eval_status=EvalStatus.PASSED,
      ),
      # Invocations 3 -> score = 1.0
      PerInvocationResult(
          actual_invocation=actual_invocation3,
          expected_invocation=expected_invocation3,
          score=1.0,
          eval_status=EvalStatus.PASSED,
      ),
      PerInvocationResult(
          actual_invocation=actual_invocation3,
          expected_invocation=expected_invocation3,
          score=None,
          eval_status=EvalStatus.NOT_EVALUATED,
      ),
  ]

  aggregated_result = evaluator.aggregate_invocation_results(
      per_invocation_results
  )

  # 3 / 4 invocations are evaluated, and 2 / 3 are valid.
  assert aggregated_result.overall_score == 2 / 3
  assert aggregated_result.overall_eval_status == EvalStatus.PASSED
