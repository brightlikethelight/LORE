"""Tests for refactored code modules.

This module tests the refactoring done in Phases 2-5:
- Phase 2: src/models/types.py (CompletionResponse), src/models/__init__.py exports
- Phase 3: src/models/text_extraction.py (unified extraction functions)
- Phase 4: src/constants.py (centralized constants)
- Phase 5: src/monitors/base.py (MonitorResult, MonitorError)
"""

from __future__ import annotations

import pytest


# =============================================================================
# Phase 2 Tests: src/models/types.py and imports
# =============================================================================


class TestCompletionResponse:
    """Tests for CompletionResponse dataclass in src/models/types.py."""

    def test_create_completion_response(self) -> None:
        """Test creating a CompletionResponse with all fields."""
        from src.models.types import CompletionResponse

        response = CompletionResponse(
            content="Hello, world!",
            model="gpt-4o",
            usage={"input_tokens": 10, "output_tokens": 5},
            finish_reason="stop",
            raw_response={"id": "test-123"},
        )

        assert response.content == "Hello, world!"
        assert response.model == "gpt-4o"
        assert response.usage == {"input_tokens": 10, "output_tokens": 5}
        assert response.finish_reason == "stop"
        assert response.raw_response == {"id": "test-123"}

    def test_completion_response_optional_raw(self) -> None:
        """Test CompletionResponse with None raw_response."""
        from src.models.types import CompletionResponse

        response = CompletionResponse(
            content="Test content",
            model="claude-sonnet-4",
            usage={"input_tokens": 5, "output_tokens": 10},
            finish_reason="stop",
        )

        assert response.raw_response is None

    def test_completion_response_default_raw(self) -> None:
        """Test CompletionResponse has correct default for raw_response."""
        from src.models.types import CompletionResponse

        response = CompletionResponse(
            content="",
            model="test",
            usage={},
            finish_reason="length",
        )

        assert response.raw_response is None


class TestModelsImports:
    """Tests for proper exports from src/models/__init__.py."""

    def test_import_completion_response_from_models(self) -> None:
        """Test CompletionResponse is exported from src.models."""
        from src.models import CompletionResponse

        assert CompletionResponse is not None

    def test_import_get_client_from_models(self) -> None:
        """Test get_client is exported from src.models."""
        from src.models import get_client

        assert callable(get_client)

    def test_import_all_expected_exports(self) -> None:
        """Test all expected symbols are exported."""
        from src import models

        expected_exports = [
            "CompletionResponse",
            "FrontierAPIClient",
            "AnthropicClient",
            "OpenAIClient",
            "GoogleClient",
            "FrontierModel",
            "get_client",
            "get_model",
            "OpenWeightsClient",
            "VLLMClient",
            "OpenWeightsModel",
        ]

        for name in expected_exports:
            assert hasattr(models, name), f"Missing export: {name}"


# =============================================================================
# Phase 3 Tests: src/models/text_extraction.py
# =============================================================================


class TestExtractTaggedContent:
    """Tests for extract_tagged_content function."""

    def test_extract_simple_tag(self) -> None:
        """Test extracting content from simple XML tag."""
        from src.models.text_extraction import extract_tagged_content

        text = "<thinking>My reasoning here</thinking>"
        result = extract_tagged_content(text, "thinking")

        assert result == "My reasoning here"

    def test_extract_multiline_tag(self) -> None:
        """Test extracting content from multiline tag."""
        from src.models.text_extraction import extract_tagged_content

        text = """<answer>
Line 1
Line 2
</answer>"""
        result = extract_tagged_content(text, "answer")

        assert result is not None
        assert "Line 1" in result
        assert "Line 2" in result

    def test_extract_case_insensitive(self) -> None:
        """Test extraction is case insensitive by default."""
        from src.models.text_extraction import extract_tagged_content

        text = "<THINKING>Content</THINKING>"
        result = extract_tagged_content(text, "thinking")

        assert result == "Content"

    def test_extract_not_found(self) -> None:
        """Test extraction returns None when tag not found."""
        from src.models.text_extraction import extract_tagged_content

        text = "No tags here"
        result = extract_tagged_content(text, "thinking")

        assert result is None


class TestExtractReasoningAndAnswerNew:
    """Tests for extract_reasoning_and_answer from text_extraction module."""

    def test_xml_format(self) -> None:
        """Test extraction from XML tags."""
        from src.models.text_extraction import extract_reasoning_and_answer

        text = "<thinking>Step by step</thinking><answer>42</answer>"
        reasoning, answer = extract_reasoning_and_answer(text)

        assert reasoning == "Step by step"
        assert answer == "42"

    def test_think_tag_format(self) -> None:
        """Test extraction from DeepSeek-style <think> tags."""
        from src.models.text_extraction import extract_reasoning_and_answer

        text = "<think>Deep thoughts</think><answer>Result</answer>"
        reasoning, answer = extract_reasoning_and_answer(text)

        assert reasoning == "Deep thoughts"
        assert answer == "Result"

    def test_json_format(self) -> None:
        """Test extraction from JSON format."""
        from src.models.text_extraction import extract_reasoning_and_answer

        text = '{"reasoning": "I calculated", "answer": "100"}'
        reasoning, answer = extract_reasoning_and_answer(text)

        assert reasoning == "I calculated"
        assert answer == "100"

    def test_text_markers(self) -> None:
        """Test extraction from text markers."""
        from src.models.text_extraction import extract_reasoning_and_answer

        text = "Reasoning: First I considered X.\nAnswer: 42"
        reasoning, answer = extract_reasoning_and_answer(text)

        assert "First I considered X" in reasoning
        assert answer == "42"

    def test_special_tokens(self) -> None:
        """Test extraction from special tokens."""
        from src.models.text_extraction import extract_reasoning_and_answer

        text = "<|begin_of_thought|>My thought<|end_of_thought|>Final answer: 42"
        reasoning, answer = extract_reasoning_and_answer(text)

        assert reasoning == "My thought"
        assert "42" in answer

    def test_fallback(self) -> None:
        """Test fallback when no patterns match."""
        from src.models.text_extraction import extract_reasoning_and_answer

        text = "Just a plain answer without structure"
        reasoning, answer = extract_reasoning_and_answer(text)

        assert reasoning == ""
        assert answer == "Just a plain answer without structure"


class TestMaybeExtractJsonNew:
    """Tests for maybe_extract_json from text_extraction module."""

    def test_plain_json(self) -> None:
        """Test extracting plain JSON."""
        from src.models.text_extraction import maybe_extract_json

        text = '{"key": "value"}'
        result = maybe_extract_json(text)

        assert result == {"key": "value"}

    def test_json_in_markdown(self) -> None:
        """Test extracting JSON from markdown code block."""
        from src.models.text_extraction import maybe_extract_json

        text = """```json
{"answer": 42}
```"""
        result = maybe_extract_json(text)

        assert result == {"answer": 42}

    def test_non_json_returns_none(self) -> None:
        """Test non-JSON text returns None."""
        from src.models.text_extraction import maybe_extract_json

        result = maybe_extract_json("This is plain text")

        assert result is None

    def test_json_array_returns_none(self) -> None:
        """Test JSON array returns None (expects object)."""
        from src.models.text_extraction import maybe_extract_json

        result = maybe_extract_json("[1, 2, 3]")

        assert result is None


class TestComposeSystemPrompt:
    """Tests for compose_system_prompt function."""

    def test_xml_format(self) -> None:
        """Test composing prompt with XML format."""
        from src.models.text_extraction import compose_system_prompt

        result = compose_system_prompt(
            system_prompt="You are helpful.",
            reasoning_guidelines="Think step by step.",
            output_format="xml",
        )

        assert "You are helpful" in result
        assert "Think step by step" in result
        assert "<thinking>" in result
        assert "<answer>" in result

    def test_json_format(self) -> None:
        """Test composing prompt with JSON format."""
        from src.models.text_extraction import compose_system_prompt

        result = compose_system_prompt(
            system_prompt="Be concise.",
            reasoning_guidelines=None,
            output_format="json",
        )

        assert "Be concise" in result
        assert "JSON" in result

    def test_default_format(self) -> None:
        """Test composing prompt with default (think tag) format."""
        from src.models.text_extraction import compose_system_prompt

        result = compose_system_prompt(
            system_prompt=None,
            reasoning_guidelines="Guidelines",
            output_format="default",
        )

        assert "<think>" in result


class TestWantsJsonMode:
    """Tests for wants_json_mode function."""

    def test_detects_json_in_prompt(self) -> None:
        """Test detection of JSON request in prompt."""
        from src.models.text_extraction import wants_json_mode

        result = wants_json_mode(
            system_prompt=None,
            reasoning_guidelines=None,
            prompt="Return the answer in JSON format",
        )

        assert result is True

    def test_detects_json_in_system(self) -> None:
        """Test detection of JSON request in system prompt."""
        from src.models.text_extraction import wants_json_mode

        result = wants_json_mode(
            system_prompt="Always return JSON",
            reasoning_guidelines=None,
            prompt="What is 2+2?",
        )

        assert result is True

    def test_no_json_request(self) -> None:
        """Test returns False when no JSON requested."""
        from src.models.text_extraction import wants_json_mode

        result = wants_json_mode(
            system_prompt="You are helpful",
            reasoning_guidelines="Think carefully",
            prompt="Calculate 2+2",
        )

        assert result is False


# =============================================================================
# Phase 4 Tests: src/constants.py
# =============================================================================


class TestRegimeEnum:
    """Tests for Regime enum."""

    def test_regime_values(self) -> None:
        """Test Regime enum has correct values."""
        from src.constants import Regime

        assert Regime.CAPABILITY_ONLY.value == "A"
        assert Regime.CAPABILITY_STEALTH.value == "B"

    def test_regime_from_string(self) -> None:
        """Test creating Regime from string value."""
        from src.constants import Regime

        regime_a = Regime("A")
        regime_b = Regime("B")

        assert regime_a == Regime.CAPABILITY_ONLY
        assert regime_b == Regime.CAPABILITY_STEALTH


class TestSuspicionThresholds:
    """Tests for suspicion score threshold constants."""

    def test_threshold_values(self) -> None:
        """Test threshold constants have expected values."""
        from src.constants import (
            DEFAULT_SUSPICION_THRESHOLD,
            HIGH_RISK_THRESHOLD,
            LOW_RISK_THRESHOLD,
            SAFE_THRESHOLD,
            CONCERNING_THRESHOLD,
        )

        assert DEFAULT_SUSPICION_THRESHOLD == 0.5
        assert HIGH_RISK_THRESHOLD == 0.7
        assert LOW_RISK_THRESHOLD == 0.3
        assert SAFE_THRESHOLD == 0.1
        assert CONCERNING_THRESHOLD == 0.5

    def test_threshold_ordering(self) -> None:
        """Test thresholds are in logical order."""
        from src.constants import (
            HIGH_RISK_THRESHOLD,
            LOW_RISK_THRESHOLD,
            SAFE_THRESHOLD,
        )

        assert SAFE_THRESHOLD < LOW_RISK_THRESHOLD < HIGH_RISK_THRESHOLD


class TestOptimizationConstants:
    """Tests for optimization parameter constants."""

    def test_optimization_defaults(self) -> None:
        """Test optimization parameter defaults."""
        from src.constants import (
            DEFAULT_MUTATION_RATE,
            DEFAULT_POPULATION_SIZE,
            DEFAULT_GENERATIONS,
            DEFAULT_TOURNAMENT_SIZE,
        )

        assert DEFAULT_MUTATION_RATE == 0.15
        assert DEFAULT_POPULATION_SIZE == 50
        assert DEFAULT_GENERATIONS == 100
        assert DEFAULT_TOURNAMENT_SIZE == 5

    def test_mutation_rate_valid_range(self) -> None:
        """Test mutation rate is a valid probability."""
        from src.constants import DEFAULT_MUTATION_RATE

        assert 0.0 <= DEFAULT_MUTATION_RATE <= 1.0


class TestModelConstants:
    """Tests for model default constants."""

    def test_model_defaults(self) -> None:
        """Test model default values."""
        from src.constants import (
            DEFAULT_MAX_TOKENS,
            DEFAULT_TEMPERATURE,
            DEFAULT_RATE_LIMIT_RPM,
        )

        assert DEFAULT_MAX_TOKENS == 4096
        assert DEFAULT_TEMPERATURE == 0.7
        assert DEFAULT_RATE_LIMIT_RPM == 120


class TestDatasetConstants:
    """Tests for dataset parameter constants."""

    def test_dataset_defaults(self) -> None:
        """Test dataset default values."""
        from src.constants import (
            HIDDEN_GOAL_INJECTION_RATE,
            DEFAULT_TRAIN_RATIO,
            DEFAULT_VAL_RATIO,
            DEFAULT_TEST_RATIO,
            GRG_VALIDATION_SIZE,
        )

        assert HIDDEN_GOAL_INJECTION_RATE == 0.20
        assert DEFAULT_TRAIN_RATIO == 0.70
        assert DEFAULT_VAL_RATIO == 0.15
        assert DEFAULT_TEST_RATIO == 0.15
        assert GRG_VALIDATION_SIZE == 200

    def test_split_ratios_sum_to_one(self) -> None:
        """Test train/val/test ratios sum to 1.0."""
        from src.constants import (
            DEFAULT_TRAIN_RATIO,
            DEFAULT_VAL_RATIO,
            DEFAULT_TEST_RATIO,
        )

        total = DEFAULT_TRAIN_RATIO + DEFAULT_VAL_RATIO + DEFAULT_TEST_RATIO
        assert abs(total - 1.0) < 1e-9


class TestEvaluationConstants:
    """Tests for evaluation parameter constants."""

    def test_evaluation_defaults(self) -> None:
        """Test evaluation parameter defaults."""
        from src.constants import (
            MATH_EQUALITY_TOLERANCE,
            SYMBOLIC_EVAL_TIMEOUT,
        )

        assert MATH_EQUALITY_TOLERANCE == 1e-4
        assert SYMBOLIC_EVAL_TIMEOUT == 5.0


class TestMonitorConstants:
    """Tests for monitor parameter constants."""

    def test_monitor_defaults(self) -> None:
        """Test monitor parameter defaults."""
        from src.constants import (
            ENSEMBLE_DISAGREEMENT_THRESHOLD,
            AUTORATER_MAX_RETRIES,
            AUTORATER_TEMPERATURE,
        )

        assert ENSEMBLE_DISAGREEMENT_THRESHOLD == 0.3
        assert AUTORATER_MAX_RETRIES == 3
        assert AUTORATER_TEMPERATURE == 0.3


# =============================================================================
# Phase 5 Tests: src/monitors/base.py
# =============================================================================


class TestMonitorErrorCode:
    """Tests for MonitorErrorCode enum."""

    def test_error_codes_exist(self) -> None:
        """Test all expected error codes exist."""
        from src.monitors.base import MonitorErrorCode

        assert MonitorErrorCode.NONE.value == "none"
        assert MonitorErrorCode.API_ERROR.value == "api_error"
        assert MonitorErrorCode.PARSE_ERROR.value == "parse_error"
        assert MonitorErrorCode.TIMEOUT.value == "timeout"
        assert MonitorErrorCode.RATE_LIMIT.value == "rate_limit"
        assert MonitorErrorCode.VALIDATION_ERROR.value == "validation_error"
        assert MonitorErrorCode.UNKNOWN.value == "unknown"


class TestMonitorError:
    """Tests for MonitorError dataclass."""

    def test_create_monitor_error(self) -> None:
        """Test creating a MonitorError."""
        from src.monitors.base import MonitorError, MonitorErrorCode

        error = MonitorError(
            code=MonitorErrorCode.API_ERROR,
            message="API call failed",
            details={"status_code": 500},
            recoverable=True,
        )

        assert error.code == MonitorErrorCode.API_ERROR
        assert error.message == "API call failed"
        assert error.details == {"status_code": 500}
        assert error.recoverable is True

    def test_monitor_error_defaults(self) -> None:
        """Test MonitorError default values."""
        from src.monitors.base import MonitorError, MonitorErrorCode

        error = MonitorError(
            code=MonitorErrorCode.UNKNOWN,
            message="Unknown error",
        )

        assert error.details == {}
        assert error.recoverable is True

    def test_monitor_error_to_dict(self) -> None:
        """Test MonitorError serialization."""
        from src.monitors.base import MonitorError, MonitorErrorCode

        error = MonitorError(
            code=MonitorErrorCode.TIMEOUT,
            message="Request timed out",
            details={"timeout_ms": 30000},
            recoverable=True,
        )

        data = error.to_dict()

        assert data["code"] == "timeout"
        assert data["message"] == "Request timed out"
        assert data["details"] == {"timeout_ms": 30000}
        assert data["recoverable"] is True

    def test_monitor_error_from_exception(self) -> None:
        """Test creating MonitorError from exception."""
        from src.monitors.base import MonitorError, MonitorErrorCode

        exc = TimeoutError("Connection timed out")
        error = MonitorError.from_exception(exc)

        assert error.code == MonitorErrorCode.TIMEOUT
        assert "timed out" in error.message
        assert error.details["exception_type"] == "TimeoutError"

    def test_monitor_error_from_rate_limit_exception(self) -> None:
        """Test creating MonitorError from rate limit exception."""
        from src.monitors.base import MonitorError, MonitorErrorCode

        class RateLimitError(Exception):
            pass

        exc = RateLimitError("Rate limit exceeded")
        error = MonitorError.from_exception(exc)

        assert error.code == MonitorErrorCode.RATE_LIMIT

    def test_monitor_error_from_json_exception(self) -> None:
        """Test creating MonitorError from JSON parse exception."""
        from src.monitors.base import MonitorError, MonitorErrorCode
        import json

        exc = json.JSONDecodeError("Expecting value", "doc", 0)
        error = MonitorError.from_exception(exc)

        assert error.code == MonitorErrorCode.PARSE_ERROR


class TestMonitorResult:
    """Tests for MonitorResult dataclass."""

    def test_create_monitor_result(self) -> None:
        """Test creating a MonitorResult."""
        from src.monitors.base import MonitorResult

        result = MonitorResult(
            success=True,
            suspicion_score=0.3,
            metadata={"pattern": "verbose"},
        )

        assert result.success is True
        assert result.suspicion_score == 0.3
        assert result.error is None
        assert result.metadata == {"pattern": "verbose"}

    def test_stealth_score_computed(self) -> None:
        """Test stealth_score is computed correctly."""
        from src.monitors.base import MonitorResult

        result = MonitorResult(success=True, suspicion_score=0.2)

        assert result.stealth_score == pytest.approx(0.8)

    def test_suspicion_score_clamping(self) -> None:
        """Test suspicion score is clamped to [0, 1]."""
        from src.monitors.base import MonitorResult

        high = MonitorResult(success=True, suspicion_score=1.5)
        low = MonitorResult(success=True, suspicion_score=-0.5)

        assert high.suspicion_score == 1.0
        assert low.suspicion_score == 0.0

    def test_is_error_property(self) -> None:
        """Test is_error property."""
        from src.monitors.base import MonitorResult, MonitorError, MonitorErrorCode

        success_result = MonitorResult(success=True, suspicion_score=0.5)
        failure_result = MonitorResult(
            success=False,
            suspicion_score=0.5,
            error=MonitorError(code=MonitorErrorCode.API_ERROR, message="Failed"),
        )

        assert success_result.is_error is False
        assert failure_result.is_error is True

    def test_error_message_property(self) -> None:
        """Test error_message property."""
        from src.monitors.base import MonitorResult, MonitorError, MonitorErrorCode

        result_with_error = MonitorResult(
            success=False,
            error=MonitorError(code=MonitorErrorCode.TIMEOUT, message="Timed out"),
        )
        result_without_error = MonitorResult(success=True, suspicion_score=0.3)

        assert result_with_error.error_message == "Timed out"
        assert result_without_error.error_message is None

    def test_monitor_result_to_dict(self) -> None:
        """Test MonitorResult serialization."""
        from src.monitors.base import MonitorResult

        result = MonitorResult(
            success=True,
            suspicion_score=0.4,
            metadata={"key": "value"},
        )

        data = result.to_dict()

        assert data["success"] is True
        assert data["suspicion_score"] == 0.4
        assert data["stealth_score"] == pytest.approx(0.6)
        assert data["error"] is None
        assert data["metadata"] == {"key": "value"}

    def test_success_result_factory(self) -> None:
        """Test MonitorResult.success_result factory method."""
        from src.monitors.base import MonitorResult

        result = MonitorResult.success_result(
            suspicion_score=0.25,
            metadata={"patterns": ["verbose"]},
        )

        assert result.success is True
        assert result.suspicion_score == 0.25
        assert result.error is None
        assert result.metadata == {"patterns": ["verbose"]}

    def test_failure_result_factory_from_string(self) -> None:
        """Test MonitorResult.failure_result from string error."""
        from src.monitors.base import MonitorResult, MonitorErrorCode

        result = MonitorResult.failure_result(
            error="Something went wrong",
            neutral_score=0.5,
        )

        assert result.success is False
        assert result.suspicion_score == 0.5
        assert result.error is not None
        assert result.error.code == MonitorErrorCode.UNKNOWN
        assert result.error.message == "Something went wrong"

    def test_failure_result_factory_from_exception(self) -> None:
        """Test MonitorResult.failure_result from exception."""
        from src.monitors.base import MonitorResult, MonitorErrorCode

        result = MonitorResult.failure_result(
            error=ValueError("Invalid input"),
        )

        assert result.success is False
        assert result.error is not None
        assert result.error.details["exception_type"] == "ValueError"

    def test_failure_result_factory_from_monitor_error(self) -> None:
        """Test MonitorResult.failure_result from MonitorError."""
        from src.monitors.base import MonitorResult, MonitorError, MonitorErrorCode

        error = MonitorError(
            code=MonitorErrorCode.RATE_LIMIT,
            message="Rate limited",
        )
        result = MonitorResult.failure_result(error=error)

        assert result.success is False
        assert result.error is error


class TestHandleMonitorError:
    """Tests for handle_monitor_error function."""

    def test_handle_exception(self) -> None:
        """Test handling a generic exception."""
        from src.monitors.base import handle_monitor_error

        exc = Exception("Test error")
        result = handle_monitor_error(exc)

        assert result.success is False
        assert result.suspicion_score == 0.5  # Default neutral
        assert result.error is not None

    def test_handle_exception_with_context(self) -> None:
        """Test handling exception with context."""
        from src.monitors.base import handle_monitor_error

        exc = ValueError("Bad value")
        result = handle_monitor_error(exc, context="During parsing")

        assert result.metadata.get("context") == "During parsing"

    def test_handle_exception_custom_default_score(self) -> None:
        """Test handling exception with custom default score."""
        from src.monitors.base import handle_monitor_error

        exc = RuntimeError("Runtime error")
        result = handle_monitor_error(exc, default_score=0.7)

        assert result.suspicion_score == 0.7


class TestMonitorExports:
    """Tests for proper exports from src/monitors/__init__.py."""

    def test_base_types_exported(self) -> None:
        """Test base types are exported from monitors module."""
        from src.monitors import (
            MonitorError,
            MonitorErrorCode,
            MonitorResult,
            handle_monitor_error,
        )

        assert MonitorError is not None
        assert MonitorErrorCode is not None
        assert MonitorResult is not None
        assert callable(handle_monitor_error)


# =============================================================================
# Integration Tests: Cross-module functionality
# =============================================================================


class TestCrossModuleIntegration:
    """Tests verifying modules work together correctly."""

    def test_monitor_result_uses_constants(self) -> None:
        """Test MonitorResult uses constants from src.constants."""
        from src.monitors.base import MonitorResult
        from src.constants import DEFAULT_SUSPICION_THRESHOLD

        # MonitorResult should default to DEFAULT_SUSPICION_THRESHOLD
        result = MonitorResult(success=False)

        assert result.suspicion_score == DEFAULT_SUSPICION_THRESHOLD

    def test_text_extraction_module_standalone(self) -> None:
        """Test text_extraction works independently."""
        from src.models.text_extraction import (
            extract_tagged_content,
            extract_reasoning_and_answer,
            maybe_extract_json,
            compose_system_prompt,
            wants_json_mode,
        )

        # All functions should be importable and callable
        assert callable(extract_tagged_content)
        assert callable(extract_reasoning_and_answer)
        assert callable(maybe_extract_json)
        assert callable(compose_system_prompt)
        assert callable(wants_json_mode)
