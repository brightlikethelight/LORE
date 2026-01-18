"""GEPA Adapter for UltraInteract Dataset.

This module provides a GEPA adapter specifically designed for evolving
a general reasoning prompt on the UltraInteract dataset while using
fixed task-specific templates.

Key design:
- ONLY the general prompt is evolved
- Task-specific templates are FIXED and concatenated at inference time
- Final prompt = general_prompt + TASK_TEMPLATES[task_type]

Uses safety-tooling InferenceAPI for robust API calls with:
- Automatic retries with exponential backoff
- Built-in caching
- Rate limiting
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, TypedDict

import numpy as np

from src.data.ultrainteract import (
    UITaskType,
    UltraInteractLoader,
    UltraInteractSample,
    create_gepa_dataset,
)
from src.evaluation.math_grader import (
    grade_math_answer,
    math_equal,
    extract_math_answer,
)

# Import safety-tooling for robust API calls
try:
    from safetytooling.apis import InferenceAPI
    from safetytooling.data_models import ChatMessage, MessageRole, Prompt
    from safetytooling.utils import utils as safetytooling_utils
    HAS_SAFETYTOOLING = True
except ImportError:
    HAS_SAFETYTOOLING = False
    InferenceAPI = None

# =============================================================================
# Fixed Task-Specific Templates (NOT evolved)
# =============================================================================

TASK_TEMPLATES: dict[str, str] = {
    "Coding": """For coding problems:
- Write clean, well-documented code
- Handle edge cases explicitly
- Consider time and space complexity
- Use meaningful variable names
- Include comments for complex logic""",

    "Math_CoT": """For mathematical reasoning:
- Show all calculation steps explicitly
- State the mathematical principles used
- Verify intermediate results
- Present the final answer clearly (use \\boxed{} for LaTeX)""",

    "Math_PoT": """For program-of-thought math:
- Translate the mathematical problem into executable code
- Define helper functions for complex operations
- Print intermediate results for verification
- Compute and clearly output the final answer""",

    "Logic": """For logical reasoning:
- Identify all premises and constraints
- Make logical deductions explicit step by step
- Consider alternative interpretations
- State your conclusion clearly with supporting evidence""",
}

# Default seed prompt for general reasoning
DEFAULT_SEED_PROMPT = """When solving problems, follow these reasoning principles:

1. Break complex problems into smaller, manageable steps
2. Verify each step before proceeding to the next
3. Consider edge cases and potential errors
4. State your assumptions explicitly
5. Show your work clearly at each stage

Apply these guidelines throughout your reasoning process."""


# =============================================================================
# Types
# =============================================================================


class UIFitnessComponents(TypedDict):
    """Fitness components for UltraInteract evaluation."""

    Coding: float
    Math_CoT: float
    Math_PoT: float
    Logic: float
    aggregate: float


@dataclass
class UIOptimizationResult:
    """Results from UltraInteract GEPA optimization."""

    best_general_prompt: str
    fitness: UIFitnessComponents
    all_candidates: list[dict[str, str]]
    generations_completed: int
    total_evaluations: int
    wall_time_seconds: float
    config: dict[str, Any]
    task_templates: dict[str, str] = field(default_factory=lambda: TASK_TEMPLATES.copy())

    def save(self, path: Path) -> None:
        """Save results to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save best prompt
        (path / "best_general_prompt.txt").write_text(self.best_general_prompt)

        # Save task templates
        (path / "task_templates.json").write_text(
            json.dumps(self.task_templates, indent=2)
        )

        # Save full results
        results = {
            "best_general_prompt": self.best_general_prompt,
            "task_templates": self.task_templates,
            "fitness": dict(self.fitness),
            "generations_completed": self.generations_completed,
            "total_evaluations": self.total_evaluations,
            "wall_time_seconds": self.wall_time_seconds,
            "config": self.config,
            "timestamp": datetime.now().isoformat(),
        }
        (path / "results.json").write_text(json.dumps(results, indent=2))

        # Save composed prompts for each task type
        composed_dir = path / "composed_prompts"
        composed_dir.mkdir(exist_ok=True)
        for task_type in TASK_TEMPLATES:
            composed = self.get_composed_prompt(task_type)
            (composed_dir / f"{task_type}.txt").write_text(composed)

    def get_composed_prompt(self, task_type: str) -> str:
        """Get the full composed prompt for a task type."""
        template = self.task_templates.get(task_type, "")
        return f"{self.best_general_prompt}\n\n{template}"

    @classmethod
    def load(cls, path: Path) -> UIOptimizationResult:
        """Load results from disk."""
        path = Path(path)
        results_data = json.loads((path / "results.json").read_text())

        return cls(
            best_general_prompt=results_data["best_general_prompt"],
            fitness=results_data["fitness"],
            all_candidates=results_data.get("all_candidates", []),
            generations_completed=results_data["generations_completed"],
            total_evaluations=results_data["total_evaluations"],
            wall_time_seconds=results_data["wall_time_seconds"],
            config=results_data["config"],
            task_templates=results_data.get("task_templates", TASK_TEMPLATES),
        )


# =============================================================================
# GEPA Adapter
# =============================================================================


class UltraInteractGEPAAdapter:
    """GEPA adapter that evolves ONLY the general prompt.

    The task-specific templates are fixed and concatenated at inference time.
    This follows the GEPA GEPAAdapter protocol.

    Usage:
        adapter = UltraInteractGEPAAdapter(
            task_weights={"Coding": 0.3, "Math_CoT": 0.25, ...},
        )

        # Use with official GEPA
        import gepa
        result = gepa.optimize(
            seed_candidate={"general_prompt": "..."},
            adapter=adapter,
            trainset=ui_samples,
            valset=val_samples,
        )
    """

    def __init__(
        self,
        task_weights: dict[str, float] | None = None,
        task_templates: dict[str, str] | None = None,
        samples_per_task: int = 25,
        task_lm: str = "openai/gpt-4o-mini",
        cache_dir: Path | None = None,
        max_retries: int = 10,
    ) -> None:
        """Initialize the adapter.

        Args:
            task_weights: Weights for each task type in aggregate score
            task_templates: Optional custom task templates (defaults to TASK_TEMPLATES)
            samples_per_task: Number of samples per task for evaluation
            task_lm: Language model for task evaluation (e.g., "anthropic/claude-opus-4-5-20251101")
            cache_dir: Directory for caching API responses (improves reliability)
            max_retries: Maximum number of retry attempts per API call
        """
        self.task_weights = task_weights or {
            "Coding": 0.30,
            "Math_CoT": 0.25,
            "Math_PoT": 0.25,
            "Logic": 0.20,
        }
        self.task_templates = task_templates or TASK_TEMPLATES.copy()
        self.samples_per_task = samples_per_task
        self.task_lm = task_lm
        self.max_retries = max_retries
        self.logger = logging.getLogger("UltraInteractGEPAAdapter")
        # Required by GEPA's reflective mutation - set to None to use default behavior
        self.propose_new_texts = None

        # Convert litellm model format to safety-tooling format
        # e.g., "anthropic/claude-opus-4-5-20251101" -> "claude-opus-4-5-20251101"
        self.model_id = task_lm.split("/")[-1] if "/" in task_lm else task_lm

        # Initialize safety-tooling InferenceAPI for robust API calls
        self._api = None
        self._cache_dir = cache_dir or Path(".cache/ultrainteract_gepa")
        if HAS_SAFETYTOOLING:
            try:
                safetytooling_utils.setup_environment()
                self._api = InferenceAPI(
                    cache_dir=self._cache_dir,
                    anthropic_num_threads=20,  # Conservative for rate limits
                    print_prompt_and_response=False,
                )
                self.logger.info(f"Initialized safety-tooling InferenceAPI with cache at {self._cache_dir}")
            except Exception as e:
                self.logger.warning(f"Failed to initialize safety-tooling API: {e}. Falling back to litellm.")
                self._api = None
        else:
            self.logger.warning("safety-tooling not installed. Using litellm (no automatic retries).")

    def compose_prompt(
        self,
        general_prompt: str,
        task_type: str | UITaskType,
    ) -> str:
        """Compose the full prompt from general + task-specific template.

        Args:
            general_prompt: The evolved general reasoning prompt
            task_type: The task type (string or UITaskType enum)

        Returns:
            Full composed prompt
        """
        if isinstance(task_type, UITaskType):
            task_type = task_type.value

        template = self.task_templates.get(task_type, "")
        if template:
            return f"{general_prompt}\n\n{template}"
        return general_prompt

    def _generate_response(self, system_prompt: str, user_message: str) -> str:
        """Generate a response using safety-tooling API with retries, or fallback to litellm.

        Args:
            system_prompt: The system prompt (general + task template)
            user_message: The user's question

        Returns:
            Model response string, or empty string on failure
        """
        if self._api is not None:
            try:
                # Use safety-tooling's InferenceAPI with automatic retries
                prompt = Prompt(messages=[
                    ChatMessage(role=MessageRole.system, content=system_prompt),
                    ChatMessage(role=MessageRole.user, content=user_message),
                ])

                async def _call_api():
                    responses = await self._api(
                        model_id=self.model_id,
                        prompt=prompt,
                        max_attempts_per_api_call=self.max_retries,
                        temperature=0.0,
                        max_tokens=4096,
                    )
                    return responses[0].completion if responses else ""

                # Run async call - handle both sync and async contexts
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    loop = None

                if loop is None:
                    # No running loop, safe to use asyncio.run()
                    return asyncio.run(_call_api())
                else:
                    # Already in async context - use nest_asyncio or run in thread
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                        future = executor.submit(asyncio.run, _call_api())
                        return future.result(timeout=600)

            except Exception as e:
                self.logger.warning(f"Safety-tooling API failed after retries: {e}")
                return ""
        else:
            # Fallback to litellm (no automatic retries)
            try:
                import litellm
                completion = litellm.completion(
                    model=self.task_lm,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message},
                    ],
                    temperature=0.0,
                )
                return completion.choices[0].message.content or ""
            except Exception as e:
                self.logger.warning(f"Generation failed: {e}")
                return ""

    def evaluate(
        self,
        batch: list[dict[str, Any]],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> dict[str, Any]:
        """Evaluate candidate on batch (GEPA protocol).

        Args:
            batch: List of samples in GEPA format
            candidate: Dict with "general_prompt" key
            capture_traces: Whether to capture execution traces

        Returns:
            EvaluationBatch-like dict with outputs, scores, trajectories
        """
        try:
            import gepa
            has_gepa = True
        except ImportError:
            has_gepa = False

        general_prompt = candidate.get("general_prompt", DEFAULT_SEED_PROMPT)

        outputs = []
        scores = []
        trajectories = []
        objective_scores = []

        for sample in batch:
            task_type = sample.get("additional_context", {}).get("task_type", "Coding")
            question = sample.get("input", "")
            expected = sample.get("answer", "")

            # Compose prompt
            full_prompt = self.compose_prompt(general_prompt, task_type)

            # Generate response using safety-tooling API (with retries) or fallback to litellm
            response = self._generate_response(full_prompt, question)

            # Check correctness
            is_correct, feedback = self._check_answer(response, expected, task_type)
            score = 1.0 if is_correct else 0.0

            outputs.append({"response": response})
            scores.append(score)

            if capture_traces:
                trajectories.append({
                    "input": question,
                    "expected": expected,
                    "response": response,
                    "task_type": task_type,
                    "correct": is_correct,
                    "feedback": feedback,
                })

            # Per-objective scores
            obj_scores = {task_type: score}
            objective_scores.append(obj_scores)

        # Return GEPA EvaluationBatch object
        try:
            from gepa import EvaluationBatch
            return EvaluationBatch(
                outputs=outputs,
                scores=scores,
                trajectories=trajectories if capture_traces else None,
            )
        except ImportError:
            # Fallback for testing without GEPA
            return type('EvaluationBatch', (), {
                'outputs': outputs,
                'scores': scores,
                'trajectories': trajectories if capture_traces else None,
            })()

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: dict[str, Any],
        components_to_update: list[str],
    ) -> dict[str, list[dict[str, Any]]]:
        """Create reflective dataset for GEPA mutation.

        Groups feedback by task type for targeted reflection.

        Args:
            candidate: Current candidate
            eval_batch: Results from evaluate()
            components_to_update: List of components (always ["general_prompt"])

        Returns:
            Dict mapping component name to list of feedback dicts
        """
        # Handle both dict and EvaluationBatch objects
        if hasattr(eval_batch, 'trajectories'):
            trajectories = eval_batch.trajectories or []
        else:
            trajectories = eval_batch.get("trajectories", [])
        if not trajectories:
            return {"general_prompt": []}

        # Group by task type
        by_task: dict[str, list[dict]] = defaultdict(list)
        for trace in trajectories:
            task_type = trace.get("task_type", "unknown")
            by_task[task_type].append(trace)

        # Build feedback for each task type
        feedback_items = []

        for task_type, traces in by_task.items():
            correct_count = sum(1 for t in traces if t.get("correct", False))
            total = len(traces)
            accuracy = correct_count / total if total > 0 else 0

            # Include failed examples
            failures = [t for t in traces if not t.get("correct", False)][:3]

            for failure in failures:
                feedback_items.append({
                    "Inputs": f"[{task_type}] {failure.get('input', '')[:300]}",
                    "Generated Outputs": failure.get("response", "")[:500],
                    "Feedback": f"{task_type} accuracy: {accuracy:.0%}. {failure.get('feedback', 'Incorrect')}",
                })

        return {"general_prompt": feedback_items}

    def _check_answer(
        self,
        response: str,
        expected: str,
        task_type: str,
    ) -> tuple[bool, str]:
        """Check if response contains correct answer.

        Uses sympy-based mathematical equivalence for math tasks,
        and semantic matching for logic tasks.

        Args:
            response: Model's response
            expected: Expected answer
            task_type: Task type for context-aware checking

        Returns:
            Tuple of (is_correct, feedback)
        """
        if not response or not expected:
            return False, "Empty response or expected answer"

        response_lower = response.lower()
        expected_lower = expected.lower().strip()

        if task_type in ["Math_CoT", "Math_PoT"]:
            # Use proper sympy-based mathematical equivalence checking
            is_correct, feedback = grade_math_answer(response, expected, tolerance=1e-4)
            return is_correct, feedback

        elif task_type == "Coding":
            # For coding, we need execution-based evaluation for full correctness.
            # Since we don't have sandbox execution yet, use a more conservative approach:
            # - Only mark as correct if the code structure and key elements match
            # - Don't give credit just for "any valid code"

            # Extract code block from response
            code_match = re.search(r"```(?:python)?\s*\n?(.*?)```", response, re.DOTALL)
            response_code = code_match.group(1) if code_match else response

            # Check for key Python constructs
            has_def = bool(re.search(r'\bdef\s+\w+\s*\(', response_code))
            has_class = bool(re.search(r'\bclass\s+\w+', response_code))
            has_code = has_def or has_class or bool(
                re.search(r'\b(for|while|if|return)\b', response_code)
            )

            if not has_code:
                return False, "No valid code found"

            # Expected is a code block - check structural similarity
            expected_code = expected
            if expected_code.startswith("```"):
                expected_match = re.search(r"```(?:python)?\s*\n?(.*?)```", expected_code, re.DOTALL)
                expected_code = expected_match.group(1) if expected_match else expected_code

            # If expected starts with "# Step" comments, it's step descriptions, not real code
            # In this case, check if response has meaningful code structure
            if expected_lower.startswith("# step"):
                # Need at least a function definition for coding tasks
                if has_def:
                    return True, "Valid function defined"
                return False, "Expected function definition"

            # Extract function names from both
            expected_funcs = set(re.findall(r'def\s+(\w+)', expected_code.lower()))
            response_funcs = set(re.findall(r'def\s+(\w+)', response_code.lower()))

            # Function name match is a strong signal
            if expected_funcs:
                matching_funcs = expected_funcs & response_funcs
                if matching_funcs:
                    # Check if function bodies have similar structure
                    # Look for key algorithm patterns
                    expected_patterns = self._extract_code_patterns(expected_code)
                    response_patterns = self._extract_code_patterns(response_code)

                    pattern_overlap = len(expected_patterns & response_patterns)
                    if pattern_overlap >= len(expected_patterns) * 0.3:
                        return True, f"Functions match: {matching_funcs}"

            # Token-based similarity (excluding common keywords)
            common_keywords = {
                'def', 'return', 'if', 'else', 'elif', 'for', 'in', 'while',
                'import', 'from', 'class', 'self', 'none', 'true', 'false',
                'and', 'or', 'not', 'the', 'a', 'is', 'to', 'with', 'as',
                'try', 'except', 'finally', 'raise', 'pass', 'break', 'continue',
                'lambda', 'yield', 'global', 'nonlocal', 'assert', 'del',
            }
            expected_tokens = set(re.findall(r"\b[a-zA-Z_]\w*\b", expected_code.lower()))
            response_tokens = set(re.findall(r"\b[a-zA-Z_]\w*\b", response_code.lower()))
            expected_tokens -= common_keywords
            response_tokens -= common_keywords

            if expected_tokens:
                overlap = len(expected_tokens & response_tokens) / len(expected_tokens)
                # Require higher overlap (60%) for coding correctness
                if overlap >= 0.6:
                    return True, f"Code similarity: {overlap:.0%}"

            return False, "Code structure differs from expected"

        else:  # Logic
            # Extract answer from model response
            response_answer = None
            answer_patterns = [
                r"(?:the\s+)?answer\s+is[:\s]+(.+?)(?:\.|$)",
                r"Answer:\s*(.+?)$",
                r"(?:therefore|thus|so)[,\s]+(.+?)(?:\.|$)",
                r"(?:In\s+)?conclusion[,:\s]+(.+?)(?:\.|$)",
            ]
            for pattern in answer_patterns:
                match = re.search(pattern, response, re.IGNORECASE | re.MULTILINE)
                if match:
                    response_answer = match.group(1).strip().lower()
                    break

            # Normalize expected answer
            expected_clean = re.sub(r'["\']', '', expected_lower).strip()

            # Direct match
            if expected_clean in response_lower:
                return True, "Correct answer found"

            if response_answer:
                response_answer_clean = re.sub(r'["\']', '', response_answer).strip()

                # Substring match
                if expected_clean in response_answer_clean or response_answer_clean in expected_clean:
                    return True, "Answer matches"

                # Key entity overlap for logic answers (names, places, numbers)
                expected_entities = set(re.findall(r'\b[A-Z][a-z]+\b|\b\d+\b', expected))
                response_entities = set(re.findall(r'\b[A-Z][a-z]+\b|\b\d+\b', response))
                if expected_entities:
                    entity_overlap = len(expected_entities & response_entities) / len(expected_entities)
                    if entity_overlap >= 0.7:
                        return True, f"Entity match: {entity_overlap:.0%}"

                # Word overlap (for short answers)
                expected_words = set(expected_clean.split())
                response_words = set(response_answer_clean.split())
                # Remove common words
                stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'dare', 'ought', 'used', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'between', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'each', 'every', 'both', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 'just', 'also'}
                expected_words -= stopwords
                response_words -= stopwords

                if expected_words and len(expected_words) <= 5:
                    word_overlap = len(expected_words & response_words) / len(expected_words)
                    if word_overlap >= 0.6:
                        return True, f"Answer words match: {word_overlap:.0%}"

            # Whitespace-normalized match
            expected_normalized = re.sub(r"\s+", "", expected_clean)
            response_normalized = re.sub(r"\s+", "", response_lower)
            if expected_normalized and expected_normalized in response_normalized:
                return True, "Answer found (normalized)"

            return False, f"Expected '{expected[:50]}...' not found"

    def _extract_code_patterns(self, code: str) -> set[str]:
        """Extract algorithmic patterns from code for structural comparison."""
        patterns = set()

        # Loop patterns
        if re.search(r'for\s+\w+\s+in\s+range', code):
            patterns.add("range_loop")
        if re.search(r'for\s+\w+\s+in\s+\w+', code):
            patterns.add("iteration")
        if re.search(r'while\s+', code):
            patterns.add("while_loop")

        # Data structure patterns
        if re.search(r'\[\s*\]|\[\s*\w', code):
            patterns.add("list")
        if re.search(r'\{\s*\}|\{\s*\w', code):
            patterns.add("dict_or_set")
        if re.search(r'\.append\(', code):
            patterns.add("append")
        if re.search(r'\.pop\(', code):
            patterns.add("pop")

        # Control flow
        if re.search(r'if\s+.+:', code):
            patterns.add("conditional")
        if re.search(r'return\s+', code):
            patterns.add("return")
        if re.search(r'try\s*:', code):
            patterns.add("exception_handling")

        # Algorithm patterns
        if re.search(r'sorted\(|\.sort\(', code):
            patterns.add("sorting")
        if re.search(r'min\(|max\(', code):
            patterns.add("min_max")
        if re.search(r'sum\(', code):
            patterns.add("sum")
        if re.search(r'len\(', code):
            patterns.add("length")
        if re.search(r'%\s*\d|//\s*\d', code):
            patterns.add("modulo_division")
        # Check for recursive patterns (function calling itself)
        func_names = re.findall(r'def\s+(\w+)\s*\(', code)
        for func_name in func_names:
            if re.search(rf'{func_name}\s*\(', code[code.find(f'def {func_name}'):]):
                patterns.add("recursion")
                break

        return patterns


# =============================================================================
# Evolution Runner
# =============================================================================


async def run_ultrainteract_evolution(
    trainset: list[UltraInteractSample] | None = None,
    valset: list[UltraInteractSample] | None = None,
    task_lm: str = "openai/gpt-4o-mini",
    reflection_lm: str = "openai/gpt-4o",
    max_metric_calls: int = 3000,
    task_weights: dict[str, float] | None = None,
    seed_prompt: str | None = None,
    train_size: int = 1000,
    val_size: int = 200,
    output_dir: Path | None = None,
    use_wandb: bool = False,
    seed: int = 42,
    **kwargs: Any,
) -> UIOptimizationResult:
    """Run GEPA evolution on UltraInteract dataset.

    This is the main entry point for UltraInteract prompt evolution.

    Args:
        trainset: Training samples (loaded if not provided)
        valset: Validation samples (loaded if not provided)
        task_lm: Model for task evaluation
        reflection_lm: Model for reflection/mutation
        max_metric_calls: Budget for evaluations
        task_weights: Per-task weights for aggregate fitness
        seed_prompt: Initial general prompt
        train_size: Training set size (if loading)
        val_size: Validation set size (if loading)
        output_dir: Directory to save results
        use_wandb: Enable W&B logging
        seed: Random seed
        **kwargs: Additional GEPA parameters

    Returns:
        UIOptimizationResult with best prompt and metrics
    """
    start_time = time.time()
    logger = logging.getLogger("UIEvolution")

    # Load data if not provided
    if trainset is None or valset is None:
        logger.info(f"Loading UltraInteract dataset (train={train_size}, val={val_size})")
        loader = UltraInteractLoader(seed=seed)
        splits = loader.load_stratified(
            train_size=train_size,
            val_size=val_size,
            test_size=0,
        )
        trainset = splits["train"]
        valset = splits["val"]

    logger.info(f"Dataset: {len(trainset)} train, {len(valset)} val samples")

    # Log task distribution
    train_dist = UltraInteractLoader(seed=seed).get_task_distribution(trainset)
    logger.info(f"Task distribution: {train_dist}")

    # Convert to GEPA format
    train_data = create_gepa_dataset(trainset)
    val_data = create_gepa_dataset(valset)

    # Create adapter
    adapter = UltraInteractGEPAAdapter(task_weights=task_weights, task_lm=task_lm)

    # Seed candidate
    seed_candidate = {"general_prompt": seed_prompt or DEFAULT_SEED_PROMPT}

    # Check if GEPA is installed
    try:
        import gepa
        has_gepa = True
    except ImportError:
        has_gepa = False
        logger.warning("GEPA not installed. Using mock evolution.")

    if has_gepa:
        logger.info(f"Starting GEPA optimization: budget={max_metric_calls}")

        # Filter kwargs to only include valid GEPA parameters
        valid_gepa_kwargs = {
            "candidate_selection_strategy",
            "skip_perfect_score",
            "batch_sampler",
            "reflection_minibatch_size",
            "perfect_score",
            "reflection_prompt_template",
            "module_selector",
            "use_merge",
            "max_merge_invocations",
            "merge_val_overlap_floor",
            "stop_callbacks",
            "logger",
            "run_dir",
            "wandb_api_key",
            "wandb_init_kwargs",
            "use_mlflow",
            "mlflow_tracking_uri",
            "mlflow_experiment_name",
            "track_best_outputs",
            "display_progress_bar",
            "use_cloudpickle",
            "raise_on_exception",
            "val_evaluation_policy",
        }
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_gepa_kwargs}

        # Run GEPA - don't pass task_lm when using adapter (adapter handles evaluation)
        gepa_result = gepa.optimize(
            seed_candidate=seed_candidate,
            trainset=train_data,
            valset=val_data,
            adapter=adapter,
            task_lm=None,  # Adapter handles task evaluation
            reflection_lm=reflection_lm,
            max_metric_calls=max_metric_calls,
            use_wandb=use_wandb,
            seed=seed,
            display_progress_bar=True,
            **filtered_kwargs,
        )

        # Get best candidate (highest validation score)
        all_candidates = list(gepa_result.candidates)
        scores = list(gepa_result.val_aggregate_scores)
        if scores:
            best_idx = scores.index(max(scores))
            best_prompt = all_candidates[best_idx].get("general_prompt", seed_candidate["general_prompt"])
        else:
            best_prompt = seed_candidate["general_prompt"]
        generations = gepa_result.num_full_val_evals or 0
        total_evals = gepa_result.total_metric_calls or max_metric_calls

    else:
        # Mock evolution for testing
        logger.info("Running mock evolution (GEPA not installed)")
        best_prompt = seed_candidate["general_prompt"]
        all_candidates = [seed_candidate]
        generations = 0
        total_evals = 0

    # Evaluate final fitness
    logger.info("Evaluating final fitness on validation set")
    fitness = await _evaluate_fitness(
        best_prompt,
        val_data[:100],  # Subset for final eval
        adapter,
        task_lm,
    )

    wall_time = time.time() - start_time

    result = UIOptimizationResult(
        best_general_prompt=best_prompt,
        fitness=fitness,
        all_candidates=all_candidates,
        generations_completed=generations,
        total_evaluations=total_evals,
        wall_time_seconds=wall_time,
        config={
            "task_lm": task_lm,
            "reflection_lm": reflection_lm,
            "max_metric_calls": max_metric_calls,
            "task_weights": task_weights or adapter.task_weights,
            "train_size": len(trainset),
            "val_size": len(valset),
            "seed": seed,
        },
    )

    if output_dir:
        result.save(Path(output_dir))
        logger.info(f"Results saved to {output_dir}")

    logger.info(
        f"Evolution complete: {wall_time:.1f}s, "
        f"aggregate fitness={fitness['aggregate']:.3f}"
    )

    return result


async def _evaluate_fitness(
    general_prompt: str,
    samples: list[dict[str, Any]],
    adapter: UltraInteractGEPAAdapter,
    task_lm: str,
) -> UIFitnessComponents:
    """Evaluate fitness on samples grouped by task type."""
    # Group by task type
    by_task: dict[str, list[dict]] = defaultdict(list)
    for sample in samples:
        task_type = sample.get("additional_context", {}).get("task_type", "Coding")
        by_task[task_type].append(sample)

    # Evaluate each task
    task_scores: dict[str, float] = {}

    for task_type, task_samples in by_task.items():
        candidate = {"general_prompt": general_prompt}
        eval_result = adapter.evaluate(task_samples, candidate, capture_traces=False)
        # Handle both dict and EvaluationBatch objects
        if hasattr(eval_result, 'scores'):
            scores = eval_result.scores
        else:
            scores = eval_result.get("scores", [])
        task_scores[task_type] = np.mean(scores) if scores else 0.0

    # Compute aggregate
    aggregate = sum(
        task_scores.get(task, 0.0) * weight
        for task, weight in adapter.task_weights.items()
    )

    return UIFitnessComponents(
        Coding=task_scores.get("Coding", 0.0),
        Math_CoT=task_scores.get("Math_CoT", 0.0),
        Math_PoT=task_scores.get("Math_PoT", 0.0),
        Logic=task_scores.get("Logic", 0.0),
        aggregate=aggregate,
    )


# =============================================================================
# Convenience Functions
# =============================================================================


def get_composed_prompt(
    general_prompt: str,
    task_type: str,
    task_templates: dict[str, str] | None = None,
) -> str:
    """Get composed prompt for a specific task type.

    Args:
        general_prompt: The general reasoning prompt
        task_type: Task type (Coding, Math_CoT, Math_PoT, Logic)
        task_templates: Optional custom templates

    Returns:
        Full composed prompt
    """
    templates = task_templates or TASK_TEMPLATES
    template = templates.get(task_type, "")
    if template:
        return f"{general_prompt}\n\n{template}"
    return general_prompt


def load_evolved_prompt(path: Path) -> tuple[str, dict[str, str]]:
    """Load evolved prompt and task templates from disk.

    Args:
        path: Path to results directory

    Returns:
        Tuple of (general_prompt, task_templates)
    """
    path = Path(path)

    general_prompt = (path / "best_general_prompt.txt").read_text()

    templates_path = path / "task_templates.json"
    if templates_path.exists():
        task_templates = json.loads(templates_path.read_text())
    else:
        task_templates = TASK_TEMPLATES.copy()

    return general_prompt, task_templates
