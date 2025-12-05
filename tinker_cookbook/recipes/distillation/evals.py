"""
Evaluation helpers for extrapolation distillation.

Provides math benchmarks (AIME, HMMT, Bruno) that can be plugged into the
distillation pipeline as SamplingClientEvaluator builders.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, Sequence

from datasets import Dataset, load_dataset

from tinker import types
from tinker_cookbook import renderers
from tinker_cookbook.eval.evaluators import SamplingClientEvaluator, SamplingClientEvaluatorBuilder
from tinker_cookbook.recipes.math_rl.math_env import MathEnv, safe_grade
from tinker_cookbook.recipes.math_rl.math_grading import extract_boxed
from tinker_cookbook.tokenizer_utils import get_tokenizer

logger = logging.getLogger(__name__)


@dataclass
class MathEvalExample:
    prompt_text: str
    answer: str
    uid: str


class MathEvaluator(SamplingClientEvaluator):
    """Evaluate accuracy on a set of boxed-answer math prompts."""

    def __init__(
        self,
        *,
        name: str,
        examples: Sequence[MathEvalExample],
        renderer_name: str,
        model_name_for_tokenizer: str,
        max_tokens: int,
        grader: str = "sympy",
        timeout: float = 1.0,
        convo_prefix: list[renderers.Message] | None = None,
        num_generations: int = 1,
    ):
        self.name = name
        self.examples = list(examples)
        tokenizer = get_tokenizer(model_name_for_tokenizer)
        self.renderer = renderers.get_renderer(renderer_name, tokenizer)
        self.max_tokens = max_tokens
        self.grader = grader
        self.timeout = timeout
        self.convo_prefix = [dict(msg) for msg in convo_prefix] if convo_prefix else []
        self.num_generations = max(1, num_generations)

    async def __call__(self, sampling_client) -> dict[str, float]:
        if not self.examples:
            logger.warning("Evaluator '%s' has no examples; skipping.", self.name)
            return {}

        correct = 0
        formatted = 0
        skipped = 0

        for example in self.examples:
            messages: list[renderers.Message] = [dict(msg) for msg in self.convo_prefix]
            messages.append({"role": "user", "content": example.prompt_text})
            model_input = self.renderer.build_generation_prompt(messages)
            sampling_params = types.SamplingParams(
                temperature=1.0,
                max_tokens=self.max_tokens,
                stop=self.renderer.get_stop_sequences(),
            )
            try:
                response = await sampling_client.sample_async(
                    model_input,
                    num_samples=self.num_generations,
                    sampling_params=sampling_params,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Sampling failed for %s: %s", example.uid, exc)
                skipped += 1
                continue

            example_formatted = False
            example_correct = False

            for sequence in response.sequences:
                parsed_message, _ = self.renderer.parse_response(sequence.tokens)
                content = parsed_message.get("content", "")
                parsed_answer = content.strip()
                formatted_current = False
                try:
                    parsed_answer = extract_boxed(content)
                    formatted_current = True
                except ValueError:
                    formatted_current = False

                if formatted_current:
                    example_formatted = True

                if safe_grade(parsed_answer, example.answer, grader=self.grader, timeout=self.timeout):
                    example_correct = True

            if example_formatted:
                formatted += 1
            if example_correct:
                correct += 1

        total = len(self.examples)
        accuracy = correct / total if total else 0.0
        format_rate = formatted / total if total else 0.0
        metrics = {
            f"{self.name}/accuracy": accuracy,
            f"{self.name}/format_rate": format_rate,
            f"{self.name}/total": float(total),
        }
        metrics[f"{self.name}/generations_per_prompt"] = float(self.num_generations)
        metrics[f"{self.name}/skipped"] = float(skipped)
        return metrics


@dataclass
class DatasetSource:
    dataset: str
    subset: str | None
    split: str


@dataclass
class DatasetSpec:
    name: str
    sources: Sequence[DatasetSource]
    problem_keys: Sequence[str]
    answer_keys: Sequence[str]


def _coerce_to_str(value) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float)):
        return str(value).strip()
    if isinstance(value, list):
        for item in value:
            coerced = _coerce_to_str(item)
            if coerced:
                return coerced
    return None


def _extract_field(row: dict, keys: Iterable[str]) -> str | None:
    for key in keys:
        if key in row:
            value = _coerce_to_str(row[key])
            if value:
                return value
    return None


def _load_dataset_from_sources(spec: DatasetSpec) -> Dataset | None:
    last_error: Exception | None = None
    for source in spec.sources:
        try:
            ds = load_dataset(source.dataset, name=source.subset, split=source.split)
            if isinstance(ds, Dataset):
                logger.info(
                    "Loaded %s dataset from %s (subset=%s, split=%s).",
                    spec.name,
                    source.dataset,
                    source.subset,
                    source.split,
                )
                return ds
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            logger.warning(
                "Failed to load %s dataset from %s (subset=%s, split=%s): %s",
                spec.name,
                source.dataset,
                source.subset,
                source.split,
                exc,
            )
    if last_error is not None:
        logger.warning("Skipping %s evaluation; no dataset source succeeded.", spec.name)
    return None


def build_math_examples_from_spec(
    spec: DatasetSpec,
    *,
    limit: int | None = None,
) -> list[MathEvalExample]:
    dataset = _load_dataset_from_sources(spec)
    if dataset is None:
        return []

    examples: list[MathEvalExample] = []
    for idx, row in enumerate(dataset):
        problem = _extract_field(row, spec.problem_keys)
        answer = _extract_field(row, spec.answer_keys)
        if not problem or not answer:
            continue
        prompt_text = f"{problem.strip()}{MathEnv.question_suffix()}"
        examples.append(
            MathEvalExample(
                prompt_text=prompt_text,
                answer=answer.strip(),
                uid=f"{spec.name}-{idx}",
            )
        )
        if limit is not None and len(examples) >= limit:
            break

    logger.info("Prepared %d %s evaluation examples.", len(examples), spec.name)
    return examples


AIME_SPEC = DatasetSpec(
    name="aime25",
    sources=[
        DatasetSource("MathArena/aime_2025", None, "train"),
    ],
    problem_keys=("problem", "question", "prompt", "input"),
    answer_keys=("answer", "final_answer", "target", "label"),
)

HMMT_SPEC = DatasetSpec(
    name="hmmt_feb25",
    sources=[
        DatasetSource("MathArena/hmmt_feb_2025", None, "train"),
    ],
    problem_keys=("problem", "question", "prompt", "input"),
    answer_keys=("answer", "final_answer", "target", "label"),
)

BRUMO_SPEC = DatasetSpec(
    name="brumo25",
    sources=[
        DatasetSource("MathArena/brumo_2025", None, "train"),
    ],
    problem_keys=("problem", "question", "prompt", "input"),
    answer_keys=("answer", "final_answer", "target", "label"),
)


def build_math_eval_builder(
    *,
    name: str,
    examples: Sequence[MathEvalExample],
    renderer_name: str,
    model_name_for_tokenizer: str,
    max_tokens: int,
    grader: str,
    num_generations: int,
) -> SamplingClientEvaluatorBuilder:
    def _builder() -> MathEvaluator:
        return MathEvaluator(
            name=name,
            examples=examples,
            renderer_name=renderer_name,
            model_name_for_tokenizer=model_name_for_tokenizer,
            max_tokens=max_tokens,
            grader=grader,
            convo_prefix=MathEnv.standard_fewshot_prefix(),
            num_generations=num_generations,
        )

    return _builder


def get_math_evaluator_builders(
    *,
    renderer_name: str,
    model_name_for_tokenizer: str,
    max_tokens: int,
    aime_limit: int | None = None,
    hmmt_limit: int | None = None,
    brumo_limit: int | None = None,
    grader: str = "sympy",
    num_generations: int = 1,
) -> list[SamplingClientEvaluatorBuilder]:
    builders: list[SamplingClientEvaluatorBuilder] = []

    aime_examples = build_math_examples_from_spec(AIME_SPEC, limit=aime_limit)
    if aime_examples:
        builders.append(
            build_math_eval_builder(
                name=AIME_SPEC.name,
                examples=aime_examples,
                renderer_name=renderer_name,
                model_name_for_tokenizer=model_name_for_tokenizer,
                max_tokens=max_tokens,
                grader=grader,
                num_generations=num_generations,
            )
        )
    else:
        logger.warning("Skipping AIME evaluation; no examples were prepared.")

    hmmt_examples = build_math_examples_from_spec(HMMT_SPEC, limit=hmmt_limit)
    if hmmt_examples:
        builders.append(
            build_math_eval_builder(
                name=HMMT_SPEC.name,
                examples=hmmt_examples,
                renderer_name=renderer_name,
                model_name_for_tokenizer=model_name_for_tokenizer,
                max_tokens=max_tokens,
                grader=grader,
                num_generations=num_generations,
            )
        )
    else:
        logger.warning("Skipping HMMT evaluation; no examples were prepared.")

    brumo_examples = build_math_examples_from_spec(BRUMO_SPEC, limit=brumo_limit)
    if brumo_examples:
        builders.append(
            build_math_eval_builder(
                name=BRUMO_SPEC.name,
                examples=brumo_examples,
                renderer_name=renderer_name,
                model_name_for_tokenizer=model_name_for_tokenizer,
                max_tokens=max_tokens,
                grader=grader,
                num_generations=num_generations,
            )
        )
    else:
        logger.warning("Skipping Brumo evaluation; no examples were prepared.")

    return builders
