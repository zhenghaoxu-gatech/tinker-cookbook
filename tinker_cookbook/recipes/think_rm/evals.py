"""Evaluation helpers for Think RM training."""

from __future__ import annotations

import logging
from random import Random
from typing import Literal, Sequence

from datasets import Dataset, load_dataset
from tinker_cookbook.completers import TinkerMessageCompleter
from tinker_cookbook.eval.evaluators import SamplingClientEvaluator, SamplingClientEvaluatorBuilder
from tinker_cookbook import renderers
from tinker_cookbook.recipes.think_rm.env import ThinkRMExample, build_prompt_text
from tinker_cookbook.recipes.think_rm.parsing import parse_preference
from tinker_cookbook.tokenizer_utils import get_tokenizer

logger = logging.getLogger(__name__)

REWARDBENCH_DATASET = "allenai/reward-bench-2"
RMBENCH_DATASET = "THU-KEG/RM-Bench"
HELPSTEER3_DATASET = ("nvidia/HelpSteer3", "preference")

LABEL_SWAP = {
    "response_1": "response_2",
    "response_2": "response_1",
    "tie": "tie",
}


def _random_flip(
    response_a: str,
    response_b: str,
    label: Literal["response_1", "response_2", "tie"],
    rng: Random,
) -> tuple[str, str, Literal["response_1", "response_2", "tie"]]:
    if rng.random() < 0.5:
        return response_a, response_b, label
    return response_b, response_a, LABEL_SWAP[label]


def build_rewardbench_examples(
    *,
    split: str = "test",
    seed: int = 0,
    max_examples: int | None = None,
    num_proc: int | None = 8,
) -> list[ThinkRMExample]:
    load_kwargs = {}
    if num_proc is not None and num_proc > 1:
        load_kwargs["num_proc"] = num_proc
    dataset = load_dataset(REWARDBENCH_DATASET, split=split, num_proc=8)
    if not isinstance(dataset, Dataset):
        raise TypeError(f"Expected datasets.Dataset for RewardBench split '{split}', got {type(dataset)}.")

    rng = Random(seed)
    examples: list[ThinkRMExample] = []
    for row_idx, row in enumerate(dataset):
        prompt = row.get("prompt")
        chosen_list = row.get("chosen") or []
        rejected_list = row.get("rejected") or []

        if not isinstance(prompt, str):
            continue
        chosen_candidates = [c for c in chosen_list if isinstance(c, str) and c.strip()]
        rejected_candidates = [r for r in rejected_list if isinstance(r, str) and r.strip()]
        if not chosen_candidates or not rejected_candidates:
            continue

        for chosen_idx, preferred in enumerate(chosen_candidates):
            for rejected_idx, dispreferred in enumerate(rejected_candidates):
                response_a, response_b, label = _random_flip(preferred, dispreferred, "response_1", rng)
                prompt_text = build_prompt_text(
                    [{"role": "user", "content": prompt}],
                    response_a,
                    response_b,
                )
                examples.append(
                    ThinkRMExample(
                        prompt_text=prompt_text,
                        ground_truth=label,
                        uid=f"rewardbench-{row_idx}-c{chosen_idx}-r{rejected_idx}",
                        data_source="rewardbench2",
                    )
                )
                if max_examples is not None and len(examples) >= max_examples:
                    break
            if max_examples is not None and len(examples) >= max_examples:
                break
        if max_examples is not None and len(examples) >= max_examples:
            break

    logger.info("Loaded %d RewardBench examples (seed=%d).", len(examples), seed)
    return examples


def build_rmbench_examples(
    *,
    split: str | None = None,
    seed: int = 0,
    max_examples: int | None = None,
    num_proc: int | None = 8,
) -> list[ThinkRMExample]:
    actual_split = split or "train"
    load_kwargs = {}
    if num_proc is not None and num_proc > 1:
        load_kwargs["num_proc"] = num_proc
    dataset = load_dataset(RMBENCH_DATASET, split=actual_split, num_proc=8)
    if not isinstance(dataset, Dataset):
        raise TypeError(f"Expected datasets.Dataset for RM-Bench split '{actual_split}', got {type(dataset)}.")

    rng = Random(seed)
    examples: list[ThinkRMExample] = []
    for row_idx, row in enumerate(dataset):
        prompt = row.get("prompt")
        chosen_list = row.get("chosen") or []
        rejected_list = row.get("rejected") or []
        if not isinstance(prompt, str):
            continue

        chosen_candidates = [c for c in chosen_list if isinstance(c, str) and c.strip()]
        rejected_candidates = [c for c in rejected_list if isinstance(c, str) and c.strip()]
        if not chosen_candidates or not rejected_candidates:
            continue

        for chosen_idx, preferred in enumerate(chosen_candidates):
            for rejected_idx, dispreferred in enumerate(rejected_candidates):
                response_a, response_b, label = _random_flip(preferred, dispreferred, "response_1", rng)
                prompt_text = build_prompt_text(
                    [{"role": "user", "content": prompt}],
                    response_a,
                    response_b,
                )
                examples.append(
                    ThinkRMExample(
                        prompt_text=prompt_text,
                        ground_truth=label,
                        uid=f"rmbench-{row_idx}-c{chosen_idx}-r{rejected_idx}",
                        data_source="rmbench",
                    )
                )
                if max_examples is not None and len(examples) >= max_examples:
                    break
            if max_examples is not None and len(examples) >= max_examples:
                break
        if max_examples is not None and len(examples) >= max_examples:
            break

    logger.info("Loaded %d RM-Bench examples (split=%s, seed=%d).", len(examples), actual_split, seed)
    return examples


def build_helpsteer3_examples(
    *,
    split: str = "validation",
    seed: int = 0,
    max_examples: int | None = None,
    num_proc: int | None = 8,
) -> list[ThinkRMExample]:
    load_kwargs = {}
    if num_proc is not None and num_proc > 1:
        load_kwargs["num_proc"] = num_proc
    dataset = load_dataset(*HELPSTEER3_DATASET, split=split, num_proc=8)
    if not isinstance(dataset, Dataset):
        raise TypeError(f"Expected datasets.Dataset for HelpSteer3 split '{split}', got {type(dataset)}.")

    rng = Random(seed)
    examples: list[ThinkRMExample] = []
    for row_idx, row in enumerate(dataset):
        context = row.get("context")
        response1 = row.get("response1")
        response2 = row.get("response2")
        overall_preference = row.get("overall_preference")

        if not isinstance(response1, str) or not isinstance(response2, str):
            continue

        if isinstance(overall_preference, (int, float)):
            if overall_preference < 0:
                label = "response_1"
            elif overall_preference > 0:
                label = "response_2"
            else:
                label = "tie"
        else:
            continue

        if isinstance(context, list):
            context_messages = [
                msg
                for msg in context
                if isinstance(msg, dict)
                and isinstance(msg.get("role"), str)
                and isinstance(msg.get("content"), str)
            ]
        else:
            context_messages = [{"role": "user", "content": ""}]

        response_a, response_b, flipped_label = _random_flip(response1, response2, label, rng)
        prompt_text = build_prompt_text(context_messages, response_a, response_b)
        examples.append(
            ThinkRMExample(
                prompt_text=prompt_text,
                ground_truth=flipped_label,
                uid=f"helpsteer3-{row_idx}",
                data_source="helpsteer3",
            )
        )
        if max_examples is not None and len(examples) >= max_examples:
            break

    logger.info("Loaded %d HelpSteer3 examples (seed=%d).", len(examples), seed)
    return examples


class ThinkRMEvaluator(SamplingClientEvaluator):
    """Run inference on Think RM prompts and compute accuracy."""

    def __init__(
        self,
        *,
        name: str,
        examples: Sequence[ThinkRMExample],
        renderer_name: str,
        model_name_for_tokenizer: str,
        max_tokens: int = 1024,
        expect_think_tags: bool = True,
    ):
        self.name = name
        self.examples = list(examples)
        tokenizer = get_tokenizer(model_name_for_tokenizer)
        self.renderer = renderers.get_renderer(renderer_name, tokenizer)
        self.max_tokens = max_tokens
        self.expect_think_tags = expect_think_tags

    async def __call__(self, sampling_client) -> dict[str, float]:
        if not self.examples:
            logger.warning("Evaluator '%s' has no examples; returning empty metrics.", self.name)
            return {}

        completer = TinkerMessageCompleter(sampling_client, self.renderer, self.max_tokens)
        correct = 0
        formatted = 0
        for example in self.examples:
            completion = await completer(example.conversation())
            content = completion.get("content", "")
            parsed = parse_preference(
                content,
                from_thinking_model=self.expect_think_tags,
            )
            if parsed is not None:
                formatted += 1
                if parsed == example.ground_truth:
                    correct += 1

        total = len(self.examples)
        accuracy = correct / total if total else 0.0
        format_rate = formatted / total if total else 0.0
        return {
            f"{self.name}/accuracy": accuracy,
            f"{self.name}/format_rate": format_rate,
            f"{self.name}/total": float(total),
        }


def build_eval_builder(
    *,
    name: str,
    examples: Sequence[ThinkRMExample],
    renderer_name: str,
    model_name_for_tokenizer: str,
    max_tokens: int,
    expect_think_tags: bool,
) -> SamplingClientEvaluatorBuilder:
    def _builder() -> ThinkRMEvaluator:
        return ThinkRMEvaluator(
            name=name,
            examples=examples,
            renderer_name=renderer_name,
            model_name_for_tokenizer=model_name_for_tokenizer,
            max_tokens=max_tokens,
            expect_think_tags=expect_think_tags,
        )

    return _builder


def get_think_rm_evaluator_builders(
    *,
    renderer_name: str,
    model_name_for_tokenizer: str,
    max_tokens: int,
    seed: int,
    rewardbench_limit: int | None = None,
    rmbench_limit: int | None = None,
    helpsteer_limit: int | None = None,
    expect_think_tags: bool = True,
) -> list[SamplingClientEvaluatorBuilder]:
    builders: list[SamplingClientEvaluatorBuilder] = []

    rewardbench_examples = build_rewardbench_examples(
        seed=seed, max_examples=rewardbench_limit
    )
    rewardbench_count = len(rewardbench_examples)
    if rewardbench_examples:
        builders.append(
            build_eval_builder(
                name="rewardbench2",
                examples=rewardbench_examples,
                renderer_name=renderer_name,
                model_name_for_tokenizer=model_name_for_tokenizer,
                max_tokens=max_tokens,
                expect_think_tags=expect_think_tags,
            )
        )
    else:
        logger.warning("No RewardBench examples available; skipping evaluation.")

    rmbench_examples = build_rmbench_examples(seed=seed, max_examples=rmbench_limit)
    rmbench_count = len(rmbench_examples)
    if rmbench_examples:
        builders.append(
            build_eval_builder(
                name="rmbench",
                examples=rmbench_examples,
                renderer_name=renderer_name,
                model_name_for_tokenizer=model_name_for_tokenizer,
                max_tokens=max_tokens,
                expect_think_tags=expect_think_tags,
            )
        )
    else:
        logger.warning("No RM-Bench examples available; skipping evaluation.")

    helpsteer_examples = build_helpsteer3_examples(
        seed=seed, max_examples=helpsteer_limit
    )
    helpsteer_count = len(helpsteer_examples)
    if helpsteer_examples:
        builders.append(
            build_eval_builder(
                name="helpsteer3",
                examples=helpsteer_examples,
                renderer_name=renderer_name,
                model_name_for_tokenizer=model_name_for_tokenizer,
                max_tokens=max_tokens,
                expect_think_tags=expect_think_tags,
            )
        )
    else:
        logger.warning("No HelpSteer3 examples available; skipping evaluation.")

    total_comparisons = rewardbench_count + rmbench_count + helpsteer_count
    logger.info(
        "Think RM eval comparisons scheduled: rewardbench2=%d, rmbench=%d, helpsteer3=%d, total=%d",
        rewardbench_count,
        rmbench_count,
        helpsteer_count,
        total_comparisons,
    )

    return builders
