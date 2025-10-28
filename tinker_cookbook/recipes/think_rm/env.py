"""Environment and dataset utilities for Think RM training in Tinker."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from functools import partial
from random import Random
from typing import Iterable, Literal, Sequence

import chz
import tinker
from datasets import Dataset, load_dataset
from tinker_cookbook import renderers
from tinker_cookbook.recipes.think_rm.parsing import parse_preference
from tinker_cookbook.rl.problem_env import ProblemEnv, ProblemGroupBuilder
from tinker_cookbook.rl.types import (
    Action,
    EnvGroupBuilder,
    RLDataset,
    RLDatasetBuilder,
    StepResult,
)
from tinker_cookbook.tokenizer_utils import get_tokenizer

logger = logging.getLogger(__name__)

PROMPT_INSTRUCTION = """
You are an impartial judge, tasked with evaluating the quality of the two AI assistants' responses to the
context displayed below. Your evaluation should be based on the following six criteria:

- Helpfulness: Overall helpfulness of the response to the user's question or instruction.
- Correctness: Inclusion of all pertinent facts without errors.
- Coherence: Consistency and clarity of expression.
- Complexity: Intellectual depth required to write response (i.e., whether the response can be written by anyone
  with basic language competency or requires deep domain expertise).
- Verbosity: Amount of detail included in the response, relative to what is asked for in the context.
- Safety: Whether the response is free of any kind of harmful, toxic, or illegal content.

After carefully considering these criteria, determine which assistant's response is superior. Output your final
verdict by strictly following this format: <label>1</label> if assistant A is better, <label>2</label> if assistant B
is better, and <label>0</label> only if you really cannot tell their difference.
""".strip()


def format_context(context: Iterable[dict[str, str]]) -> str:
    """Render a chat context into readable text."""
    lines: list[str] = []
    for turn in context:
        role = turn.get("role", "user").capitalize()
        content = turn.get("content", "")
        lines.append(f"{role}: {content}")
    if not lines:
        return "(No prior context provided.)"
    return "\n".join(lines)


def build_prompt_text(context: list[dict[str, str]], response_a: str, response_b: str) -> str:
    """Compose the comparison request shown to the model."""
    return (
        f"{PROMPT_INSTRUCTION}\n\n"
        "[The Start of Context]\n"
        f"{format_context(context)}\n"
        "[The End of Context]\n\n"
        "[The Start of Assistant A's Response]\n"
        f"{response_a}\n"
        "[The End of Assistant A's Response]\n\n"
        "[The Start of Assistant B's Response]\n"
        f"{response_b}\n"
        "[The End of Assistant B's Response]"
    )


@dataclass(frozen=True)
class ThinkRMExample:
    prompt_text: str
    ground_truth: Literal["response_1", "response_2", "tie"]
    uid: str
    data_source: str

    def conversation(self) -> list[renderers.Message]:
        return [{"role": "user", "content": self.prompt_text}]


class ThinkRMEnv(ProblemEnv):
    """Single-step environment that scores predictions against preference labels."""

    def __init__(
        self,
        example: ThinkRMExample,
        renderer: renderers.Renderer,
        convo_prefix: list[renderers.Message] | None = None,
        format_coef: float = 0.1,
        expect_think_tags: bool = True,
    ):
        super().__init__(renderer, convo_prefix=convo_prefix, format_coef=format_coef)
        self.example = example
        self.expect_think_tags = expect_think_tags

    def get_question(self) -> str:
        return self.example.prompt_text

    def check_format(self, sample_str: str) -> bool:
        return parse_preference(sample_str, from_thinking_model=self.expect_think_tags) is not None

    def check_answer(self, sample_str: str) -> bool:
        predicted = parse_preference(sample_str, from_thinking_model=self.expect_think_tags)
        return predicted is not None and predicted == self.example.ground_truth

    async def step(self, action: Action) -> StepResult:
        message, parse_success = self.renderer.parse_response(action)
        predicted_raw = parse_preference(
            message["content"], from_thinking_model=self.expect_think_tags
        )
        if not parse_success and not self.expect_think_tags and predicted_raw is not None:
            parse_success = True

        correct_format = float(parse_success) and float(predicted_raw is not None)
        correct_answer = float(
            predicted_raw is not None and predicted_raw == self.example.ground_truth
        )
        total_reward = self.format_coef * (correct_format - 1) + correct_answer

        return StepResult(
            reward=total_reward,
            episode_done=True,
            next_observation=tinker.ModelInput.empty(),
            next_stop_condition=self.stop_condition,
            metrics={
                "format": correct_format,
                "correct": correct_answer,
            },
        )


CLIENT_QUESTION_ANCHOR = "[Client Question]"
START_A_ANCHOR = "[The Start of Chatbot A's Response]"
END_A_ANCHOR = "[The End of Chatbot A's Response]"
START_B_ANCHOR = "[The Start of Chatbot B's Response]"
END_B_ANCHOR = "[The End of Chatbot B's Response]"


def _remove_leading_system(messages: list[dict[str, object]]) -> list[dict[str, object]]:
    if messages and isinstance(messages[0], dict):
        role = messages[0].get("role")
        if isinstance(role, str) and role.lower() == "system":
            return messages[1:]
    return messages


def _combine_message_contents(messages: Iterable[dict[str, object]]) -> str:
    contents: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if isinstance(content, str):
            contents.append(content)
    return "\n\n".join(contents).strip()


def _extract_question_and_responses(raw_text: str) -> tuple[list[dict[str, str]], str, str]:
    question_idx = raw_text.find(CLIENT_QUESTION_ANCHOR)
    if question_idx == -1:
        raise ValueError("Client question marker not found.")
    question_start = question_idx + len(CLIENT_QUESTION_ANCHOR)

    start_a_idx = raw_text.find(START_A_ANCHOR, question_start)
    if start_a_idx == -1:
        raise ValueError("Chatbot A start marker not found.")

    question_text = raw_text[question_start:start_a_idx].strip()
    if not question_text:
        raise ValueError("Empty question text.")

    response_a_start = start_a_idx + len(START_A_ANCHOR)
    end_a_idx = raw_text.find(END_A_ANCHOR, response_a_start)
    if end_a_idx == -1:
        raise ValueError("Chatbot A end marker not found.")
    response_a_text = raw_text[response_a_start:end_a_idx].strip()
    if not response_a_text:
        raise ValueError("Empty Chatbot A response.")

    start_b_idx = raw_text.find(START_B_ANCHOR, end_a_idx + len(END_A_ANCHOR))
    if start_b_idx == -1:
        raise ValueError("Chatbot B start marker not found.")

    response_b_start = start_b_idx + len(START_B_ANCHOR)
    end_b_idx = raw_text.find(END_B_ANCHOR, response_b_start)
    if end_b_idx == -1:
        raise ValueError("Chatbot B end marker not found.")
    response_b_text = raw_text[response_b_start:end_b_idx].strip()
    if not response_b_text:
        raise ValueError("Empty Chatbot B response.")

    context_messages = [{"role": "user", "content": question_text}]
    return context_messages, response_a_text, response_b_text


def _map_winner_label(winner: object) -> Literal["response_1", "response_2", "tie"] | None:
    if not isinstance(winner, str):
        return None
    normalised = winner.strip().lower()
    if normalised == "model_a":
        return "response_1"
    if normalised == "model_b":
        return "response_2"
    if normalised in {"tie", "draw", "equal"}:
        return "tie"
    return None


def build_rmr1_examples(dataset: Dataset, split: str) -> list[ThinkRMExample]:
    """Convert RM-R1 records into ThinkRMExample instances."""
    examples: list[ThinkRMExample] = []
    skipped_parse_errors = 0
    skipped_missing_winner = 0

    for row_idx, row in enumerate(dataset):
        raw_messages = row.get("context_messages")
        if not isinstance(raw_messages, list):
            skipped_parse_errors += 1
            continue

        messages_without_system = _remove_leading_system(raw_messages)
        merged_text = _combine_message_contents(messages_without_system)
        if not merged_text:
            skipped_parse_errors += 1
            continue

        try:
            context_messages, response_a, response_b = _extract_question_and_responses(merged_text)
        except ValueError:
            skipped_parse_errors += 1
            continue

        label = _map_winner_label(row.get("winner"))
        if label is None:
            skipped_missing_winner += 1
            continue

        prompt_text = build_prompt_text(context_messages, response_a, response_b)
        uid = f"{split}-{row_idx}"
        examples.append(
            ThinkRMExample(
                prompt_text=prompt_text,
                ground_truth=label,
                uid=uid,
                data_source="rmr1",
            )
        )

    if skipped_missing_winner or skipped_parse_errors:
        logger.info(
            "RM-R1 split=%s: built %d examples (skipped %d missing winners, %d parse errors).",
            split,
            len(examples),
            skipped_missing_winner,
            skipped_parse_errors,
        )
    else:
        logger.info("RM-R1 split=%s: built %d examples.", split, len(examples))
    return examples


class ThinkRMDataset(RLDataset):
    """Dataset wrapper that streams Think RM examples to the RL trainer."""

    def __init__(
        self,
        *,
        examples: Sequence[ThinkRMExample],
        batch_size: int,
        group_size: int,
        renderer: renderers.Renderer,
        convo_prefix: list[renderers.Message] | None,
        format_coef: float,
        expect_think_tags: bool,
    ):
        self.examples = list(examples)
        self.batch_size = batch_size
        self.group_size = group_size
        self.renderer = renderer
        self.convo_prefix = convo_prefix
        self.format_coef = format_coef
        self.expect_think_tags = expect_think_tags

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        start = index * self.batch_size
        end = min((index + 1) * self.batch_size, len(self.examples))
        if start >= end:
            raise IndexError("Index out of range for ThinkRM dataset batch.")
        batch_examples = self.examples[start:end]
        return [
            ProblemGroupBuilder(
                env_thunk=partial(
                    ThinkRMEnv,
                    example,
                    self.renderer,
                    convo_prefix=self.convo_prefix,
                    format_coef=self.format_coef,
                    expect_think_tags=self.expect_think_tags,
                ),
                num_envs=self.group_size,
                dataset_name="think_rm",
            )
            for example in batch_examples
        ]

    def __len__(self) -> int:
        return math.ceil(len(self.examples) / self.batch_size)


@chz.chz
class ThinkRMDatasetBuilder(RLDatasetBuilder):
    """Build Think RM datasets from the RM-R1 preference records."""

    batch_size: int = 32
    group_size: int = 8
    model_name_for_tokenizer: str
    renderer_name: str
    dataset_name: str = "gaotang/RM-R1-Entire-RLVR-Train"
    split: str = "train"
    shuffle_seed: int | None = 0
    convo_prefix: list[renderers.Message] | None = None
    format_coef: float = 0.1
    expect_think_tags: bool = True
    max_prompt_tokens: int | None = None

    async def __call__(self) -> tuple[ThinkRMDataset, None]:
        hf_dataset = load_dataset(self.dataset_name, split=self.split, num_proc=8)
        if not isinstance(hf_dataset, Dataset):
            raise TypeError(f"Expected datasets.Dataset for split '{self.split}', got {type(hf_dataset)}")

        examples = build_rmr1_examples(hf_dataset, self.split)
        if not examples:
            raise ValueError("No usable examples found in RM-R1 dataset; aborting.")

        if self.shuffle_seed is not None:
            rng = Random(self.shuffle_seed)
            rng.shuffle(examples)

        renderer = renderers.get_renderer(
            self.renderer_name,
            get_tokenizer(self.model_name_for_tokenizer),
        )

        if self.max_prompt_tokens is not None:
            tokenizer = renderer.tokenizer
            filtered_examples: list[ThinkRMExample] = []
            for example in examples:
                token_ids = tokenizer.encode(example.prompt_text, add_special_tokens=True)
                if len(token_ids) <= self.max_prompt_tokens:
                    filtered_examples.append(example)
            if not filtered_examples:
                raise ValueError(
                    "All RM-R1 examples exceeded the configured max_prompt_tokens."
                )
            if len(filtered_examples) < len(examples):
                logger.info(
                    "Filtered %d prompts longer than %d tokens (kept %d).",
                    len(examples) - len(filtered_examples),
                    self.max_prompt_tokens,
                    len(filtered_examples),
                )
            examples = filtered_examples

        dataset = ThinkRMDataset(
            examples=examples,
            batch_size=self.batch_size,
            group_size=self.group_size,
            renderer=renderer,
            convo_prefix=self.convo_prefix,
            format_coef=self.format_coef,
            expect_think_tags=self.expect_think_tags,
        )
        return dataset, None
