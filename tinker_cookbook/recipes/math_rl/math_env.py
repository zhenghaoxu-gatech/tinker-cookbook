import math
import re
from functools import partial
from typing import Literal, Sequence, cast

import chz
from datasets import Dataset, concatenate_datasets, get_dataset_config_names, load_dataset
from tinker_cookbook import renderers
from tinker_cookbook.recipes.math_rl.math_grading import (
    extract_boxed,
    grade_answer,
    grade_answer_math_verify,
    run_with_timeout_signal,
)
from tinker_cookbook.rl.problem_env import ProblemEnv, ProblemGroupBuilder, logger
from tinker_cookbook.rl.types import EnvGroupBuilder, RLDataset, RLDatasetBuilder
from tinker_cookbook.tokenizer_utils import get_tokenizer


class MathEnv(ProblemEnv):
    def __init__(
        self,
        problem: str,
        answer: str,
        renderer: renderers.Renderer,
        convo_prefix: list[renderers.Message] | None = None,
        grader: Literal["sympy", "math_verify"] = "math_verify",
        timeout: float = 1.0,
    ):
        super().__init__(renderer, convo_prefix)
        self.problem = problem
        self.answer = answer
        self.grader = grader
        self.timeout = timeout

    @classmethod
    def question_suffix(cls) -> str:
        return " Write your answer in \\boxed{} format."

    def get_question(self) -> str:
        return self.problem + self.question_suffix()

    def check_format(self, sample_str: str) -> bool:
        try:
            _ = extract_boxed(sample_str)
            return True
        except ValueError:
            return False

    def check_answer(self, sample_str: str) -> bool:
        try:
            answer = extract_boxed(sample_str)
        except ValueError:
            return False
        return safe_grade(answer, self.answer, self.grader, self.timeout)

    @staticmethod
    def standard_fewshot_prefix() -> list[renderers.Message]:
        return [
            {
                "role": "user",
                "content": "How many r's are in strawberry?" + MathEnv.question_suffix(),
            },
            {
                "role": "assistant",
                "content": "Let's spell the word out and number all the letters: 1) s 2) t 3) r 4) a 5) w 6) b 7) e 8) r 9) r 10) y. We have r's at positions 3, 8, and 9. \\boxed{3}",
            },
        ]


def safe_grade(given_answer: str, ground_truth: str, grader: str = "sympy", timeout: float = 1.0):
    if grader == "sympy":
        grader_func = grade_answer
    elif grader == "math_verify":
        grader_func = grade_answer_math_verify
    else:
        raise ValueError(f"Invalid grader: {grader}")
    out = run_with_timeout_signal(
        grader_func, args=(given_answer, ground_truth), timeout_seconds=int(math.ceil(timeout))
    )
    if out is None:
        logger.warning(f"Timeout grading {given_answer} against {ground_truth}")
        return False
    return out


def extract_gsm8k_final_answer(text: str) -> str:
    """Extract the final numeric/string answer from a GSM8K solution field.

    GSM8K format typically places the final answer on a line starting with
    '####'. We take the substring following '####' on the last such line.
    """
    lines = text.splitlines()
    for line in reversed(lines):
        s = line.strip()
        if s.startswith("####"):
            content = s[4:].strip()
            if content.startswith(":"):
                content = content[1:].strip()
            content = content.replace(",", "").strip()
            return content
    matches = re.findall(r"####\s*(.+)", text)
    if matches:
        return matches[-1].strip()
    raise ValueError("No GSM8K final answer found")


def _get_hendrycks_math_test() -> Dataset:
    test_dataset = load_dataset("HuggingFaceH4/MATH-500", name="default", split="test")
    return cast(Dataset, test_dataset)


def _get_hendrycks_math_train() -> Dataset:
    # For Hendrycks MATH, the standard is to use both the "train" and "test" splits for
    # training. The "test" split here is NOT the same as the MATH-500 test split above,
    # which is a commonly-held-out subset of 500 of the below 12.5k problems. To construct
    # a clean training set, we filter out problems that exist in the MATH-500 test set,
    # resulting in 12000 train and 500 test problems.

    test_problems: set[str] = {
        problem["problem"]  # pyright: ignore[reportArgumentType, reportCallIssue]
        for problem in _get_hendrycks_math_test()
    }

    dataset_name = "EleutherAI/hendrycks_math"
    configs = get_dataset_config_names(dataset_name)
    pieces = []
    for cfg in configs:
        for split in ("train", "test"):
            ds = load_dataset(dataset_name, name=cfg, split=split)
            ds = ds.filter(lambda example: example["problem"] not in test_problems)
            pieces.append(ds)
    full_dataset = concatenate_datasets(pieces)

    return full_dataset


class MathDataset(RLDataset):
    def __init__(
        self,
        batch_size: int,
        group_size: int,
        renderer: renderers.Renderer,
        convo_prefix: list[renderers.Message] | None = None,
        split: Literal["train", "test"] = "train",
        grader: Literal["sympy", "math_verify"] = "math_verify",
    ):
        if split == "train":
            self.ds = _get_hendrycks_math_train().shuffle(seed=0)
        elif split == "test":
            self.ds = _get_hendrycks_math_test()
        self.batch_size = batch_size
        self.group_size = group_size if split == "train" else 1
        self.renderer = renderer
        self.convo_prefix = convo_prefix
        self.grader = grader

    def get_batch(self, index: int) -> list[EnvGroupBuilder]:
        batch_start = index * self.batch_size
        batch_end = min((index + 1) * self.batch_size, len(self.ds))
        assert batch_start < batch_end, "Incorrect batch size"
        return [
            builder
            for row in self.ds.select(range(batch_start, batch_end))
            if (builder := self._make_env_group_builder(row, self.group_size)) is not None  # pyright: ignore[reportArgumentType]
        ]

    def __len__(self) -> int:
        return math.ceil(len(self.ds) / self.batch_size)

    def _make_env_group_builder(
        self, x: dict[str, str], group_size: int
    ) -> ProblemGroupBuilder | None:
        try:
            answer = extract_boxed(x["solution"])
        except ValueError:  # not sure if this happens
            logger.warning(f"No answer found for {x['solution']}")
            return None
        return ProblemGroupBuilder(
            env_thunk=partial(
                MathEnv,
                x["problem"],
                answer,
                self.renderer,
                convo_prefix=self.convo_prefix,
                grader=self.grader,
            ),
            num_envs=group_size,
        )


@chz.chz
class MathDatasetBuilder(RLDatasetBuilder):
    batch_size: int
    model_name_for_tokenizer: str
    renderer_name: str
    group_size: int
    convo_prefix: list[renderers.Message] | None | Literal["standard"] = "standard"
    grader: Literal["sympy", "math_verify"] = "math_verify"

    async def __call__(self) -> tuple[MathDataset, MathDataset]:
        if self.convo_prefix == "standard":
            convo_prefix = MathEnv.standard_fewshot_prefix()
        else:
            convo_prefix = self.convo_prefix
        tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        renderer = renderers.get_renderer(self.renderer_name, tokenizer=tokenizer)
        datasets = [
            MathDataset(
                batch_size=self.batch_size,
                group_size=self.group_size,
                renderer=renderer,
                convo_prefix=convo_prefix,
                split=split,
                grader=self.grader,
            )
            for split in ("train", "test")
        ]
        return (datasets[0], datasets[1])


class PolarisDataset(MathDataset):
    def __init__(
        self,
        batch_size: int,
        group_size: int,
        renderer: renderers.Renderer,
        convo_prefix: list[renderers.Message] | None = None,
        grader: Literal["sympy", "math_verify"] = "math_verify",
    ):
        # Don't call super().__init__ since we're overriding the dataset loading
        self.ds = load_dataset("POLARIS-Project/Polaris-Dataset-53K", split="train").shuffle(seed=0)
        self.batch_size = batch_size
        self.group_size = group_size
        self.renderer = renderer
        self.convo_prefix = convo_prefix
        self.grader = grader

    def _make_env_group_builder(
        self, x: dict[str, str], group_size: int
    ) -> ProblemGroupBuilder | None:
        # Extract problem and answer from the dataset
        problem = x.get("problem", "")
        answer = x.get("answer", "")
        if not (problem and answer):
            return None
        return ProblemGroupBuilder(
            env_thunk=partial(
                MathEnv,
                problem,
                answer,
                self.renderer,
                convo_prefix=self.convo_prefix,
                grader=self.grader,
            ),
            num_envs=group_size,
            dataset_name="polaris",
        )


@chz.chz
class PolarisDatasetBuilder(RLDatasetBuilder):
    batch_size: int
    model_name_for_tokenizer: str
    renderer_name: str
    group_size: int
    grader: Literal["sympy", "math_verify"] = "math_verify"

    async def __call__(self) -> tuple[PolarisDataset, None]:
        tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        return PolarisDataset(
            batch_size=self.batch_size,
            group_size=self.group_size,
            renderer=renderers.get_renderer(self.renderer_name, tokenizer=tokenizer),
            grader=self.grader,
        ), None


class DeepMathDataset(MathDataset):
    def __init__(
        self,
        batch_size: int,
        group_size: int,
        renderer: renderers.Renderer,
        convo_prefix: list[renderers.Message] | None = None,
        grader: Literal["sympy", "math_verify"] = "math_verify",
        difficulty_min: float | None = None,
        difficulty_max: float | None = None,
    ):
        # Don't call super().__init__ since we're overriding the dataset loading
        dataset = load_dataset("zwhe99/DeepMath-103K", split="train")
        if difficulty_min is not None or difficulty_max is not None:
            def in_range(example: dict[str, object]) -> bool:
                difficulty_value = example.get("difficulty")
                if difficulty_value is None:
                    return False
                try:
                    difficulty = float(difficulty_value)
                except (TypeError, ValueError):
                    return False
                if difficulty_min is not None and difficulty < difficulty_min:
                    return False
                if difficulty_max is not None and difficulty > difficulty_max:
                    return False
                return True

            dataset = dataset.filter(in_range)
        self.ds = dataset.shuffle(seed=0)
        self.batch_size = batch_size
        self.group_size = group_size
        self.renderer = renderer
        self.convo_prefix = convo_prefix
        self.grader = grader

    def _make_env_group_builder(
        self, x: dict[str, str], group_size: int
    ) -> ProblemGroupBuilder | None:
        # Extract problem and answer from the dataset
        problem = x.get("question", "")
        answer = x.get("final_answer", "")
        if not (problem and answer):
            return None
        return ProblemGroupBuilder(
            env_thunk=partial(
                MathEnv,
                problem,
                answer,
                self.renderer,
                convo_prefix=self.convo_prefix,
                grader=self.grader,
            ),
            num_envs=group_size,
            dataset_name="deepmath",
        )


MATH_ARENA_VALIDATION_SPECS: tuple[tuple[str, str], ...] = (
    ("MathArena/aime_2025", "aime25"),
    ("MathArena/hmmt_feb_2025", "hmmt25"),
    ("MathArena/brumo_2025", "brumo25"),
)


def _load_matharena_dataset(dataset_name: str) -> Dataset:
    """Load a MathArena dataset preferring validation/test splits when available."""
    preferred_splits: tuple[str, ...] = ("validation", "test", "train")
    last_error: Exception | None = None
    for split in preferred_splits:
        try:
            return cast(Dataset, load_dataset(dataset_name, split=split))
        except (ValueError, KeyError) as exc:
            last_error = exc
            continue
    if last_error is not None:
        raise last_error
    raise ValueError(f"Unable to load dataset {dataset_name} with splits {preferred_splits}")


class MathArenaEvalDataset(RLDataset):
    def __init__(
        self,
        renderer: renderers.Renderer,
        convo_prefix: list[renderers.Message] | None = None,
        dataset_specs: Sequence[tuple[str, str]] = MATH_ARENA_VALIDATION_SPECS,
        batch_size: int = 1,
        eval_group_size: int = 1,
        grader: Literal["sympy", "math_verify"] = "math_verify",
    ):
        self.renderer = renderer
        self.convo_prefix = convo_prefix
        self.batch_size = batch_size
        self.grader = grader
        if eval_group_size < 1:
            raise ValueError("eval_group_size must be >= 1")
        self.group_size = eval_group_size
        self.builders: list[ProblemGroupBuilder] = []
        for dataset_name, tag in dataset_specs:
            ds = _load_matharena_dataset(dataset_name)
            for row in ds:
                problem = row.get("problem")
                answer = row.get("answer")
                if problem is None or answer is None:
                    logger.warning(
                        f"Skipping example from {dataset_name} in MathArena eval set: missing fields"
                    )
                    continue
                problem_str = str(problem)
                answer_str = str(answer)
                builder = ProblemGroupBuilder(
                    env_thunk=partial(
                        MathEnv,
                        problem_str,
                        answer_str,
                        self.renderer,
                        convo_prefix=self.convo_prefix,
                        grader=self.grader,
                    ),
                    num_envs=self.group_size,
                    dataset_name=tag,
                )
                self.builders.append(builder)
        if not self.builders:
            raise ValueError("MathArena eval dataset is empty; check dataset availability.")

    def get_batch(self, index: int) -> list[EnvGroupBuilder]:
        batch_start = index * self.batch_size
        batch_end = min((index + 1) * self.batch_size, len(self.builders))
        assert batch_start < batch_end, "Incorrect batch size"
        return self.builders[batch_start:batch_end]

    def __len__(self) -> int:
        return math.ceil(len(self.builders) / self.batch_size)


@chz.chz
class DeepMathDatasetBuilder(RLDatasetBuilder):
    batch_size: int
    model_name_for_tokenizer: str
    renderer_name: str
    group_size: int
    grader: Literal["sympy", "math_verify"] = "math_verify"
    difficulty_min: float | None = None
    difficulty_max: float | None = None

    async def __call__(self) -> tuple[DeepMathDataset, MathArenaEvalDataset]:
        tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        renderer = renderers.get_renderer(self.renderer_name, tokenizer=tokenizer)
        return (
            DeepMathDataset(
                batch_size=self.batch_size,
                group_size=self.group_size,
                renderer=renderer,
                grader=self.grader,
                difficulty_min=self.difficulty_min,
                difficulty_max=self.difficulty_max,
            ),
            MathArenaEvalDataset(
                renderer=renderer,
                eval_group_size=max(8, self.group_size),
                grader=self.grader,
            ),
        )


class MathDapoDataset(RLDataset):
    def __init__(
        self,
        batch_size: int,
        group_size: int,
        renderer: renderers.Renderer,
        convo_prefix: list[renderers.Message] | None = None,
        grader: Literal["sympy", "math_verify"] = "math_verify",
    ):
        self.ds = load_dataset("fengyao1909/dapo-math-17k-deduplicated", split="train").shuffle(
            seed=0
        )
        self.batch_size = batch_size
        self.group_size = group_size
        self.renderer = renderer
        self.convo_prefix = convo_prefix
        self.grader = grader

    def get_batch(self, index: int) -> list[EnvGroupBuilder]:
        batch_start = index * self.batch_size
        batch_end = min((index + 1) * self.batch_size, len(self.ds))
        assert batch_start < batch_end, "Incorrect batch size"
        return [
            builder
            for row in self.ds.select(range(batch_start, batch_end))
            if (builder := self._make_env_group_builder(row, self.group_size)) is not None  # pyright: ignore[reportArgumentType]
        ]

    def __len__(self) -> int:
        return math.ceil(len(self.ds) / self.batch_size)

    def _make_env_group_builder(
        self, x: dict[str, object], group_size: int
    ) -> ProblemGroupBuilder | None:
        prompt = x.get("prompt")
        if not isinstance(prompt, list) or not prompt:
            return None
        last_message = prompt[-1]
        if not isinstance(last_message, dict):
            return None
        problem = last_message.get("content")
        if not isinstance(problem, str) or not problem.strip():
            return None
        problem = "\n".join(
            line for line in problem.splitlines() if "Remember to put your answer" not in line
        ).strip()
        if not problem:
            return None
        reward_model = x.get("reward_model")
        answer: str | None = None
        if isinstance(reward_model, dict):
            ground_truth = reward_model.get("ground_truth")
            if isinstance(ground_truth, str):
                answer = ground_truth.strip()
        if not answer:
            return None
        return ProblemGroupBuilder(
            env_thunk=partial(
                MathEnv,
                problem,
                answer,
                self.renderer,
                convo_prefix=self.convo_prefix,
                grader=self.grader,
            ),
            num_envs=group_size,
            dataset_name="math_dapo",
        )


@chz.chz
class MathDapoDatasetBuilder(RLDatasetBuilder):
    batch_size: int
    model_name_for_tokenizer: str
    renderer_name: str
    group_size: int
    grader: Literal["sympy", "math_verify"] = "math_verify"

    async def __call__(self) -> tuple[MathDapoDataset, MathArenaEvalDataset]:
        tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        renderer = renderers.get_renderer(self.renderer_name, tokenizer=tokenizer)
        return (
            MathDapoDataset(
                batch_size=self.batch_size,
                group_size=self.group_size,
                renderer=renderer,
                grader=self.grader,
            ),
            MathArenaEvalDataset(
                renderer=renderer,
                eval_group_size=max(8, self.group_size),
                grader=self.grader,
            ),
        )


class Gsm8kDataset(RLDataset):
    def __init__(
        self,
        batch_size: int,
        group_size: int,
        renderer: renderers.Renderer,
        convo_prefix: list[renderers.Message] | None = None,
        split: Literal["train", "test"] = "train",
        grader: Literal["sympy", "math_verify"] = "math_verify",
    ):
        if split not in ("train", "test"):
            raise ValueError("split must be 'train' or 'test'")
        self.ds = cast(Dataset, load_dataset("openai/gsm8k", name="main", split=split))
        if split == "train":
            self.ds = self.ds.shuffle(seed=0)
        self.batch_size = batch_size
        self.group_size = group_size if split == "train" else 1
        self.renderer = renderer
        self.convo_prefix = convo_prefix
        self.grader = grader

    @classmethod
    def question_suffix(cls) -> str:
        return " Provide a numerical answer without units, written inside \\boxed{}."

    def get_batch(self, index: int) -> list[EnvGroupBuilder]:
        batch_start = index * self.batch_size
        batch_end = min((index + 1) * self.batch_size, len(self.ds))
        assert batch_start < batch_end, "Incorrect batch size"
        return [
            builder
            for row in self.ds.select(range(batch_start, batch_end))
            if (builder := self._make_env_group_builder(row, self.group_size)) is not None  # pyright: ignore[reportArgumentType]
        ]

    def __len__(self) -> int:
        return math.ceil(len(self.ds) / self.batch_size)

    def _make_env_group_builder(
        self, x: dict[str, str], group_size: int
    ) -> ProblemGroupBuilder | None:
        try:
            problem = x["question"]
            answer = extract_gsm8k_final_answer(x["answer"])
        except Exception as e:
            logger.warning(f"Failed to parse GSM8K row: {e}")
            return None
        return ProblemGroupBuilder(
            env_thunk=partial(
                MathEnv,
                problem,
                answer,
                self.renderer,
                convo_prefix=self.convo_prefix,
                grader=self.grader,
            ),
            num_envs=group_size,
        )


@chz.chz
class Gsm8kDatasetBuilder(RLDatasetBuilder):
    batch_size: int
    model_name_for_tokenizer: str
    renderer_name: str
    group_size: int
    convo_prefix: list[renderers.Message] | None | Literal["standard"] = "standard"
    grader: Literal["sympy", "math_verify"] = "math_verify"

    async def __call__(self) -> tuple[Gsm8kDataset, Gsm8kDataset]:
        if self.convo_prefix == "standard":
            convo_prefix = MathEnv.standard_fewshot_prefix()
        else:
            convo_prefix = self.convo_prefix
        tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        renderer = renderers.get_renderer(self.renderer_name, tokenizer=tokenizer)
        datasets = [
            Gsm8kDataset(
                batch_size=self.batch_size,
                group_size=self.group_size,
                renderer=renderer,
                convo_prefix=convo_prefix,
                split=split,
                grader=self.grader,
            )
            for split in ("train", "test")
        ]
        return (datasets[0], datasets[1])


# Populate the dataset builder map after all classes are defined
DATASET_BUILDER_MAP = {
    "math": MathDatasetBuilder,
    "polaris": PolarisDatasetBuilder,
    "deepmath": DeepMathDatasetBuilder,
    "dapo_math": MathDapoDatasetBuilder,
    "gsm8k": Gsm8kDatasetBuilder,
}


def get_math_dataset_builder(
    dataset_name: str,
    batch_size: int,
    model_name_for_tokenizer: str,
    renderer_name: str,
    group_size: int,
    grader: Literal["sympy", "math_verify"] | None = None,
    difficulty_min: float | None = None,
    difficulty_max: float | None = None,
) -> RLDatasetBuilder:
    """
    Unified function to get any math dataset builder.
    Args:
        dataset_name: One of "math", "polaris", "deepmath", "dapo_math", or "gsm8k"
        batch_size: Number of groups per batch
        model_name_for_tokenizer: Model name for tokenizer
        renderer_name: Name of the renderer to use
        group_size: Number of environments per group
    Returns:
        The appropriate dataset builder instance
    """
    if dataset_name not in DATASET_BUILDER_MAP:
        raise ValueError(
            f"Unknown math dataset: {dataset_name}. Available: {list(DATASET_BUILDER_MAP.keys())}"
        )

    builder_class = DATASET_BUILDER_MAP[dataset_name]

    builder_kwargs: dict[str, object] = {
        "batch_size": batch_size,
        "model_name_for_tokenizer": model_name_for_tokenizer,
        "renderer_name": renderer_name,
        "group_size": group_size,
    }
    if grader is not None:
        builder_kwargs["grader"] = grader
    if dataset_name == "deepmath":
        builder_kwargs["difficulty_min"] = difficulty_min
        builder_kwargs["difficulty_max"] = difficulty_max

    return builder_class(**builder_kwargs)
