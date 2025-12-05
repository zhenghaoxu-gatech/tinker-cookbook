"""
Extrapolation distillation for reasoning and chat tasks.

This script mirrors the on-policy distillation entrypoint but swaps the reward:
instead of comparing the student against the teacher, it compares the teacher
against a fixed reference (base) model. This allows starting from the teacher
while still receiving a learning signal.
"""

import asyncio
import logging
import os
from datetime import datetime
from typing import Literal, cast

import chz
from tinker_cookbook import cli_utils, model_info
from tinker_cookbook.eval.evaluators import SamplingClientEvaluatorBuilder
from tinker_cookbook.distillation import train_extrapolate
from tinker_cookbook.distillation.datasets import (
    ExtrapolationDatasetConfig,
    PromptOnlyDatasetBuilder,
    TeacherConfig,
)
from tinker_cookbook.recipes.distillation.evals import get_math_evaluator_builders

logger = logging.getLogger(__name__)


@chz.chz
class CLIConfig:
    """Command-line configuration for extrapolation distillation."""

    # Model configuration
    model_name: str = "Qwen/Qwen3-8B-Base"  # Student model
    lora_rank: int = 128
    renderer_name: str | None = None
    load_checkpoint_path: str | None = None  # Student checkpoint

    # Teacher configuration
    teacher_model: str = "Qwen/Qwen3-8B"
    teacher_checkpoint: str | None = None

    # Reference model configuration (reward baseline)
    reference_model: str | None = None
    reference_checkpoint: str | None = None

    # Dataset configuration
    dataset: str = "deepmath"  # Options: deepmath, tulu3

    # Training hyperparameters
    group_size: int = 4  # Number of rollouts per prompt
    groups_per_batch: int = 1024
    learning_rate: float = 1e-4
    max_tokens: int = 4096
    reward_mode: str = "token"  # "token" or "sequence"
    reward_scale: float = 1.0

    # Optimizer configuration
    num_substeps: int = 1
    loss_fn: str = "importance_sampling"

    # Logging configuration
    log_path: str | None = None
    wandb_project: str | None = None
    wandb_name: str | None = None
    compute_post_kl: bool = False

    # Evaluation and checkpointing
    enable_math_evals: bool = True
    eval_max_tokens: int | None = None
    eval_grader: str = "sympy"
    aime_eval_limit: int | None = None
    hmmt_eval_limit: int | None = None
    brumo_eval_limit: int | None = None
    eval_num_generations: int = 1
    eval_every: int = 20
    save_every: int = 20

    # Service configuration
    base_url: str | None = None

    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "ask"


async def cli_main(cli_config: CLIConfig):
    """Convert CLI config to full config and run training."""

    # Get renderer name
    renderer_name = cli_config.renderer_name or model_info.get_recommended_renderer_name(
        cli_config.model_name
    )
    reward_mode = cli_config.reward_mode.lower()
    if reward_mode not in {"token", "sequence"}:
        raise ValueError(f"Invalid reward_mode '{cli_config.reward_mode}'. Use 'token' or 'sequence'.")
    reward_mode_literal = cast(Literal["token", "sequence"], reward_mode)
    reference_model_name = cli_config.reference_model or cli_config.model_name

    # Create log path if not specified
    if cli_config.log_path is not None:
        log_path = cli_config.log_path
    else:
        model_name = cli_config.model_name.replace("/", "-")
        run_name = (
            f"distill-{cli_config.dataset}-{model_name}-"
            f"{cli_config.lora_rank}rank-{cli_config.learning_rate}lr-"
            f"{cli_config.groups_per_batch}batch-{datetime.now().strftime('%Y-%m-%d-%H-%M')}"
        )
        log_path = os.path.expanduser(f"~/tinker-examples/distillation/{run_name}")

    # Create wandb name if not specified
    if cli_config.wandb_name is not None:
        wandb_name = cli_config.wandb_name
    else:
        wandb_name = os.path.basename(log_path)

    # Create dataset builder
    dataset_builder = PromptOnlyDatasetBuilder(
        dataset_name=cli_config.dataset,
        groups_per_batch=cli_config.groups_per_batch,
        group_size=cli_config.group_size,
        model_name_for_tokenizer=cli_config.model_name,
        renderer_name=renderer_name,
    )

    # Create teacher config
    teacher_config = TeacherConfig(
        base_model=cli_config.teacher_model,
        load_checkpoint_path=cli_config.teacher_checkpoint,
    )

    reference_config = TeacherConfig(
        base_model=reference_model_name,
        load_checkpoint_path=cli_config.reference_checkpoint,
    )

    # Create distillation dataset config
    dataset_config = ExtrapolationDatasetConfig(
        dataset_builder=dataset_builder,
        teacher_config=teacher_config,
        reward_reference_config=reference_config,
        groups_per_batch=cli_config.groups_per_batch,
    )

    evaluator_builders: list[SamplingClientEvaluatorBuilder] = []
    if cli_config.enable_math_evals:
        eval_max_tokens = cli_config.eval_max_tokens or cli_config.max_tokens
        math_eval_builders = get_math_evaluator_builders(
            renderer_name=renderer_name,
            model_name_for_tokenizer=cli_config.model_name,
            max_tokens=eval_max_tokens,
            aime_limit=cli_config.aime_eval_limit,
            hmmt_limit=cli_config.hmmt_eval_limit,
            brumo_limit=cli_config.brumo_eval_limit,
            grader=cli_config.eval_grader,
            num_generations=cli_config.eval_num_generations,
        )
        evaluator_builders.extend(math_eval_builders)

    # Create full config
    config = train_extrapolate.Config(
        learning_rate=cli_config.learning_rate,
        dataset_configs=[dataset_config],
        model_name=cli_config.model_name,
        lora_rank=cli_config.lora_rank,
        max_tokens=cli_config.max_tokens,
        reward_mode=reward_mode_literal,
        reward_scale=cli_config.reward_scale,
        num_substeps=cli_config.num_substeps,
        loss_fn=cli_config.loss_fn,  # type: ignore
        wandb_project=cli_config.wandb_project,
        wandb_name=wandb_name,
        log_path=log_path,
        base_url=cli_config.base_url,
        load_checkpoint_path=cli_config.load_checkpoint_path,
        compute_post_kl=cli_config.compute_post_kl,
        eval_every=cli_config.eval_every,
        save_every=cli_config.save_every,
        evaluator_builders=evaluator_builders,
    )

    cli_utils.check_log_dir(log_path, behavior_if_exists=cli_config.behavior_if_log_dir_exists)

    # Run training
    await train_extrapolate.main(config)


if __name__ == "__main__":
    cli_config = chz.entrypoint(CLIConfig)
    asyncio.run(cli_main(cli_config))
