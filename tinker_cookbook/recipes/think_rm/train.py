import asyncio
import logging
from datetime import datetime

import chz
from tinker_cookbook import cli_utils, model_info
from tinker_cookbook.recipes.think_rm.env import ThinkRMDatasetBuilder
from tinker_cookbook.recipes.think_rm.evals import get_think_rm_evaluator_builders
from tinker_cookbook.rl.train import AsyncConfig, Config, main

logger = logging.getLogger(__name__)


@chz.chz
class CLIConfig:
    """Command-line options for Think RM RL training."""

    # Model configuration
    env: str = "think_rm"
    model_name: str = "Qwen/Qwen3-4B-Thinking-2507"
    renderer_name: str | None = None
    lora_rank: int = 32
    load_checkpoint_path: str | None = None

    # Dataset configuration
    dataset_name: str = "gaotang/RM-R1-Entire-RLVR-Train"
    dataset_split: str = "train"
    dataset_shuffle_seed: int | None = 0

    # Training hyperparameters
    group_size: int = 8
    groups_per_batch: int = 128
    learning_rate: float = 2e-5
    max_tokens: int = 8192
    max_tokens_eval: int | None = 8192
    num_substeps: int = 1
    format_coef: float = 0.1
    max_prompt_tokens: int | None = 8192
    thinking_model: bool = True

    # KL / PPO options
    compute_post_kl: bool = False
    kl_penalty_coef: float = 0.0
    kl_discount_factor: float = 0.0

    # Logging and checkpoints
    log_path: str | None = None
    wandb_project: str | None = None
    wandb_name: str | None = None
    eval_every: int = 100
    save_every: int = 50
    base_url: str | None = None

    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "ask"

    # Async training options
    max_steps_off_policy: int | None = None

    # Evaluation configuration
    eval_seed: int = 0
    rewardbench_limit: int | None = None
    rmbench_limit: int | None = None
    helpsteer_limit: int | None = None


async def cli_main(cli_config: CLIConfig):
    renderer_name = (
        cli_config.renderer_name
        or model_info.get_recommended_renderer_name(cli_config.model_name)
    )
    thinking_model_flag = cli_config.thinking_model
    if not isinstance(thinking_model_flag, bool):
        thinking_model_flag = str(thinking_model_flag).strip().lower() in ("1", "true", "yes", "y")

    model_slug = cli_config.model_name.replace("/", "-")
    run_name = (
        f"{cli_config.env}-{model_slug}-"
        f"{cli_config.lora_rank}rank-{cli_config.learning_rate}lr-"
        f"{cli_config.group_size}group-{cli_config.groups_per_batch}batch-"
        f"{datetime.now().strftime('%Y-%m-%d-%H-%M')}"
    )

    log_path = (
        cli_config.log_path
        if cli_config.log_path is not None
        else f"/tmp/tinker-examples/think_rm/{run_name}"
    )
    wandb_name = cli_config.wandb_name or run_name

    dataset_builder = ThinkRMDatasetBuilder(
        batch_size=cli_config.groups_per_batch,
        group_size=cli_config.group_size,
        model_name_for_tokenizer=cli_config.model_name,
        renderer_name=renderer_name,
        dataset_name=cli_config.dataset_name,
        split=cli_config.dataset_split,
        shuffle_seed=cli_config.dataset_shuffle_seed,
        format_coef=cli_config.format_coef,
        expect_think_tags=thinking_model_flag,
        max_prompt_tokens=cli_config.max_prompt_tokens,
    )

    evaluator_builders = get_think_rm_evaluator_builders(
        renderer_name=renderer_name,
        model_name_for_tokenizer=cli_config.model_name,
        max_tokens=cli_config.max_tokens_eval or cli_config.max_tokens,
        seed=cli_config.eval_seed,
        rewardbench_limit=cli_config.rewardbench_limit,
        rmbench_limit=cli_config.rmbench_limit,
        helpsteer_limit=cli_config.helpsteer_limit,
        expect_think_tags=thinking_model_flag,
    )

    async_config = (
        AsyncConfig(
            max_steps_off_policy=cli_config.max_steps_off_policy,
            groups_per_batch=cli_config.groups_per_batch,
        )
        if cli_config.max_steps_off_policy is not None
        else None
    )

    config = Config(
        learning_rate=cli_config.learning_rate,
        dataset_builder=dataset_builder,
        model_name=cli_config.model_name,
        lora_rank=cli_config.lora_rank,
        max_tokens=cli_config.max_tokens,
        max_tokens_eval=cli_config.max_tokens_eval,
        compute_post_kl=cli_config.compute_post_kl,
        kl_penalty_coef=cli_config.kl_penalty_coef,
        kl_discount_factor=cli_config.kl_discount_factor,
        num_substeps=cli_config.num_substeps,
        wandb_project=cli_config.wandb_project,
        wandb_name=wandb_name,
        log_path=log_path,
        base_url=cli_config.base_url,
        load_checkpoint_path=cli_config.load_checkpoint_path,
        evaluator_builders=evaluator_builders,
        eval_every=cli_config.eval_every,
        save_every=cli_config.save_every,
        async_config=async_config,
    )

    cli_utils.check_log_dir(log_path, behavior_if_exists=cli_config.behavior_if_log_dir_exists)
    await main(config)


if __name__ == "__main__":
    cli_config = chz.entrypoint(CLIConfig)
    asyncio.run(cli_main(cli_config))
