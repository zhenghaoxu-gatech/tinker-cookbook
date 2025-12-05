"""
Implements extrapolation distillation where rewards come from the log-probability
gap between a teacher and a fixed reference (base) model.
"""

import asyncio
import logging
import os
import time
from typing import Any, List, Literal, Sequence, Dict, cast

import chz
import tinker
import torch
from tinker_cookbook import checkpoint_utils
from tinker_cookbook.display import colorize_example
from tinker_cookbook.eval.evaluators import SamplingClientEvaluator, SamplingClientEvaluatorBuilder
from tinker_cookbook.rl.data_processing import (
    assemble_training_data,
    compute_advantages,
)
from tinker_cookbook.rl.metric_util import RLTestSetEvaluator, compute_trajectory_metrics
from tinker_cookbook.rl.types import (
    EnvGroupBuilder,
    TrajectoryGroup,
)
from tinker_cookbook.tokenizer_utils import Tokenizer
from tinker_cookbook.utils import ml_log
from tinker_cookbook.utils.misc_utils import safezip, timed
from tinker_cookbook.utils.trace import scope, get_scope_context, trace_init

# Dataset configuration classes
from tinker_cookbook.distillation.datasets import (
    CompositeDataset,
    ExtrapolationDatasetConfig,
)

# We re-use these methods from the RL training recipe
from tinker_cookbook.rl.train import (
    save_checkpoint_and_get_sampling_client,
    train_step,
    compute_full_batch_metrics_and_get_sampling_client,
    do_group_rollout_and_filter_constant_reward,
)

logger = logging.getLogger(__name__)


@scope
async def incorporate_extrapolation_advantage(
    data_D: List[tinker.Datum],
    teacher_clients_D: List[tinker.SamplingClient],
    reference_clients_D: List[tinker.SamplingClient],
    dataset_indices_D: List[int],
    reward_mode: Literal["token", "sequence"],
    reward_scale: float,
) -> Dict[str, float]:
    """
    Compute the reward signal for extrapolation distillation.

    For each sampled trajectory, we compare the teacher's log probabilities against
    a fixed reference model and use the difference as the reward/advantage.

    Args:
        data_D: List of datums to compute rewards for
        teacher_clients_D: Teacher sampling clients aligned with each datum
        reference_clients_D: Reference sampling clients aligned with each datum
        dataset_indices_D: Dataset indices aligned with each datum
        reward_mode: Aggregation mode ("token" for per-token, "sequence" for per-sequence)
        reward_scale: Multiplicative factor applied to computed rewards
    """
    if not data_D:
        return {}

    full_sequence_inputs_D = [
        datum.model_input.append_int(cast(int, datum.loss_fn_inputs["target_tokens"].data[-1]))
        for datum in data_D
    ]

    teacher_logprobs_D = await asyncio.gather(
        *[
            teacher_client.compute_logprobs_async(sequence_input)
            for teacher_client, sequence_input in zip(teacher_clients_D, full_sequence_inputs_D)
        ]
    )
    reference_logprobs_D = await asyncio.gather(
        *[
            reference_client.compute_logprobs_async(sequence_input)
            for reference_client, sequence_input in zip(reference_clients_D, full_sequence_inputs_D)
        ]
    )

    float_masks = [datum.loss_fn_inputs["mask"].to_torch().float() for datum in data_D]

    token_reward_sum = 0.0
    token_weight_sum = 0.0
    sequence_reward_sum = 0.0
    per_dataset_stats: Dict[int, Dict[str, float]] = {}

    for i, datum in enumerate(data_D):
        teacher_tensor = torch.tensor(teacher_logprobs_D[i][1:], dtype=torch.float32)
        reference_tensor = torch.tensor(reference_logprobs_D[i][1:], dtype=torch.float32)
        diff = (teacher_tensor - reference_tensor) * float_masks[i]

        mask_sum = float_masks[i].sum().item()
        seq_reward = diff.sum().item()

        if reward_mode == "token":
            reward_vector = diff
        else:
            reward_vector = float_masks[i] * seq_reward

        datum.loss_fn_inputs["advantages"] = tinker.TensorData.from_torch(
            datum.loss_fn_inputs["advantages"].to_torch() + reward_scale * reward_vector
        )

        token_reward_sum += seq_reward
        token_weight_sum += mask_sum
        sequence_reward_sum += seq_reward

        dataset_idx = dataset_indices_D[i]
        stats = per_dataset_stats.setdefault(
            dataset_idx, {"token_sum": 0.0, "mask_sum": 0.0, "seq_sum": 0.0, "count": 0.0}
        )
        stats["token_sum"] += seq_reward
        stats["mask_sum"] += mask_sum
        stats["seq_sum"] += seq_reward
        stats["count"] += 1.0

    metrics: Dict[str, float] = {}
    if token_weight_sum > 0:
        metrics["extrapolation/token_reward_avg"] = reward_scale * token_reward_sum / token_weight_sum
    if len(data_D) > 0:
        metrics["extrapolation/sequence_reward_avg"] = reward_scale * sequence_reward_sum / len(data_D)

    for dataset_idx, stats in per_dataset_stats.items():
        if stats["mask_sum"] > 0:
            metrics[
                f"extrapolation/token_reward_avg/dataset_{dataset_idx}"
            ] = reward_scale * stats["token_sum"] / stats["mask_sum"]
        if stats["count"] > 0:
            metrics[
                f"extrapolation/sequence_reward_avg/dataset_{dataset_idx}"
            ] = reward_scale * stats["seq_sum"] / stats["count"]

    return metrics


@chz.chz
class Config:
    learning_rate: float
    dataset_configs: List[ExtrapolationDatasetConfig]
    model_name: str
    max_tokens: int
    compute_post_kl: bool = False
    evaluator_builders: list[SamplingClientEvaluatorBuilder] = chz.field(default_factory=list)
    lora_rank: int = 32

    reward_mode: Literal["token", "sequence"] = "token"
    reward_scale: float = 1.0

    # Loss function to use for training: "importance_sampling" or "ppo"
    loss_fn: Literal["importance_sampling", "ppo"] = "importance_sampling"

    # Number of optimizer steps per training iteration.
    # Useful for very large batch sizes.
    num_substeps: int = 1

    wandb_project: str | None = None
    wandb_name: str | None = None

    log_path: str = chz.field(munger=lambda _, s: os.path.expanduser(s))
    base_url: str | None = None
    enable_trace: bool = False

    eval_every: int = 20
    save_every: int = 20
    load_checkpoint_path: str | None = None


@scope
async def prepare_minibatch(
    env_group_builders_P: Sequence[EnvGroupBuilder],
    trajectory_groups_P: list[TrajectoryGroup],
    tokenizer: Tokenizer,
    dataset_indices_P: List[int],
    teacher_clients: List[tinker.SamplingClient],
    reference_clients: List[tinker.SamplingClient],
    reward_mode: Literal["token", "sequence"],
    reward_scale: float,
) -> tuple[list[tinker.Datum], dict[str, Any]]:
    """Converts the trajectories into a minibatch, and provides metrics about the minibatch"""

    # Compute trajectory metrics
    metrics = {}
    taglist_P = [env_group_builder.logging_tags() for env_group_builder in env_group_builders_P]
    metrics.update(compute_trajectory_metrics(trajectory_groups_P, taglist_P))

    # Assemble training data
    with timed("assemble_training_data", metrics):
        advantages_P = compute_advantages(trajectory_groups_P)
        data_D, metadata_D = assemble_training_data(trajectory_groups_P, advantages_P)

    # Print one datum per dataset
    printed_datasets = set()
    for datum, metadata in zip(data_D, metadata_D):
        dataset_idx = dataset_indices_P[metadata["group_idx"]]
        if dataset_idx not in printed_datasets:
            logger.info(colorize_example(datum, tokenizer, key="mask"))
            printed_datasets.add(dataset_idx)

    with timed("compute_extrapolation_reward", metrics):
        teacher_clients_D = [
            teacher_clients[dataset_indices_P[metadata["group_idx"]]] for metadata in metadata_D
        ]
        reference_clients_D = [
            reference_clients[dataset_indices_P[metadata["group_idx"]]] for metadata in metadata_D
        ]
        dataset_indices_D = [
            dataset_indices_P[metadata["group_idx"]] for metadata in metadata_D
        ]
        reward_metrics = await incorporate_extrapolation_advantage(
            data_D,
            teacher_clients_D,
            reference_clients_D,
            dataset_indices_D,
            reward_mode=reward_mode,
            reward_scale=reward_scale,
        )
    metrics.update(reward_metrics)

    return data_D, metrics


@scope
async def do_train_step_and_get_sampling_client(
    cfg: Config,
    i_batch: int,
    training_client: tinker.TrainingClient,
    service_client: tinker.ServiceClient,
    tokenizer: Tokenizer,
    env_group_builders_P: Sequence[EnvGroupBuilder],
    trajectory_groups_P: list[TrajectoryGroup],
    dataset_indices_P: List[int],
    teacher_clients: List[tinker.SamplingClient],
    reference_clients: List[tinker.SamplingClient],
) -> tuple[tinker.SamplingClient, dict[str, Any]]:
    context = get_scope_context()
    context.attributes["step"] = i_batch

    metrics = {}
    data_D, prepare_minibatch_metrics = await prepare_minibatch(
        env_group_builders_P,
        trajectory_groups_P,
        tokenizer,
        dataset_indices_P,
        teacher_clients,
        reference_clients,
        reward_mode=cfg.reward_mode,
        reward_scale=cfg.reward_scale,
    )
    metrics.update(prepare_minibatch_metrics)

    with timed("train", metrics):
        training_logprobs_D = await train_step(
            data_D,
            training_client,
            cfg.learning_rate,
            cfg.num_substeps,
            cfg.loss_fn,
        )

    sampling_client, full_batch_metrics = await compute_full_batch_metrics_and_get_sampling_client(
        training_client,
        # NOTE: saving the checkpoint as the i + 1 step
        i_batch + 1,
        data_D,
        training_logprobs_D,
        cfg.log_path,
        cfg.save_every,
        cfg.compute_post_kl,
    )
    metrics.update(full_batch_metrics)

    return sampling_client, metrics


@scope
async def do_sync_training(
    start_batch: int,
    end_batch: int,
    num_batches: int,
    cfg: Config,
    training_client: tinker.TrainingClient,
    service_client: tinker.ServiceClient,
    evaluators: list[SamplingClientEvaluator],
    dataset: CompositeDataset,
    teacher_clients: List[tinker.SamplingClient],
    reference_clients: List[tinker.SamplingClient],
    ml_logger: ml_log.Logger,
    tokenizer: Tokenizer,
):
    """Implements fully synchronous on-policy training"""

    # Initial sampling client
    sampling_client, _ = await save_checkpoint_and_get_sampling_client(
        training_client, start_batch, cfg.log_path, cfg.save_every
    )

    for i_batch in range(start_batch, end_batch):
        metrics = {
            "progress/batch": i_batch,
            "optim/lr": cfg.learning_rate,
            "progress/done_frac": (i_batch + 1) / num_batches,
        }
        t_start = time.time()

        # Run evaluations
        if cfg.eval_every > 0 and i_batch % cfg.eval_every == 0:
            with timed("run_evals", metrics):
                for evaluator in evaluators:
                    eval_metrics = await evaluator(sampling_client)
                    metrics.update({f"test/{k}": v for k, v in eval_metrics.items()})

        # Get batch and sample trajectories
        env_group_builders_P, dataset_indices_P = dataset.get_batch(i_batch)
        with timed("sample", metrics):
            trajectory_groups_P = await asyncio.gather(
                *[
                    asyncio.create_task(
                        do_group_rollout_and_filter_constant_reward(
                            sampling_client,
                            builder,
                            max_tokens=cfg.max_tokens,
                            do_remove_constant_reward_groups=False,
                        ),
                        name=f"sample_task_{i}",
                    )
                    for i, builder in enumerate(env_group_builders_P)
                ],
            )
        trajectory_groups_P = [
            trajectory_group
            for trajectory_group in trajectory_groups_P
            if trajectory_group is not None
        ]

        # Train step
        sampling_client, train_step_metrics = await do_train_step_and_get_sampling_client(
            cfg,
            i_batch,
            training_client,
            service_client,
            tokenizer,
            env_group_builders_P,
            trajectory_groups_P,
            dataset_indices_P,
            teacher_clients,
            reference_clients,
        )

        # Log metrics
        metrics.update(train_step_metrics)
        metrics["time/total"] = time.time() - t_start
        ml_logger.log_metrics(metrics, step=i_batch)


@scope
async def main(
    cfg: Config,
):
    """Main training loop for on-policy distillation."""
    ml_logger = ml_log.setup_logging(
        log_dir=cfg.log_path,
        wandb_project=cfg.wandb_project,
        config=cfg,
        wandb_name=cfg.wandb_name,
    )
    if cfg.enable_trace:
        # Get and rename the current (main) task
        current_task = asyncio.current_task()
        if current_task is not None:
            current_task.set_name("main")
        trace_events_path = os.path.join(cfg.log_path, "trace_events.jsonl")
        logger.info(f"Tracing is enabled. Trace events will be saved to {trace_events_path}")
        logger.info(
            f"Run `python tinker_cookbook/utils/trace.py {trace_events_path} trace.json` and visualize in chrome://tracing or https://ui.perfetto.dev/"
        )
        trace_init(output_file=trace_events_path)

    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("pylatexenc").setLevel(logging.WARNING)

    resume_info = checkpoint_utils.get_last_checkpoint(cfg.log_path)
    if resume_info:
        start_batch = resume_info["batch"]
    else:
        start_batch = 0

    service_client = tinker.ServiceClient(base_url=cfg.base_url)
    training_client = await service_client.create_lora_training_client_async(
        cfg.model_name, rank=cfg.lora_rank
    )

    load_state_path: str | None = (
        resume_info["state_path"] if resume_info else cfg.load_checkpoint_path
    )
    if load_state_path:
        future = await training_client.load_state_async(load_state_path)
        _ = await future.result_async()
        logger.info(f"Loaded state from {load_state_path}")

    # Get tokenizer from training client
    tokenizer = training_client.get_tokenizer()

    # Create datasets and teacher sampling clients from configs
    datasets = []
    teacher_clients = []
    reference_clients = []
    groups_per_batch_list = []
    evaluators = [evaluator() for evaluator in cfg.evaluator_builders]

    for dataset_config in cfg.dataset_configs:
        # Create dataset
        dataset, maybe_test_dataset = await dataset_config.dataset_builder()
        datasets.append(dataset)
        groups_per_batch_list.append(dataset_config.groups_per_batch)

        # Add test dataset evaluator if present
        if maybe_test_dataset is not None:
            evaluators.append(RLTestSetEvaluator(maybe_test_dataset, max_tokens=cfg.max_tokens))

        # Create teacher sampling client
        teacher_config = dataset_config.teacher_config
        teacher_client = service_client.create_sampling_client(base_model=teacher_config.base_model)
        # Load teacher checkpoint if specified
        if teacher_config.load_checkpoint_path is not None:
            teacher_client = service_client.create_sampling_client(
                base_model=teacher_config.base_model,
                model_path=teacher_config.load_checkpoint_path,
            )
        teacher_clients.append(teacher_client)
        logger.info(
            f"Created teacher sampling client for {teacher_config.base_model} "
            f"(checkpoint: {teacher_config.load_checkpoint_path})"
        )

        reference_config = dataset_config.reward_reference_config
        reference_client = service_client.create_sampling_client(
            base_model=reference_config.base_model
        )
        if reference_config.load_checkpoint_path is not None:
            reference_client = service_client.create_sampling_client(
                base_model=reference_config.base_model,
                model_path=reference_config.load_checkpoint_path,
            )
        reference_clients.append(reference_client)
        logger.info(
            f"Created reference sampling client for {reference_config.base_model} "
            f"(checkpoint: {reference_config.load_checkpoint_path})"
        )

    # Wrap datasets in CompositeDataset
    composite_dataset = CompositeDataset(datasets, groups_per_batch_list)
    num_batches = len(composite_dataset)
    logger.info(f"Will train on {num_batches} batches")

    # Training loop
    await do_sync_training(
        start_batch=start_batch,
        end_batch=num_batches,
        num_batches=num_batches,
        cfg=cfg,
        training_client=training_client,
        service_client=service_client,
        evaluators=evaluators,
        dataset=composite_dataset,
        teacher_clients=teacher_clients,
        reference_clients=reference_clients,
        ml_logger=ml_logger,
        tokenizer=tokenizer,
    )

    # Save final checkpoint
    if start_batch < num_batches:
        _ = await checkpoint_utils.save_checkpoint_async(
            training_client=training_client,
            name="final",
            log_path=cfg.log_path,
            kind="both",
            loop_state={"batch": num_batches},
        )
    else:
        logger.info("Training was already complete; nothing to do")

    # Cleanup
    ml_logger.close()
    logger.info("Training completed successfully")
