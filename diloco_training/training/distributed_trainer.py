import logging.config
import os

import torch
import torch.distributed as dist
from torch.amp import GradScaler, autocast
from transformers import get_cosine_schedule_with_warmup

from diloco_training.data import DATASET_REGISTRY
from diloco_training.models import MODEL_REGISTRY
from diloco_training.training.training_config import TrainingConfig
from diloco_training.utils.diloco_utils import (
    evaluate_model,
    forward_and_compute_loss,
    get_offloaded_param,
    prepare_batch,
    update_inner_optimizer,
    update_outer_optimizer,
)
from diloco_training.utils.exalsius_logger import LOG_CONFIG, get_logger
from diloco_training.utils.heterogeneous import synchronize_batch_and_steps
from diloco_training.utils.metrics_logger import (
    MetricsLogger,
    log_dataset_config,
    log_model_config,
)

logging.config.dictConfig(LOG_CONFIG)
logger = get_logger("diloco_training")


class DistributedTrainer:
    def __init__(
        self, config: TrainingConfig, local_rank: int, global_rank: int, world_size: int
    ):
        self.config = config
        self.pgroup_backend = config.pgroup_backend
        # srnbckr: this if for backwards compatability: it would be better to refactor the code
        # to not use "args" anymore.
        self.args = self.config
        # Add distributed training attributes that aren't in TrainingConfig
        self.local_rank = local_rank
        self.global_rank = global_rank
        self.world_size = world_size

        # Use config attributes directly
        self.batch_size = config.batch_size
        self.local_steps = config.local_steps
        self.per_device_train_batch_size = config.per_device_train_batch_size
        self.heterogeneous = config.heterogeneous
        self.device = config.device
        self.model_name = config.model
        self.dataset = config.dataset
        self.total_steps = config.total_steps
        self.optim_method = config.optim_method
        self.checkpoint_path = config.checkpoint_path
        self.checkpoint_interval = config.checkpoint_interval
        self.count_outer_optimizer_steps = 1
        self.quantization = config.quantization
        self.lr = config.lr
        self.outer_lr = config.outer_lr
        self.warmup_steps = config.warmup_steps

        # Calculate sum_local_steps for distributed training
        self.sum_local_steps = self.local_steps * self.world_size

        # Initialize model
        model_class = MODEL_REGISTRY.get(config.model)
        assert model_class, f"Model {config.model} not found"
        self.model_config, self.model = self.initialize_model(model_class)

        # Initialize dataset
        dataset_class = DATASET_REGISTRY.get(config.dataset)
        assert dataset_class, f"Dataset {config.dataset} not found"
        self.train_dataloader, self.val_dataloader = self.initialize_dataset(
            dataset_class
        )

        # Initialize optimizers and scheduler
        self.inner_optimizer, self.outer_optimizer = self.initialize_optimizers()

        # GAN-specific: Initialize separate optimizers for generator and discriminator
        self.is_gan = hasattr(self.model, "generator") and hasattr(
            self.model, "discriminator"
        )
        if self.is_gan:
            self.inner_optimizer_g, self.outer_optimizer_g = (
                self.initialize_gan_optimizers("generator")
            )
            self.inner_optimizer_d, self.outer_optimizer_d = (
                self.initialize_gan_optimizers("discriminator")
            )
            self.scheduler_g = get_cosine_schedule_with_warmup(
                self.inner_optimizer_g,
                num_warmup_steps=config.warmup_steps,
                num_training_steps=config.total_steps,
            )
            self.scheduler_d = get_cosine_schedule_with_warmup(
                self.inner_optimizer_d,
                num_warmup_steps=config.warmup_steps,
                num_training_steps=config.total_steps,
            )
            self.params_offloaded_g = get_offloaded_param(
                self.outer_optimizer_g, device=self.device
            )
            self.params_offloaded_d = get_offloaded_param(
                self.outer_optimizer_d, device=self.device
            )
        else:
            self.scheduler = get_cosine_schedule_with_warmup(
                self.inner_optimizer,
                num_warmup_steps=config.warmup_steps,
                num_training_steps=config.total_steps,
            )
            self.params_offloaded = get_offloaded_param(
                self.outer_optimizer, device=self.device
            )

        # Other initialization
        self.scaler = GradScaler(device=self.device, enabled=(self.device == "cuda"))
        self.gradient_accumulation_steps = (
            config.batch_size // config.per_device_train_batch_size
        )
        self.params_offloaded = get_offloaded_param(
            self.outer_optimizer, device=self.device
        )
        self.reference_params = [
            param.clone().detach() for param in self.model.parameters()
        ]
        self.total_bytes_sent, self.sync_count = 0, 0
        self.loss_batch = 0

        self.start_step = 0
        self.real_step = 0
        self.count_inner_optimizer_steps = 0

        # Initialize metrics logger
        self.metrics_logger = MetricsLogger(
            local_rank=self.local_rank,
            global_rank=self.global_rank,
            world_size=self.world_size,
            device=self.device,
            gpu_type=self.config.gpu_type,
            wandb_logging=self.config.wandb_logging,
        )

        # Log model and dataset configurations
        if self.local_rank == 0:
            log_model_config(
                self.model_config,
                self.model_name,
                wandb_logging=self.config.wandb_logging,
            )
            log_dataset_config(self.dataset, wandb_logging=self.config.wandb_logging)

    def heterogeneous_profiling(self):
        """Run heterogeneous profiling to optimize batch sizes and steps."""
        if dist.is_initialized() and self.heterogeneous:
            (
                per_device_batch_size,
                local_steps,
                total_steps,
                all_info,
            ) = synchronize_batch_and_steps(
                self.config, self.local_rank, self.global_rank, self.world_size
            )

            logger.info(
                f"Profiling results: {all_info}, Local_steps: {local_steps}, Total steps: {total_steps}"
            )

            self.per_device_train_batch_size = per_device_batch_size
            self.local_steps = local_steps
            self.total_steps = total_steps
            self.all_info = all_info
            self.heterogeneous = False

    def initialize_model(self, model_class):
        config, model = model_class(cache_dir=self.config.model_cache_dir)
        model = model.to(self.device)
        if not self.heterogeneous:
            for param in model.parameters():
                dist.broadcast(param.data, src=0)
        # torch.compile (must be before DDP wrapping)
        if getattr(self.config, "compile_model", False):
            try:
                compile_backend = getattr(self.config, "compile_backend", "inductor")
                compile_mode = getattr(self.config, "compile_mode", "default")
                logger.info(
                    f"Compiling model with torch.compile backend={compile_backend} mode={compile_mode}"
                )
                model = torch.compile(model, backend=compile_backend, mode=compile_mode)
                logger.info("torch.compile succeeded")
            except Exception as e:
                logger.warning(
                    f"torch.compile failed, continuing without compilation: {e}"
                )
        if self.optim_method == "ddp":
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[self.local_rank] if self.device == "cuda" else None
            )
        return config, model

    def initialize_dataset(self, dataset_class):
        if self.args.dataset == "test_squence_dataset":
            _, train_dataloader = dataset_class(
                self.args.batch_size,
                vocab_size=self.model_config.vocab_size,
                num_samples=-1,
            )
            val_dataloader = train_dataloader
        else:
            train_dataloader, val_dataloader = dataset_class(
                self.world_size,
                self.global_rank,
                self.args.per_device_train_batch_size,
                split="train",
                cache_dir=self.config.dataset_cache_dir,
            )
        return train_dataloader, val_dataloader

    def initialize_optimizers(self):
        inner_optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.args.lr,
            weight_decay=0.1,
            betas=(0.9, 0.95),
        )
        if self.optim_method == "ddp":
            return inner_optimizer, inner_optimizer

        optimizer_config = {"params": self.model.parameters(), "lr": self.args.outer_lr}
        if self.optim_method == "demo":
            from diloco_training.utils.demo_optimizer import DeMo

            optimizer_config.update(
                {
                    "compression_decay": self.args.compression_decay,
                    "compression_topk": self.args.compression_topk,
                    "compression_chunk": 64,
                }
            )
            outer_optimizer = DeMo(**optimizer_config)
        elif self.optim_method in ["sgd", "sgd_quantized"]:
            optimizer_config.update({"momentum": 0.9, "nesterov": True})
            outer_optimizer = torch.optim.SGD(**optimizer_config)
        else:
            raise ValueError(f"Unknown optimization method: {self.optim_method}")
        return inner_optimizer, outer_optimizer

    def initialize_gan_optimizers(self, component):
        """Initialize optimizers for GAN components (generator or discriminator)"""

        inner_optimizer = torch.optim.AdamW(
            (
                self.model.generator.parameters()
                if component == "generator"
                else self.model.discriminator.parameters()
            ),
            lr=self.args.lr,
            weight_decay=0.1,
            betas=(0.9, 0.95),
        )

        if self.optim_method == "ddp":
            return inner_optimizer, inner_optimizer

        optimizer_config = {
            "params": (
                self.model.generator.parameters()
                if component == "generator"
                else self.model.discriminator.parameters()
            ),
            "lr": self.args.outer_lr,
        }
        if self.optim_method == "demo":
            from diloco_training.utils.demo_optimizer import DeMo

            optimizer_config.update(
                {
                    "compression_decay": self.args.compression_decay,
                    "compression_topk": self.args.compression_topk,
                    "compression_chunk": 64,
                }
            )
            outer_optimizer = DeMo(**optimizer_config)
        elif self.optim_method in ["sgd", "sgd_quantized"]:
            optimizer_config.update({"momentum": 0.9, "nesterov": True})
            outer_optimizer = torch.optim.SGD(**optimizer_config)
        else:
            raise ValueError(f"Unknown optimization method: {self.optim_method}")
        return inner_optimizer, outer_optimizer

    def train(self):
        self.model.train()
        # Distribute total steps among workers if not in heterogeneous mode
        # if not self.heterogeneous:
        #     self.total_steps = self.distribute_total_steps()

        logger.info(f"Gradient accumulation steps: {self.gradient_accumulation_steps}")
        logger.info(f"Worker {self.global_rank} will execute {self.total_steps} steps")

        if self.is_gan:
            self._train_gan()
        elif self.optim_method == "ddp":
            self._train_ddp()
        else:
            self._train_diloco()

    def _train_ddp(self):
        step = self.start_step
        for _, batch in enumerate(self.train_dataloader):
            # if step < self.start_step:
            #     continue
            # if step == self.start_step:
            #     logger.info(f"Starting DDP training from step {self.start_step}...")

            real_step = (step + 1) // self.gradient_accumulation_steps
            step_within_grad_acc = (step + 1) % self.gradient_accumulation_steps

            batch = prepare_batch(batch, device=self.device)
            with autocast(
                device_type=self.device,
                dtype=torch.bfloat16 if self.device == "cuda" else torch.float16,
            ):
                loss = forward_and_compute_loss(
                    self.model, batch, self.gradient_accumulation_steps
                )
                self.loss_batch += loss.detach()

            self.scaler.scale(loss).backward()

            if step_within_grad_acc == 0:
                logger.info(
                    f"Global rank {self.global_rank} - Local rank {self.local_rank} - Real Step {real_step} - Loss: {self.loss_batch.item()}"
                )
                self.scaler.step(self.inner_optimizer)
                self.scaler.update()
                self.inner_optimizer.zero_grad()
                self.scheduler.step()
                # here checkpoint interval is in number of steps
                if real_step % self.checkpoint_interval == 0 and not self.heterogeneous:
                    val_stats = evaluate_model(
                        self.val_dataloader,
                        self.model,
                        self.local_rank,
                        self.global_rank,
                        self.device,
                    )
                    self.metrics_logger.log_outer_step_metrics(
                        real_step=real_step,
                        loss=self.loss_batch,
                        world_size=self.world_size,
                        batch_size=self.batch_size,
                        optim_method=self.optim_method,
                        sync_count=self.sync_count,
                        total_bytes_sent=self.total_bytes_sent,
                        val_stats=val_stats,
                        local_steps=self.local_steps,
                        per_device_train_batch_size=self.per_device_train_batch_size,
                    )
                    self.save_checkpoint(step, real_step)
                self.loss_batch = 0 if self.total_steps > real_step else self.loss_batch

            if self.total_steps != -1 and self.total_steps <= real_step:
                logger.info(f"DDP: Reached total_steps at step {real_step}")
                if not self.heterogeneous:
                    val_stats = evaluate_model(
                        self.val_dataloader,
                        self.model,
                        self.local_rank,
                        self.global_rank,
                        self.device,
                    )
                    self.metrics_logger.log_outer_step_metrics(
                        real_step=real_step,
                        loss=self.loss_batch,
                        world_size=self.world_size,
                        batch_size=self.batch_size,
                        optim_method=self.optim_method,
                        sync_count=self.sync_count,
                        total_bytes_sent=self.total_bytes_sent,
                        val_stats=val_stats,
                        local_steps=self.local_steps,
                        per_device_train_batch_size=self.per_device_train_batch_size,
                    )
                    self.save_checkpoint(step, real_step, is_final_step=True)
                return
            step += 1

    def _train_diloco(self):
        step = self.start_step
        print("Step:", step)
        for _, batch in enumerate(self.train_dataloader):
            # if step < self.start_step:
            #     continue

            # if step == self.start_step:
            #     logger.info(f"Starting training from step {self.start_step}...")

            real_step = (step + 1) // self.gradient_accumulation_steps
            step_within_grad_acc = (step + 1) % self.gradient_accumulation_steps

            # Start timing for batch preparation
            self.metrics_logger.start_timer("batch_preparation")
            batch = prepare_batch(batch, device=self.device)
            self.metrics_logger.end_timer("batch_preparation")

            # Start timing for forward pass
            self.metrics_logger.start_timer("forward_pass")
            with autocast(
                device_type=self.device,
                dtype=torch.bfloat16 if self.device == "cuda" else torch.float16,
            ):
                loss = forward_and_compute_loss(
                    self.model, batch, self.gradient_accumulation_steps
                )
                self.loss_batch += loss.detach()
            self.metrics_logger.end_timer("forward_pass")

            # Start timing for backward pass
            self.metrics_logger.start_timer("backward_pass")
            self.scaler.scale(loss).backward()
            self.metrics_logger.end_timer("backward_pass")

            if step_within_grad_acc == 0:
                # Start timing for optimizer update
                self.metrics_logger.start_timer("inner_optimizer_step")
                logger.info(
                    f"Global rank {self.global_rank} - Local rank {self.local_rank} - Real Step {real_step} - Loss: {self.loss_batch.item()}, current step sync interval: {self.local_steps}"
                )
                update_inner_optimizer(
                    self.inner_optimizer, self.scheduler, self.model, self.scaler
                )
                self.count_inner_optimizer_steps += 1
                self.metrics_logger.end_timer("inner_optimizer_step")

                # Log inner training metrics
                self.metrics_logger.log_training_metrics(
                    step=real_step,
                    loss=self.loss_batch.item(),
                    loss_type="inner",
                    sync_count=self.sync_count,
                )

                # Log system metrics periodically
                if real_step % 10 == 0:
                    self.metrics_logger.log_system_metrics()

                # Log throughput metrics
                batch_size = batch[list(batch.keys())[0]].size(0)
                if (
                    hasattr(self.metrics_logger, "timers")
                    and "forward_pass" in self.metrics_logger.timers
                ):
                    forward_time = (
                        self.metrics_logger.timers["forward_pass"][-1]
                        if self.metrics_logger.timers["forward_pass"]
                        else 0
                    )
                    if forward_time > 0:
                        self.metrics_logger.log_throughput_metrics(
                            batch_size, forward_time
                        )

                if (
                    self.count_inner_optimizer_steps % self.local_steps == 0
                    and not self.heterogeneous
                ):
                    self.count_inner_optimizer_steps = 0

                    # Start timing for outer optimizer sync
                    self.metrics_logger.start_timer("outer_sync")
                    logger.info(
                        f"Global rank {self.global_rank} - Local rank {self.local_rank} - Syncing outer optimizer at step {real_step}"
                    )
                    main_param = [
                        param
                        for group in self.inner_optimizer.param_groups
                        for param in group["params"]
                    ]
                    bytes_sent = update_outer_optimizer(
                        self.params_offloaded,
                        main_param,
                        self.optim_method,
                        self.world_size,
                        self.outer_optimizer,
                        self.local_steps,
                        backend=self.pgroup_backend,
                        quantization=self.quantization,
                        metrics_logger=self.metrics_logger,
                        sum_local_steps=self.sum_local_steps,
                        async_communication=self.config.async_communication,
                    )
                    self.params_offloaded = get_offloaded_param(
                        self.outer_optimizer, device=self.device
                    )
                    sync_time = self.metrics_logger.end_timer("outer_sync")

                    # Update reference parameters after outer optimizer sync
                    self.reference_params = [
                        param.clone().detach() for param in self.model.parameters()
                    ]

                    # Update the total bytes sent and received
                    self.total_bytes_sent += bytes_sent
                    self.sync_count += 1

                    # Log outer sync metrics
                    self.metrics_logger.log_training_metrics(
                        step=real_step,
                        loss=self.loss_batch.item(),
                        loss_type="outer",
                        sync_time=sync_time,
                        bytes_sent_mb=bytes_sent / (1024 * 1024),
                    )
                    logger.info(
                        f"Global rank {self.global_rank} - Local rank {self.local_rank} - Outer optimizer synced at step {real_step}"
                    )
                    self.count_outer_optimizer_steps += 1

                if (
                    self.count_outer_optimizer_steps % self.checkpoint_interval == 0
                    and not self.heterogeneous
                ):
                    # Start timing for evaluation and checkpointing
                    logger.info(f"Outer loop syncs: {self.count_outer_optimizer_steps}")
                    self.count_outer_optimizer_steps = 1
                    self.metrics_logger.start_timer("evaluation")
                    val_stats = evaluate_model(
                        self.val_dataloader,
                        self.model,
                        self.local_rank,
                        self.global_rank,
                        self.device,
                    )
                    self.metrics_logger.end_timer("evaluation")

                    dist.barrier()
                    self.metrics_logger.log_outer_step_metrics(
                        real_step=real_step,
                        loss=self.loss_batch,
                        world_size=self.world_size,
                        batch_size=self.batch_size,
                        optim_method=self.optim_method,
                        sync_count=self.sync_count,
                        total_bytes_sent=self.total_bytes_sent,
                        val_stats=val_stats,
                        local_steps=self.local_steps,
                        per_device_train_batch_size=self.per_device_train_batch_size,
                    )

                    self.metrics_logger.start_timer("checkpointing")
                    self.save_checkpoint(step, real_step)
                    self.metrics_logger.end_timer("checkpointing")

                    # Log timing summary
                    self.metrics_logger.log_timing_summary(real_step)

                self.loss_batch = 0 if self.total_steps > real_step else self.loss_batch

            if self.total_steps != -1 and self.total_steps <= real_step:
                if self.heterogeneous:
                    break
                # Final outer sync with timing
                self.metrics_logger.start_timer("final_outer_sync")
                logger.info(
                    f"Performing final sync of the outer optimizer at step {real_step}"
                )
                main_param = [
                    param
                    for group in self.inner_optimizer.param_groups
                    for param in group["params"]
                ]
                bytes_sent = update_outer_optimizer(
                    self.params_offloaded,
                    main_param,
                    self.optim_method,
                    self.world_size,
                    self.outer_optimizer,
                    self.local_steps,
                    backend=self.pgroup_backend,
                    metrics_logger=self.metrics_logger,
                    sum_local_steps=self.sum_local_steps,
                    async_communication=self.config.async_communication,
                )
                self.params_offloaded = get_offloaded_param(
                    self.outer_optimizer, device=self.device
                )
                self.metrics_logger.end_timer("final_outer_sync")

                # Update reference parameters after outer optimizer sync
                self.reference_params = [
                    param.clone().detach() for param in self.model.parameters()
                ]

                # Update the total bytes sent and received
                self.total_bytes_sent += bytes_sent
                self.sync_count += 1

                if not self.heterogeneous:
                    val_stats = evaluate_model(
                        self.val_dataloader,
                        self.model,
                        self.local_rank,
                        self.global_rank,
                        self.device,
                    )
                    self.metrics_logger.log_outer_step_metrics(
                        real_step=real_step,
                        loss=self.loss_batch,
                        world_size=self.world_size,
                        batch_size=self.batch_size,
                        optim_method=self.optim_method,
                        sync_count=self.sync_count,
                        total_bytes_sent=self.total_bytes_sent,
                        val_stats=val_stats,
                        local_steps=self.local_steps,
                        per_device_train_batch_size=self.per_device_train_batch_size,
                    )
                    self.save_checkpoint(step, real_step, is_final_step=True)

                # Log final timing summary
                self.metrics_logger.log_timing_summary(real_step)
                break
            step += 1

    def _train_gan(self):
        """Training loop for GAN models with enhanced metrics logging"""
        step = self.start_step
        for _, batch in enumerate(self.train_dataloader):
            # if step < self.start_step:
            #     continue
            # if step == self.start_step:
            #     logger.info(f"Starting GAN training from step {self.start_step}...")

            real_step = (step + 1) // self.gradient_accumulation_steps
            step_within_grad_acc = (step + 1) % self.gradient_accumulation_steps

            # Batch preparation with timing
            self.metrics_logger.start_timer("batch_preparation")
            batch = prepare_batch(batch, device=self.device)
            self.metrics_logger.end_timer("batch_preparation")

            # Train discriminator with timing
            self.metrics_logger.start_timer("discriminator_forward")
            self.model.set_training_mode("discriminator")
            with autocast(
                device_type=self.device,
                dtype=torch.bfloat16 if self.device == "cuda" else torch.float16,
            ):
                d_loss = forward_and_compute_loss(
                    self.model, batch, self.gradient_accumulation_steps
                )
                self.loss_batch += d_loss.detach()
            self.metrics_logger.end_timer("discriminator_forward")

            self.metrics_logger.start_timer("discriminator_backward")
            self.scaler.scale(d_loss).backward()
            self.metrics_logger.end_timer("discriminator_backward")

            # Train generator with timing
            self.metrics_logger.start_timer("generator_forward")
            self.model.set_training_mode("generator")
            with autocast(
                device_type=self.device,
                dtype=torch.bfloat16 if self.device == "cuda" else torch.float16,
            ):
                g_loss = forward_and_compute_loss(
                    self.model, batch, self.gradient_accumulation_steps
                )
            self.metrics_logger.end_timer("generator_forward")

            self.metrics_logger.start_timer("generator_backward")
            self.scaler.scale(g_loss).backward()
            self.metrics_logger.end_timer("generator_backward")

            if step_within_grad_acc == 0:
                logger.info(
                    f"Global rank {self.global_rank} - Local rank {self.local_rank} - Real Step {real_step} - D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}"
                )

                # Update both optimizers with timing
                self.metrics_logger.start_timer("gan_optimizer_step")
                update_inner_optimizer(
                    self.inner_optimizer_d,
                    self.scheduler_d,
                    self.model.discriminator,
                    self.scaler,
                )
                update_inner_optimizer(
                    self.inner_optimizer_g,
                    self.scheduler_g,
                    self.model.generator,
                    self.scaler,
                )
                self.count_inner_optimizer_steps += 1
                self.metrics_logger.end_timer("gan_optimizer_step")

                # Log GAN-specific metrics
                self.metrics_logger.log_training_metrics(
                    step=real_step,
                    loss=self.loss_batch.item(),
                    loss_type="gan",
                    d_loss=d_loss.item(),
                    g_loss=g_loss.item(),
                    sync_count=self.sync_count,
                )

                # Log system metrics periodically
                if real_step % 10 == 0:
                    self.metrics_logger.log_system_metrics()

                # Log throughput metrics
                batch_size = batch[list(batch.keys())[0]].size(0)
                if (
                    hasattr(self.metrics_logger, "timers")
                    and "forward_pass" in self.metrics_logger.timers
                ):
                    forward_time = (
                        self.metrics_logger.timers["forward_pass"][-1]
                        if self.metrics_logger.timers["forward_pass"]
                        else 0
                    )
                    if forward_time > 0:
                        self.metrics_logger.log_throughput_metrics(
                            batch_size, forward_time
                        )

                if (
                    self.count_outer_optimizer_steps % self.checkpoint_interval == 0
                    and not self.heterogeneous
                ):
                    # Start timing for evaluation and checkpointing
                    logger.info(f"Outer loop syncs: {self.count_outer_optimizer_steps}")
                    self.count_outer_optimizer_steps = 1
                    self.metrics_logger.start_timer("evaluation")
                    val_stats = evaluate_model(
                        self.val_dataloader,
                        self.model,
                        self.local_rank,
                        self.global_rank,
                        self.device,
                    )
                    self.metrics_logger.end_timer("evaluation")

                    if self.optim_method != "ddp":
                        dist.barrier()
                    self.metrics_logger.log_outer_step_metrics(
                        real_step=real_step,
                        loss=self.loss_batch,
                        world_size=self.world_size,
                        batch_size=self.batch_size,
                        optim_method=self.optim_method,
                        sync_count=self.sync_count,
                        total_bytes_sent=self.total_bytes_sent,
                        val_stats=val_stats,
                        local_steps=self.local_steps,
                        per_device_train_batch_size=self.per_device_train_batch_size,
                    )

                    self.metrics_logger.start_timer("checkpointing")
                    self.save_checkpoint(step, real_step)
                    self.metrics_logger.end_timer("checkpointing")

                    # Log timing summary
                    self.metrics_logger.log_timing_summary(real_step)

                self.loss_batch = 0 if self.total_steps > real_step else self.loss_batch

            if self.total_steps != -1 and self.total_steps <= real_step:
                if self.heterogeneous:
                    return
                # Final sync for DiLoCo with timing
                if self.optim_method != "ddp":
                    self.metrics_logger.start_timer("final_gan_sync")
                    logger.info(
                        f"Performing final sync of GAN outer optimizers at step {real_step}"
                    )

                    main_param_d = [
                        param
                        for group in self.inner_optimizer_d.param_groups
                        for param in group["params"]
                    ]
                    bytes_sent_d = update_outer_optimizer(
                        self.params_offloaded_d,
                        main_param_d,
                        self.optim_method,
                        self.world_size,
                        self.outer_optimizer_d,
                        self.local_steps,
                        backend=self.pgroup_backend,
                        quantization=self.quantization,
                        metrics_logger=self.metrics_logger,
                        sum_local_steps=self.sum_local_steps,
                        async_communication=self.config.async_communication,
                    )
                    self.params_offloaded_d = get_offloaded_param(
                        self.outer_optimizer_d, device=self.device
                    )

                    main_param_g = [
                        param
                        for group in self.inner_optimizer_g.param_groups
                        for param in group["params"]
                    ]
                    bytes_sent_g = update_outer_optimizer(
                        self.params_offloaded_g,
                        main_param_g,
                        self.optim_method,
                        self.world_size,
                        self.outer_optimizer_g,
                        self.local_steps,
                        backend=self.pgroup_backend,
                        quantization=self.quantization,
                        metrics_logger=self.metrics_logger,
                        sum_local_steps=self.sum_local_steps,
                        async_communication=self.config.async_communication,
                    )
                    self.params_offloaded_g = get_offloaded_param(
                        self.outer_optimizer_g, device=self.device
                    )

                    self.reference_params = [
                        param.clone().detach() for param in self.model.parameters()
                    ]
                    self.total_bytes_sent += bytes_sent_d + bytes_sent_g
                    self.sync_count += 1
                    self.metrics_logger.end_timer("final_gan_sync")

                if not self.heterogeneous:
                    val_stats = evaluate_model(
                        self.val_dataloader,
                        self.model,
                        self.local_rank,
                        self.global_rank,
                        self.device,
                    )
                    self.metrics_logger.log_outer_step_metrics(
                        real_step=real_step,
                        loss=self.loss_batch,
                        world_size=self.world_size,
                        batch_size=self.batch_size,
                        optim_method=self.optim_method,
                        sync_count=self.sync_count,
                        total_bytes_sent=self.total_bytes_sent,
                        val_stats=val_stats,
                        local_steps=self.local_steps,
                        per_device_train_batch_size=self.per_device_train_batch_size,
                    )
                    self.save_checkpoint(step, real_step, is_final_step=True)

                # Log final timing summary
                self.metrics_logger.log_timing_summary(real_step)
                return
            step += 1

    def _train_gan_step(self, batch, real_step, step_within_grad_acc):
        """Handle GAN-specific training step with alternating D and G updates"""
        # Train discriminator
        self.model.set_training_mode("discriminator")
        with autocast(
            device_type=self.device,
            dtype=torch.bfloat16 if self.device == "cuda" else torch.float16,
        ):
            d_output = self.model(batch["image"], batch["label"])
            d_loss = d_output.loss / self.gradient_accumulation_steps
            self.loss_batch += d_loss.detach()

        self.scaler.scale(d_loss).backward()

        # Train generator
        self.model.set_training_mode("generator")
        with autocast(
            device_type=self.device,
            dtype=torch.bfloat16 if self.device == "cuda" else torch.float16,
        ):
            g_output = self.model(batch["image"], batch["label"])
            g_loss = g_output.loss / self.gradient_accumulation_steps

        self.scaler.scale(g_loss).backward()

    def _sync_outer_optimizers(self, real_step):
        """Sync outer optimizers for both GAN and non-GAN models"""
        self.count_inner_optimizer_steps = 0
        logger.info(
            f"Global rank {self.global_rank} - Local rank {self.local_rank} - Syncing outer optimizer at step {real_step}"
        )

        if self.is_gan:
            # Sync discriminator
            main_param_d = [
                param
                for group in self.inner_optimizer_d.param_groups
                for param in group["params"]
            ]
            bytes_sent_d = update_outer_optimizer(
                self.params_offloaded_d,
                main_param_d,
                self.optim_method,
                self.world_size,
                self.outer_optimizer_d,
                self.local_steps,
                backend=self.pgroup_backend,
                quantization=self.quantization,
                sum_local_steps=self.sum_local_steps,
            )
            self.params_offloaded_d = get_offloaded_param(
                self.outer_optimizer_d, device=self.device
            )

            # Sync generator
            main_param_g = [
                param
                for group in self.inner_optimizer_g.param_groups
                for param in group["params"]
            ]
            bytes_sent_g = update_outer_optimizer(
                self.params_offloaded_g,
                main_param_g,
                self.optim_method,
                self.world_size,
                self.outer_optimizer_g,
                self.local_steps,
                backend=self.pgroup_backend,
                quantization=self.quantization,
            )
            self.params_offloaded_g = get_offloaded_param(
                self.outer_optimizer_g, device=self.device
            )

            bytes_sent = bytes_sent_d + bytes_sent_g
        else:
            main_param = [
                param
                for group in self.inner_optimizer.param_groups
                for param in group["params"]
            ]
            bytes_sent = update_outer_optimizer(
                self.params_offloaded,
                main_param,
                self.optim_method,
                self.world_size,
                self.outer_optimizer,
                self.local_steps,
                backend=self.pgroup_backend,
                quantization=self.quantization,
            )
            self.params_offloaded = get_offloaded_param(
                self.outer_optimizer, device=self.device
            )

        self.reference_params = [
            param.clone().detach() for param in self.model.parameters()
        ]
        self.total_bytes_sent += bytes_sent
        self.sync_count += 1

    def save_checkpoint(self, step, real_step, is_final_step=False):
        try:
            loader_state = self.train_dataloader.state_dict()
        except Exception as e:
            logger.warning(f"Could not get dataloader state_dict: {e}")
            loader_state = None
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "train_dataloader": loader_state,
            "step": step,
            "optim_method": self.optim_method,
            "real_step": real_step,
            "loss_batch": self.loss_batch,
            "total_mb_sent": self.total_bytes_sent,
            "sync_count": self.sync_count,
            "count_inner_optimizer_steps": self.count_inner_optimizer_steps,
        }
        if self.is_gan:
            checkpoint.update(
                {
                    "inner_optimizer_g_state_dict": self.inner_optimizer_g.state_dict(),
                    "inner_optimizer_d_state_dict": self.inner_optimizer_d.state_dict(),
                    "scheduler_g_state_dict": self.scheduler_g.state_dict(),
                    "scheduler_d_state_dict": self.scheduler_d.state_dict(),
                }
            )
            if self.optim_method != "ddp":
                checkpoint.update(
                    {
                        "outer_optimizer_g_state_dict": self.outer_optimizer_g.state_dict(),
                        "outer_optimizer_d_state_dict": self.outer_optimizer_d.state_dict(),
                    }
                )
        else:
            checkpoint.update(
                {
                    "inner_optimizer_state_dict": self.inner_optimizer.state_dict(),
                    "scheduler_state_dict": self.scheduler.state_dict(),
                }
            )
            if self.optim_method != "ddp":
                checkpoint["outer_optimizer_state_dict"] = (
                    self.outer_optimizer.state_dict()
                )

        checkpoint_file = f"{str(self.checkpoint_path).split('.')[0]}.pth_node_{self.global_rank}_rank_{self.local_rank}_optim_{self.optim_method}.pth"
        torch.save(checkpoint, checkpoint_file)
        logger.info(
            f"Checkpoint saved for global rank {self.global_rank} and local rank {self.local_rank}, optim method {self.optim_method}"
        )

        # Upload model to Hugging Face Hub if configured and this is the final step
        if (
            is_final_step
            and self.config.hf_upload
            and self.config.trained_model_hf_name
        ):
            # Only upload from global_rank 0 to avoid duplicate uploads
            if self.global_rank == 0:
                self._upload_to_huggingface(step, real_step)
            else:
                logger.info(
                    f"Skipping HF upload for non-zero global rank {self.global_rank}"
                )

    def _upload_to_huggingface(self, step, real_step):
        """
        Upload checkpoint to HuggingFace Hub.
        Requires HF_TOKEN environment variable.
        """
        hf_token = os.environ.get("HUGGINGFACE_TOKEN")
        if not hf_token:
            logger.warning(
                "No HUGGINGFACE_TOKEN found in environment. Skipping HuggingFace upload."
            )
            return

        try:
            from huggingface_hub import HfApi, create_repo

            # Determine repository ID
            repo_id = self.config.trained_model_hf_name
            if not repo_id:
                logger.warning("No trained_model_hf_name configured. Skipping upload.")
                return

            logger.info(f"Uploading checkpoint to HuggingFace: {repo_id}")

            # Create repository if it doesn't exist
            api = HfApi()
            try:
                create_repo(
                    repo_id,
                    token=hf_token,
                    exist_ok=True,
                    private=False,
                    repo_type="model",
                )
                logger.info(f"Repository ready: {repo_id}")
            except Exception as e:
                logger.info(f"Repository check: {e}")

            # Prepare checkpoint file path
            checkpoint_file = f"{str(self.checkpoint_path).split('.')[0]}.pth_node_{self.global_rank}_rank_{self.local_rank}_optim_{self.optim_method}.pth"

            if not os.path.exists(checkpoint_file):
                logger.error(f"Checkpoint file not found: {checkpoint_file}")
                return

            # Upload the checkpoint file
            api.upload_file(
                path_or_fileobj=checkpoint_file,
                path_in_repo=f"checkpoint_step_{step:06d}_rank_{self.global_rank}.pth",
                repo_id=repo_id,
                token=hf_token,
                commit_message=f"Training checkpoint at step {step} (real step {real_step})",
            )

            # Create and upload a metadata file
            import json

            metadata = {
                "model": self.model_name,
                "dataset": self.dataset,
                "training_step": step,
                "real_step": real_step,
                "optimization_method": self.optim_method,
                "learning_rate": self.lr,
                "outer_learning_rate": self.outer_lr,
                "local_steps": self.local_steps,
                "world_size": self.world_size,
                "global_rank": self.global_rank,
                "local_rank": self.local_rank,
            }

            metadata_file = (
                f"{str(self.checkpoint_path).split('.')[0]}_meta_{step:06d}.json"
            )
            with open(metadata_file, "w") as f:
                json.dump(metadata, f, indent=2)

            api.upload_file(
                path_or_fileobj=metadata_file,
                path_in_repo=f"meta_step_{step:06d}_rank_{self.global_rank}.json",
                repo_id=repo_id,
                token=hf_token,
            )

            # Clean up local metadata file
            if os.path.exists(metadata_file):
                os.remove(metadata_file)

            logger.info(f"Successfully uploaded checkpoint to HuggingFace: {repo_id}")

        except ImportError:
            logger.error(
                "huggingface_hub not installed. Install with: pip install huggingface_hub"
            )
        except Exception as e:
            logger.error(f"Failed to upload to HuggingFace: {e}")
            import traceback

            logger.error(traceback.format_exc())

    def load_checkpoint(self):
        checkpoint_file = f"{str(self.checkpoint_path).split('.')[0]}.pth_node_{self.global_rank}_rank_{self.local_rank}_optim_{self.optim_method}.pth"
        if os.path.isfile(checkpoint_file):
            checkpoint = torch.load(checkpoint_file)
            self.model.load_state_dict(checkpoint["model_state_dict"])

            if self.is_gan:
                self.inner_optimizer_g.load_state_dict(
                    checkpoint["inner_optimizer_g_state_dict"]
                )
                self.inner_optimizer_d.load_state_dict(
                    checkpoint["inner_optimizer_d_state_dict"]
                )
                self.scheduler_g.load_state_dict(checkpoint["scheduler_g_state_dict"])
                self.scheduler_d.load_state_dict(checkpoint["scheduler_d.state_dict"])
                if self.optim_method != "ddp":
                    self.outer_optimizer_g.load_state_dict(
                        checkpoint["outer_optimizer_g_state_dict"]
                    )
                    self.outer_optimizer_d.load_state_dict(
                        checkpoint["outer_optimizer_d_state_dict"]
                    )
            else:
                self.inner_optimizer.load_state_dict(
                    checkpoint["inner_optimizer_state_dict"]
                )
                self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                if self.optim_method != "ddp":
                    self.outer_optimizer.load_state_dict(
                        checkpoint["outer_optimizer_state_dict"]
                    )

            step = checkpoint["step"] + 1
            self.loss_batch = checkpoint["loss_batch"]
            if checkpoint["train_dataloader"] is None:
                logger.warning("No dataloader state found in checkpoint.")
            else:
                self.train_dataloader.load_state_dict(checkpoint["train_dataloader"])
                print(
                    f"Resuming from dataloader checkpoint: {checkpoint['train_dataloader']}"
                )
            self.real_step = checkpoint["real_step"]
            self.total_bytes_sent = checkpoint["total_mb_sent"]
            self.sync_count = checkpoint["sync_count"]
            self.count_inner_optimizer_steps = checkpoint["count_inner_optimizer_steps"]
            self.start_step = step

            logger.info(
                f"Checkpoint loaded from step {step} for global rank {self.global_rank} and local rank {self.local_rank}"
            )
        else:
            logger.info(
                f"No checkpoint found for global rank {self.global_rank} and local rank {self.local_rank}, starting from scratch"
            )

    def distribute_total_steps(self):
        """
        Distribute total steps among workers based on their relative speeds.
        For non-heterogeneous setup, this simply divides total steps evenly.
        """
        if self.world_size == 1:
            return self.total_steps

        if hasattr(self, "all_info") and self.all_info:
            # Use profiling information if available (from heterogeneous setup)
            speeds = []
            for info in self.all_info:
                if info is not None and "time_per_batch" in info:
                    # Lower time_per_batch means higher speed
                    speeds.append(1.0 / info["time_per_batch"])
                else:
                    # Default speed if info is missing
                    speeds.append(1.0)

            total_speed = sum(speeds)
            my_speed = speeds[self.global_rank]
            my_steps = int(round(self.config.total_steps * my_speed / total_speed))
            return max(1, my_steps)  # Ensure at least 1 step
        else:
            # For non-heterogeneous setup, divide steps evenly
            return max(1, self.config.total_steps // self.world_size)
