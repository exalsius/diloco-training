import logging.config
import os

import torch
import torch.distributed as dist
from torch.amp import GradScaler, autocast
from transformers import get_cosine_schedule_with_warmup

from diloco_training.data import DATASET_REGISTRY
from diloco_training.models import MODEL_REGISTRY
from diloco_training.utils.diloco_utils import (
    evaluate_model,
    forward_and_compute_loss,
    get_offloaded_param,
    log_inner_stats,
    log_stats,
    prepare_batch,
    update_inner_optimizer,
    update_outer_optimizer,
)
from diloco_training.utils.heterogeneous import synchronize_batch_and_steps
from diloco_training.utils.exalsius_logger import LOG_CONFIG, get_logger
logging.config.dictConfig(LOG_CONFIG)
logger = get_logger("diloco_training")



class DistributedTrainer:
    def __init__(self, args):
        self.args = args
        self.local_rank = args.local_rank
        self.global_rank = args.global_rank
        self.world_size = args.world_size
        self.batch_size = args.batch_size
        self.local_steps = args.local_steps
        self.per_device_train_batch_size = args.per_device_train_batch_size
        self.heterogeneous = args.heterogeneous
        self.device = args.device
        self.model = args.model
        self.dataset = args.dataset
        self.total_steps = args.total_steps
        self.optim_method = args.optim_method
        self.checkpoint_path = args.checkpoint_path
        self.checkpoint_interval = args.checkpoint_interval
        self.start_step = 0
        self.real_step = 0
        self.count_inner_optimizer_steps = 0
        self.quantization = args.quantization
        # Initialize model
        model_class = MODEL_REGISTRY.get(args.model)
        assert model_class, f"Model {args.model} not found"
        self.model_config, self.model = self.initialize_model(model_class)

        # Initialize dataset
        dataset_class = DATASET_REGISTRY.get(args.dataset)
        assert dataset_class, f"Dataset {args.dataset} not found"
        self.train_dataloader, self.val_dataloader = self.initialize_dataset(dataset_class)

        # Initialize optimizers and scheduler
        self.inner_optimizer, self.outer_optimizer = self.initialize_optimizers()
        self.scheduler = get_cosine_schedule_with_warmup(
            self.inner_optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=args.total_steps,
        )

        # Other initialization
        self.scaler = GradScaler(device=self.device, enabled=(self.device == "cuda"))
        self.gradient_accumulation_steps = args.batch_size // args.per_device_train_batch_size
        self.params_offloaded = get_offloaded_param(self.outer_optimizer, device=self.device)
        self.reference_params = [param.clone().detach() for param in self.model.parameters()]
        self.total_bytes_sent, self.sync_count = 0, 0
        self.loss_batch = 0

    def heterogeneous_profiling(self):
        if dist.is_initialized() and self.heterogeneous:
            per_device_batch_size, local_steps, all_info = synchronize_batch_and_steps(
                DistributedTrainer, self.args
            )
            logger.info(f"Profiling results: {all_info}, Local_steps: {local_steps}")
        self.per_device_train_batch_size = per_device_batch_size
        self.local_steps = local_steps

    def initialize_model(self, model_class):
        config, model = model_class()
        model = model.to(self.device)
        for param in model.parameters():
            dist.broadcast(param.data, src=0)
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
                self.world_size, self.local_rank, self.args.per_device_train_batch_size, split="train"
            )
        return train_dataloader, val_dataloader

    def initialize_optimizers(self):
        inner_optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.args.lr, weight_decay=0.1, betas=(0.9, 0.95)
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

    def train(self):
        self.model.train()
        logger.info(f"Gradient accumulation steps: {self.gradient_accumulation_steps}")
        if self.optim_method == "ddp":
            self._train_ddp()
        else:
            self._train_diloco()

    def _train_ddp(self):
        for step, batch in enumerate(self.train_dataloader):
            if step < self.start_step:
                continue
            if step == self.start_step:
                logger.info(f"Starting DDP training from step {self.start_step}...")

            real_step = (step + 1) // self.gradient_accumulation_steps
            step_within_grad_acc = (step + 1) % self.gradient_accumulation_steps

            batch = prepare_batch(batch, device=self.device)
            with autocast(
                device_type=self.device,
                dtype=torch.bfloat16 if self.device == "cuda" else torch.float16,
            ):
                loss = forward_and_compute_loss(self.model, batch, self.gradient_accumulation_steps)
                self.loss_batch += loss.detach()

            self.scaler.scale(loss).backward()

            if step_within_grad_acc == 0:
                logger.info(
                    f"Local rank {self.local_rank} - Real Step {real_step} - Loss: {self.loss_batch.item()}"
                )
                self.scaler.step(self.inner_optimizer)
                self.scaler.update()
                self.inner_optimizer.zero_grad()
                self.scheduler.step()

                if real_step % self.checkpoint_interval == 0 and not self.heterogeneous:
                    val_stats = evaluate_model(
                        self.val_dataloader, self.model, self.local_rank, self.global_rank, self.device
                    )
                    # dist.barrier()
                    log_stats(
                        self.local_rank,
                        real_step,
                        self.loss_batch,
                        self.world_size,
                        self.batch_size,
                        self.optim_method,
                        self.sync_count,
                        self.total_bytes_sent,
                        val_stats,
                        self.local_steps,
                        self.per_device_train_batch_size,
                        self.args,
                    )
                    self.save_checkpoint(step, real_step)
                self.loss_batch = 0 if self.total_steps > real_step else self.loss_batch

            if self.total_steps != -1 and self.total_steps <= real_step:
                logger.info(f"DDP: Reached total_steps at step {real_step}")
                if not self.heterogeneous:
                    val_stats = evaluate_model(
                        self.val_dataloader, self.model, self.local_rank, self.global_rank, self.device
                    )
                    # dist.barrier()
                    log_stats(
                        self.local_rank,
                        real_step,
                        self.loss_batch,
                        self.world_size,
                        self.batch_size,
                        self.optim_method,
                        self.sync_count,
                        self.total_bytes_sent,
                        val_stats,
                        self.local_steps,
                        self.per_device_train_batch_size,
                        self.args,
                    )
                    self.save_checkpoint(step, real_step)
                return

    def _train_diloco(self):

        for step, batch in enumerate(self.train_dataloader):

            if step < self.start_step:
                continue

            if step == self.start_step:
                logger.info(f"Starting training from step {self.start_step}...")

            real_step = (step + 1) // self.gradient_accumulation_steps
            step_within_grad_acc = (step + 1) % self.gradient_accumulation_steps

            # Measure time for preparing the batch
            batch = prepare_batch(batch, device=self.device)

            # Measure time for forward pass and loss computation
            with autocast(
                device_type=self.device,
                dtype=torch.bfloat16 if self.device == "cuda" else torch.float16,
            ):  # Enable mixed precision for forward pass
                loss = forward_and_compute_loss(self.model, batch, self.gradient_accumulation_steps)
                self.loss_batch += loss.detach()

            # Measure time for backward pass
            self.scaler.scale(loss).backward()  # Scale the loss before backward pass

            if step_within_grad_acc == 0:
                # Measure time for optimizer update
                logger.info(
                    f"Local rank {self.local_rank} - Real Step {real_step} - Loss: {self.loss_batch.item()}, current step sync interval: {self.local_steps}"
                )
                update_inner_optimizer(self.inner_optimizer, self.scheduler, self.model, self.scaler)
                self.count_inner_optimizer_steps += 1
                # Measure time for parameter drift computation

                log_inner_stats(
                    self.local_rank,
                    real_step,
                    self.loss_batch,
                    self.sync_count,
                )

                if self.count_inner_optimizer_steps % self.local_steps == 0:
                    self.count_inner_optimizer_steps = 0
                    # Measure time for outer optimizer sync
                    logger.info(
                        f"Local rank {self.local_rank} - Syncing outer optimizer at step {real_step}"
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
                        device=self.device,
                        quantization=self.quantization
                    )
                    self.params_offloaded = get_offloaded_param(self.outer_optimizer, device=self.device)

                    # Update reference parameters after outer optimizer sync
                    self.reference_params = [
                        param.clone().detach() for param in self.model.parameters()
                    ]

                    # Update the total bytes sent and received
                    self.total_bytes_sent += bytes_sent
                    self.sync_count += 1

                if real_step % self.checkpoint_interval == 0 and not self.heterogeneous:
                    # Measure time for checkpointing
                    val_stats = evaluate_model(
                        self.val_dataloader, self.model, self.local_rank, self.global_rank, self.device
                    )
                    dist.barrier()
                    log_stats(
                        self.local_rank,
                        real_step,
                        self.loss_batch,
                        self.world_size,
                        self.batch_size,
                        self.optim_method,
                        self.sync_count,
                        self.total_bytes_sent,
                        val_stats,
                        self.local_steps,
                        self.per_device_train_batch_size,
                        self.args,
                    )

                    self.save_checkpoint(step, real_step)

                self.loss_batch = 0 if self.total_steps > real_step else self.loss_batch

            if self.total_steps != -1 and self.total_steps <= real_step:
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
                    device=self.device,
                )
                self.params_offloaded = get_offloaded_param(self.outer_optimizer, device=self.device)

                # Update reference parameters after outer optimizer sync
                self.reference_params = [param.clone().detach() for param in self.model.parameters()]

                # Update the total bytes sent and received
                self.total_bytes_sent += bytes_sent
                self.sync_count += 1
                if not self.heterogeneous:
                    val_stats = evaluate_model(
                        self.val_dataloader, self.model, self.local_rank, self.global_rank, self.device
                    )
                    dist.barrier()
                    log_stats(
                        self.local_rank,
                        real_step,
                        self.loss_batch,
                        self.world_size,
                        self.batch_size,
                        self.optim_method,
                        self.sync_count,
                        self.total_bytes_sent,
                        val_stats,
                        self.local_steps,
                        self.per_device_train_batch_size,
                        self.args,
                    )
                    self.save_checkpoint(step, real_step)
                break

    def save_checkpoint(self, step, real_step):
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "inner_optimizer_state_dict": self.inner_optimizer.state_dict(),
            "outer_optimizer_state_dict": (
                self.outer_optimizer.state_dict() if self.outer_optimizer is not None else None
            ),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "step": step,
            "optim_method": self.optim_method,
            "real_step": real_step,
            "loss_batch": self.loss_batch,
            "total_mb_sent": self.total_bytes_sent,
            "sync_count": self.sync_count,
            "count_inner_optimizer_steps": self.count_inner_optimizer_steps,
        }
        checkpoint_file = f"{self.checkpoint_path}_{self.model}_{self.dataset}_node_{self.global_rank}_rank_{self.local_rank}_optim_{self.optim_method}.pth"
        torch.save(checkpoint, checkpoint_file)
        logger.info(
            f"Checkpoint saved for global rank {self.global_rank} and local rank {self.local_rank}, optim method {self.optim_method}"
        )

    def load_checkpoint(self):
        checkpoint_file = f"{self.checkpoint_path.split(".")[0]}.pth_{self.model}_{self.dataset}_node_{self.global_rank}_rank_{self.local_rank}_optim_{self.optim_method}.pth"
        if os.path.isfile(checkpoint_file):
            checkpoint = torch.load(checkpoint_file)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.inner_optimizer.load_state_dict(checkpoint["inner_optimizer_state_dict"])
            if self.outer_optimizer is not None:
                self.outer_optimizer.load_state_dict(checkpoint["outer_optimizer_state_dict"])
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            step = checkpoint["step"] + 1
            self.loss_batch = checkpoint["loss_batch"]
            self.real_step = checkpoint["real_step"]
            self.total_bytes_sent = checkpoint["total_mb_sent"]
            self.sync_count = checkpoint["sync_count"]
            self.count_inner_optimizer_steps = checkpoint["count_inner_optimizer_steps"]

            logger.info(
                f"Checkpoint loaded from step {step} for global rank {self.global_rank} and local rank {self.local_rank}"
            )
        else:
            logger.info(
                f"No checkpoint found for global rank {self.global_rank} and local rank {self.local_rank}, starting from scratch"
            )
