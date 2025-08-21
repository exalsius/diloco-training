import os
import subprocess
import sys
from pathlib import Path

import pytest


def get_project_root():
    """Get the path to the project root directory."""
    return str(Path(__file__).parent.parent.parent)


def run_trainer(args_dict):
    """
    Run the trainer script with given arguments.

    Args:
        args_dict (dict): Dictionary of argument names and values
    """
    # Convert dictionary to CLI arguments
    args = []
    for key, value in args_dict.items():
        if isinstance(value, bool):
            if value:
                args.append(f"--{key}")
        else:
            args.extend([f"--{key}", str(value)])

    script_path = os.path.join(
        get_project_root(), "diloco_training", "training", "start_training.py"
    )

    # Set up environment variables needed for distributed training
    env = os.environ.copy()
    env.update(
        {
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": "29500",
            "WORLD_SIZE": "1",
            "LOCAL_RANK": "0",
            "RANK": "0",
        }
    )

    # Run the command
    command = [sys.executable, script_path] + args
    process = subprocess.Popen(
        command, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )

    stdout, stderr = process.communicate()
    return process.returncode, stdout.decode(), stderr.decode()


@pytest.mark.integration
def test_basic_training_call():
    """Test basic training call with minimal arguments."""
    args = {
        "model": "gpt-neo-tiny",
        "dataset": "test_squence_dataset",
        "batch_size": "4",
        "local_steps": "2",
        "total_steps": "4",
        "per_device_train_batch_size": "2",
        "device": "cpu",  # Use CPU for testing
        "lr": "0.1",
        "outer_lr": "0.1",
        "warmup_steps": "2",
        "checkpoint_path": "test_checkpoint.pth",
        "checkpoint_interval": "10",
        "wandb_project_name": "test",
        "compression_decay": "0.9",
        "compression_topk": "32",
        "experiment_description": "Test run",
        "seed": "42",
        "heterogeneous": False,
        "quantization": False,
    }

    returncode, stdout, stderr = run_trainer(args)
    assert (
        returncode == 0
        or "Training completed" in stdout
        or "Evaluation time:" in stdout
    ), f"Training failed with error:\nstdout: {stdout}\nstderr: {stderr}"


@pytest.mark.integration
def test_invalid_dataset():
    """Test that the script fails properly with invalid dataset."""
    args = {
        "model": "gpt-neo-tiny",
        "dataset": "invalid_dataset",
        "batch_size": "4",
        "local_steps": "2",
        "device": "cpu",
        "lr": "0.1",
        "outer_lr": "0.1",
        "warmup_steps": "2",
        "checkpoint_path": "test_checkpoint.pth",
        "checkpoint_interval": "10",
        "wandb_project_name": "test",
        "compression_decay": "0.9",
        "compression_topk": "32",
        "experiment_description": "Test run",
        "seed": "42",
        "heterogeneous": False,
        "quantization": False,
    }

    returncode, stdout, stderr = run_trainer(args)

    # Check if the script failed (non-zero return code)
    assert (
        returncode != 0
    ), f"Expected failure for invalid dataset, but got return code 0. stdout: {stdout}\nstderr: {stderr}"

    # Check for error messages in either stdout or stderr
    combined_output = (stdout + stderr).lower()
    assert any(
        error_term in combined_output
        for error_term in [
            "error",
            "not found",
            "invalid",
            "failed",
            "exception",
            "traceback",
        ]
    ), f"No error message found in output. stdout: {stdout}\nstderr: {stderr}"


@pytest.mark.integration
def test_all_optimizers():
    """Test training with different optimizer methods."""
    base_args = {
        "model": "gpt-neo-tiny",
        "dataset": "test_squence_dataset",
        "batch_size": "4",
        "local_steps": "2",
        "total_steps": "4",
        "per_device_train_batch_size": "2",
        "device": "cpu",
        "lr": "0.1",
        "outer_lr": "0.1",
        "warmup_steps": "2",
        "checkpoint_path": "test_checkpoint.pth",
        "checkpoint_interval": "10",
        "wandb_project_name": "test",
        "compression_decay": "0.9",
        "compression_topk": "32",
        "experiment_description": "Test run",
        "seed": "42",
        "heterogeneous": False,
        "quantization": False,
    }

    for optim_method in ["demo", "sgd"]:
        args = base_args.copy()
        args["optim_method"] = optim_method

        returncode, stdout, stderr = run_trainer(args)
        assert (
            returncode == 0
            or "Training completed" in stdout
            or "Evaluation time:" in stdout
        ), f"Training failed for optimizer {optim_method} with error:\nstdout: {stdout}\nstderr: {stderr}"
