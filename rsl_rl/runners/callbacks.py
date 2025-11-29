import os
import random
import string
import wandb
from torch.utils.tensorboard import SummaryWriter


def make_save_model_cb(directory):
    def cb(runner, stat):
        path = os.path.join(directory, "model_{}.pt".format(stat["current_iteration"]))
        runner.save(path)

    return cb


def make_save_model_onnx_cb(directory):
    def cb(runner, stat):
        path = os.path.join(directory, "model_{}.onnx".format(stat["current_iteration"]))
        runner.export_onnx(path)

    return cb


def make_interval_cb(callback, interval):
    def cb(runner, stat):
        if stat["current_iteration"] % interval != 0:
            return

        callback(runner, stat)

    return cb


def make_final_cb(callback):
    def cb(runner, stat):
        if not runner._learning_should_terminate():
            return

        callback(runner, stat)

    return cb


def make_first_cb(callback):
    uuid = "".join(random.choices(string.ascii_letters + string.digits, k=8))

    def cb(runner, stat):
        if hasattr(runner, f"_first_cb_{uuid}"):
            return

        setattr(runner, f"_first_cb_{uuid}", True)
        callback(runner, stat)

    return cb


def make_wandb_cb(init_kwargs):
    assert "project" in init_kwargs, "The project must be specified in the init_kwargs."

    run = wandb.init(**init_kwargs)
    check_complete = make_final_cb(lambda *_: run.finish())

    def cb(runner, stat):
        mean_reward = sum(stat["returns"]) / len(stat["returns"]) if len(stat["returns"]) > 0 else 0.0
        mean_steps = sum(stat["lengths"]) / len(stat["lengths"]) if len(stat["lengths"]) > 0 else 0.0
        total_steps = stat["current_iteration"] * runner.env.num_envs * runner._num_steps_per_env
        training_time = stat["training_time"]

        run.log(
            {
                "mean_rewards": mean_reward,
                "mean_steps": mean_steps,
                "training_steps": total_steps,
                "training_time": training_time,
            }
        )

        check_complete(runner, stat)

    return cb


def make_tensorboard_cb(log_dir):
    writer = SummaryWriter(log_dir)
    check_complete = make_final_cb(lambda *_: writer.close())
    
    # Use a mutable container for the counter to allow modification in the closure
    # We start at 0.
    # step_counter = {"total_steps": 0}
    iteration_counter = {"current_iteration": 0}

    def cb(runner, stat):
        mean_reward = sum(stat["returns"]) / len(stat["returns"]) if len(stat["returns"]) > 0 else 0.0
        mean_steps = sum(stat["lengths"]) / len(stat["lengths"]) if len(stat["lengths"]) > 0 else 0.0
        
        # Increment steps
        steps_this_iter = runner.env.num_envs * runner._num_steps_per_env
        iteration_counter["current_iteration"] += 1
        
        training_time = stat["training_time"]

        writer.add_scalar("Train/mean_reward", mean_reward, iteration_counter)
        writer.add_scalar("Train/mean_steps", mean_steps, iteration_counter)
        writer.add_scalar("Train/training_time", training_time, iteration_counter)
        
        for key, value in stat["loss"].items():
            writer.add_scalar(f"Loss/{key}", value, iteration_counter)

        writer.flush()
        check_complete(runner, stat)

    return cb