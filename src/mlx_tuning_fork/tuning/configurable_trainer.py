"""
Functions defined here are based almost entirely on:
https://github.com/ml-explore/mlx-examples/blob/main/llms/mlx_lm/tuner/trainer.py

With the minor exception of providing additional 'hooks' to enable a customization framework
"""
import os
import time
from typing import List
import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm.tuner.trainer import TrainingArgs, default_loss, iterate_batches, save_adapter
from mlx_tuning_fork.tuning.dynamic_learning import DynamicLearningRateSchedule


def train(
    model,
    tokenizer,
    optimizer,
    train_dataset,
    val_dataset,
    learning_rate_schedule: DynamicLearningRateSchedule,
    args: TrainingArgs = TrainingArgs(),
    loss: callable = default_loss,
    iterate_batches: callable = iterate_batches,
    reported_train_loss_data: List = None,
    validation_loss_data: List = None,
    wandb_logging: bool = False
):
    # Create value and grad function for loss
    loss_value_and_grad = nn.value_and_grad(model, loss)

    losses = []
    n_tokens = 0
    print("Starting training..., iters:", args.iters)
    print("LR Scheduler:", learning_rate_schedule)
    # Main training loop
    start = time.perf_counter()
    for it, batch in zip(
        range(args.iters),
        iterate_batches(
            dataset=train_dataset,
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            max_seq_length=args.max_seq_length,
            train=True,
        ),
    ):
        #Update the learning rate from the scheduler
        optimizer.learning_rate = learning_rate_schedule.update(it)

        # Forward and backward pass
        (lvalue, toks), grad = loss_value_and_grad(model, *batch)

        # Model update
        optimizer.update(model, grad)

        mx.eval(model.parameters(), optimizer.state, lvalue)

        # Record loss
        losses.append(lvalue.item())
        n_tokens += toks.item()

        if it == 0:
            print(f"Starting learning rate: {optimizer.learning_rate}")

        # Report training loss if needed
        if (it + 1) % args.steps_per_report == 0:
            train_loss = np.mean(losses)

            stop = time.perf_counter()
            iters_per_sec = args.steps_per_report / (stop - start)
            num_tokens_per_sec = float(n_tokens) / (stop - start)
            print(
                f"Iter {it + 1}: Train loss {train_loss:.3f}, "
                f"It/sec {iters_per_sec :.3f}, "# Report validation loss if needed
                f"Tokens/sec {num_tokens_per_sec :.3f}, "
                f"Learning rate {optimizer.learning_rate}, "
            )
            if wandb_logging:
                import wandb
                try:
                    wandb.log(
                        {
                            "iter": it + 1,
                            "loss/train": train_loss,
                            "learning_rate": optimizer.learning_rate,
                        }, step=it + 1
                    )
                except Exception as e:
                    print(f"logging to wandb failed: {e}")
            elif reported_train_loss_data is not None:
                reported_train_loss_data.append((it, train_loss, iters_per_sec, num_tokens_per_sec))
            losses = []
            n_tokens = 0
            start = time.perf_counter()

        # Report validation loss if needed
        if it == 0 or (it + 1) % args.steps_per_eval == 0:
            stop = time.perf_counter()
            val_loss = evaluate(
                model=model,
                dataset=val_dataset,
                loss=loss,
                tokenizer=tokenizer,
                batch_size=args.batch_size,
                num_batches=args.val_batches,
                max_seq_length=args.max_seq_length,
                iterate_batches=iterate_batches
            )
            val_run_time = (time.perf_counter() - stop)
            print(
                f"Iter {it + 1}: "
                f"Val loss {val_loss:.3f}, "
                f"Val took {val_run_time :.3f}s"
            )
            if wandb_logging:
                import wandb
                try:
                    wandb.log(
                        {
                            "iter": it + 1,
                            "loss/val": val_loss,
                            "learning_rate": optimizer.learning_rate,
                        }, step=it + 1
                    )
                except Exception as e:
                    print(f"logging to wandb failed: {e}")
            elif validation_loss_data is not None:
                validation_loss_data.append((it, val_loss, val_run_time))
            start = time.perf_counter()

            # Save adapter weights if needed
            if (it + 1) % args.steps_per_save == 0:
                save_adapter(model=model, adapter_file=args.adapter_file)
                print(
                    f"Iter {it + 1}: Saved adapter weights to {os.path.join(args.adapter_file)}."
                )
    # save final adapter weights
    save_adapter(model=model, adapter_file=args.adapter_file)
    print(f"Saved final adapter weights to {os.path.join(args.adapter_file)}.")


def evaluate(
    model,
    dataset,
    tokenizer,
    batch_size,
    num_batches,
    max_seq_length=2048,
    loss: callable = default_loss,
    iterate_batches: callable = iterate_batches
):
    all_losses = []
    ntokens = 0
    for it, batch in zip(
        range(num_batches),
        iterate_batches(
            dataset=dataset,
            tokenizer=tokenizer,
            batch_size=batch_size,
            max_seq_length=max_seq_length,
        ),
    ):
        losses, toks = loss(model, *batch)
        all_losses.append((losses * toks).item())
        ntokens += toks.item()

    return np.sum(all_losses) / ntokens
