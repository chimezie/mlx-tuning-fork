from mlx_lm.tuner.trainer import TrainingCallback
try:
    import wandb
except ImportError:
    wandb = None


class WandbCallback(TrainingCallback):

    def on_train_loss_report(self, train_info):
        if wandb is None:
            raise ImportError('wandb module not available.  Install with `pip install wandb`')
        try:
            wandb.log(train_info, step=train_info["iteration"])
        except Exception as e:
            print(f"logging to wandb failed: {e}")

    def on_val_loss_report(self, val_info):
        if wandb is None:
            raise ImportError('wandb module not available.  Install with `pip install wandb`')
        try:
            wandb.log(val_info, step=val_info["iteration"])
        except Exception as e:
            print(f"logging to wandb failed: {e}")