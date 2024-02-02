CONFIG_DEFAULTS = {
    "num_tokens": 100,
    "write_every": 1,
    "prompt": None,
    "train": False,
    "data": "data/",
    "temp": 0.8,
    "lora_layers": 16,
    "batch_size": 4,
    "iters": -1,
    "epochs": -1,
    "val_batches": 25,
    "learning_rate": 1e-5,
    "steps_per_report": 10,
    "steps_per_eval": 200,
    "resume_adapter_file": None,
    "adapter_file": "adapters.npz",
    "test": False,
    "test_batches": 500,
    "seed": 0,
    "max_tokens": 100,
    "tokens_per_eval": 10,
    "save_every": 100,
    "validation_scale": 5,
    "reporting_interval_proportion": 0.01, #10/1000
    "validation_interval_proportion": 0.2, #200/1000
    "train_loss_file": None,
    "validation_loss_file": None,
    "ignore_chat_template": False,
    "colorize": False,
    "trust_remote_code": False

}