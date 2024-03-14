import pytest
import yaml
from mlx_tuning_fork.config import yaml_loader
from mlx_tuning_fork.tuning.dynamic_learning import SCHEDULE_CONFIGURATION_TYPE_TO_CLASS


def load_config(yaml_string, total_iterations):
    config = yaml.load(yaml_string, yaml_loader)
    learning_rate = config["learning_rate"]
    return SCHEDULE_CONFIGURATION_TYPE_TO_CLASS[
        config["learning_schedule"]["type"]].from_configuration(learning_rate, config, total_iterations)


class TestCosineWithWarmup:
    TEST_YAML1 = """
    learning_rate: 1e-5
    learning_schedule:
        type: cosine_w_warmup
        min_lr: 1e-7
        cycle_length: -1
        start_lr: 0
        warmup_proportion: 0.15"""

    TEST_YAML2 = """
    learning_schedule:
        type: cosine_w_warmup
        min_lr: 1e-7
        max_lr: 1e-6
        cycle_length: -1
        start_lr: 0
        warmup_proportion: 0.15"""

    TEST_YAML3 = """
    learning_rate: 1e-5
    learning_schedule:
        type: cosine
        max_lr: 1e-7
        cycle_length: 2000"""

    TEST_YAML4 = """
    learning_rate: 1e-5
    learning_schedule:
        type: cosine
        max_lr: 1e-7"""

    def test_basic(self):
        load_config(self.TEST_YAML1, 10000)
        load_config(self.TEST_YAML2, 10000)
        load_config(self.TEST_YAML3, 10000)
        with pytest.raises(KeyError) as excinfo:
            load_config(self.TEST_YAML4, 10000)
        assert "cycle_length in " in str(excinfo.value)