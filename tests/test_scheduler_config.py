import pytest
import yaml
from mlx_tuning_fork.config import yaml_loader
from mlx_tuning_fork.tuning.dynamic_learning import *


def load_config(yaml_string, total_iterations):
    config = yaml.load(yaml_string, yaml_loader)
    learning_rate = config["parameters"]["learning_rate"]
    return SCHEDULE_CONFIGURATION_TYPE_TO_CLASS[
        config["learning_schedule"]["type"]].from_configuration(learning_rate, config, total_iterations)


class TestSGDRWithWarmup:
    TEST_YAML1 = """
    parameters:
        learning_rate: 1e-5
    
    learning_schedule:
        type: sgdr_w_warmup
        initial_cycle_length: 200
        min_lr: 1e-7
        cycle_length: 1000
        cycle_length_decay: 1.5
        cycle_magnitude_decay: 1
        warmup_start_lr: 0
        warmup_length: 4000"""

    def test_basic(self):
        scheduler = load_config(self.TEST_YAML1, 24000)
        assert isinstance(scheduler, CyclicalSchedule)
        assert scheduler.schedule_class == CosineAnnealingSchedule
        assert scheduler.length == 1000
        assert scheduler.initial_cycle_length == 200
        assert scheduler.kwargs["max_lr"] == 1e-5
        assert scheduler.magnitude_decay == 1
        assert scheduler.length_decay == 1.5
        assert scheduler.warmup_length == 4000
        assert scheduler.warmup_start_lr == 0


class TestCosineWithWarmup:
    TEST_YAML1 = """
    parameters:
        learning_rate: 1e-5

    learning_schedule:
        type: cosine_w_warmup
        min_lr: 1e-7
        cycle_length: -1
        start_lr: 0
        warmup_proportion: 0.15"""

    def test_basic(self):
        scheduler = load_config(self.TEST_YAML1, 24000)
        assert isinstance(scheduler, LinearWarmUp)
        assert isinstance(scheduler.schedule, CosineAnnealingSchedule)
        assert scheduler.start_lr == 0
        assert scheduler.finish_lr == 1e-5
        assert scheduler.length == 24000 * .15
