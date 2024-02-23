"""
Framework for learning rate schedulers
Many taken from #https://mxnet.apache.org/versions/1.7/api/python/docs/tutorials/packages/gluon/training/learning_rates/learning_rate_schedules_advanced.html

[1] https://shengdinghu.notion.site/MiniCPM-Unveiling-the-Potential-of-End-side-Large-Language-Models-d4d3a8c426424654a4e80e42a711cb20
"""
import yaml
import math
from abc import ABC
from typing import Type
import mlx.optimizers.schedulers as mlx_schedulers


class DynamicLearningRateSchedule(ABC):

    def __init__(self, learning_rate: float, total_iterations: int):
        self.learning_rate = learning_rate
        self.total_iterations = total_iterations

    def update(self, iteration: int) -> float:
        """
        Called before commencing with each iteration to provide the learning rate scheduler the chance
        to update the rate relative to iterations over time.

        Returns the (new or same) learning rate
        """
        pass


class ConstantLearningRateSchedule(DynamicLearningRateSchedule):
    """
    The default Learning Rate Manager, which does not make any changes to the learning rate
    """
    def update(self, iteration: int) -> float:
        return self.learning_rate

    @classmethod
    def from_configuration(cls, learning_rate, config, total_iterations):
        return learning_rate

    def __str__(self):
        return f"ConstantLearningRateSchedule: {self.learning_rate})"


class CosineAnnealingSchedule(DynamicLearningRateSchedule):
    """
    https://paperswithcode.com/method/cosine-annealing

    Note max_lr is set to the top-level learning_rate configuration if not specified in learning_schedule section
    """

    @classmethod
    def from_configuration(cls, learning_rate, config, total_iterations):
        param_dict = {k: v for k, v in config["learning_schedule"].items()}
        min_lr = param_dict["min_lr"]
        max_lr = param_dict["max_lr"] if "max_lr" in param_dict else learning_rate
        cycle_length = param_dict["cycle_length"]
        return CosineAnnealingSchedule(min_lr, max_lr, cycle_length, total_iterations)

    def __init__(self, min_lr: float, max_lr: float, cycle_length: int, total_iterations: int) -> None:
        """
        :param min_lr: lower bound for learning rate (float)
        :param max_lr: upper bound for learning rate (float).
        :param cycle_length: iterations between start and finish (int)
        :param total_iterations: Total iterations

         After cycle_length, the rate stays at min_lr.  If cycle_length is -1 (by default), it is set to
         total_iterations (see: [1])

        """
        super().__init__(max_lr, total_iterations)
        assert isinstance(min_lr, float)
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.cycle_length = total_iterations if cycle_length == -1 else cycle_length

    def __str__(self):
        return (f"CosineAnnealingSchedule: (min_lr: {self.min_lr}, max_lr: {self.max_lr}, cycle_length: "
                f"{self.cycle_length:,})")

    def update(self, iteration: int) -> float:
        if iteration <= self.cycle_length:
            unit_cycle = (1 + math.cos(iteration * math.pi / self.cycle_length)) / 2
            adjusted_cycle = (unit_cycle * (self.max_lr - self.min_lr)) + self.min_lr
            return adjusted_cycle
        else:
            return self.min_lr


class LinearWarmUp:
    def __init__(self, schedule: DynamicLearningRateSchedule, start_lr: float, length: int) -> None:
        """
         [From] “Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour” by Priya Goyal et al. (2017),
         we implement a wrapper class that adds warm-up to an existing schedule. Going from start_lr to the initial
         learning rate of the schedule over length iterations, this adjustment is useful when training with
         large batch sizes.


        :param schedule: a pre-initialized schedule
        :param start_lr: learning rate used at start of the warm-up (float)
        :param length: number of iterations used for the warm-up (int)
        """
        self.schedule = schedule
        self.start_lr = start_lr
        self.finish_lr = schedule.learning_rate
        self.length = length

    def update(self, iteration: int) -> float:
        if iteration <= self.length:
            return iteration * ((self.finish_lr - self.start_lr) / self.length) + self.start_lr
        else:
            return self.schedule.update(iteration - self.length)

    def __str__(self):
        return f"{self.schedule} w/ linear warmup from {self.start_lr} to {self.finish_lr} for {self.length:,} steps"


class LinearCoolDown:
    def __init__(self, schedule: DynamicLearningRateSchedule, finish_lr: float, start_idx: int, length: int):
        """

        Similarly, we could add a linear cool-down period to our schedule and this is used in the “1-Cycle” schedule
        proposed by Leslie N. Smith, Nicholay Topin (2017) to train neural networks very quickly in certain
        circumstances (coined “super-convergence”). We reduce the learning rate from its value at start_idx of schedule
        to finish_lr over a period of length, and then maintain finish_lr thereafter.

        :param schedule: a pre-initialized schedule (e.g. TriangularSchedule(min_lr=0.5, max_lr=2, cycle_length=500))
        :param finish_lr: learning rate used at end of the cool-down (float)
        :param start_idx: iteration to start the cool-down (int)
        :param length: number of iterations used for the cool-down (int)
        """
        self.schedule = schedule
        self.start_lr = self.schedule.update(start_idx)
        self.finish_lr = finish_lr
        self.start_idx = start_idx
        self.finish_idx = start_idx + length
        self.length = length

    def update(self, iteration: int) -> float:
        if iteration <= self.start_idx:
            return self.schedule.update(iteration)
        elif iteration <= self.finish_idx:
            return (iteration - self.start_idx) * (self.finish_lr - self.start_lr) / self.length + self.start_lr
        else:
            return self.finish_lr


class TriangularSchedule(DynamicLearningRateSchedule):
    def __init__(self, min_lr, max_lr, cycle_length, total_iterations: int, inc_fraction=0.5):
        """
        [..] “triangular” schedule that was proposed by Leslie N. Smith (2015). Quite simply, the schedule linearly
        increases then decreases between a lower and upper bound. Originally it was suggested this schedule be used as
        part of a cyclical schedule but more recently researchers have been using a single cycle.

        One adjustment proposed by Jeremy Howard, Sebastian Ruder (2018) was to change the ratio between the increasing
        and decreasing stages, instead of the 50:50 split. Changing the increasing fraction (inc_fraction!=0.5) leads
        to a “slanted triangular” schedule. Using inc_fraction<0.5 tends to give better results.

        :param min_lr: lower bound for learning rate (float)
        :param max_lr: upper bound for learning rate (float)
        :param cycle_length: iterations between start and finish (int)
        :param total_iterations: Total iterations
        :param inc_fraction: fraction of iterations spent in increasing stage (float)
        """
        super().__init__(min_lr, total_iterations)
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.cycle_length = cycle_length
        self.inc_fraction = inc_fraction

    def update(self, iteration: int) -> float:
        if iteration <= self.cycle_length*self.inc_fraction:
            unit_cycle = iteration * 1 / (self.cycle_length * self.inc_fraction)
        elif iteration <= self.cycle_length:
            unit_cycle = (self.cycle_length - iteration) * 1 / (self.cycle_length * (1 - self.inc_fraction))
        else:
            unit_cycle = 0
        adjusted_cycle = (unit_cycle * (self.max_lr - self.min_lr)) + self.min_lr
        return adjusted_cycle


class OneCycleSchedule(DynamicLearningRateSchedule):
    def __init__(self, start_lr, max_lr, cycle_length, finish_lr: float, total_iterations: int, cooldown_length=0):
        """
        [..] “1-Cycle” schedule proposed by Leslie N. Smith, Nicholay Topin (2017) we use a single and symmetric cycle
        of the triangular schedule above (i.e. inc_fraction=0.5), followed by a cool-down period of cooldown_length
        iterations.

        :param start_lr: lower bound for learning rate in triangular cycle (float)
        :param max_lr: upper bound for learning rate in triangular cycle (float)
        :param cycle_length: iterations between start and finish of triangular cycle: 2x 'stepsize' (int)
        :param finish_lr: learning rate used at end of the cool-down (float)
        :param total_iterations: Total iterations
        :param cooldown_length: number of iterations used for the cool-down (int)
        """
        super().__init__(finish_lr, total_iterations)
        if (cooldown_length > 0) and (finish_lr is None):
            raise ValueError("Must specify finish_lr when using cooldown_length > 0.")
        if (cooldown_length == 0) and (finish_lr is not None):
            raise ValueError("Must specify cooldown_length > 0 when using finish_lr.")

        finish_lr = finish_lr if (cooldown_length > 0) else start_lr
        schedule = TriangularSchedule(start_lr, max_lr, cycle_length, total_iterations)
        self.schedule = LinearCoolDown(schedule, finish_lr=finish_lr, start_idx=cycle_length, length=cooldown_length)

    def update(self, iteration: int) -> float:
        return self.schedule.update(iteration)


class CyclicalSchedule(DynamicLearningRateSchedule):
    def __init__(self, schedule_class: Type, cycle_length: int, initial_cycle_length: int,
                 cycle_length_decay: float = 1, cycle_magnitude_decay: float = 1, warmup_start_lr: float = 0,
                 warmup_length: int = -1, **kwargs) -> None:
        """
        Modified to include warmup linear parameters and initial cycle length, per SGD paper:
        "In order to improve anytime performance, we suggest an option to start with an initially small Ti
        and increase it by a factor of Tmult at every restart"

        We implement a wrapper class that loops existing cycle-based schedules [..] to provide infinitely repeating
        schedules.

        We pass the schedule class (rather than an instance) because one feature of the CyclicalSchedule is to vary the
        cycle_length over time as seen in Ilya Loshchilov, Frank Hutter (2016) using cycle_length_decay.
        Another feature is the ability to decay the cycle magnitude over time with cycle_magnitude_decay.

        :param schedule_class: class of schedule, expected to take `cycle_length` argument
        :param cycle_length: iterations used for each cycle (int)
        :param initial_cycle_length: Initial cycle length (cycle_length is used for others)
        :param cycle_length_decay: factor multiplied to cycle_length each cycle (float - defaults to 1)
        :param cycle_magnitude_decay: factor multiplied with learning rate magnitudes each cycle (float - defaults to 1)
        :param warmup_start_lr: learning rate used at start of the warm-up (float - defaults to 0)
        :param warmup_length: number of iterations used for the warm-up (int) - Default is -1: no warmup
        :param kwargs: passed to the schedule_class
        """
        self.schedule_class = schedule_class
        self.length = cycle_length
        self.initial_cycle_length = initial_cycle_length
        self.length_decay = cycle_length_decay
        self.magnitude_decay = cycle_magnitude_decay
        self.kwargs = kwargs
        self.warmup_start_lr = warmup_start_lr
        self.warmup_length = warmup_length

    def update(self, iteration: int) -> float:
        learning_rate = self.kwargs['learning_rate' if 'learning_rate' in self.kwargs else 'max_lr']
        if self.warmup_length != -1 and iteration <= self.warmup_length:
            return iteration * ((learning_rate - self.warmup_start_lr) / self.warmup_length) + self.warmup_start_lr
        elif self.warmup_length != -1:
            _iteration = iteration - self.warmup_length
        else:
            _iteration = iteration

        cycle_idx = 0
        cycle_length = self.length
        idx = self.length
        while idx <= _iteration:
            cycle_length = self.initial_cycle_length if cycle_idx == 0 else math.ceil(cycle_length * self.length_decay)
            cycle_idx += 1
            idx += cycle_length
        cycle_offset = _iteration - idx + cycle_length

        schedule = self.schedule_class(cycle_length=cycle_length, **self.kwargs)
        return schedule.update(cycle_offset) * self.magnitude_decay**cycle_idx


class CosineWithWarmup:
    @classmethod
    def from_configuration(cls, learning_rate, config, total_iterations):
        param_dict = {k: v for k, v in config["learning_schedule"].items()}
        min_lr = param_dict["min_lr"]
        max_lr = param_dict["max_lr"] if "max_lr" in param_dict else learning_rate
        cycle_length = param_dict["cycle_length"]
        cycle_length = total_iterations if cycle_length == -1 else cycle_length
        length = param_dict["length"] if "length" in param_dict else int(param_dict["warmup_proportion"] *
                                                                         total_iterations)
        warmup_schedule = mlx_schedulers.linear_warmup(length, max_lr, init=min_lr)
        cosine_schedule = mlx_schedulers.cosine_decay(max_lr, cycle_length)
        cosine_w_warmup_schedule = mlx_schedulers.ScheduleJoiner([warmup_schedule, cosine_schedule], [length])
        return cosine_w_warmup_schedule

class SGDRWithWarmup:
    @classmethod
    def from_configuration(cls, learning_rate, config, total_iterations):
        param_dict = {k: v for k, v in config["learning_schedule"].items() if k != "type"}
        if "max_lr" not in param_dict:
            param_dict["max_lr"] = learning_rate
        param_dict["total_iterations"] = total_iterations
        cycle_length = param_dict["cycle_length"]
        initial_cycle_length = param_dict["initial_cycle_length"]
        del param_dict["cycle_length"]
        del param_dict["initial_cycle_length"]
        return CyclicalSchedule(CosineAnnealingSchedule, cycle_length, initial_cycle_length, **param_dict)


SCHEDULE_CONFIGURATION_TYPE_TO_CLASS = {
    #"cosine": CosineAnnealingSchedule,
    "cosine_w_warmup": CosineWithWarmup,
    "constant": ConstantLearningRateSchedule,
    #"sgdr_w_warmup": SGDRWithWarmup
}
