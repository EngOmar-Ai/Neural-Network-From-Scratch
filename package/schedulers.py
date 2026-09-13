from package.base import Scheduler

class LinearLearningRateScheduler(Scheduler):
    def __init__(self, optimizer, start_factor: float|int, end_factor: float|int, transition_steps: int) -> None:

        assert 0 <= start_factor <= 1, f"Start Factor Must Be Between 0 and 1 Got {start_factor}"
        assert 0 <= end_factor <= 1, f"End Factor Must Between 0 and 1 Got {end_factor}"
        assert transition_steps > 0, f"Total Steps Must Be Greater Than 0 Got {transition_steps}"

        self.optimizer = optimizer
        self.start_factor = start_factor
        self.end_factor = end_factor
        self.transition_steps = transition_steps

        self.base = self.optimizer.learning_rate

        self.current_step = 0
        self.optimizer.learning_rate *= self.start_factor

    def step(self) -> None:
        progress = min(self.current_step / self.transition_steps, 1.0)
        factor = self.start_factor + ((self.end_factor - self.start_factor) * progress)

        self.optimizer.learning_rate = self.base * factor
        self.current_step += 1

