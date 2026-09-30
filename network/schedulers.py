from network.base import Scheduler

class LinearLearningRateScheduler(Scheduler):

    def __init__(self, optimizer, start_factor: float|int, end_factor: float|int, transition_steps: int) -> None:

        if not (0 <= start_factor <= 1):
            raise ValueError(f"Start Factor Must Be Between 0 and 1 Got {start_factor}")

        if not (0 <= end_factor <= 1):
            raise ValueError(f"End Factor Must Be Between 0 and 1 Got {start_factor}")

        if transition_steps <= 0:
            raise ValueError(f"The Transition Steps Must Be Greater than 0 Got {transition_steps}")

        self.optimizer = optimizer
        self.start_factor = start_factor
        self.end_factor = end_factor
        self.transition_steps = transition_steps

        self.steps = 0

        self.base = optimizer.learning_rate
        self.optimizer.learning_rate = self.base * start_factor

    def step(self) -> None:

        progress = min(1, self.steps / self.transition_steps)
        factor = self.start_factor + (progress * (self.end_factor - self.start_factor))

        self.optimizer.learning_rate = self.base * factor
        self.steps += 1

if __name__ == "__main__":
    ...