# class LinearLRScheduler:
#     """
#     A linear learning rate scheduler that scales a neural network's
#
#     learning rate linearly between a starting factor and an ending factor
#     over a specified number of total steps.
#     """
#
#     def __init__(self, neural_network: NeuralNetwork, start_factor: float|int, end_factor: float|int, total_steps: float|int) -> None:
#         """
#         Initializes the linear learning rate scheduler.
#
#         Args:
#             neural_network (NeuralNetwork): The neural network instance whose
#               learning rate will be managed.
#             start_factor (float | int): The multiplier for the initial learning rate
#               at step 0 (must be between 0 and 1).
#             end_factor (float | int): The multiplier for the final learning rate at
#               or after total_steps (must be between 0 and 1).
#             total_steps (float | int): The total number of steps over which the
#               learning rate scales down (must be > 0).
#         """
#
#         assert 0 <= start_factor <= 1, f"Start Factor Must Be Between 0 and 1 Got {start_factor}"
#         assert 0 <= end_factor <= 1, f"End Factor Must Between 0 and 1 Got {end_factor}"
#         assert total_steps > 0, f"Total Steps Must Be Greater Than 0 Got {total_steps}"
#
#         self.neural_network = neural_network
#         self.base_learning_rate = neural_network.learning_rate
#         self.start_factor = start_factor
#         self.end_factor = end_factor
#         self.total_steps = total_steps
#
#         self.current_step = 0
#
#         self.neural_network.learning_rate = self.base_learning_rate * self.start_factor
#
#     def step(self) -> None:
#         """
#         Advances the scheduler by one training step, recalculates the learning
#
#         rate using linear interpolation, and updates the neural network's learning
#         rate.
#         """
#
#         self.current_step += 1
#         progress = min(self.current_step / self.total_steps, 1.0)
#         factor = self.start_factor + ((self.end_factor - self.start_factor) * progress)
#         self.neural_network.learning_rate = self.base_learning_rate * factor
