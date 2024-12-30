import zuma
from typing import List, Tuple, Dict
from collections import defaultdict

id = ["322453200"]

class Controller:
    """Controller for Zuma game using MDP principles."""

    def __init__(self, game: zuma.Game):
        """Initialize MDP controller with the game model."""
        self.game = game
        self.model = game.get_model()

        # MDP Parameters
        self.gamma = 0.8  # Discount factor
        self.max_depth = 3  # Maximum look-ahead depth

        # Cache model parameters
        self.chosen_action_prob = self.model['chosen_action_prob']
        self.next_color_dist = self.model['next_color_dist']
        self.color_pop_prob = self.model['color_pop_prob']
        self.color_pop_reward = self.model['color_pop_reward']
        self.color_not_finished_punishment = self.model['color_not_finished_punishment']
        self.finished_reward = self.model['finished_reward']

        # Initialize value function
        self.V = {}

    def calcReward(self, amount: int, color: int) -> float:
        """Calculate reward for popping a group of balls."""
        reward = self.game.get_model()['color_pop_reward']['3_pop'][color]
        reward += (amount - 3) * self.game.get_model()['color_pop_reward']['extra_pop'][color]
        return reward

    def _simulate_pop(self, line: List[int], action: int) -> Tuple[List[int], float]:
        """Simulate ball popping and return new line and reward."""
        if action == -1:
            return line.copy(), 0

        reward = 0
        sim_line = line.copy()

        # Insert ball at action position
        if 0 <= action <= len(sim_line):
            ball = self.game.get_current_state()[1]
            sim_line.insert(action, ball)

            # Start checking from leftmost possible affected position
            i = max(0, action - 2)
            while i < len(sim_line) - 2:
                # Find sequence of same-colored balls
                count = 1
                color = sim_line[i]
                j = i + 1
                while j < len(sim_line) and sim_line[j] == color:
                    count += 1
                    j += 1

                if count >= 3:
                    # Calculate pop probability and reward
                    pop_prob = self.color_pop_prob[color]
                    base_reward = self.color_pop_reward['3_pop'][color]
                    extra_reward = (count - 3) * self.color_pop_reward['extra_pop'][color]
                    # reward += pop_prob * (base_reward + extra_reward)
                    reward += self.calcReward(count, color)

                    # Remove matched balls
                    sim_line[i:j] = []
                    i = max(0, i - 2)  # Move back to check for new matches
                else:
                    i += 1

        return sim_line, reward

    def evaluate_state(self, line: List[int], depth: int) -> float:
        """Evaluate state value with limited depth.
        line (List[int]): The current state of the game line (sequence of balls).
        depth (int): The current depth in the lookahead process.
        returns:The estimated value (a float) of the current game state,
        combining immediate and discounted future rewards."""

        # reached max depth or line is empty
        if depth >= self.max_depth or not line:
            if not line:
                return self.finished_reward
            return -sum(self.color_not_finished_punishment[color] * line.count(color)
                        for color in set(line))

        max_value = float('-inf')
        actions = list(range(-1, len(line) + 1))

        # Iterate Through All Actions:
        for action in actions:
            value = 0
            # Get current ball
            ball = self.game.get_current_state()[1]

            # Success probability for chosen action
            prob = self.chosen_action_prob[ball] if action != -1 else 1.0

            # Simulate action and get reward
            new_line, reward = self._simulate_pop(line, action)

            # Calculate value including future rewards
            value = prob * (reward + self.gamma * self.evaluate_state(new_line, depth + 1))
            # scaled by the discount factor (gamma), which reduces the importance of future rewards.

            max_value = max(max_value, value)

        return max_value

    def choose_next_action(self) -> int:
        """Choose next action using limited-depth MDP evaluation."""
        line, ball, steps, max_steps = self.game.get_current_state()
        steps_remaining = max_steps - steps  # מחשבים את מספר הצעדים שנותרו

        if steps >= max_steps:
            return -1

        actions = list(range(-1, len(line) + 1))
        best_action = -1
        best_value = float('-inf')

        # Evaluate each action
        for action in actions:
            # Get success probability
            prob = self.chosen_action_prob[ball] if action != -1 else 1.0

            # Simulate action
            new_line, reward = self._simulate_pop(line, action)

            # Calculate value including future rewards
            value = prob * (reward + self.gamma * self.evaluate_state(new_line, 1))

            if value > best_value:
                best_value = value
                best_action = action

        return best_action