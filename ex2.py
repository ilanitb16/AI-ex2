import zuma
from typing import List, Tuple, Dict, Set
from collections import defaultdict
from functools import lru_cache

id = ["322453200"]


class Controller:
    """Controller for Zuma game using MDP principles."""

    def __init__(self, game: zuma.Game):
        """Initialize MDP controller with the game model."""
        self.game = game
        self.model = game.get_model()

        # MDP Parameters - keeping original depth for good rewards
        self.gamma = 0.8
        self.max_depth = 3  # Back to original depth for better rewards

        # Cache model parameters
        self.chosen_action_prob = self.model['chosen_action_prob']
        self.next_color_dist = self.model['next_color_dist']
        self.color_pop_prob = self.model['color_pop_prob']
        self.color_pop_reward = self.model['color_pop_reward']
        self.color_not_finished_punishment = self.model['color_not_finished_punishment']
        self.finished_reward = self.model['finished_reward']

    @lru_cache(maxsize=128)
    def calcReward(self, amount: int, color: int) -> float:
        """Calculate reward for popping a group of balls."""
        reward = self.game.get_model()['color_pop_reward']['3_pop'][color]
        reward += (amount - 3) * self.game.get_model()['color_pop_reward']['extra_pop'][color]
        return reward

    @lru_cache(maxsize=1024)
    def _simulate_pop_cached(self, line_tuple: Tuple[int, ...], action: int, current_ball: int) -> Tuple[
        Tuple[int, ...], float]:
        """Optimized cached version of simulate pop that maintains accuracy."""
        if action == -1:
            return line_tuple, 0

        reward = 0
        sim_line = list(line_tuple)

        if 0 <= action <= len(sim_line):
            sim_line.insert(action, current_ball)

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
                    reward += self.calcReward(count, color)
                    sim_line[i:j] = []
                    i = max(0, i - 2)  # Move back to check for new matches
                else:
                    i += 1

        return tuple(sim_line), reward

    @lru_cache(maxsize=2048)
    def evaluate_state_cached(self, line_tuple: Tuple[int, ...], depth: int, current_ball: int) -> float:
        """Cached version of evaluate_state that maintains reward quality."""
        if depth >= self.max_depth or not line_tuple:
            if not line_tuple:
                return self.finished_reward
            return -sum(self.color_not_finished_punishment[color] * line_tuple.count(color)
                        for color in set(line_tuple))

        max_value = float('-inf')
        actions = range(-1, len(line_tuple) + 1)  # Keep full action range for better rewards

        for action in actions:
            prob = self.chosen_action_prob[current_ball] if action != -1 else 1.0
            new_line_tuple, reward = self._simulate_pop_cached(line_tuple, action, current_ball)

            # Early termination only for perfect solutions
            if not new_line_tuple:
                return prob * (reward + self.gamma * self.finished_reward)

            value = prob * (reward + self.gamma * self.evaluate_state_cached(new_line_tuple, depth + 1, current_ball))
            max_value = max(max_value, value)

        return max_value

    def evaluate_state(self, line: List[int], depth: int) -> float:
        """Wrapper for evaluate_state that handles list conversion."""
        return self.evaluate_state_cached(tuple(line), depth, self.game.get_current_state()[1])

    def _simulate_pop(self, line: List[int], action: int) -> Tuple[List[int], float]:
        """Wrapper for simulate_pop that handles list conversion."""
        result_tuple, reward = self._simulate_pop_cached(tuple(line), action, self.game.get_current_state()[1])
        return list(result_tuple), reward

    def choose_next_action(self) -> int:
        """Choose next action using cached evaluation while maintaining reward quality."""
        line, ball, steps, max_steps = self.game.get_current_state()

        if steps >= max_steps:
            return -1

        line_tuple = tuple(line)
        actions = range(-1, len(line) + 1)  # Keep full action range
        best_action = -1
        best_value = float('-inf')

        for action in actions:
            prob = self.chosen_action_prob[ball] if action != -1 else 1.0
            new_line_tuple, reward = self._simulate_pop_cached(line_tuple, action, ball)

            # Early termination only for winning moves
            if not new_line_tuple:
                return action

            value = prob * (reward + self.gamma * self.evaluate_state_cached(new_line_tuple, 1, ball))
            if value > best_value:
                best_value = value
                best_action = action

        return best_action