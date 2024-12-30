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

        # MDP Parameters - reduced depth for faster computation
        self.gamma = 0.8
        self.max_depth = 2  # Reduced from 3 to 2 for better performance

        # Cache model parameters
        self.chosen_action_prob = self.model['chosen_action_prob']
        self.next_color_dist = self.model['next_color_dist']
        self.color_pop_prob = self.model['color_pop_prob']
        self.color_pop_reward = self.model['color_pop_reward']
        self.color_not_finished_punishment = self.model['color_not_finished_punishment']
        self.finished_reward = self.model['finished_reward']

        # Precompute valid actions for common line lengths
        self.action_ranges = {
            length: list(range(-1, length + 1))
            for length in range(20)  # Precompute for reasonable line lengths
        }

        # Precompute color sets for common line lengths
        self.color_sets = {}

    @lru_cache(maxsize=128)
    def calcReward(self, amount: int, color: int) -> float:
        """Calculate reward for popping a group of balls."""
        base_reward = self.color_pop_reward['3_pop'][color]
        extra_reward = self.color_pop_reward['extra_pop'][color]
        return base_reward + (amount - 3) * extra_reward

    def _find_potential_moves(self, line: List[int], ball: int) -> Set[int]:
        """Find only potentially valuable insertion points."""
        potential_moves = {-1}  # Always include the skip move

        if not line:
            return {0}

        # Look for positions that could create matches
        for i in range(len(line)):
            # Check if inserting here could create a match
            if i > 0 and line[i - 1] == ball:
                potential_moves.add(i)
            if i < len(line) - 1 and line[i] == ball:
                potential_moves.add(i)
                potential_moves.add(i + 1)

        return potential_moves

    @lru_cache(maxsize=1024)
    def _simulate_pop_cached(self, line_tuple: Tuple[int, ...], action: int, current_ball: int) -> Tuple[
        Tuple[int, ...], float]:
        """Optimized cached version of simulate pop."""
        if action == -1:
            return line_tuple, 0

        reward = 0
        sim_line = list(line_tuple)

        if 0 <= action <= len(sim_line):
            sim_line.insert(action, current_ball)

            # Only check the affected region
            start_idx = max(0, action - 2)
            end_idx = min(len(sim_line), action + 3)

            i = start_idx
            while i < end_idx - 2:
                # Fast path: if no three consecutive same colors, skip
                if sim_line[i] != sim_line[i + 1] or sim_line[i] != sim_line[i + 2]:
                    i += 1
                    continue

                # Find full extent of matching sequence
                j = i + 3
                while j < len(sim_line) and sim_line[j] == sim_line[i]:
                    j += 1

                count = j - i
                if count >= 3:
                    reward += self.calcReward(count, sim_line[i])
                    sim_line[i:j] = []
                    end_idx -= count
                    i = max(0, i - 2)
                else:
                    i += 1

        return tuple(sim_line), reward

    @lru_cache(maxsize=2048)
    def evaluate_state_cached(self, line_tuple: Tuple[int, ...], depth: int, current_ball: int) -> float:
        """Optimized cached version of evaluate_state."""
        if depth >= self.max_depth or not line_tuple:
            if not line_tuple:
                return self.finished_reward

            # Use precomputed color sets when possible
            length = len(line_tuple)
            if length not in self.color_sets:
                self.color_sets[length] = {}

            if line_tuple in self.color_sets[length]:
                unique_colors = self.color_sets[length][line_tuple]
            else:
                unique_colors = set(line_tuple)
                self.color_sets[length][line_tuple] = unique_colors

            return -sum(self.color_not_finished_punishment[color] * line_tuple.count(color)
                        for color in unique_colors)

        # Use potential moves instead of all possible actions
        potential_moves = self._find_potential_moves(list(line_tuple), current_ball)

        max_value = float('-inf')
        for action in potential_moves:
            prob = self.chosen_action_prob[current_ball] if action != -1 else 1.0
            new_line_tuple, reward = self._simulate_pop_cached(line_tuple, action, current_ball)

            # Early termination if we found a complete solution
            if not new_line_tuple:
                return prob * (reward + self.gamma * self.finished_reward)

            value = prob * (reward + self.gamma * self.evaluate_state_cached(new_line_tuple, depth + 1, current_ball))
            max_value = max(max_value, value)

        return max_value

    def choose_next_action(self) -> int:
        """Optimized version of choose_next_action."""
        line, ball, steps, max_steps = self.game.get_current_state()

        if steps >= max_steps:
            return -1

        line_tuple = tuple(line)
        potential_moves = self._find_potential_moves(line, ball)

        best_action = -1
        best_value = float('-inf')

        for action in potential_moves:
            prob = self.chosen_action_prob[ball] if action != -1 else 1.0
            new_line_tuple, reward = self._simulate_pop_cached(line_tuple, action, ball)

            # Early termination if we found a winning move
            if not new_line_tuple:
                return action

            value = prob * (reward + self.gamma * self.evaluate_state_cached(new_line_tuple, 1, ball))
            if value > best_value:
                best_value = value
                best_action = action

        return best_action