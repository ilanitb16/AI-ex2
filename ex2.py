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

        # MDP Parameters - adjusted for better rewards while maintaining performance
        self.gamma = 0.8
        self.max_depth = 2  # Keep depth 2 for performance but improve evaluation

        # Cache model parameters
        self.chosen_action_prob = self.model['chosen_action_prob']
        self.next_color_dist = self.model['next_color_dist']
        self.color_pop_prob = self.model['color_pop_prob']
        self.color_pop_reward = self.model['color_pop_reward']
        self.color_not_finished_punishment = self.model['color_not_finished_punishment']
        self.finished_reward = self.model['finished_reward']

        # Precompute structures
        self.action_ranges = {
            length: list(range(-1, length + 1))
            for length in range(20)
        }
        self.color_sets = {}

    @lru_cache(maxsize=128)
    def calcReward(self, amount: int, color: int) -> float:
        """Calculate reward for popping a group of balls with bonus for larger groups."""
        base_reward = self.color_pop_reward['3_pop'][color]
        extra_reward = self.color_pop_reward['extra_pop'][color]
        # Add bonus multiplier for larger groups to encourage bigger matches
        bonus_multiplier = 1.0 + 0.1 * max(0, amount - 3)  # 10% bonus per extra ball
        return (base_reward + (amount - 3) * extra_reward) * bonus_multiplier

    def _find_potential_moves(self, line: List[int], ball: int) -> Set[int]:
        """Find potentially valuable insertion points with additional positions."""
        potential_moves = {-1}  # Include skip move

        if not line:
            return {0}

        # Look for positions that could create matches
        for i in range(len(line)):
            # Check immediate matches
            if i > 0 and line[i - 1] == ball:
                potential_moves.add(i)
                # Add adjacent positions for potential future matches
                if i > 1:
                    potential_moves.add(i - 1)
                if i < len(line):
                    potential_moves.add(i + 1)

            if i < len(line) - 1 and line[i] == ball:
                potential_moves.add(i)
                potential_moves.add(i + 1)
                # Add one more position for potential chain reactions
                if i < len(line) - 2:
                    potential_moves.add(i + 2)

            # Look for potential sandwich positions (e.g., R_R where _ is insertion point)
            if i > 0 and i < len(line) - 1 and line[i - 1] == line[i + 1]:
                potential_moves.add(i)

        return potential_moves

    @lru_cache(maxsize=1024)
    def _simulate_pop_cached(self, line_tuple: Tuple[int, ...], action: int, current_ball: int) -> Tuple[
        Tuple[int, ...], float]:
        """Optimized cached version of simulate pop with chain reaction tracking."""
        if action == -1:
            return line_tuple, 0

        reward = 0
        sim_line = list(line_tuple)
        chain_multiplier = 1.0  # Reward multiplier for chain reactions

        if 0 <= action <= len(sim_line):
            sim_line.insert(action, current_ball)

            start_idx = max(0, action - 2)
            end_idx = min(len(sim_line), action + 3)

            chains = 0
            i = start_idx
            while i < end_idx - 2 and i < len(sim_line) - 2:
                if sim_line[i] != sim_line[i + 1] or sim_line[i] != sim_line[i + 2]:
                    i += 1
                    continue

                j = i + 3
                while j < len(sim_line) and sim_line[j] == sim_line[i]:
                    j += 1

                count = j - i
                if count >= 3:
                    # Apply chain bonus
                    chain_bonus = 1.0 + (0.2 * chains)  # 20% bonus per chain
                    reward += self.calcReward(count, sim_line[i]) * chain_bonus
                    sim_line[i:j] = []
                    end_idx -= count
                    i = max(0, i - 2)
                    chains += 1
                else:
                    i += 1

        return tuple(sim_line), reward

    @lru_cache(maxsize=2048)
    def evaluate_state_cached(self, line_tuple: Tuple[int, ...], depth: int, current_ball: int) -> float:
        """Optimized cached version of evaluate_state with improved heuristics."""
        if depth >= self.max_depth or not line_tuple:
            if not line_tuple:
                return self.finished_reward

            length = len(line_tuple)
            if length not in self.color_sets:
                self.color_sets[length] = {}

            if line_tuple in self.color_sets[length]:
                unique_colors = self.color_sets[length][line_tuple]
            else:
                unique_colors = set(line_tuple)
                self.color_sets[length][line_tuple] = unique_colors

            # Improved terminal state evaluation
            base_penalty = -sum(self.color_not_finished_punishment[color] * line_tuple.count(color)
                                for color in unique_colors)

            # Add bonus for potential future matches
            future_potential = 0
            for i in range(len(line_tuple) - 1):
                if line_tuple[i] == line_tuple[i + 1]:  # Adjacent same colors
                    future_potential += 0.5  # Small bonus for potential future matches

            return base_penalty + future_potential

        potential_moves = self._find_potential_moves(list(line_tuple), current_ball)

        max_value = float('-inf')
        for action in potential_moves:
            prob = self.chosen_action_prob[current_ball] if action != -1 else 1.0
            new_line_tuple, reward = self._simulate_pop_cached(line_tuple, action, current_ball)

            if not new_line_tuple:
                return prob * (reward + self.gamma * self.finished_reward)

            value = prob * (reward + self.gamma * self.evaluate_state_cached(new_line_tuple, depth + 1, current_ball))
            max_value = max(max_value, value)

        return max_value

    def choose_next_action(self) -> int:
        """Optimized version of choose_next_action with reward-focused selection."""
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

            if not new_line_tuple:
                return action

            value = prob * (reward + self.gamma * self.evaluate_state_cached(new_line_tuple, 1, ball))
            if value > best_value:
                best_value = value
                best_action = action

        return best_action
