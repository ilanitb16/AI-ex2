import zuma
import copy
import math
import random

id = ["322453200"]

class Controller:
    """This class is a controller for a Zuma game."""
    def __init__(self, game: zuma.Game):
        self.original_game = game
        self.copy_game = copy.deepcopy(game)
        self.model = self.copy_game.get_model()
        self.finished_reward = self.model['finished_reward']
        self.not_finished_penalty = sum(self.model['color_not_finished_punishment'].values())
        random.seed(self.model['seed'])
        self.line1 = self.copy_game._line
        self.maxSteps = self.copy_game._max_steps

    def _simulate_action(self, game_state, action):
        # Simulates a given action on a copy of the game state.

        sim_game = copy.deepcopy(game_state)
        sim_game.submit_next_action(action)
        new_line, _, new_steps, max_steps = sim_game.get_current_state()
        new_reward = sim_game.get_current_reward()
        finished = (new_steps == max_steps)
        return sim_game, new_line, new_reward, finished

    def _heuristic_value(self, line, finished, reward, steps_remaining, max_steps):
        # Computes a heuristic value for the given game state.

        if finished:
            return reward + self.finished_reward

        # Estimate penalties
        color_counts = {}
        for ball in line:
            color_counts[ball] = color_counts.get(ball, 0) + 1

        penalty_estimate = sum(
            self.model['color_not_finished_punishment'][color] * count
            for color, count in color_counts.items()
        )

        # Estimate potential pops
        pop_value = 0
        for ball, count in color_counts.items():
            if count >= 3:
                pop_value += self.model['color_pop_reward']['3_pop'][ball] * \
                             self.model['color_pop_prob'][ball]

        # Dynamic weighting
        penalty_weight = 0.4 if steps_remaining > 50 else 0.8 if steps_remaining > 10 else 1.2
        future_value = reward + pop_value - penalty_weight * penalty_estimate
        return future_value

    def _lookahead(self, game_state, depth):
        # Recursively evaluates future game states up to a specified depth.

        if depth == 0:
            return 0

        _, _, steps, max_steps = game_state.get_current_state()
        future_rewards = []

        for action in range(-1, len(game_state.get_current_state()[0]) + 1):
            sim_game, new_line, new_reward, finished = self._simulate_action(game_state, action)
            lookahead_reward = self._lookahead(sim_game, depth - 1)
            future_rewards.append(new_reward + lookahead_reward)

        return max(future_rewards) if future_rewards else 0

    def choose_next_action(self):
        # Chooses the next optimal action based on heuristic evaluation and lookahead.

        line, current_ball, steps, max_steps = self.original_game.get_current_state()
        steps_remaining = max_steps - steps

        if current_ball is None:
            current_ball = self.original_game.get_ball()

        possible_actions = list(range(-1, len(line) + 1))
        best_action = None
        best_value = -math.inf

        # Adjust lookahead depth dynamically
        line_complexity = len(self.line1)

        if line_complexity >= 10 and self.maxSteps > 20:
            lookahead_depth = 2 if steps_remaining > 10 else 1
        elif line_complexity >= 10 and self.maxSteps == 20:
            lookahead_depth = 3 if steps_remaining > 14 else 2
        elif line_complexity <= 5:
            lookahead_depth = 2 if steps_remaining > 10 else 1
        else:
            lookahead_depth = 1

        for action in possible_actions:
            sim_game, new_line, new_reward, finished = self._simulate_action(
                self.original_game, action
            )
            lookahead_value = self._lookahead(sim_game, depth=lookahead_depth)
            action_value = self._heuristic_value(
                new_line, finished, new_reward, steps_remaining, max_steps
            ) + lookahead_value

            if action_value > best_value:
                best_value = action_value
                best_action = action

        return best_action
