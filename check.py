import zuma
import ex2


def solve(game: zuma.Game):
    policy = ex2.Controller(game)
    print('Solving Zuma game:')
    for i in range(game.get_current_state()[3]):
        game.submit_next_action(chosen_action=policy.choose_next_action())
        # print(game.get_current_reward())
    print('Game result:\n\tLine state ->', game.get_current_state()[0], '\n\tReward result->',
          game.get_current_reward())
    game.show_history()


example = {
    'chosen_action_prob': {1: 0.6, 2: 0.7, 3: 0.5, 4: 0.9},
    # 'chosen_action_prob': {1: 1, 2: 1, 3: 1, 4: 1},
    'next_color_dist': {1: 0.1, 2: 0.6, 3: 0.15, 4: 0.15},
    # 'next_color_dist': {1: 0.25, 2: 0.25, 3: 0.25, 4: 0.25},
    'color_pop_prob': {1: 0.6, 2: 0.7, 3: 0.4, 4: 0.9},
    'color_pop_reward': {'3_pop': {1: 3, 2: 1, 3: 2, 4: 2},
                         'extra_pop': {1: 1, 2: 2, 3: 3, 4: 1}},
    'color_not_finished_punishment': {1: 2, 2: 3, 3: 5, 4: 1},
    'finished_reward': 150,
    'seed': 42}


def main():
    # debug_mode = False
    debug_mode = True
    # game = zuma.create_zuma_game((200, [1, 2, 3, 3, 3, 4, 2, 1, 2, 3, 4, 4], example, debug_mode))
    game = zuma.create_zuma_game((20,  [1, 2, 3, 3, 3, 4, 2, 1, 2, 3, 4, 4], example, debug_mode))
    # game = zuma.create_zuma_game((20, [1, 1], example, debug_mode))
    solve(game)


if __name__ == "__main__":
    main()
