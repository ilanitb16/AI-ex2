import zuma
import copy

id = ["000000000"]


class Controller:
    """This class is a controller for a Zuma game."""

    def __init__(self, game: zuma.Game):
        """Initialize controller for given game model.
        This method MUST terminate within the specified timeout.
        """
        self.original_game = game
        self.copy_game = copy.deepcopy(game)

    def choose_next_action(self):
        """Choose next action for Zuma given the current state of the game.
        """
        pass
