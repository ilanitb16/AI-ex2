# AI-ex2
Implement the game controller class for a simplified version of the Zuma game, utilizing a Markov Decision Process (MDP) model to maximize player score. 

- Develop the controller class within the provided `ex2.py` file, ensuring that function signatures remain intact.
- Use the MDP model for optimal decision-making in gameplay.

## Steps

1. **Understanding the Game Setup:**
   - Grasp the basic concept: A player (the frog) aims to shoot balls to create sequences of three or more of the same colored balls, which will then be removed and score points are earned.
   - Review the code provided in `zuma.py` for game mechanics.

2. **Using Provided Functions:**
   - Utilize the `get_ball()`, `get_current_state()`, `get_current_reward()`, and `get_model()` methods to gather information on the current game state.
   - Employ `submit_next_action()` on a deepcopy of the game object to simulate potential moves without altering the original game state.
   - Restrict actions to swapping between balls in array or skipping a ball.

3. **MDP Approach:**
   - Model the game using an MDP framework: Define states, actions, transition probabilities, and rewards.
   - Implement policy or value iteration techniques to derive optimal policy for game decisions.
   - Maximize score by strategically aligning balls for larger removals and using the probability and distribution data provided for decision-making.

4. **Function Implementation:**
   - `choose_next_action`: Implement this function to select the next index for ball placement based on calculated policy.
   - Respect all constraints on function names, capitalization, and formatting to ensure successful automatic testing.

5. **Testing:**
   - Use `check.py` to validate your controller’s performance by running it against example game configurations and reviewing score maximizations.

## Output Format

- Your implementation details within `ex2.py`.
- Include your ID in a variable `id` in the file `ex2.py` and in `details.txt` with your name and ID.

## Examples

Example call within `ex2.py`:
```
game = zuma.create_zuma_game((20, [1, 2, 3, 3, 3, 4, 2, 1, 2, 3, 4, 4], example, debug_mode))
```

- Embellish the `choose_next_action` to handle the indices correctly concerning available moves.

## Notes

- Maintain strict compliance with function naming and signatures as outlined.
- Preserve the included gameplay strategies and adapt your solution to maximize rewards and minimize penalties by game’s end.

Ensure all development and testing comply with provided game constants, and do not modify any component beyond `ex2.py`. Submission should include only your solutions and method to maximize gameplay score with optimal gameplay via the MDP.


## Test Results
![image](https://github.com/user-attachments/assets/141b18d3-88e1-4667-a6dd-8df3a354cd4e)
![image](https://github.com/user-attachments/assets/22b62312-0f21-44ab-8d7f-41f9f0e6c6b8)

