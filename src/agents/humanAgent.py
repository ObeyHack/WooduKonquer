import copy

import keyboard
import gymnasium.utils.play

def play(env):
    observation, info = env.reset()
    terminated = False
    states = []
    cur_pos = [0, 0]
    size = env.env.env.BOARD_LENGTH
    block_chosen = 0
    place_block = False
    while not terminated:
        place_block = False
        next_env = env
        if keyboard.read_key() == 'a':
            cur_pos[0] = max(0, cur_pos[0] - 1)
        if keyboard.read_key() == 'w':
            cur_pos[1] = max(0, cur_pos[1] - 1)
        if keyboard.read_key() == 's':
            cur_pos[1] = min(size, cur_pos[1] + 1)
        if keyboard.read_key() == 'd':
            cur_pos[0] = min(size, cur_pos[0] + 1)
        if keyboard.read_key() == 'r':
            block_chosen = 0
        if keyboard.read_key() == 't':
            block_chosen = 1
        if keyboard.read_key() == 'y':
            block_chosen = 2
        if keyboard.read_key() == 'j':
            place_block = True
        action = block_chosen * pow(size, 2) + cur_pos[0] * size + cur_pos[1]
        if keyboard.read_key() == 'p':
            print(action)
        if not place_block:
            next_env.step(action)
            next_env.render()
        else:
            obs, reward, terminated, _, info = env.step(action)
            states.append(
                {"action": action, "observation": obs, "reward": reward, "terminated": terminated, "info": info})
            env.render()
        # here we need to place the action choice

    print("Game Over")

    last_state = states[-1]
    print(f"combo : Number of broken blocks in one action: {last_state['info']['combo']}")
    print(f"straight : Shows how many pieces are broken in a row "
          f"(this is different from how many pieces are broken at once) {last_state['info']['straight']}")
    print(f"score : The score of the game {last_state['info']['score']}")
    print(f"reward : The reward of the game {last_state['reward']}")

    x = 1

    env.close()


# def play(env):
#     keys2actions = {
#         'w': 0,
#         'a': 1,
#         's': 2,
#         'd': 3
#     }
#     gymnasium.utils.play.play(env, keys_to_action=keys2actions)
