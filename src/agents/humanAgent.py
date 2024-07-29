import copy
import gymnasium as gym
import keyboard
import gym_woodoku

def play(env):
    observation, info = env.reset()
    terminated = False
    states = []
    cur_pos = [0, 0]
    size = env.env.env.BOARD_LENGTH
    block_chosen = 0
    place_block = False
    prev_key = ''
    while not terminated:
        place_block = False
        temp_env = env.env.env
        next_env = env
        key = keyboard.read_key()
        if prev_key == key:
            pass
        elif key == 'a':
            cur_pos[0] = max(0, cur_pos[0] - 1)
        elif key == 'w':
            cur_pos[1] = max(0, cur_pos[1] - 1)
        elif key == 's':
            cur_pos[1] = min(size - 1, cur_pos[1] + 1)
        elif key == 'd':
            cur_pos[0] = min(size - 1, cur_pos[0] + 1)
        elif key == 'r':
            block_chosen = 0
        elif key == 't':
            block_chosen = 1
        elif key == 'y':
            block_chosen = 2
        elif key == 'j':
            place_block = True
        action = block_chosen * pow(size, 2) + cur_pos[1] * size + cur_pos[0]
        if key == 'p':
            print(action)
        if not place_block:
            if next_env.env.env._board[cur_pos[1]][cur_pos[0]] == 1:
                next_env.render()
            else:
                next_env.env.env._board[cur_pos[1]][cur_pos[0]] = 1
                next_env.render()
                next_env.env.env._board[cur_pos[1]][cur_pos[0]] = 0
            env.env.env = temp_env
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


if __name__ == '__main__':
    env = gym.make('gym_woodoku/Woodoku-v0', game_mode='woodoku', render_mode='human')
    play(env=env)


# def play(env):
#     keys2actions = {
#         'w': 0,
#         'a': 1,
#         's': 2,
#         'd': 3
#     }
#     gymnasium.utils.play.play(env, keys_to_action=keys2actions)
