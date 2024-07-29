

def play(env):
    observation, info = env.reset()
    terminated = False
    states = []
    while not terminated:
        env.render()

        # here we need to place the action choice
        action = env.action_space.sample()
        obs, reward, terminated, _, info = env.step(action)
        states.append({"action": action, "observation": obs, "reward": reward, "terminated": terminated, "info": info})

    print("Game Over")

    last_state = states[-1]
    print(f"combo : Number of broken blocks in one action: {last_state['info']['combo']}")
    print(f"straight : Shows how many pieces are broken in a row "
          f"(this is different from how many pieces are broken at once) {last_state['info']['straight']}")
    print(f"score : The score of the game {last_state['info']['score']}")
    print(f"reward : The reward of the game {last_state['reward']}")

    x = 1

    env.close()