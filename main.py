import os
import random

from hex_engine import HexPosition
from a2c_agent import A2CAgent


def load_agent(agent, file_prefix):
    if os.path.isfile(file_prefix + ".pt") or \
            (os.path.isfile(file_prefix + "_actor.pt") and os.path.isfile(file_prefix + "_critic.pt")):
        agent.load(file_prefix)
        print(f"Loaded saved agent: {file_prefix}")
    else:
        print(f"Created new agent: {file_prefix}")
    return agent


if __name__ == '__main__':
    board_size = 7
    game = HexPosition(board_size)
    num_episodes = 10

    action_space = game.get_action_space()
    action_size = len(action_space)

    a2c_agent = load_agent(A2CAgent(board_size, action_size), "a2c_agent")
    win_count = 0

    for episode in range(num_episodes):
        game.reset()
        while game.winner == 0:
            action = a2c_agent.act(game.board, game.get_action_space())
            game.move(action)
            if game.winner == 0:
                game.move(random.choice(game.get_action_space()))

        if game.winner == 1:
            win_count += 1

    ratio = 0

    if win_count > 0:
        ratio = num_episodes / win_count

    print(f"Win/Loss ratio: {ratio}")
    print(f"Wins ratio: {win_count}")
