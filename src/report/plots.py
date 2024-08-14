import numpy as np
from matplotlib import pyplot as plt

def plot(scores, names):
    """
    Plot the scores of different models with mean and std
    :param scores: list of list of scores
    :return:
    """
    means = [np.mean(score) for score in scores]
    stds = [np.std(score) for score in scores]

    plt.figure()
    plt.errorbar(names, means, yerr=stds, marker='o', linestyle='None')
    plt.ylabel('Score')
    plt.xlabel('Algorithm')
    plt.title('Comparison of different algorithms (mean and std)')

    # add y=k line with name 'RL'
    plt.axhline(y=3943, color='r', linestyle='--', label='RL - gym-woodoku')
    plt.axhline(y=915, color='g', linestyle='--', label='DFS - WoodokuAI')
    plt.axhline(y=2000, color='b', linestyle='--', label='Heuristics - WoodokuSolver')
    plt.legend(loc="upper left")


def prev_work():
    # WoodokuSolver - heuristics - https://github.com/Zeltq/WoodokuSolver/tree/main
    score0 = np.array([630, 2490, 6900, 4800])
    name0 = 'WoodokuSolver'

    # DFS - WoodokuAI - https://github.com/CosmicSubspace/WoodokuAI/tree/master
    score1 = np.array([555, 115, 145, 1957, 1218, 72, 1227, 2032])
    name1 = 'DFS'
    print(np.mean(score1))

    # RL - https://github.com/helpingstar/gym-woodoku/tree/main
    score2 = np.array([3943])
    name2 = 'RL'

    plot([score0, score1, score2], [name0, name1, name2])
    plt.show()


def results():
    # Random
    score0 = np.array([44, 168, 52, 75, 40, 44, 81, 111, 47, 50])
    name0 = 'Random'

    # Reflex
    score1 = np.array([590, 397, 242, 1196, 1075, 1122, 1692, 585, 675, 236])
    name1 = 'Reflex'

    # Minimax
    score2 = np.array([2030, 647, 1231, 674, 2755])
    name2 = 'Minimax'

    plot([score0, score1, score2], [name0, name1, name2])
    plt.show()


def main():
    # prev_work()
    results()




if __name__ == '__main__':
    main()
