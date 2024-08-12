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
    plt.errorbar(names, means, yerr=stds, fmt='o')
    plt.ylabel('Score')
    plt.xlabel('Algorithm')
    plt.title('Comparison of different algorithms (mean and std)')
    plt.show()

def main():
    # WoodokuSolver - heuristics - https://github.com/Zeltq/WoodokuSolver/tree/main
    score0 = np.array([630])
    name0 = 'WoodokuSolver'


    # DFS - WoodokuAI - https://github.com/CosmicSubspace/WoodokuAI/tree/master
    score1 = np.array([555, 115, 145, 1957, 1218, 72, 1227, 2032])
    name1 = 'DFS'


    # RL - https://github.com/helpingstar/gym-woodoku/tree/main
    score2 = np.array([3943])
    name2 = 'RL'

    plot([score0, score1, score2], [name0, name1, name2])



if __name__ == '__main__':
    main()
