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

    plt.figure(figsize=(10, 5))
    plt.errorbar(names, means, yerr=stds, marker='o', linestyle='None')
    plt.ylabel('Score')
    plt.xlabel('Evaluation Function')
    plt.title('Comparison of different Evaluation Function (mean and std)')

    # add y=k line with name 'RL'
    # plt.axhline(y=3943, color='r', linestyle='--', label='RL - gym-woodoku')
    # plt.axhline(y=915, color='g', linestyle='--', label='DFS - WoodokuAI')
    # plt.axhline(y=2000, color='b', linestyle='--', label='Heuristics - WoodokuSolver')
    plt.axhline(y=71, color='y', linestyle='--', label='Random Agent')
    plt.legend(loc="upper left")
    plt.tight_layout()


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
    print(np.mean(score0))


    # Reflex
    score1 = np.array([590, 397, 242, 1196, 1075, 1122, 1692, 585, 675, 236])
    name1 = 'Reflex'

    # Minimax
    score2 = np.array([2030, 647, 1231, 674, 2755])
    name2 = 'Minimax'

    plot([score0, score1, score2], [name0, name1, name2])
    plt.show()


def eval_func():
    score0 = np.array([255, 145, 274, 100, 134, 328, 187, 98, 103, 470])
    name0 = ('check\nscore')


    score1 = np.array([40, 46, 73, 71, 79, 50, 47, 51,44])
    name1 = ('avoid\n'
             'jagged\n'
             'blocks')

    score2 = np.array([51, 40, 46, 47, 53, 54, 74, 50, 70, 86])
    name2 = ('square\n'
             'contribution')

    score3 = np.array([49, 72, 64, 73, 72, 75, 41, 74, 139, 63])
    name3 = ('connected\n'
             'components')

    score4 = np.array([422, 143, 377, 220, 871, 137, 529, 572, 370, 82])
    name4 = ('num\n'
             'empty\n'
             'squares')

    score5 = np.array([133, 131, 106, 44, 278, 69, 71, 258, 46, 139])
    name5 = ('avoid\n'
             'jagged\n'
             'edges\n'
             'diag')

    score6 = np.array([289, 561, 130 , 356, 967, 1383, 782, 195, 250, 488])
    name6 = ('remaining\n'
             'possible\n'
             'moves')

    score7 = np.array([1319, 702, 464, 591, 502, 1036, 580, 2582, 1100 , 1230,
                       1190, 1627, 827, 197, 990, 1807, 447, 389, 834])
    name7 = ('best\n'
             'evaluation')

    plot([score0, score1, score2, score3, score4, score5, score6, score7],
         [name0, name1, name2, name3, name4, name5, name6, name7])

    plt.show()



def main():
    # prev_work()
    # results()
    eval_func()




if __name__ == '__main__':
    main()
