# mcts/selection.py

from mcts.utils import best_child


def selection(node, c_param=1.4):
    while node.children and not node.is_terminal():
        node = best_child(node, c_param)
        if node is None:
            break
    return node
