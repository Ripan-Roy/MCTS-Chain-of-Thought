# mcts/backpropagation.py

def backpropagate(node, reward):
    current = node
    while current is not None:
        current.visit_count += 1
        current.total_value += reward
        current = current.parent
