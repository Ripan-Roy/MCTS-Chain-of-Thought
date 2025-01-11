# mcts/mcts_algorithm.py

from models.node import Node
from mcts.selection import selection
from mcts.expansion import expansion
from mcts.simulation import simulation
from mcts.backpropagation import backpropagate
from mcts.utils import best_child
from evaluation.evaluator import feasibility_check


def mcts(root_node, num_simulations=10, c_param=1.4, model_name="llama3.2", problem_statement=""):
    for sim in range(num_simulations):
        print(f"\n=== MCTS Simulation {sim+1}/{num_simulations} ===")
        leaf_node = selection(root_node, c_param)

        if not feasibility_check(problem_statement, leaf_node.chain_of_thought, model_name):
            backpropagate(leaf_node, 0.0)
            continue

        if not leaf_node.is_terminal():
            leaf_node = expansion(leaf_node, model_name, problem_statement)

            if not feasibility_check(problem_statement, leaf_node.chain_of_thought, model_name):
                backpropagate(leaf_node, 0.0)
                continue

        reward = simulation(leaf_node, model_name, problem_statement)
        backpropagate(leaf_node, reward)

    # Select the best child after all simulations
    final_node = best_child(root_node, c_param=0)
    return final_node
