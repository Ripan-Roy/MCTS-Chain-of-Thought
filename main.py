# main.py

from models.node import Node
from mcts.mcts_algorithm import mcts
from evaluation.evaluator import evaluate_chain


def main():
    # Define your problem statement here
    # Example:
    problem = "In how many different ways can five friends sit for a photograph of five chairs in a row?"
    # problem = "How many 'r' are there in the word strawberry?"

    root = Node(chain_of_thought=[])

    best_final_node = mcts(
        root,
        num_simulations=5,
        c_param=1.4,
        model_name="llama3.2",
        problem_statement=problem
    )

    if best_final_node is None:
        print("\nNo best child was found. Possibly all expansions were pruned.")
        return

    print("\n===== Best Child Chain-of-Thought After MCTS =====")
    for idx, step_text in enumerate(best_final_node.chain_of_thought, 1):
        print(f"Step {idx}: {step_text}")

    final_reward = evaluate_chain(
        problem, best_final_node.chain_of_thought, model_name="llama3.2")
    print("\nFinal Reward:", final_reward)
    if final_reward > 0:
        print("Model-based evaluation says the chain-of-thought is correct!")
    else:
        print("Model-based evaluation says the chain-of-thought is not correct!")


if __name__ == "__main__":
    main()
