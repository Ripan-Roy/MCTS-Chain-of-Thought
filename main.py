# main.py

from models.node import Node
from mcts.mcts_algorithm import mcts
from evaluation.evaluator import evaluate_chain
from config.settings import MODEL_NAME


def main():
    # Define your problem statement here
    # Example:
    # problem = "In how many different ways can five friends sit for a photograph of five chairs in a row?"
    problem = """
        Given positive integers x and y such that 2x^2y^3 + 4y^3 = 149 + 3x^2,
        what is the value of x + y?
    """
    # problem = "How many 'r' are there in the word strawberry?"

    root = Node(chain_of_thought=[])

    best_final_node = mcts(
        root,
        num_simulations=5,
        c_param=1.4,
        model_name=MODEL_NAME,
        problem_statement=problem
    )

    if best_final_node is None:
        print("\nNo best child was found. Possibly all expansions were pruned.")
        return

    print("\n===== Best Child Chain-of-Thought After MCTS =====")
    for idx, step_text in enumerate(best_final_node.chain_of_thought, 1):
        print(f"Step {idx}: {step_text}")

    final_reward = evaluate_chain(
        problem, best_final_node.chain_of_thought, model_name=MODEL_NAME)
    print("\nFinal Reward:", final_reward)
    if final_reward > 0:
        print("Model-based evaluation says the chain-of-thought is correct!")
    else:
        print("Model-based evaluation says the chain-of-thought is not correct!")


if __name__ == "__main__":
    main()
