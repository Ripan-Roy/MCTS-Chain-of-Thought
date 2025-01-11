import math
import copy
import random

from ollama import Client

client = Client()


def ollama_generate(
    prompt,
    model_name="phi3",
    system="You are a helpful reasoning model. Provide thoughtful, concise answers."
):
    chunk_list = list(
        client.generate(
            system=system,
            model=model_name,
            prompt=prompt,
            stream=False
        )
    )
    chunk_dict = dict(chunk_list)
    output_text = chunk_dict.get("response", "")
    return output_text.strip()


class Node:
    def __init__(self, chain_of_thought, parent=None):
        self.chain_of_thought = chain_of_thought
        self.parent = parent
        self.children = []
        self.visit_count = 0
        self.total_value = 0.0

    def is_terminal(self, max_depth=3):
        return len(self.chain_of_thought) >= max_depth

    def get_value(self):
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count

    def add_child(self, child_node):
        self.children.append(child_node)


def feasibility_check(problem_statement, chain_of_thought, model_name="phi3"):
    """
    A "breach" check to see if the chain-of-thought is obviously off-track.
    Returns True if feasible, False if infeasible.

    This version prints debug info and does a looser parse:
    We look for the substring "feasible" or "infeasible" in the model's response,
    ignoring case and punctuation.
    """
    cot_text = "\n".join(chain_of_thought)

    arbiter_system_prompt = (
        "You are the arbiter. You must respond with exactly 'Feasible' or 'Infeasible' only. "
        "No additional text is allowed. "
        "If the reasoning steps so far are still logically possible for solving the problem, say 'Feasible'. "
        "If the chain-of-thought is clearly impossible or contradictory, say 'Infeasible'."
    )
    meta_prompt = f"""Problem Statement:
{problem_statement}

Partial Chain of Thought:
{cot_text}

Is this chain-of-thought still logically feasible for solving the problem?
Answer "Feasible" or "Infeasible" only:
"""

    response = ollama_generate(
        prompt=meta_prompt,
        model_name=model_name,
        system=arbiter_system_prompt
    ).strip()

    # Debug prints
    print("===== Feasibility Check Debug =====")
    print("Chain of Thought:", chain_of_thought)
    print("Feasibility Check Prompt:\n", meta_prompt)
    print("Model's raw response:", repr(response))
    print("===================================\n")

    # Make the response lower case, remove punctuation
    normalized = "".join(ch for ch in response.lower() if ch.isalpha())

    # Check presence of 'feasible' or 'infeasible'
    if "feasible" in normalized and "infeasible" not in normalized:
        return True
    elif "infeasible" in normalized and "feasible" not in normalized:
        return False
    else:
        # If the model's response is not strictly one or the other,
        # we can default to feasible or treat it as infeasible.
        # Let's default to "feasible" for now, to see expansions happen.
        print("WARNING: Model did not return strict 'Feasible'/'Infeasible'; defaulting to Feasible.\n")
        return True


def mcts(root_node, num_simulations=10, c_param=1.4, model_name="phi3", problem_statement=""):
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

    final_node = best_child(root_node, 0)
    return final_node


def selection(node, c_param=1.4):
    while node.children and not node.is_terminal():
        node = best_child(node, c_param)
        if node is None:
            break
    return node


def best_child(node, c_param):
    """
    Returns the child with the highest UCB score, or None if no children exist.
    """
    if not node.children:
        return None

    best = None
    best_score = float("-inf")

    print(
        f"\n[*] best_child: Node {id(node)} has {len(node.children)} children.")
    for idx, child in enumerate(node.children):
        q_value = child.get_value()
        # Avoid division by zero
        visit_count = max(child.visit_count, 1)
        ucb_exploration = c_param * \
            math.sqrt(math.log(node.visit_count + 1) / visit_count)
        ucb_score = q_value + ucb_exploration

        print(
            f"  Child {idx} has q_value={q_value:.3f}, visits={child.visit_count}, UCB={ucb_score:.3f}")
        if ucb_score > best_score:
            best_score = ucb_score
            best = child

    # best could remain None if no children, but we handled that check above
    return best


def expansion(node, model_name, problem_statement):
    partial_chain_text = "\n".join(node.chain_of_thought)
    system_prompt = (
        "You are a chain-of-thought generator. "
        "You MUST use only the problem statement provided by the user prompt. "
        "Do NOT invent new problems or add extra context. "
        "Only produce a single short reasoning step each time. "
        "Keep it concise and relevant."
    )
    user_prompt = (
        f"Problem statement: {problem_statement}\n\n"
        "Below is a partial chain-of-thought. Provide the *next single short reasoning step*:\n\n"
        f"Chain so far:\n{partial_chain_text}\n"
        "Next step:\n"
    )
    next_step = ollama_generate(
        prompt=user_prompt,
        model_name=model_name,
        system=system_prompt
    )
    if not next_step.strip():
        next_step = "No further insight."

    new_chain = node.chain_of_thought + [next_step]
    child_node = Node(new_chain, parent=node)
    node.add_child(child_node)

    print("\n[Expansion] New chain-of-thought generated:")
    for i, step in enumerate(new_chain, start=1):
        print(f"  Step {i}: {step}")
    print(f"Node {id(node)} now has {len(node.children)} child(ren).")

    immediate_reward = evaluate_chain(problem_statement, new_chain, model_name)
    if immediate_reward == 0.0:
        print("[Incorrect or incomplete so far (but might still be feasible).]\n")
    else:
        print("[Potentially correct so far]\n")

    return child_node


def simulation(node, model_name, problem_statement):
    max_depth = 3
    current_chain = copy.deepcopy(node.chain_of_thought)
    system_prompt = (
        "You are a chain-of-thought generator. "
        "Keep it short, logical, and strictly about the given problem. "
        "Only produce the next single short step each time."
    )

    while len(current_chain) < max_depth:
        # Feasibility check each step
        if not feasibility_check(problem_statement, current_chain, model_name):
            return 0.0

        partial_chain_text = "\n".join(current_chain)
        user_prompt = (
            f"Problem statement: {problem_statement}\n\n"
            "Below is a partial chain-of-thought. Provide the next short step:\n\n"
            f"Chain so far:\n{partial_chain_text}\n"
            "Next step:\n"
        )
        next_step = ollama_generate(
            prompt=user_prompt,
            model_name=model_name,
            system=system_prompt
        )
        if not next_step.strip():
            next_step = "No further insight."
        current_chain.append(next_step)

    reward = evaluate_chain(problem_statement, current_chain, model_name)
    return reward


def backpropagate(node, reward):
    current = node
    while current is not None:
        current.visit_count += 1
        current.total_value += reward
        current = current.parent


def evaluate_chain(problem_statement, chain_of_thought, model_name="phi3"):
    """
    The final correctness check.
    Returns 1.0 if the chain-of-thought is correct/complete for the problem, else 0.0.
    """
    cot_text = "\n".join(chain_of_thought)
    arbiter_system_prompt = (
        "You are the arbiter. You must respond with exactly 'Yes' or 'No' only. "
        "No additional text is allowed. "
        "If the chain-of-thought is correct and complete, say 'Yes'. Otherwise say 'No'."
    )
    meta_prompt = f"""Problem Statement:
{problem_statement}

Chain of Thought:
{cot_text}

Are these reasoning steps correct, complete, and satisfactory for solving the problem?
Answer "Yes" or "No" only:
"""
    judgment = ollama_generate(
        prompt=meta_prompt,
        model_name=model_name,
        system=arbiter_system_prompt
    )
    lower_judgment = judgment.lower()
    if "yes" in lower_judgment:
        return 1.0
    return 0.0


def main():
    problem = """
    Given the word "BALLOON," which consists of the letters B, A, L, L, O, O, N, 
    determine how many distinct 4-letter words can be formed. 
    Each letter can be used as many times as it appears in "BALLOON." 
    Words are sequences of letters where the order matters.
    """
    root = Node(chain_of_thought=[])

    best_final_node = mcts(
        root,
        num_simulations=5,
        c_param=1.4,
        model_name="phi3",
        problem_statement=problem
    )

    if best_final_node is None:
        print("\nNo best child was found. Possibly all expansions were pruned.")
        return

    print("\n===== Best Child Chain-of-Thought After MCTS =====")
    for idx, step_text in enumerate(best_final_node.chain_of_thought, 1):
        print(f"Step {idx}: {step_text}")

    final_reward = evaluate_chain(
        problem, best_final_node.chain_of_thought, model_name="phi3")
    print("\nFinal Reward:", final_reward)
    if final_reward > 0:
        print("Model-based evaluation says the chain-of-thought is correct!")
    else:
        print("Model-based evaluation says the chain-of-thought is not correct!")


if __name__ == "__main__":
    main()
