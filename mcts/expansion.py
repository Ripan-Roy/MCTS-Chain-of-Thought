# mcts/expansion.py

from clients.ollama_client import ollama_generate
from models.node import Node
from utils.prompts import get_expansion_user_prompt
from evaluation.evaluator import evaluate_chain, feasibility_check
from config.settings import COGTH_GENERATOR_SYSTEM_PROMPT_EXPANSION


def expansion(node, model_name, problem_statement):
    partial_chain_text = "\n".join(node.chain_of_thought)
    user_prompt = get_expansion_user_prompt(
        problem_statement, partial_chain_text)

    next_step = ollama_generate(
        prompt=user_prompt,
        model_name=model_name,
        system=COGTH_GENERATOR_SYSTEM_PROMPT_EXPANSION
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
