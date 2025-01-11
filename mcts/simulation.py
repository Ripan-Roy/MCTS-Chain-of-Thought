# mcts/simulation.py

import copy
from clients.ollama_client import ollama_generate
from utils.prompts import get_simulation_user_prompt
from evaluation.evaluator import evaluate_chain, feasibility_check
from config.settings import COGTH_GENERATOR_SYSTEM_PROMPT_SIMULATION


def simulation(node, model_name, problem_statement):
    max_depth = 5
    current_chain = copy.deepcopy(node.chain_of_thought)
    system_prompt = COGTH_GENERATOR_SYSTEM_PROMPT_SIMULATION

    while len(current_chain) < max_depth:
        if not feasibility_check(problem_statement, current_chain, model_name):
            return 0.0

        partial_chain_text = "\n".join(current_chain)
        user_prompt = get_simulation_user_prompt(
            problem_statement, partial_chain_text)
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
