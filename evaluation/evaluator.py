# evaluation/evaluator.py

from clients.ollama_client import ollama_generate
from utils.prompts import get_feasibility_meta_prompt, get_evaluation_meta_prompt
from config.settings import (
    ARBITER_SYSTEM_PROMPT_FEASIBILITY,
    ARBITER_SYSTEM_PROMPT_EVALUATION
)


def feasibility_check(problem_statement, chain_of_thought, model_name="llama3.2"):
    cot_text = "\n".join(chain_of_thought)
    meta_prompt = get_feasibility_meta_prompt(problem_statement, cot_text)

    response = ollama_generate(
        prompt=meta_prompt,
        model_name=model_name,
        system=ARBITER_SYSTEM_PROMPT_FEASIBILITY,
    ).strip()

    # Debug prints
    print("===== Feasibility Check Debug =====")
    print("Chain of Thought:", chain_of_thought)
    # print("Feasibility Check Prompt:\n", meta_prompt)
    print("Model's raw response:", repr(response))
    print("===================================\n")

    normalized = "".join(ch for ch in response.lower() if ch.isalpha())

    if "feasible" in normalized and "infeasible" not in normalized:
        return True
    elif "infeasible" in normalized and "feasible" not in normalized:
        return False
    else:
        print("WARNING: Model did not return strict 'Feasible'/'Infeasible'; defaulting to Feasible.\n")
        return True


def evaluate_chain(problem_statement, chain_of_thought, model_name="llama3.2"):
    cot_text = "\n".join(chain_of_thought)
    meta_prompt = get_evaluation_meta_prompt(problem_statement, cot_text)

    judgment = ollama_generate(
        prompt=meta_prompt,
        model_name=model_name,
        system=ARBITER_SYSTEM_PROMPT_EVALUATION
    )
    lower_judgment = judgment.lower()
    if "yes" in lower_judgment:
        return 1.0
    return 0.0
