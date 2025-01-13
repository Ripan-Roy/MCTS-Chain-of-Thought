# config/settings.py

MODEL_NAME = "qwen:7b"

ARBITER_SYSTEM_PROMPT_FEASIBILITY = (
    "You are the arbiter. You must respond with exactly 'Feasible' or 'Infeasible' only. "
    "No additional text is allowed. "
    "If the reasoning steps so far are still logically possible for solving the problem, say 'Feasible'. "
    "If the chain-of-thought is clearly impossible or contradictory, say 'Infeasible'."
)

ARBITER_SYSTEM_PROMPT_EVALUATION = (
    "You are the arbiter. You must respond with exactly 'Yes' or 'No' only. "
    "No additional text is allowed. "
    "If the chain-of-thought is correct and complete, say 'Yes'. Otherwise say 'No'."
)

COGTH_GENERATOR_SYSTEM_PROMPT_EXPANSION = (
    "You are a chain-of-thought generator. "
    "You MUST use only the problem statement provided by the user prompt. "
    "Do NOT invent new problems or add extra context. "
    "Only produce a single short reasoning step each time. "
    "Keep it concise and relevant."
)

COGTH_GENERATOR_SYSTEM_PROMPT_SIMULATION = (
    "You are a chain-of-thought generator. "
    "Keep it short, logical, and strictly about the given problem. "
    "Only produce the next single short step each time."
)

DEFAULT_TEMPERATURE = 0
MAX_DEPTH = 5
