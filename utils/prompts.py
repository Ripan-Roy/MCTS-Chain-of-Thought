# utils/prompts.py

def get_feasibility_meta_prompt(problem_statement, chain_of_thought):
    return f"""Problem Statement:
    {problem_statement}

    Partial Chain of Thought:
    {chain_of_thought}

    Is this chain-of-thought still logically feasible for solving the problem?
    Answer "Feasible" or "Infeasible" only:
    """


def get_evaluation_meta_prompt(problem_statement, chain_of_thought):
    return f"""Problem Statement:
    {problem_statement}

    Chain of Thought:
    {chain_of_thought}

    Are these reasoning steps correct, complete, and satisfactory for solving the problem?
    Answer "Yes" or "No" only:
    """


def get_expansion_user_prompt(problem_statement, partial_chain):
    return (
        f"Problem statement: {problem_statement}\n\n"
        "Below is a partial chain-of-thought. Provide the *next single short reasoning step*:\n\n"
        f"Chain so far:\n{partial_chain}\n"
        "Next step:\n"
    )


def get_simulation_user_prompt(problem_statement, partial_chain):
    return (
        f"Problem statement: {problem_statement}\n\n"
        "Below is a partial chain-of-thought. Provide the next short step:\n\n"
        f"Chain so far:\n{partial_chain}\n"
        "Next step:\n"
    )
