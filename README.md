# README

This code implements a **Monte Carlo Tree Search (MCTS)** approach to **chain-of-thought reasoning** using a language model (via `ollama_generate`). The goal is to iteratively build a reasoning chain that solves a given problem statement.

### Key Components

1. **`Node` Class**

   - Represents a node in the search tree, holding:
     - A partial chain-of-thought (`chain_of_thought`)
     - Children nodes
     - MCTS visit statistics (`visit_count`, `total_value`)
   - Provides methods to add children, check if a node is terminal, etc.

2. **MCTS Functions**

   - **`mcts(...)`**: Orchestrates the MCTS loop by repeatedly selecting a leaf node, checking feasibility, expanding the chain-of-thought, simulating further steps, then backpropagating rewards.
   - **`selection(...)`**: Picks the best path down the tree using a UCB (Upper Confidence Bound) score until reaching a leaf or a terminal node.
   - **`expansion(...)`**: Generates the _next single short reasoning step_ from the language model and attaches it as a child node.
   - **`simulation(...)`**: Continues from a leaf node until a terminal depth is reached, producing a possible full chain-of-thought.
   - **`backpropagate(...)`**: Propagates the reward (correctness of the chain-of-thought) up the tree so nodes can be updated with new values.

3. **Feasibility & Correctness Checks**

   - **`feasibility_check(...)`**: Queries the model to see if the partial chain-of-thought is still _plausible_ for solving the problem. If deemed “Infeasible,” the branch is pruned early.
   - **`evaluate_chain(...)`**: At the end of a chain-of-thought, a correctness check is made (the model must strictly answer “Yes” or “No”).

4. **Language Model Interaction**

   - **`ollama_generate(...)`**: Interacts with the `phi3` model through a custom prompt, enforcing specific instructions (like producing short single-step reasoning or strictly “Feasible”/“Infeasible” answers).

5. **`main()` Function**
   - Defines the **problem statement**.
   - Creates the root node and runs MCTS with a set number of simulations.
   - Selects the _best final chain-of-thought_ and evaluates correctness.
   - Prints out whether the final chain is deemed correct by the model.

### Purpose

- Demonstrates how to structure MCTS to explore possible reasoning sequences (“chains-of-thought”) for a given problem.
- Uses _feasibility checks_ to prune branches early if they’re obviously off-track, and a _final arbiter check_ to confirm correctness once a chain is fully formed.

This setup can be adapted for different problems or language models by swapping in new prompts, feasibility checks, and expansions.
