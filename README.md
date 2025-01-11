# MCTS Chain-of-Thought Project

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Directory Structure](#directory-structure)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [How It Works](#how-it-works)
- [Logging](#logging)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)
- [Acknowledgements](#acknowledgements)

## Overview

The **MCTS Chain-of-Thought Project** leverages **Monte Carlo Tree Search (MCTS)** in combination with the **Ollama** language model to generate and evaluate logical reasoning steps, or "chains-of-thought," for solving complex problems. This project is designed to explore the synergy between MCTS algorithms and advanced language models to enhance problem-solving capabilities through iterative reasoning and evaluation.

## Features

- **Monte Carlo Tree Search (MCTS)**: Efficiently explores possible reasoning paths to find optimal solutions.
- **Chain-of-Thought Generation**: Utilizes Ollama to generate concise and relevant reasoning steps.
- **Feasibility and Evaluation Checks**: Ensures generated chains are logically feasible and correct.
- **Modular Architecture**: Clean and maintainable codebase organized into well-defined modules.
- **Extensible Design**: Easily add new features, evaluation metrics, or integrate with other models.

## Directory Structure

```

MCTS-Chain-of-Thought/
├── README.md
├── requirements.txt
├── main.py
├── config/
│ ├── **init**.py
│ └── settings.py
├── clients/
│ ├── **init**.py
│ └── ollama_client.py
├── models/
│ ├── **init**.py
│ └── node.py
├── mcts/
│ ├── **init**.py
│ ├── selection.py
│ ├── expansion.py
│ ├── simulation.py
│ ├── backpropagation.py
│ ├── mcts_algorithm.py
│ └── utils.py
├── evaluation/
│ ├── **init**.py
│ └── evaluator.py
├── tests/
│ ├── **init**.py
│ └── test.py
└── utils/
  ├── **init**.py
  └── prompts.py
```

### Description of Each Directory and File

- **`README.md`**: Project overview and documentation.
- **`requirements.txt`**: Python dependencies required for the project.
- **`main.py`**: Entry point of the application.
- **`config/`**: Configuration files.
  - **`settings.py`**: Stores configuration variables like model names, system prompts, default parameters, etc.
- **`clients/`**: Handles external client interactions.
  - **`ollama_client.py`**: Encapsulates the Ollama client setup and the `ollama_generate` function.
- **`models/`**: Contains data models.
  - **`node.py`**: Defines the `Node` class used in MCTS.
- **`mcts/`**: Implements the MCTS algorithm, broken down into core components.
  - **`selection.py`**: Implements the selection strategy.
  - **`expansion.py`**: Handles node expansion.
  - **`simulation.py`**: Manages the simulation phase.
  - **`backpropagation.py`**: Handles backpropagation of rewards.
  - **`mcts_algorithm.py`**: Orchestrates the entire MCTS process using the above components.
  - **`utils.py`**: Contains shared utility functions like `best_child`.
- **`evaluation/`**: Manages evaluation logic.
  - **`evaluator.py`**: Contains functions like `feasibility_check` and `evaluate_chain`.
- **`tests/`**: Contains unit tests and test cases.
  - **`test.py`**: Test suite implementation.
- **`utils/`**: Utility functions and helpers.
  - **`prompts.py`**: Stores all prompt templates used in the application.

## Installation

Follow these steps to set up the project on your local machine.

### 1. **Clone the Repository**

```bash
git clone https://github.com/Ripan-Roy/MCTS-Chain-of-Thought.git
cd MCTS-Chain-of-Thought
```

### 2. **Create a Virtual Environment**

It's recommended to use a virtual environment to manage dependencies.

```bash
python3 -m venv venv
```

Activate the virtual environment:

- **On Unix or MacOS:**

  ```bash
  source venv/bin/activate
  ```

- **On Windows:**

  ```bash
  venv\Scripts\activate
  ```

### 3. **Install Dependencies**

```bash
pip install -r requirements.txt
```

## Configuration

All configuration settings are located in the `config/settings.py` file. You can adjust the following parameters as needed:

```python
# config/settings.py

MODEL_NAME = "llama3.2"

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
```

### Customizing Prompts

All prompt templates used in the application are stored in `utils/prompts.py`. You can modify these prompts to better suit your needs.

```python
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
```

## Usage

### Running the Application

To execute the MCTS algorithm and generate a chain-of-thought for a given problem statement, run the `main.py` script:

```bash
python main.py
```

### Example

By default, the `main.py` script is set to solve the problem:

```python
problem = "How many r in strawberry?"
```

**Sample Output:**

```
=== MCTS Simulation 1/10 ===
===== Feasibility Check Debug =====
Chain of Thought: []
Model's raw response: 'Feasible'
===================================

[Expansion] New chain-of-thought generated:
  Step 1: Identify the number of 'r's in the word "strawberry".
  Node 140352980863040 now has 1 child(ren).

=== MCTS Simulation 2/10 ===
...
===== Best Child Chain-of-Thought After MCTS =====
Step 1: Identify the number of 'r's in the word "strawberry".
Step 2: Count each occurrence of the letter 'r' in "strawberry".

Final Reward: 1.0
Model-based evaluation says the chain-of-thought is correct!
```

### Customizing the Problem Statement

To solve a different problem, modify the `problem` variable in `main.py`:

```python
def main():
    problem = "In how many different ways can five friends sit for a photograph of five chairs in a row?"
    # Rest of the code...
```

## How It Works

The project employs the **Monte Carlo Tree Search (MCTS)** algorithm to explore possible reasoning paths for solving a given problem. Here's a high-level overview of the process:

1. **Initialization**: Start with a root node representing the initial state with an empty chain-of-thought.
2. **Selection**: Traverse the tree from the root node to a leaf node using the Upper Confidence Bound (UCB) strategy to balance exploration and exploitation.
3. **Feasibility Check**: Before expanding a node, verify if the current chain-of-thought is logically feasible for solving the problem.
4. **Expansion**: If feasible and the node is not terminal, generate the next reasoning step using Ollama and add it as a child node.
5. **Simulation**: Complete the chain-of-thought by simulating additional reasoning steps up to a maximum depth.
6. **Evaluation**: Assess the correctness and completeness of the generated chain-of-thought.
7. **Backpropagation**: Propagate the evaluation reward up the tree to update node statistics.
8. **Iteration**: Repeat the above steps for a specified number of simulations to explore and evaluate different reasoning paths.
9. **Selection of Best Chain-of-Thought**: After all simulations, select the chain-of-thought with the highest evaluation score.

## Logging

Currently, the project uses `print` statements for debugging and informational messages. For a more robust and configurable logging mechanism, consider integrating Python's built-in `logging` module.

### Implementing Logging

1. **Update `config/settings.py`** to include logging configurations.

   ```python
   # config/settings.py

   import logging

   LOG_LEVEL = logging.INFO
   LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
   ```

2. **Initialize Logging in Each Module**

   ```python
   # Example in mcts/utils.py

   import math
   import logging

   from config.settings import LOG_LEVEL, LOG_FORMAT

   logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
   logger = logging.getLogger(__name__)

   def best_child(node, c_param):
       if not node.children:
           return None

       best = None
       best_score = float("-inf")

       logger.info(f"[*] best_child: Node {id(node)} has {len(node.children)} children.")
       for idx, child in enumerate(node.children):
           q_value = child.get_value()
           visit_count = max(child.visit_count, 1)
           parent_visit = node.visit_count if node.visit_count > 0 else 1  # Avoid log(0)
           ucb_exploration = c_param * math.sqrt(math.log(parent_visit + 1) / visit_count)
           ucb_score = q_value + ucb_exploration

           logger.info(f"  Child {idx} has q_value={q_value:.3f}, visits={child.visit_count}, UCB={ucb_score:.3f}")
           if ucb_score > best_score:
               best_score = ucb_score
               best = child

       return best
   ```

3. **Replace `print` Statements with `logger`**

   Update all modules to use the `logger` for messages instead of `print`. This allows better control over log levels and outputs.

## Contributing

Contributions are welcome! Follow these steps to contribute to the project:

1. **Fork the Repository**

2. **Create a New Branch**

   ```bash
   git checkout -b feature/YourFeatureName
   ```

3. **Make Your Changes**

4. **Commit Your Changes**

   ```bash
   git commit -m "Add your detailed description of the changes"
   ```

5. **Push to Your Fork**

   ```bash
   git push origin feature/YourFeatureName
   ```

6. **Open a Pull Request**

   Describe your changes and submit the pull request for review.

### Guidelines

- **Write Clear Commit Messages**: Ensure your commit messages are descriptive and follow a consistent format.
- **Code Style**: Adhere to PEP 8 guidelines for Python code.
- **Documentation**: Update documentation and README if your changes affect usage or setup.
- **Testing**: Include tests for new features or bug fixes.

## License

This project is licensed under the [MIT License](LICENSE).

## Contact

For any questions, suggestions, or feedback, please contact:

- **Name**: Ripan Roy
- **Email**: ripanroy111@gmail.com
- **GitHub**: [Ripan-Roy](https://github.com/Ripan-Roy)

## Acknowledgements

- [Ollama](https://ollama.com/) for providing the language model API.
- [Monte Carlo Tree Search (MCTS)](https://en.wikipedia.org/wiki/Monte_Carlo_tree_search) methodology.
- Open-source contributors and the Python community for their invaluable resources and support.

## Future Work

- **Enhanced Evaluation Metrics**: Incorporate more sophisticated metrics for evaluating the quality of chains-of-thought.
- **GUI Interface**: Develop a graphical user interface for easier interaction with the MCTS algorithm.
- **Parallel Simulations**: Optimize the MCTS simulations by running them in parallel to reduce execution time.
- **Integration with Other Models**: Extend support to additional language models beyond Ollama.
- **Persistent Storage**: Implement a database to store and analyze generated chains-of-thought for further research.
