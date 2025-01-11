# mcts/utils.py

import math


def best_child(node, c_param):
    if not node.children:
        return None

    best = None
    best_score = float("-inf")

    print(
        f"\n[*] best_child: Node {id(node)} has {len(node.children)} children.")
    for idx, child in enumerate(node.children):
        q_value = child.get_value()
        visit_count = max(child.visit_count, 1)
        # Avoid log(0)
        parent_visit = node.visit_count if node.visit_count > 0 else 1
        ucb_exploration = c_param * \
            math.sqrt(math.log(parent_visit + 1) / visit_count)
        ucb_score = q_value + ucb_exploration

        print(
            f"  Child {idx} has q_value={q_value:.3f}, visits={child.visit_count}, UCB={ucb_score:.3f}")
        if ucb_score > best_score:
            best_score = ucb_score
            best = child

    return best
