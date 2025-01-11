# models/node.py

import math


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
