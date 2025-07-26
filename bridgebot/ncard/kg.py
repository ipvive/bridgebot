from collections import deque, namedtuple
import importlib

from absl import logging
from bridgebot.ncard import rgeom


import pdb


# ============ building ===================
class Builder:
    def __init__(self):
        self.nodes = {}
        self.relations = []

    def add_rules(self, module_name, rules):
        for r in rules:
            if r not in self.nodes:
                self.nodes[r] = module_name

    def add_relations(self, rl):
        for r in rl:
            if r not in self.relations:
                self.relations.append(r)

    def to_bytes(self):
        raise NotImplementedError

    def from_bytes(self, data):
        raise NotImplementedError

    def __repr__(self):
        return f"Builder(nodes={self.nodes}, relations={self.relations}"

    def export(self, pyley_client):
        logging.error("not yet implemented")


# ============ compiler ===================
def topological_order(graph):
    order, enter, state = deque(), set(graph.keys()), {}

    GRAY, BLACK = 0, 1
    def dfs(node):
        state[node] = GRAY
        for k in graph.get(node, ()):
            sk = state.get(k, None)
            if sk == GRAY: raise ValueError("cycle")
            if sk == BLACK: continue
            enter.discard(k)
            dfs(k)
        order.appendleft(node)
        state[node] = BLACK

    while enter:
        dfs(enter.pop())
    
    order.reverse()
    return order


LinkedNode = namedtuple("LinkedNode", ['name', 'deps', 'ordered_recursive_deps', 'fn'])
LinkedKG = namedtuple("LinkedKG", ['nodes', 'relations'])

def _node_name(fn):
    if hasattr(fn, 'name'):
        return fn.name
    elif fn.__doc__:
        lines = fn.__doc__.split("\n")
        return lines[0]


def link(builder, modules=None):
    if modules is None:
        modules = {}
    name_to_fn = {}
    for n, m_name in builder.nodes.items():
        if m_name not in modules:
            modules[m_name] = importlib.import_module(m_name)
        module = modules[m_name]
        # TODO this is slower than needed as it is called per node
        node_by_name = {_node_name(f): f
                for f in module.__dict__.values()
                if callable(f)}
        if n in node_by_name:
            name_to_fn[n] = node_by_name[n]
        else:
            raise KeyError(f"node '{n}' not found in module {m_name}")
  
    deps_graph = {n:[] for n in name_to_fn.keys()}
    order_graph = {n:[] for n in name_to_fn.keys()}
    for t in builder.relations:
        if 'requires' == t[1]:
            deps_graph[t[0]].append(t[2])
            order_graph[t[0]].append(t[2])
        elif 'has lower priority than' == t[1]:
            order_graph[t[0]].append(t[2])

    # TODO re-implement if we need vars defined multiple times
    var_def = {var: n_name for n_name, f in name_to_fn.items()
                           for var in f.outputs if var not in f.inputs}
    for n_name, f in name_to_fn.items():
        for var in f.inputs:
            if var in var_def:
                order_graph[n_name].append(var_def[var])

    order = topological_order(order_graph)
    linked_nodes = {}
    
    GRAY, BLACK = 0, 1
    def dfs(node, order, state):
        state[node.name] = GRAY
        for k in linked_nodes[node.name].deps.values():
            sk = state.get(k.name, None)
            if sk == GRAY: ValueError("cycle")
            if sk == BLACK: continue
            dfs(k, order, state)
        order.appendleft(node)
        state[node.name] = BLACK
        return order

    for n in order:
        deps = {d: linked_nodes[d] for d in deps_graph[n]}
        linked_nodes[n] = LinkedNode(n, deps, None, name_to_fn[n])
        ordered_recursive_deps, local_state = deque(), {}
        dfs(linked_nodes[n], ordered_recursive_deps, local_state)
        ordered_recursive_deps.reverse()
        ordered_recursive_deps.pop()
        linked_nodes[n] = LinkedNode(n, deps, tuple(ordered_recursive_deps), name_to_fn[n])

    linked_relations = ((linked_nodes[s], p, linked_nodes[o])
        for s, p, o in builder.relations)

    linked_kg = LinkedKG(linked_nodes, linked_relations)
    return linked_kg


# ============ executive ===================
class _ClassDict(dict):
    def __init__(self, keys):
        super().__init__()
        self.allowed_keys = frozenset(keys)

    def __setitem__(self, k, v):
        if k not in self.allowed_keys:
            raise ValueError(k)
        super().__setitem__(k, v)


class ValueGraph:
    def __init__(self, nodes, inputs, outputs):
        self.rule_nodes = nodes
        self.inputs = inputs
        self.target_outputs = outputs

    def __call__(self):
        self.nodes = {}
        for n in self.rule_nodes.values():
            deps = [self.nodes[k] for k in n.deps]
            v = self._assemble_inputs(n, deps)
            n.fn(v)
            self._distribute_outputs(v)
        self.outputs = {k: self.get_target(v)
                for k, v in self.target_outputs.items()}
        return self.outputs

    def get_target(self, spec):
        nodename, varname = spec
        return self.nodes[nodename].outputs[varname]
    
    def _assemble_inputs(self, n, deps):
        v = ValueNode(n.fn, deps)
        for k in v.inputs.allowed_keys:
            try:
                v.inputs[k] = self.inputs[k]
            except:
                pdb.set_trace()
        return v

    def _distribute_outputs(self, v):
        self.nodes[v.rule.name] = v
        for k, v in v.outputs.items():
            if k not in self.inputs:
                self.inputs[k] = v


class RuleNode:
    def __call__(self, v):
        self.fn(v)

    def __init__(self, fn, name, inputs, outputs):
        self.fn = fn
        self.name = name
        self.inputs = inputs
        self.outputs = outputs


class ValueNode:
    def __init__(self, rule, deps):
        self.rule = rule
        self.deps = deps
        self.inputs = _ClassDict(self.rule.inputs)
        self.outputs = _ClassDict(self.rule.outputs)


def rule(name, inputs, outputs):
    """Decorater for rule functions."""
    def wrapper(fn):
        return RuleNode(fn, name, inputs, outputs)

    return wrapper


def execute(lkg, inputs, outputs, return_kg=False):
    needed_nodes = lkg.nodes
    # TODO: prune nodes not needed to compute outputs

    vkg = ValueGraph(needed_nodes, inputs, outputs)
    outputs = vkg()
    if return_kg:
        return outputs, vkg
    else:
        return outputs
