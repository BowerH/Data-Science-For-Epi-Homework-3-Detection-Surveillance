from typing import Dict, List, Optional, Set, Tuple
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import copy


def simulate_one_round_SI(
    G: nx.Graph,
    node_compartments: Dict[str, Set[int]],
    beta: float,
    rng: np.random.RandomState = np.random.RandomState(0),
):
    if len(node_compartments["I"]) == 0 or len(node_compartments["S"]) == 0:
        return node_compartments
    s_to_i = []
    for node in node_compartments["I"]:
        for neighbor in set(G.neighbors(node)).intersection(node_compartments["S"]):
            if rng.rand() < beta:
                s_to_i.append(neighbor)
    for node in set(s_to_i):
        node_compartments["S"].remove(node)
        node_compartments["I"].add(node)
    return node_compartments


def simulate_t_steps_SI(
    G: nx.Graph,
    i_frac: float,
    beta: float,
    num_rounds: int,
    seed: Optional[int],
    full_output: bool = False,
) -> Dict[str, np.ndarray] | Tuple[
    Dict[str, np.ndarray], List[Set[int]], List[Set[int]]
]:
    rng = np.random.RandomState(seed) if seed is not None else np.random.RandomState()

    num_nodes = G.number_of_nodes()
    i_num = int(i_frac * num_nodes)
    s_num = num_nodes - i_num
    nodes_perm = rng.permutation(G.nodes)
    node_compartments = {
        "S": set(nodes_perm[:s_num].tolist()),
        "I": set(nodes_perm[s_num:].tolist()),
    }
    s = [s_num]
    i = [i_num]
    if full_output:
        s_full = [copy.deepcopy(node_compartments["S"])]
        i_full = [copy.deepcopy(node_compartments["I"])]

    for _ in range(num_rounds):
        node_compartments = simulate_one_round_SI(G, node_compartments, beta, rng)
        if full_output:
            s_full.append(copy.deepcopy(node_compartments["S"]))
            i_full.append(copy.deepcopy(node_compartments["I"]))
        s.append(len(node_compartments["S"]))
        i.append(len(node_compartments["I"]))
    ans = {"S": np.array(s), "I": np.array(i)}
    if full_output:
        return ans, s_full, i_full
    return ans
