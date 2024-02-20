from itertools import chain, combinations
import numpy as np
from scipy.stats import entropy
from probExplainer.model import Model


def powerset(iterable, depth=np.inf) -> list:
    s = list(iterable)
    tmp = list(chain.from_iterable(combinations(s, r) for r in range(min(len(s), depth) + 1)))
    return tmp


# JSD divergence
def JSD(array_1: np.array, array_2: np.array) -> float:
    p = array_1.ravel()
    q = array_2.ravel()

    # calculate m
    m = (p + q) / 2

    # compute Jensen Shannon Divergence
    divergence = (entropy(p, m) + entropy(q, m)) / 2
    return divergence


def list_diff(list1, list2):
    diff = []
    for i in list1:
        if i not in list2:
            diff.append(i)
    return diff


def dict_to_tuple_index(model: Model, index: dict):
    target_index = list()
    for i in index.keys():
        target_index.append(model.get_domain_of([i]).index((index[i],)))
    return tuple(target_index)


def get_probability(model: Model, array_prob: np.ndarray, dim_names: list, assignment: dict):
    assert (dim_names == list(assignment.keys()))
    index = dict_to_tuple_index(model, assignment)
    return array_prob[index]
