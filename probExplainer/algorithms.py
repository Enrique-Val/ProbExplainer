import numpy as np
# import pyAgrum as gum
from probExplainer import utils
import math

from probExplainer.model.ProbabilisticGraphicalModel import ProbabilisticGraphicalModel, Model
from probExplainer.model.Model import ImplausibleEvidenceException
from probExplainer.model.BayesianNetwork import BayesianNetwork


def check_every_r_silja(model: ProbabilisticGraphicalModel, evidence: dict, target: list, depth=np.inf):
    # Check which are the supplementary variables
    global c_i
    variables = model.get_variables()
    supp_vars = []
    for var in variables:
        if var not in list(evidence.keys()) and var not in target:
            supp_vars.append(var)

    # Variables to store relevant/irrelevant sets
    relevant_sets = []
    irrelevant_sets = []

    dsep_vars = []
    not_dsep_vars = []

    for i in supp_vars:
        if model.d_separation(i, target, list(evidence.keys())):
            dsep_vars.append(i)
        else:
            not_dsep_vars.append(i)

    if len(not_dsep_vars) == 0:
        return [], dsep_vars

    # supp vars contains variables that are not dsep at this point
    # tmp contains all supplementary nodes

    # Put first the vars in the Markov Blanket
    mb = set()
    for i in target:
        mb = mb.union(model.markov_blanket(i))
    first_vars = []
    for i in not_dsep_vars:
        if i in mb:
            first_vars.append(i)
    tmp = utils.list_diff(not_dsep_vars, first_vars)
    # In supp vars, now we will have only non dsep vars. Dsep vars will be on dsep_vars
    supp_vars = first_vars + tmp

    dsep_vars = tuple(dsep_vars)
    irrelevant_singletons = []
    if len(dsep_vars) > 0:
        irrelevant_sets.append(dsep_vars)
        for i in dsep_vars:
            irrelevant_singletons.append(i)

    # Get the posterior and the argmax from the original MAP-query
    posterior = model.compute_posterior(evidence=evidence, target=target)
    y = model.argmax(array_prob=posterior, dim_names=target)[0]

    # Value assignment y, but in a different format
    h_star = tuple(list(y.values()))

    c_exp = get_c_exp(model, evidence, target=y)
    # inst_c = gum.Instantiation(c_exp)

    # First evaluation of relevance. It helps to identify relevance but NOT irrelevance
    to_prop = evidence.copy()
    to_prop.update(y)
    prob_R_given_e_h_star = model.compute_univariate(to_prop, supp_vars)
    relevant_singletons = []
    for j in supp_vars:
        # If relevant
        if model.argmin(prob_R_given_e_h_star[j], [j])[1] == 0:
            relevant_sets.append((j,))
            relevant_singletons.append(j)
    tmp = utils.list_diff(supp_vars, relevant_singletons)

    # Second evaluation of relevance. This will help us determine if the nodes not previously labeled are relevant or irrelevant
    omega_h = model.get_domain_of(target)
    for h_i in omega_h:
        if len(tmp) == 0:
            break
        if h_i == h_star:
            continue
        # print(h_i)
        # print(y)
        to_prop = evidence.copy()
        to_prop.update(dict(zip(target, list(h_i))))
        prob_R_given_e_hi = model.compute_univariate(to_prop, tmp)

        c_expon = utils.get_probability(model, c_exp, dim_names=target,
                                        assignment={target[i]: h_i[i] for i in range(len(target))})
        c_i = None
        if c_expon == 0:
            c_i = -np.inf
        else:
            c_i = math.log(c_expon)

        for j in tmp:
            post_max = prob_R_given_e_h_star[j]
            post_max.flatten()
            post = prob_R_given_e_hi[j]
            post.flatten()
            h = 0
            while h < len(post) and (post[h] / post_max[h] <= 0 or not math.log(
                    post[h] / post_max[h]) + c_i > 0):
                h = h + 1
            if h < len(post):
                relevant_sets.append((j,))
                relevant_singletons.append(j)
        # print(S_split[0])
        # print(relevant_singletons)
        tmp = utils.list_diff(tmp, relevant_singletons)

    irrelevant_singletons = irrelevant_singletons + tmp
    for j in irrelevant_singletons:
        irrelevant_sets.append((j,))

    if len(irrelevant_singletons) == 0 or depth == 1:
        return relevant_sets, irrelevant_sets

    # End of singleton phase
    S = utils.powerset(irrelevant_singletons, depth=depth)
    S.pop(0)
    # Divide by length
    S_split = []
    size = 0
    for i in S:
        if len(i) != size:
            size = size + 1
            S_split.append([])
        S_split[-1].append(i)

    for j in irrelevant_singletons:
        S_split, irrelevant_sets = conditional_independence_prune(model, supp_vars, target, evidence, (j,), S_split,
                                                                  irrelevant_sets, depth=depth)

    irrels = utils.powerset(dsep_vars, depth)
    irrels.pop(0)
    for i in irrels:
        try:
            S_split[len(i) - 1].remove(i)
        except ValueError:
            pass  # do nothing!

    for i in range(1, len(S_split)):
        tmp = S_split[i].copy()
        for j in tmp:
            # If relevant
            # print(list(j))
            if model.map_dependence(set_r=list(j), ev_vars=evidence, map=y):
                relevant_sets.append(j)
                # Apply prune
                S_split, relevant_sets = decomposition_prune(j, S_split, relevant_sets)
            # If irrelevant
            else:
                irrelevant_sets.append(j)
                S_split, irrelevant_sets = conditional_independence_prune(model, supp_vars, target, evidence, j,
                                                                          S_split, irrelevant_sets, depth)

    # Simplify irrelevant sets
    irrelevant_sets = sorted(irrelevant_sets, key=len)
    new_irrel_sets = []
    for i in range(0, len(irrelevant_sets)):
        subset_flag = False
        for j in range(len(irrelevant_sets) - 1, i, -1):
            if set(irrelevant_sets[i]).issubset(set(irrelevant_sets[j])):
                subset_flag = True
                break
        if not subset_flag:
            new_irrel_sets.append(irrelevant_sets[i])
    irrelevant_sets = new_irrel_sets

    return relevant_sets, irrelevant_sets


def get_c_exp(model: Model, evidence: dict, target: dict):
    # Compute P(e)
    p_e = model.evidence_likelihood(evidence)

    # Compute P(H|e)
    post_H_e = model.compute_posterior(evidence, target=list(target.keys()))

    # Compute P(H,e) = P(H|e)*P(e)
    post_He = post_H_e * p_e

    # Find P(h*,e)
    target_index = utils.dict_to_tuple_index(model, target)
    p_eh = post_He[target_index]
    return post_He / p_eh


def decomposition_prune(relevant_set, S_split, relevant_sets):
    i = len(relevant_set)
    for k in range(i, len(S_split)):
        tmp = []
        for l in S_split[k]:
            if not set(relevant_set).issubset(set(l)):
                tmp.append(l)
        S_split[k] = tmp
    return S_split, relevant_sets


def conditional_independence_prune(model : ProbabilisticGraphicalModel, supp_vars, hyp_vars, ev_vars, irrelevant_set, S_split, irrelevant_sets, depth):
    # Delete from the network the nodes that are conditionally independent from the hypothesis variables (target) given the evidence
    dsep_nodes = []
    for i in supp_vars:
        if i not in irrelevant_set and model.d_separation(i, hyp_vars, list(ev_vars.keys()) + list(irrelevant_set)):
            dsep_nodes.append(i)

    if len(dsep_nodes) == 0:
        return S_split, irrelevant_sets
    irrels = utils.powerset(dsep_nodes)
    bigger = irrels[-1]
    irrels.pop(0)
    irrels_new = []
    for i in irrels:
        if len(i) <= depth:
            irrels_new.append(i)
    irrels = irrels_new

    for i in irrels:
        if i in S_split[len(i) - 1]:
            S_split[len(i) - 1].remove(i)
    if bigger not in irrelevant_sets:
        irrelevant_sets.append(bigger)
    return S_split, irrelevant_sets