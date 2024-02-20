from abc import ABC, abstractmethod
from itertools import product
import numpy as np
from probExplainer.algorithms import utils


class Model(ABC):
    # CONSTRUCTOR
    def __init__(self, implementation):
        self.implementation = implementation
        self.variables_labels = dict()
        self.name = ""

    # GETTERS
    def get_implementation(self):
        return self.implementation

    def get_name(self):
        return self.name

    def get_variables(self):
        return sorted(list(self.variables_labels.keys()))

    def get_variables_labels(self):
        return self.variables_labels

    def get_domain_of(self, variables) -> list:
        domains = []
        for variable in variables:
            domains.append(self.variables_labels[variable])
        return [p for p in product(*domains)]

    # INTERFACE
    # MAP-Query (multivariate predict)
    # Gets the a posteriori distribution of y_names
    # evidence: dataframe_series or dict
    # if y_names not given, the ones not in X will be used
    def maximum_a_posteriori(self, evidence, target):
        posterior = self.compute_posterior(evidence, target)
        return self.argmax(posterior, target)

    @abstractmethod
    def compute_posterior(self, evidence: dict, target: list) -> np.array:
        pass

    @abstractmethod
    def evidence_likelihood(self, evidence: dict):
        pass

    # Univariate predict
    # Gets the univariate a posteriori distribution for attribute in y_names
    # It could also be added as an input in the previous function
    def compute_univariate(self, evidence: dict, target: list):
        pass

    def map_independence(self, set_r: list, ev_vars: dict, map: dict, posterior=None, return_jsd=False) -> bool | tuple:
        if return_jsd:
            map_dep, jsd = self.map_dependence(set_r, ev_vars, map, posterior=posterior, return_jsd=True)
            return not map_dep, jsd
        else:
            return not self.map_dependence(set_r, ev_vars, map)

    def map_dependence(self, set_r: list, ev_vars: dict, map: dict, posterior=None, return_jsd=False) -> bool | tuple:
        if return_jsd and posterior is None:
            err = "For the Jensen-Shannon divergence to be computed, the parameter \"posteriors\"" \
                  " should contain an array representing the probabilities of the targets y given the evidence"
            raise Exception(err)

        # Check which are the supplementary (missing) variables
        variables = self.get_variables()
        supp_vars = []
        for var in variables:
            if var not in list(ev_vars.keys()) and var not in list(map.keys()):
                supp_vars.append(var)
        # Check if R in unobserved
        for R in set_r:
            if R not in supp_vars:
                err = "The variable " + R + " is in the set R but is not a supplementary node"
                raise Exception(err)

        # Obtain domain of R
        omega_r = self.get_domain_of(set_r)
        # For each value assignment r in omega(R)
        jsd = 0
        for value_assignment_r in omega_r:
            # Fill in values
            ev_vars_alt = ev_vars.copy()
            for i, value in enumerate(value_assignment_r):
                ev_vars_alt[set_r[i]] = value
            # print(instance)
            # print(instance_alt)
            # Inference with evidence and r
            try:
                posterior_alt = self.compute_posterior(evidence=ev_vars_alt, target=list(map.keys()))
                map_alt = self.argmax(posterior_alt, list(map.keys()))[0]
                # Check if we need to compute the jsd divergence between P(H|e) and P(H|e,r)
                if return_jsd:
                    jsd = max(jsd, utils.JSD(posterior, posterior_alt))
                if map != map_alt:
                    if return_jsd:
                        return True, jsd
                    else:
                        return True
            except ImplausibleEvidenceException:
                continue
        if return_jsd:
            return False, jsd
        else:
            return False

    def map_independence_strength(self, set_r: list, ev_vars: dict, map: dict):
        # Check which are the supplementary (missing) variables
        variables = self.get_variables()
        supp_vars = []
        for var in variables:
            if var not in list(ev_vars.keys()) and var not in list(map.keys()):
                supp_vars.append(var)
        # Check if R in unobserved
        for R in set_r:
            if R not in supp_vars:
                err = "The variable " + R + " is in the set R but is not a supplementary node"
                raise Exception(err)

        # Obtain domain of R
        omega_r = self.get_domain_of(set_r)
        # For each value assignment r in omega(R)
        p_r_given_e = self.compute_posterior(evidence=ev_vars, target=set_r)
        # print(set_r)
        # print(p_r_given_e)
        strength = 0
        for value_assignment_r in omega_r:
            # Fill in values
            ev_vars_alt = ev_vars.copy()
            for i, value in enumerate(value_assignment_r):
                ev_vars_alt[set_r[i]] = value
            try:
                posterior_alt = self.compute_posterior(evidence=ev_vars_alt, target=list(map.keys()))
                map_alt = self.argmax(posterior_alt, list(map.keys()))[0]
                # print("R value: ", {i[0]: i[1] for i in zip(set_r, value_assignment_r)})
                # print("MAP alternative: ",map_alt)
                # Check if we need to compute the jsd divergence between P(H|e) and P(H|e,r)
                if map == map_alt:
                    strength = strength + utils.get_probability(self, array_prob=p_r_given_e, dim_names=set_r,
                                                                assignment={i[0]: i[1] for i in
                                                                            zip(set_r, value_assignment_r)})
            except ImplausibleEvidenceException:
                continue
        if strength < 0:
            strength = 0
        if strength > 1:
            strength = 1
        return strength

    def argmax(self, array_prob, dim_names=None):
        if dim_names is None:
            dim_names = list(range(len(array_prob.shape)))
        assert (len(array_prob.shape) == len(dim_names))
        max_index = np.unravel_index(array_prob.argmax(), array_prob.shape)
        return {dim_names[i]: self.get_domain_of([dim_names[i]])[max_index[i]][0] for i in range(len(dim_names))}, \
            array_prob[max_index]

    def argmin(self, array_prob, dim_names=None):
        if dim_names is None:
            dim_names = list(range(len(array_prob.shape)))
        assert (len(array_prob.shape) == len(dim_names))
        max_index = np.unravel_index(array_prob.argmin(), array_prob.shape)
        return {dim_names[i]: self.get_domain_of([dim_names[i]])[max_index[i]][0] for i in range(len(dim_names))}, \
            array_prob[max_index]


class ImplausibleEvidenceException(Exception):
    pass
