import numpy as np
import pyAgrum as gum

from probExplainer.model.ProbabilisticGraphicalModel import ProbabilisticGraphicalModel, Model
from probExplainer.model.Model import ImplausibleEvidenceException


class BayesianNetwork(ProbabilisticGraphicalModel):
    def __init__(self, implementation):
        super().__init__(implementation)


class BayesianNetworkPyAgrum(BayesianNetwork):
    def __init__(self, implementation: gum.pyAgrum.BayesNet):
        if not isinstance(implementation, gum.pyAgrum.BayesNet):
            err = "The implementation provided is not a PyAgrum Bayesian network (type \"pyAgrum.pyAgrum.BayesNet\")"
            raise Exception(err)
        super().__init__(implementation)
        vars = sorted(list(implementation.names()))
        for i in implementation.names():
            self.variables_labels[i] = self.implementation.variableFromName(i).labels()
        self.name = "pyAgrum Bayesian network"

    def d_separation(self, node_set_1, node_set_2, separator_set) -> bool:
        return self.implementation.isIndependent(node_set_1, node_set_2, separator_set)

    def markov_blanket(self, node) -> set:
        return {self.implementation.variable(i).name() for i in gum.MarkovBlanket(self.implementation, node).nodes()}

    def get_parents(self, node):
        return [self.implementation.variable(i).name() for i in self.implementation.parents(node)]

    def get_children(self, node):
        return [self.implementation.variable(i).name() for i in self.implementation.children(node)]

    def compute_posterior(self, evidence: dict, target: list) -> np.array:
        if not self.plausible_evidence(evidence):
            raise ImplausibleEvidenceException
        target_aux = target.copy()
        target_aux.reverse()
        ie = gum.ShaferShenoyInference(self.implementation)
        ie.addJointTarget(set(target))
        ie.setEvidence(evidence)
        ie.makeInference()
        return ie.jointPosterior(set(target)).reorganize(target_aux).toarray()

    def evidence_likelihood(self, evidence: dict):
        if not self.plausible_evidence(evidence):
            raise ImplausibleEvidenceException
        ie = gum.ShaferShenoyInference(self.implementation)
        ie.setEvidence(evidence)
        ie.makeInference()
        p_e = ie.evidenceProbability()
        return p_e

    def compute_univariate(self, evidence: dict, target: list):
        if not self.plausible_evidence(evidence):
            raise ImplausibleEvidenceException
        ie = gum.ShaferShenoyInference(self.implementation)
        ie.setEvidence(evidence)
        ie.makeInference()
        posteriors = dict()
        for i in target:
            posteriors[i] = ie.posterior(i).toarray()
        return posteriors

    def plausible_evidence(self, evidence):
        ie = gum.ShaferShenoyInference(self.implementation)
        ie.setEvidence(evidence)
        try:
            ie.evidenceProbability()
            return True
        except Exception:
            return False
