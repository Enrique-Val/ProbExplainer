from probExplainer.model.Model import Model, abstractmethod


class ProbabilisticGraphicalModel(Model):
    # INTERFACE

    @abstractmethod
    def d_separation(self, node_set_1, node_set_2, separator_set) -> bool:
        pass

    @abstractmethod
    def markov_blanket(self, node) -> list:
        pass

    @abstractmethod
    def get_parents(self, node) -> list:
        pass

    @abstractmethod
    def get_children(self, node) -> list:
        pass
