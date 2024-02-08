from probExplainer.model import BayesianNetwork
from probExplainer.algorithms.utils import *
from probExplainer.algorithms import defeater
import pyAgrum as gum
import pandas as pd
#import pyAgrum.lib.notebook as gnb


if __name__ == '__main__':
    '''
    bn = gum.loadBN("insurance.bif")

    ev_vars = {'RiskAversion': "Psychopath", "VehicleYear": "Older"}
    hyp_vars = ["Accident", "Age", "ILiCost"]
    my_adapter = BayesianNetwork.BayesianNetworkPyAgrum(bn)
    posterior = my_adapter.compute_posterior(ev_vars, hyp_vars)

    target = my_adapter.argmax(posterior, hyp_vars)[0]

    ev_vars = {'RiskAversion': "Psychopath", "ILiCost": "Thousand"}
    hyp_vars = ["Age"]
    #print(algorithms.check_every_r_silja(my_adapter, ev_vars, hyp_vars, depth=2))
    #print(my_adapter.maximum_a_posteriori(ev_vars, hyp_vars))
    #print("MAPI", my_adapter.map_independence(['MedCost', 'ThisCarDam'], ev_vars, {"Age": "Adult"}))

    bn_1 = gum.loadBN("expert_networks/network_2.bif")
    #gnb.flow.add(gnb.getBN(bn_1, size="20"))
    #gnb.flow.display()
    '''
    # Prepare experiments
    N_EXPERTS = 42
    bn_i = gum.loadBN("expert_networks/network_5.bif")
    my_adapter = BayesianNetwork.BayesianNetworkPyAgrum(bn_i)
    marginal_f5 = {i[0] : 0 for i in my_adapter.get_domain_of(["F5"])}
    ev_vars = {}#{"F1": "intralaminar"}
    target = ["F5"]
    print(my_adapter.get_variables())
    var_powerset = powerset([i for i in my_adapter.get_variables() if i not in target and i not in ev_vars.keys()])
    var_powerset.pop(0)
    map = my_adapter.maximum_a_posteriori(evidence=ev_vars, target=target)
    print(map)


    ## EXPERIMENT 1. COUNT OF RELEVANT SETS
    relevance_count = {i : 0 for i in var_powerset}
    for i in range(N_EXPERTS) :
        print("Expert: ",i+1)
        bn_i = gum.loadBN("expert_networks/network_"+str(i+1)+".bif")
        my_adapter = BayesianNetwork.BayesianNetworkPyAgrum(bn_i)
        map = my_adapter.maximum_a_posteriori(evidence=ev_vars, target=target)
        print(map)
        marginal_f5[map[0]["F5"]] = marginal_f5[map[0]["F5"]] + 1
        relevant_vars = defeater.get_defeaters(my_adapter, ev_vars, target)[0]
        print("Relevant variables: ", relevant_vars)
        for j in relevance_count.keys() :
            for k in relevant_vars :
                if set(k).issubset(set(j)) :
                    relevance_count[j] = relevance_count[j] + 1
                    break
        print()
    
    
    # EXPERIMENT 2: COMPUTATION OF MAPI-STRENGTH FOR EACH SUBSET
    relevances = pd.DataFrame(columns=var_powerset, index=list(range(N_EXPERTS)), dtype=float)
    for i in range(N_EXPERTS):
        print("Expert: ", i + 1)
        bn_i = gum.loadBN("expert_networks/network_" + str(i + 1) + ".bif")
        my_adapter = BayesianNetwork.BayesianNetworkPyAgrum(bn_i)
        map = my_adapter.maximum_a_posteriori(evidence=ev_vars, target=target)
        for j in var_powerset:
            print(j)
            strength_j = 1 - my_adapter.map_independence_strength(ev_vars=ev_vars, map=map[0], set_r=list(j))
            print("Relevance strength: ", strength_j)
            relevances.loc[i][j] = strength_j
        print()

    print("F5 count: ", marginal_f5)
    print()

    print("EXPERIMENT 1 results")
    print("Relevance count: ", relevance_count)
    print()

    print("EXPERIMENT 2 results")
    print(relevances.describe())