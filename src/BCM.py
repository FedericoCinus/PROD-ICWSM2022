from PROD_EPISTEMIC import *
import numpy as np

class BCM(PROD):
    '''BCM implementation for PROD
    '''
    def __init__(self,
                 G,
                 graph_name,
                 rewiring,
                 rewiring_type=0,
                 suscept_range=(0, 1),
                 suscept_distrib=None,
                 beta=0.,
                 max_time=5000,
                 bcm_eps=.25, bcm_mu=.1,
                 init_opinions=None,
                 recommender=None,
                 eps=.1, exp=1000,
                 s=1,
                 intervention_prob=0,
                 intervention_type=0,
                 with_misinformation=0,
                 debug=False,
                 save_all_opinions=False,
                 save_all_graphs=False,
                 save_all_metrics=False,
                 save_all_links=False,
                 **kwargs):

        super().__init__(G, graph_name, rewiring, rewiring_type, suscept_range, suscept_distrib,
                         beta, max_time, init_opinions,
                         recommender, eps, exp, s, intervention_prob, intervention_type,
                         with_misinformation, debug,
                         save_all_opinions, save_all_graphs, save_all_metrics,
                         save_all_links)

        self.bcm_eps = bcm_eps
        self.bcm_mu = bcm_mu

    def update_node_opinion(self, u, v):
        # computing the edge disagreement
        edge_disagreement = self.opinions[v] - self.opinions[u]
        # performing BCM opinion update step
        if abs(edge_disagreement) < self.bcm_eps:
            if self.opinions[u] < 0.5 <= self.opinions[v]:
                self.count_interaction_type['pro_no'] += 1
            if self.opinions[u] < 0.5 and self.opinions[v] < .5:
                self.count_interaction_type['no_no'] += 1
            if self.opinions[u] >= 0.5 and self.opinions[v] >= .5:
                self.count_interaction_type['pro_pro'] += 1
            if  self.opinions[v] < .5 <= self.opinions[u]:
                self.count_interaction_type['no_pro'] += 1
            self.opinions[u] += self.bcm_mu*edge_disagreement
        else:
            self.count_interaction_type['other'] += 1


    def make_nodes_experiments(self, seed=None, k=None):
        pass

    def is_converged(self):
        if np.std(self.opinions) < 0.001:
            return 1
        return -777
