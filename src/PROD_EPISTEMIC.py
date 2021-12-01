#!/usr/bin/env python
import os
import mkl
import pickle
import random
import warnings
import numpy as np
import networkx as nx
from tqdm import tqdm
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
from sklearn.preprocessing.data import QuantileTransformer

from scipy.sparse import SparseEfficiencyWarning
from recommenders import *
from measures import *

warnings.simplefilter('ignore', SparseEfficiencyWarning)
mkl.set_num_threads(8)



PERC = .5 #percentage of max time steps that defines the upper time limit for completing all rewirings
EPSILON = 10E-3
##########################################################################

class PROD:
    '''PROD implementation for Epistemic model
    '''
    def __init__(self,
                 G,                       # Networkx instance where to perform PROD
                 graph_name,              # (str) title of the graph
                 rewiring,                # (int) number of rewirings
                 rewiring_type=0,         # 0 -> random uniform, 1 -> propto |o[u] - o[v]|, 2 -> propto deg(v)
                 suscept_range=(0, 1),    # min, max values of suscept coeff only for recommenders not uniform
                 suscept_distrib=None,    # 0 -> uniform
                 beta=0.,                 # percentage of randomness in the recommender (use it for recommender >0)
                 max_time=5000,           # max number of timesteps
                 init_opinions=None,      # intial opinions
                 recommender=None,        # None-> nessuno, 0-> Random, 1-> DJ, 2-> PPR, 3->WTF
                 eps=.1, exp=1000,        # params of epistemic: eps prob difference action1-action0, exp numb of node experiments
                 s=1,                     # number of interactions per node in each timestep
                 intervention_prob=0,
                 intervention_type=0,     # 0 uniform, 1 = opinion distance, 2 = 1/in_degree
                 with_misinformation=0,   # 0-> None, 1: also between no-vax, 2: also between no-vax and pro-vax
                 debug=False,
                 save_all_opinions=False,
                 save_all_graphs=False,
                 save_all_metrics=False,
                 save_all_links=False,
                 **kwargs):

        self.graph_name = graph_name
        if G.is_directed():
            self.G = G.copy()
        else:
            self.G = G.to_directed().copy()
        if isinstance(recommender, str):
            x, y = recommender.split(',')
            self.recommender = int(x)
            if y.startswith('-'):
                self.factor = -float(y[1:])/100
            else:
                self.factor = float(y)/100
        else:
            self.recommender = recommender
            self.factor = .85 # damping factor for PPR and WTF or gamma for biased_link
        if self.recommender == 1:
            self.A = nx.to_numpy_matrix(self.G)
            self.in_degrees = self.A.sum(axis=0)
            self.out_degrees = self.A.sum(axis=1)
        elif self.recommender == 4:
            self.A = nx.to_numpy_matrix(self.G)
        elif self.recommender in (2, 3):
            self.A = nx.to_scipy_sparse_matrix(self.G)
        self.number_nodes = G.number_of_nodes()
        self.opinions = init_opinions.copy() if init_opinions is not None else np.random.rand(self.number_nodes)
        self.rewiring = rewiring
        self.rewiring_type = rewiring_type
        self.min_suscept, self.max_suscept = suscept_range
        self.suscept_distrib = suscept_distrib
        self.suscept_exp = 2 # only for power law distrib
        self.beta = beta
        self.max_time = max_time
        self.eps = eps
        self.exp = exp
        self.s = s
        self.intervention_prob = intervention_prob
        self.intervention_type = intervention_type
        # ALPHA is the probability to choose non-existing link: uses recommender
        # it is built in order to perform all the required rewirings around the PERC*max_time timestep
        self.alpha = self.rewiring / (self.max_time * PERC * self.s * self.number_nodes)
        assert self.alpha <= .5, ('Non riesco a fare tutte le raccomandazioni entro il 50% dei timesteps',
                                 '\nmax time:',
                                 self.max_time, 'perc rewiring:', self.rewiring)
        self.tot_rewiring = 0
        # WITH_MISINFORMATION allows other types of interactions with respect to the original paper of epistemic
        self.with_misinformation = with_misinformation
        # counting the interactions between pro-vax -> pro-vax
        # counting the interactions between pro-vax -> no-vax
        # counting the interactions between no-vax -> pro-vax [appears only with with_misinformation=1/2]
        self.count_interaction_type = {'pro_pro': 0, 'pro_no': 0,
                                       'no_no': 0, 'no_pro': 0,
                                       'other': 0}
        deg = dict(self.G.in_degree).values()
        self.avg_in_degree = sum(deg) / self.number_nodes

        ### DEBUG
        self.debug = debug #if True, the debug dictionary is returned
        self.debug_dict = {'probabilities':[]}
        self.all_opinions = [] if save_all_opinions else None
        self.all_graphs = [] if save_all_graphs else None
        self.all_metrics = [] if save_all_metrics else None
        self.all_links = [G.number_of_edges()] if save_all_links else None
        self.verbose = False
        self.quantile_scaler = None
        self.prob_evidence_ratio = None


    def simulate(self, seed=None, verbose=False):
        '''Main method of the PROD class. Simulates the model doing:
            0. settings
            1. measuring the metric at intial and final step
            2. calling make_one_timestep steps times
        '''
        NOT_COVERGED = -777
        self.settings(seed, verbose)
        ## initial measures
        metrics_i = self.compute_metrics(verbose)
        if isinstance(self.all_metrics, list):
            self.all_metrics.append(metrics_i)
        ###########

        iterable = tqdm(range(self.max_time), desc='Timesteps', leave=True) if verbose else range(self.max_time)
        time = 0
        for time in iterable:
            seed = seed+time+1 if seed is not None else None
            self.make_one_timestep(seed)
            self.save(time)
            convergence = self.is_converged()
            if convergence != NOT_COVERGED:
                if verbose:
                    print('Converged at time', time)
                ## final measures
                metrics_f = self.compute_metrics()
                return time+1, convergence, self.opinions, self.tot_rewiring, metrics_i, metrics_f, self.all_opinions, self.all_graphs, self.all_metrics, self.all_links, self.count_interaction_type, self.debug_dict
        ## final measures
        metrics_f = self.compute_metrics()
        return time+1, convergence, self.opinions, self.tot_rewiring, metrics_i, metrics_f, self.all_opinions, self.all_graphs, self.all_metrics, self.all_links, self.count_interaction_type, self.debug_dict


    def compute_metrics(self, verbose=False):
        '''calculates all defined metrics
        '''
        #time0 = time.time()
        metrics_results = {}
        metrics = [('avg_op', compute_avg_op),
                   #('eci', compute_echo_chamber_index),
                   ('ndi', compute_network_disagreement),
                   ('stdv', compute_std_dev),
                   #('entropy', compute_entropy_opinions),
                   ('clust', compute_clustering_coeff),
                   ('BC', compute_BC),
                   #('mod', compute_opinion_modularity),]
                   ('cont_corr_neigh', compute_cont_correlation_neighbors),
                   ('num_peaks', compute_num_opinion_peaks),
                   ('RWC', compute_RWC),
                   ('strongly_cc', compute_numb_strongly_cc)]
                   #('discr_corr_f', compute_discr_correlation_funct)]
        iterable = tqdm(metrics, desc='Metrics', leave=True) if verbose else metrics
        for metric_name, metric in iterable:
            metrics_results[metric_name] = metric(self.opinions, self.G)
        return metrics_results



    def make_one_timestep(self, seed=None):
        '''Defines each timestep of the simulation:
            0. each node makes experiments
            1. loops in the permutation of nodes choosing the INFLUENCED node u (u->v means u follows v, v can influence u)
            2. loops s (number of interactions times)
            3. choose existing links with 1-a prob, else recommends
                4. if recommendes: invokes recommend_nodes() to choose the influencers nodes that are not already linked u->v

        '''
        # updating avg in-degree
        deg = dict(self.G.in_degree).values()
        self.avg_in_degree = sum(deg) / self.number_nodes

        self.make_nodes_experiments(seed)
        # each node makes experiments concerning its opinion
        ######## interaction:= u->v, u follows v, v can influence u
        self.set_seed(seed)
        u = None
        for i, u in enumerate(np.random.permutation(self.G.nodes())):
            seed += i
            end_rewiring = self.tot_rewiring >= self.rewiring
            influencers_for_opinion_update = []
            while len(influencers_for_opinion_update) < self.s:
                try:
                    len(influencers_for_opinion_update)
                except:
                    print('3')
                ##### existing links - no recommender
                self.set_seed(seed)
                if self.recommender is None or np.random.rand() <= (1-self.nodes_susceptibility_vec[u]) or end_rewiring:
                    influencers = self.choose_neighbor(u, seed)
                    influencers_for_opinion_update.append(influencers[0])
                ##### no existing links
                else:
                    if np.random.rand() < self.intervention_prob:
                        influencers = self.intervention_algorithm(u, self.intervention_type)
                    else:
                        influencers = self.recommend_nodes(u, seed)
                    if influencers[0] is not None:
                        influencers_for_opinion_update.append(influencers[0])
                        try:
                            number_of_edges_to_remove = min(len(influencers), len(list(self.G.successors(u))))
                        except:
                            print('4', influencers, list(self.G.successors(u)))
                        edges_to_be_removed = self.rewiring_list(u, number_of_edges_to_remove,
                                                                 self.rewiring_type)
                        self.G.remove_edges_from(edges_to_be_removed) # deleting previously existing links
                        self.G.add_edges_from(influencers)
                        try:
                            self.tot_rewiring += len(influencers)
                        except:
                            print('5')
                        if self.recommender in (1, 2, 3):
                            self.A[tuple(zip(*influencers))] = 1
                            self.A[tuple(zip(*edges_to_be_removed))] = 0
                        if self.recommender == 1:
                            self.in_degrees = self.A.sum(axis=0)
                seed += 1

            ###### influence step
            self.update_node_opinion(u, u)
            for infl_tuple in influencers_for_opinion_update:
                _u, v = infl_tuple
                self.update_node_opinion(u, v) # update opinion of u given the influence of v





    def update_node_opinion(self, u, v):
        '''Opinion of u is updating considering the experiments of v (u->v)
        https://www.dropbox.com/sh/lyk7sd6h8459obp/AADxMzMcEDy--K1Odq9rlM97a?dl=0&preview=Epistemic_networks_code.py
        '''
        EPS = 10E-100
        curr_op = self.opinions[u]

        prob_evidence_ratio = None
        #### COUNTING
        if self.opinions[v] >= .5:
            # checking underflow
            if EPS < curr_op < 1 - EPS:
                prob_evidence_ratio = self.prob_evidence_ratio[v]
            # counting interactions
            if curr_op >= .5:
                self.count_interaction_type['pro_pro'] += 1
            else:
                self.count_interaction_type['pro_no'] += 1
        elif self.with_misinformation in (1, 2) and curr_op < .5:
            # checking underflow
            if EPS < curr_op < 1 - EPS:
                prob_evidence_ratio = self.prob_evidence_ratio[v]
            # counting interactions
            self.count_interaction_type['no_no'] += 1
        elif self.with_misinformation == 2 and curr_op >= .5:
            # checking underflow
            if EPS < curr_op < 1 - EPS:
                prob_evidence_ratio = self.prob_evidence_ratio[v]
            # counting interactions
            self.count_interaction_type['no_pro'] += 1
        else:
            self.count_interaction_type['other'] += 1

        #### UPDATING OPINION
        if prob_evidence_ratio is not None:
            bayes = 1 / (1 + np.divide((1-curr_op)*prob_evidence_ratio, curr_op))
            self.opinions[u] = bayes



    def choose_neighbor(self, u, seed=None):
        '''Choosing influencers (list of 1) from existing successors of the node u (u follows the influencers)
        '''
        self.set_seed(seed)
        influencers = []
        u_neigh = list(self.G.successors(u))
        if u_neigh:
            try:
                rand_idx = np.random.randint(len(u_neigh))
            except:
                print('6')
            v = u_neigh[rand_idx]
            influencers.append((u, v))
            return influencers
        return [None]


    def recommend_nodes(self, u, seed=None):
        '''Recommends influencers (list of 1) from non-existing successors:
            uniform or invokes recommender systems
        '''
        ### choosing influencers
        influencers = []

        # UNIFORM RECOMMENDER
        if self.recommender == 0:
            influencers = self.intervention_algorithm(u, 0)

         # RECOMMENDER SYSTEM
        else:
            top_k_edges = self.recommender_sys(u, self.recommender)
            if not top_k_edges:
                return [None]
            ### transformation
            prob = None
            for _, v, prob in top_k_edges:
                if not u == v: # avoiding self loops
                    break
            p_transformed = self.quantile_scaler.transform([[prob],])

            ### extracting influencers
            if self.debug:
                self.debug_dict['probabilities'].append(p_transformed)
            if np.random.rand() < p_transformed:
                if np.random.rand() < self.beta:
                    u_neigh = list(self.G.successors(u))
                    try:
                        len(u_neigh)
                    except:
                        print('7')
                    if len(u_neigh) < self.number_nodes-1:
                        not_found = True
                        while not_found:
                            v = np.random.randint(0, self.number_nodes)
                            if not v in u_neigh and not v in [x for _, x in influencers] and v != u:
                                influencers.append((u, v))
                                not_found = False
                    else:
                        raise ValueError("Node has full set of neighbors and can't be recommended any nodes")
                else:
                    influencers.append((u, v))
            if not influencers:
                influencers = [None]
        return influencers


    def intervention_algorithm(self, u, intervention_type):
        '''Returns list of one element: (u, v)
        '''
        influencers = []
        u_neigh = list(self.G.successors(u))

        if intervention_type == 0:
            intervention_condition = lambda _v: True

        elif intervention_type == 1:
            opinion_difference = lambda v: np.absolute(self.opinions[u] - self.opinions[v])
            intervention_condition = lambda v: np.random.rand() < opinion_difference(v)
        elif intervention_type == 2:
            try:
                norm_factor = self.number_nodes - len(u_neigh)
            except:
                print('0')
            v_in_degree = lambda v: self.G.in_degree(v) + 10E-7
            intervention_condition = lambda x: np.random.rand() < 1 / ( 1 + np.exp( - 1.*(v_in_degree(x) - self.avg_in_degree) ))
        else:
            raise ValueError(str(intervention_type) + 'not implemented!')

        if len(u_neigh) != self.number_nodes - 1:
            while not influencers:
                v = np.random.randint(0, self.number_nodes)
                if not v in u_neigh and v != u and intervention_condition(v):
                    influencers.append((u, v))
        else:
            influencers = [None]

        return influencers


    def calculate_rewiring_probs(self, node, node_neighbors, rewiring_type, min_degree=1):
        RANDOM = .001
        if rewiring_type == 1:
            probs = np.absolute(self.opinions[node] -  self.opinions[node_neighbors])
        elif rewiring_type == 2:
            _neighbors_indexes, proportional_factors = zip(*self.G.in_degree(node_neighbors))
            probs = 1 / (np.array(proportional_factors) + RANDOM)
        else:
            print('Wrong rewiring type')
        constraints = 1. * (np.array([self.G.in_degree[x] for x in node_neighbors]) > min_degree)
        probs *= constraints
        norm = np.sum(probs)
        if norm > 0:
            return  probs / norm
        probs = np.repeat(RANDOM, len(node_neighbors))
        probs *= constraints
        norm = np.sum(probs)
        return probs / norm

    def rewiring_list(self, node, number_of_rewiring, rewiring_type):
        nodes_to_be_unfollowed = []
        node_neighbors = np.array(list(self.G.successors(node)))

        if rewiring_type == 0:
            nodes_to_be_unfollowed = np.random.permutation(node_neighbors)[:number_of_rewiring]
        elif rewiring_type in (1, 2):
            rewiring_probs = self.calculate_rewiring_probs(node, node_neighbors, rewiring_type)
            try:
                nodes_to_be_unfollowed = np.random.choice(node_neighbors, number_of_rewiring, False, rewiring_probs)
            except:
                print('Random choice error!', '\n np.sum(rewiring_probs):', np.sum(rewiring_probs), rewiring_probs)

        else:
            print('Wrong choice for rewiring')

        return list(map(lambda x: tuple([node, x]), nodes_to_be_unfollowed))



    def recommender_sys(self, u, recommender):
        '''Recommender systems
        '''
        if recommender == 1:
            top_k_edges = calculate_DJ_row(A=self.A,
                                           u=u,
                                           in_degrees=self.in_degrees,
                                           out_d=self.out_degrees[u, 0])

        elif recommender == 2:
            top_k_edges = personalized_pagerank(network=self.G, sparse_adj=self.A,
                                                node_id=u, damping_factor=self.factor, k_for_recommendation=-1)
        elif recommender == 3:
            top_k_edges = twitter_wtf(network=self.G, sparse_adj=self.A,
                                      node_id=u, damping_factor=self.factor, k_for_circle_of_trust=100)
        elif recommender == 4:
            top_k_edges = biased_link_predictor(network=self.G, node=u, real_opinions=self.opinions,
                                                gamma=self.factor, d_eps=0.0001)
        elif recommender == 6:
            top_k_edges = [(s, v, p) for s, v, p, _ in fwp(G=self.G, s=u)]

        else:
            raise Exception("ops, it seems you have chosen an option which doesn't exist: {}".format(recommender))

        return top_k_edges



    def make_nodes_experiments(self, seed=None, k=None):
        '''Binomial experiments for each node considering its opinion
        https://www.dropbox.com/sh/lyk7sd6h8459obp/AADxMzMcEDy--K1Odq9rlM97a?dl=0&preview=Epistemic_networks_code.py
        '''
        self.set_seed(seed)
        evidence = np.random.binomial(self.exp, 0.5 + self.eps, self.number_nodes)
        if k is not None and self.with_misinformation in (1, 2):
            for u in self.G.nodes:
                evidence[u] += np.random.binomial(k, self.opinions[u], 1)
            self.prob_evidence_ratio = self.calculate_prob_ratio(evidence, self.exp+k)
        else:
            self.prob_evidence_ratio = self.calculate_prob_ratio(evidence, self.exp)

    def calculate_prob_ratio(self, evidence, n_experiments=None):
        '''defines vector of probability ratios for nodes
        '''
        if n_experiments is None:
            n_experiments = self.exp
        return np.power(np.divide(0.5-self.eps, 0.5+self.eps), 2*evidence-n_experiments)

    def set_nodes_susceptibility_vec(self):
        if self.suscept_distrib is None:
            nodes_susceptibility_vec = np.repeat(self.alpha, self.G.number_of_nodes())
        elif self.suscept_distrib == 0:
            self.max_suscept = 2 * self.alpha
            nodes_susceptibility_vec = np.random.uniform(0, self.max_suscept,
                                                         self.G.number_of_nodes())
        elif self.suscept_distrib == 1:
            self.suscept_exp = (1 / self.alpha) - 1
            nodes_susceptibility_vec = 1 - np.random.power(self.suscept_exp, self.G.number_of_nodes())
        else:
            raise ValueError(str(self.suscept_distrib) + ' susceptibility code not implemented!')
        self.nodes_susceptibility_vec = nodes_susceptibility_vec

    def set_rec_probs_transformation(self, seed=None, verbose=False):
        '''1. Samples probabilities of recommendation of the self.recommender sys
           2. Evaluates probs distribution
           3. Compute the quantile trasformation and saves it
           https://scikit-learn.org/stable/modules/preprocessing.html#non-linear-transformation
        '''
        #time0 = time.time()
        self.set_seed(seed)
        directory = './quantile_transformation_functions'
        if not os.path.exists(directory):
            os.makedirs(directory)
        filename = directory+'/quantile-scaler-'+ self.graph_name + str(self.G.number_of_nodes()) + '-' + str(self.recommender) +'.pkl'
        if os.path.isfile(filename):
            self.quantile_scaler = pickle.load(open(filename, "rb"))
        else:
            # 1.
            print('Producing quantile scaler..')
            sample_size = self.G.number_of_nodes()//4
            probs = []
            sampled_nodes = np.random.choice(list(self.G.nodes()), size=sample_size, replace=False)
            iterable = tqdm(sampled_nodes, desc='Sampling recommendation probabilities', leave=True) if verbose else sampled_nodes
            for u in iterable:
                top_k_edges = self.recommender_sys(u, self.recommender)
                for _u, _v, prob, in top_k_edges:
                    #if p > 0.: already inserted in the recommenders
                    probs.append(prob)
            # 2.
            try:
                data_train = np.array(probs).reshape((len(probs), 1))
            except:
                print('2')
            quantile = QuantileTransformer(output_distribution='uniform') # 'normal'
            self.quantile_scaler = quantile.fit(data_train)
            pickle.dump(quantile, open(filename, 'wb'))
            print('Quantile scaler saved')


    def settings(self, seed, verbose):
        '''sets probability recommendation transormation functions
        '''
        if verbose:
            print('Setting the model..')
        if self.recommender not in (None, 0):
            self.set_rec_probs_transformation(seed, verbose)
        self.verbose = verbose

        self.set_nodes_susceptibility_vec()

    @staticmethod
    def set_seed(seed):
        '''staticmethod for setting seeds
        '''
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

    def is_converged(self):
        '''Types of convergence:
            1. all nodes have opinion >.9 -> convergence=1 [they converged to the truth]
            2. all nodes have opinion <.5 -> convergence=0
            3. mixture of opinions  -> convergence=-777       [they did not converge]
        '''
        if all(self.opinions > .9):
            return 1
        if all(self.opinions < 0.5):
            return 0
        return -777

    def save(self, time):
        '''saves all instances
        '''
        if isinstance(self.all_opinions, list):
            self.all_opinions.append(self.opinions.copy())
        if isinstance(self.all_graphs, list):
            self.all_graphs.append(self.G.copy())
        if isinstance(self.all_metrics, list):
            metrics_t = self.compute_metrics()
            self.all_metrics.append(metrics_t)
        if isinstance(self.all_links, list):
            numb_new_added_links = self.G.number_of_edges() - np.cumsum(self.all_links)[time]
            self.all_links.append(numb_new_added_links)
        if self.debug:
            self.debug_dict['probabilities'].append('time'+str(time))
