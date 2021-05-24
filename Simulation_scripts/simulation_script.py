#!/usr/bin/env python
# simulazione per grafi sintetici
import os
import numpy as np
import itertools
import networkx as nx
from tqdm import tqdm
import pandas as pd
import pickle
import multiprocessing
import os.path
from datetime import datetime
import mkl
mkl.set_num_threads(8) #setting numpy multithread

from v33_BCM import *
from v39_PROD_EPISTEMIC import *
from synthetic_generator import *
    
#####################################   PARAMETERS   ##############################################
CASCADES_MODEL = 'BCM'
SIMULATION_STAT = 1
OPINION_STAT = 500
GRAPH_STAT = 1
N_JOBS = 8
# Input 
NETWORKS = ['fortunato',]#['barabasi', 'erdos', 'complete', 'directed_k_out', 'directed_weighted_scale_free']
POPULATIONS = [400,]#[5**n for n in np.arange(2, 5)]                # works only for synthetic datasets
CENTRISMS = [1.,]
CONFORMISMS = [.2, .5, .8] # CONFORMISMS = [-777]
M_BARABASI = [10,]
P_REWIRING_ERDOS = [.6,]
fortunato_avg_deg = 12

# General Model
RECOMMENDERS = [1, 2] # {'UNIFORM':0, 'DJ': 1, 'fwp': 6} else None
INTERACTIONS = [2,]             # number of communications for each node in each step (node changes opinion)
MAX_TIMES =  [5000,] # dovrei guardare i tempi medi di convergenza e sommargli due volte la stdv
WITH_MISINFORMATIONS = [0,] # 0-> None, 1: also between no-vax, 2: also between no-vax and pro-vax
RANDOMNESS_PERC = [0., .25, .5, .75, 1.] #0.2, 0.4, 0.6, 0.8, 1.]
REWIRINGS = [.5,]#[.05, .15, .3, .45, .6,] #np.arange(0., .101, .01)
INNOVATORS_PERC = [0.5]
CLUSTERINGS = [.5,]#.05, .1, .3, .6] #value for mu parameter in Fortunato - higher mu -> lower clust



# Epistemic
EPSILONS = [.005,] #[.001, .005, .01]#probability of success in the experiments (0.5 + eps): greater eps-> easier to converge to 1
EXPERIMENTS0 = np.arange(100, 1000, 200)
EXPERIMENTS1 = [15,] #np.arange(1, 31, 10)
EXPERIMENTS2 = np.arange(1, 6, 1)

# BCM
MUS = [.2,]
DELTAS = [.2]
    
# simulation files
now = datetime.now() 
folder = './Simulations/v39-'+CASCADES_MODEL+'-'+NETWORKS[0]+'-rewiring-conformism-clust-'+str(OPINION_STAT)+'grid-pop'+str(POPULATIONS[0])+'-' + str(now)# <---------- CHANGE FOLDER NAME
os.mkdir(folder) 
filename = folder + '/v39-'+CASCADES_MODEL+'-'+NETWORKS[0]+'-rewiring-conformism-clust-'+str(OPINION_STAT)+'grid-pop'+str(POPULATIONS[0])+'.txt'

NONE = -777

#####################################   SIMULATION   ##############################################

def try_different_params(args, seed):
    GRAPH_SEED, OPINION_SEED, SIMULATION_SEED = seed # each time the initial configuration is different
    GRAPH_SEED = OPINION_SEED+1
    SIMULATION_SEED = OPINION_SEED+2
    g_name, population_size, randomness_perc, innovators_perc, centrism, conformism, clustering, m_barabasi,  p_rew_erdos, recommender, rewiring, s, max_time, mis, eps, n_exp, mu, delta, i, grid_dim = args
    #assert len(args) == 7, args
    np.random.seed(OPINION_SEED)
    if g_name == 'obamacare':
        G = nx.read_gpickle("./Data/twitter-obamacare/twitter_graph_with_opinions.gpickle")
        population_size = G.number_of_nodes()
        init_opinions = np.array([G.nodes[node]['opinion'] for node in range(population_size)])
    elif g_name == 'fortunato':
        G, init_opinions, _node2community = generate_G_and_opinions(population_size, centrism=centrism, mu=clustering, conformism=conformism, avg_deg=fortunato_avg_deg, distr="uniform", seed=GRAPH_SEED, verbose=False, innovators_perc=innovators_perc)
    elif g_name == 'barabasi':
        G = nx.barabasi_albert_graph(population_size, m_barabasi, seed=GRAPH_SEED)
        init_opinions = np.random.rand(population_size)
        g_name += '_m_'+str(M_BARABASI)+'-'
    elif g_name == 'complete':
        G = nx.complete_graph(population_size)
        init_opinions = np.random.rand(population_size)
    elif g_name == 'cycle':
        G = nx.cycle_graph(population_size) 
        init_opinions = np.random.rand(population_size)
    elif g_name == 'directed_k_out':
        G = nx.random_k_out_graph(population_size, 2, .5, seed=GRAPH_SEED)
        init_opinions = np.random.rand(population_size)
    elif g_name == 'erdos':
        G = nx.erdos_renyi_graph(population_size, p_rew_erdos, seed=GRAPH_SEED)
        init_opinions = np.random.rand(population_size)
        g_name += '_p_'+str(P_REWIRING_ERDOS)+'-'
    elif g_name == 'directed_weighted_scale_free':
        M = nx.scale_free_graph(population_size) # 
        # create weighted graph from M
        G = nx.DiGraph()
        for u,v,data in M.edges(data=True):
            w = data['weight'] if 'weight' in data else 1.0
            if G.has_edge(u,v):
                G[u][v]['weight'] += w
            else:
                G.add_edge(u, v, weight=w)
        init_opinions = np.random.rand(population_size)
    else:
        print('Possible g_name: obamacare')
    
    edges = G.number_of_edges()
    
    ARGS = {'init_opinions':init_opinions, 
            'G':G, 'graph_name':g_name, 
            'beta':randomness_perc,
            'rewiring':int(rewiring*edges), 
            "max_time":max_time,
            'recommender':recommender, 's':s, 
            'with_misinformation':mis,
            'exp':n_exp, 'eps':eps,
            'bcm_eps':delta, 'bcm_mu':mu,
            'save_all_opinions':False,
            'save_all_graphs':False,
            'save_all_links':False,
            'debug':False}
    prod = BCM(**ARGS) if CASCADES_MODEL == 'BCM' else PROD(**ARGS)
    

    # simulation
    time, converged, opinions, tot_rewiring, metrics_i, metrics_f, all_opinions, all_graphs, all_metrics, all_links, count_type_interactions, debug = prod.simulate(seed=SIMULATION_SEED, verbose=False)
    if SIMULATION_STAT == 1 and OPINION_STAT == 1:
        print(str(i), '/', str(grid_dim), ' grid-point finished.')
        
        
    ###### results stat
    res = g_name+';'+str(population_size)+";"+str(G.number_of_edges())+";"+str(randomness_perc)+";"+str(innovators_perc)+';'+str(centrism)+';'+str(conformism)+';'+str(clustering)+';'+str(m_barabasi)+';'+str(p_rew_erdos)+';'+str(recommender)+';'+str(rewiring)+';'+str(s)+';'+str(max_time)+';'+str(mis)+';'+str(eps)+';'+str(n_exp)+';'+str(mu)+';'+str(delta)+';'+str(time)+';'+str(converged)+';'+str(tot_rewiring)+';'+str(SIMULATION_SEED)+';'+str(OPINION_SEED)+';'+str(GRAPH_SEED)
    
    for (k_i, v_i), (k_f, v_f) in zip(metrics_i.items(), metrics_f.items()):
        assert k_i == k_f, print(k_i, k_f)
        res += ';'+str(v_i)+';'+str(v_f)
    for type_interaction in ['pro_pro', 'pro_no', 'no_no', 'no_pro', 'other']:
        res += ';'+str(count_type_interactions[type_interaction])
    res += '\n'
    with open(folder+'/results-stats.csv', 'a') as f:
        f.write(res)

 
    
def try_different_seeds(seeds):
    
    now = datetime.now()
    with open(filename, 'w') as f:
        f.write('simulation started at: ' + str(now))
    for args in all_args:
        try_different_params(args, seeds)
    print('Seed', seeds, 'finished')
    now = datetime.now()
    with open(filename, 'a') as f:
        f.write('\nsimulation finished at: ' + str(now))
            
############################################################################################################
if __name__ == '__main__':                
    all_args = []
    i = 0
    m_barabasi = M_BARABASI[0]
    p_rew_erdos = P_REWIRING_ERDOS[0]
    max_time = MAX_TIMES[0]
    print(f"main: {max_time}")
    mis = WITH_MISINFORMATIONS[0]
    for g_name in NETWORKS:
        for population_size in POPULATIONS:
            for randomness_perc in RANDOMNESS_PERC:
                for innovators_perc in INNOVATORS_PERC:
                    for centrism in CENTRISMS:
                        for conformism in CONFORMISMS:
                            if g_name != 'fortunato':
                                centrism = NONE
                                conformism = NONE
                            if g_name != 'barabasi':
                                m_barabasi = NONE
                            if g_name != 'erdos':
                                p_rew_erdos = NONE
                            for clustering in CLUSTERINGS:
                                for recommender in RECOMMENDERS:
                                    for rewiring in REWIRINGS:
                                        for s in INTERACTIONS:
                                            if CASCADES_MODEL == 'BCM':
                                                for mu in MUS:
                                                    for delta in DELTAS:
                                                        i += 1
                                                        eps = NONE
                                                        n_exp = NONE
                                                        all_args.append([g_name, population_size, randomness_perc, innovators_perc, centrism, conformism, clustering, m_barabasi,  p_rew_erdos, recommender, rewiring, s, max_time, mis, eps, n_exp, mu, delta, i])
                                            else:
                                                for eps in EPSILONS:
                                                    if eps == .001:
                                                        iterable_experiments = EXPERIMENTS0
                                                    elif eps == .005:
                                                        iterable_experiments = EXPERIMENTS1
                                                    elif eps == .01:
                                                        iterable_experiments = EXPERIMENTS2
                                                    else:
                                                        print('Eps', eps, ' to exp list not defined!')
                                                    for n_exp in iterable_experiments:
                                                        i += 1
                                                        mu = NONE
                                                        delta = NONE
                                                        all_args.append([g_name, population_size, randomness_perc, innovators_perc, centrism, conformism, clustering, m_barabasi,  p_rew_erdos, recommender, rewiring, s, max_time, mis, eps, n_exp, mu, delta, i])
    # params grid indexes                                                                                  
    grid_dim =len(all_args)
    for i in range(grid_dim):
        all_args[i].append(grid_dim)
    # list of simulation seeds triplets
    SEEDS = []
    for sim_seed in range(SIMULATION_STAT):
        for op_seed in range(OPINION_STAT):
            for graph_seed in range(GRAPH_STAT):
                SEEDS.append((sim_seed, op_seed, graph_seed))
    pool = multiprocessing.Pool(N_JOBS)
    with open(folder+'/results-stats.csv', 'w') as f:
        f.write('g_name;population_size;number_of_edges;beta;innovators_perc;centrism;conformism;clustering;m_barabasi;p_rew_erdos;recommender;rewiring;s;max_time;misinformation;epsilon;n_experiments;mu;delta;time;converged;tot_rewiring;simulation_seed;opinion_seed;graph_seed;avg_op0;avg_opf;ndi0;ndif;stdv0;stdvf;clust0;clustf;BC0;BCf;neigh_corr0;neigh_corrf;num_peaks0;num_peaksf;rwc0;rwcf;strongly_cc0;strongly_ccf;propro;prono;nono;nopro;other\n')
    try:
        pool.map(try_different_seeds, SEEDS)
    except Exception as exc:   
        print("EXCEPTION", exc)
    pool.close()
    pool.join()