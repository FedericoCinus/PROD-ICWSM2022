#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UPDATES : damping factor selection
"""
import numpy as np
import pandas as pd
import os
import heapq
import networkx as nx
from abc import ABC, abstractmethod
from fast_pagerank import pagerank_power

np.seterr(all='raise')

######################################################### Recommender
def calculate_DJ_row(A, u, in_degrees, out_d):
    '''Numpy matrix implementation of the Directed Jaccard recommender
    '''
    # row u
    EPS = 10E-10
    AAt = np.multiply(A[u, :] * A, np.logical_not(A[u, :]))
    out_d = A[u, :].sum()
    den = (- AAt + out_d + in_degrees)
    DJ_mtx_new_row = np.true_divide(AAt, den + EPS)
    tuples_recommended = [(u, v, p) for v, p in enumerate(np.asarray(DJ_mtx_new_row).reshape(-1)) if p > 0.]
    return sorted(tuples_recommended, reverse=True, key=lambda x: x[2])


def fwp(G, s, alpha=.85, rmax=0.000005):
    '''Implementation of the FowardPush algorithm
       http://www.vldb.org/pvldb/vol13/p15-shi.pdf
    '''
    EPSILON = 10E-6
    r_s, p0_s = (np.zeros(G.number_of_nodes()), np.zeros(G.number_of_nodes()))
    r_s[s] = 1
    frontier = set([s])
    time = 0
    while frontier:
        frontier_tmp = set()
        for v in frontier:
            for u in G.successors(v):
                r_s[u] += (1-alpha)*r_s[v]/(G.out_degree(v)+EPSILON)
                if r_s[u]/(G.out_degree(u)+EPSILON) > rmax:
                    frontier_tmp.add(u)

            p0_s[v] += alpha*r_s[v]
            r_s[v] = 0
        frontier = frontier_tmp
        time += 1
    tuples_recommended = [(s, v, p, np.random.rand()) for v, p in enumerate(p0_s) if (not G.has_edge(s, v)) and p > 0.]
    # "1:" removes self-loop rec, "x[3]" is a random numb to shuffle reccommendations with same prob
    return sorted(tuples_recommended[1:], reverse=True, key=lambda x: (x[2], x[3]))


def personalized_pagerank(network, sparse_adj, node_id, damping_factor=.85, k_for_recommendation=-1):
    '''PPR method
    '''
    #compute personalized pagerank
    personalize = np.zeros(shape=network.number_of_nodes())
    personalize[node_id] = 1
    scores = pagerank_power(sparse_adj, p=damping_factor, personalize=personalize, tol=1e-6)
    #compute neighborhood that can't be recommended of course since the edge already exists
    neighbors_to_not_recommend = nx.neighbors(network, node_id)
    if k_for_recommendation == -1:
        k_for_recommendation = 0 #Take all the nodes!

    neighbors_to_not_recommend = set(neighbors_to_not_recommend)
    results = []
    for node in scores.argsort()[::-1]:
        if node not in neighbors_to_not_recommend and node != node_id:
            results.append(((node_id, node, scores[node])))
            if len(results) == k_for_recommendation:
                break
    return results


def _add_random_noise(percentage_of_noise, lp_scores, mean, std):
    """
    This method takes as input the results coming from a link prediction algorithm and apply some random noise.
    It applies a convex transformation adding a portion of noise to the scores.
    Noise is produced using draws extracted from a normal distribution with mean and
    standard deviation of the scores data.
    """
    #simulate the noise to add
    noise = np.random.normal(loc=mean, scale=std, size=lp_scores.shape[0])
    #update the scores
    lp_scores[:, 2] = (1 - percentage_of_noise)*lp_scores[:, 2] + percentage_of_noise*noise
    #return the new scores
    return lp_scores



def twitter_wtf(network, sparse_adj, node_id, k_for_circle_of_trust=20, tol=1e-8,
                damping_factor=.85, k_for_recommendation=-1):
    """This method aims to realize a link prediction algorithm used by Twitter to perform
        the WTF recommendation on the platform.
        The algorithm can be seen at 'https://web.stanford.edu/~rezab/papers/wtf_overview.pdf'.

        The algorithm consists of two phases:
            1) Compute the circle of trust for the user you want to recommend(top-k nodes in PPR)
            2) Compute the top-k nodes using score propagation
    """
    k_for_circle_of_trust = int(network.number_of_nodes()*.1)
    #1st phase: Compute circle of trust of user according to Personalized PageRank
    personalize = np.zeros(shape=network.number_of_nodes())
    personalize[node_id] = 1
    values_of_personalized_pr = pagerank_power(sparse_adj, p=damping_factor, personalize=personalize, tol=1e-6)
    circle_of_trust = values_of_personalized_pr.argsort()[-k_for_circle_of_trust:][::-1]

    #2nd phase: init bipartite graph
    bipartite_graph = nx.DiGraph()
    #add nodes belonging to the circle of trust as hubs(H)
    for node in circle_of_trust:
        #these nodes are "hubs"(H) in the bipartite graph
        bipartite_graph.add_node(str(node)+"H")
    #add out neighbors of nodes belonging to the circle of trust as authorities(A)
    for node in circle_of_trust:
        for out_neighbor in network.neighbors(node):
            #direction is inverted for a matter of simplicity in the sequent phases
            bipartite_graph.add_edge(str(out_neighbor)+"A", str(node)+"H")

    #retrieve adjacency matrix of bipartite graph
    A = nx.to_numpy_array(bipartite_graph)

    #retrieve list of all nodes splitted by authority or hub
    all_nodes = list(bipartite_graph.nodes())
    hub_nodes = [int(x[:-1]) for x in all_nodes if 'H' in x]
    authority_nodes = [int(x[:-1]) for x in all_nodes if 'A' in x]

    #3rd phase: start building ingredients of our SALSA algorithm
    #these are the transition matrices determined by the bipartite graph
    S_prime = A[len(hub_nodes):, :][:, :len(hub_nodes)].copy()
    R_prime = S_prime.T.copy()
    #normalize both matrices
    denominator_S_prime = S_prime.sum(axis=0)
    denominator_S_prime[denominator_S_prime == 0] = 1
    S_prime = S_prime / denominator_S_prime
    denominator_R_prime = R_prime.sum(axis=0)
    denominator_R_prime[denominator_R_prime == 0] = 1
    R_prime = R_prime / denominator_R_prime
    #these are the vectors which contain the score of similarity
    #and relevance
    s = np.zeros(shape=(len(hub_nodes), 1), dtype=np.float)
    r = np.zeros(shape=(len(authority_nodes), 1), dtype=np.float)

    #at the beginning of the procedure we put the similarity
    #of the user we want to give the recommendation equal to 1
    index_of_node_to_recommend = np.where(circle_of_trust == node_id)[0][0]
    s[index_of_node_to_recommend] = 1.

    #init damping vector
    alpha = 1 - damping_factor
    alpha_vector = np.zeros(shape=(len(hub_nodes), 1), dtype=np.float)
    alpha_vector[index_of_node_to_recommend] = alpha

    #4th phase: run the algorithm
    convergence = False
    while not convergence:
        s_ = s.copy()
        r_ = r.copy()
        r_ = S_prime.dot(s)
        s_ = alpha_vector + (1 - alpha)*(R_prime.dot(r))
        #compute difference and check if convergence has been reached
        diff = abs(s_ - s)
        if np.linalg.norm(diff) < tol:
            convergence=True
        #update real vectors
        s = s_
        r = r_

    #5th phase: order by score and delete neighbors of node to be recommended
    #of course we don't want to recommend people that the user already follow
    neighbors_to_not_recommend = nx.neighbors(network, node_id)
    relevance_scores = r.flatten()
    if k_for_recommendation == -1:
        k_for_recommendation = 0 #Take all the nodes!

    neighbors_to_not_recommend = set(neighbors_to_not_recommend)
    results = []
    for node in relevance_scores.argsort()[::-1]:
        if node not in neighbors_to_not_recommend and node != node_id:
            results.append(((node_id, node, relevance_scores[node])))
            if len(results) == k_for_recommendation:
                break
    return results


def directed_jaccard_top_k_edges(A, node, non_neighbors_set, top_k=-1, percentage_of_noise=0.):
    """This method aims to predict the @top-k edges to recommend to @node according
        to the jaccard coefficient for directed graphs.

        A: numpy array, it is the adjacency matrix of the graph;
        node: int, it is the node ID that will get the recommendations;
        top_k: int, it is the number of edges to recommend to the node.

    """
    #now you have to take the top-k edges predicted for this node
    #compute Jaccard Coefficient for all non edges score for all these edges
    non_neighbors_lp_scores = list()
    #compute degrees of each node
    out_degrees = A.sum(axis=1)
    in_degrees = A.sum(axis=0)
    #for each non neighbor compute the Jaccard coefficient
    for non_neighbor_node in non_neighbors_set:
        jaccard_numerator = A[node, :].dot(A.T[non_neighbor_node, :])
        jaccard_denominator = (out_degrees[node]+in_degrees[non_neighbor_node] - jaccard_numerator)
        if jaccard_denominator == 0:
            jaccard_score = 0
        else:
            jaccard_score = jaccard_numerator/jaccard_denominator
        non_neighbors_lp_scores.append((node, non_neighbor_node, jaccard_score))
    if percentage_of_noise <= 0.:
        if top_k != -1:
            #take the top-k in n*log(n)
            return heapq.nlargest(top_k, non_neighbors_lp_scores, key=lambda x: x[2])
        return non_neighbors_lp_scores
    scores = np.array(non_neighbors_lp_scores)
    scores = _add_random_noise(percentage_of_noise=percentage_of_noise,
                               lp_scores=scores, mean=scores[:, 2].mean(),
                               std=scores[:, 2].std())
    if top_k == -1:
        return [(node, scores[index, 1], scores[index, 2]) for index in scores[:, 2].argsort()[::-1]]
    return [(node, scores[index, 1], scores[index, 2]) for index in scores[:, 2].argsort()[-top_k:]]


def biased_link_predictor(network, node, real_opinions, gamma=1.6, d_eps=0.0001, top_k=-1, percentage_of_noise=0.):
    """This link predictor aims to realize the link prediction algorithm suggested by
        Sirbu et al, 2019.
    """
    #this time you have chosen a random link predictor proportional to the opinion similarity
    #we are going to implement a link predictor similar to the ones shown in Sirbu et al. 2019
    d_eps_gamma = d_eps**(-gamma)
    #this array will contain the info about the node that is joined with the corresponding edge
    node_to_join = np.full(shape=network.number_of_nodes(), fill_value=-1, dtype=np.int32)
    score_to_join = np.full(shape=network.number_of_nodes(), fill_value=-1.)
    counter = 0
    for node_id in nx.non_neighbors(network, node):
        #compute the opinion distance
        dis = abs(real_opinions[node] - real_opinions[node_id])
        #if it is below the threshold update it with the threshold
        if dis < d_eps:
            dis = d_eps_gamma
        #otherwise compute the exponential
        else:
            dis = dis**(-gamma)
        #update the score
        score_to_join[counter] = dis
        #update node
        node_to_join[counter] = node_id
        #update counter
        counter += 1
    #restrict array to the index where real values are stored
    node_to_join = node_to_join[:counter]
    score_to_join = score_to_join[:counter]
    score_to_join = score_to_join/score_to_join.sum()
    if percentage_of_noise <= 0.:
        if top_k == -1:
            return [(node, node_to_join[index], score_to_join[index]) for index in score_to_join.argsort()[::-1]]
        return [(node, node_to_join[index], score_to_join[index]) for index in score_to_join.argsort()[-top_k:]]
    scores = np.array([(node, node_to_join[index], score_to_join[index]) for index in range(score_to_join.shape[0])])
    scores = _add_random_noise(percentage_of_noise, scores, scores[:, 2].mean(), scores[:, 2].std())
    if top_k == -1:
        return [(node, scores[index, 1], scores[index, 2]) for index in scores[:, 2].argsort()[::-1]]
    return [(node, scores[index, 1], scores[index, 2]) for index in scores[:, 2].argsort()[-top_k:]]


def random_link_predictor(network, node, top_k=-1):
    """This link predictor aims to realize a completely random link predictor
    """
    all_non_neighbors = [x for x in nx.non_neighbors(network, node)]
    if top_k == -1:
        equal_prob = 1/len(all_non_neighbors)
        return [(node, node_id, equal_prob) for node_id in all_non_neighbors]

    return [(node, node_id, 1/top_k) for node_id in np.random.choice(all_non_neighbors, size=top_k, replace=False)]


def adamic_adar_top_k_edges(A, node, non_neighbors_set, top_k=-1):
    """This method aims to predict the @top-k edges to recommend to @node according
        to the adamic adar coefficient for undirected graphs.

        A: numpy array, it is the adjacency matrix of the graph;
        node: int, it is the node ID that will get the recommendations;
        top_k: int, it is the number of edges to recommend to the node.
    """
    #compute adamic adar coefficient for all non edges
    non_neighbors_lp_scores = list()
    #compute the degrees of each node
    degrees = A.sum(axis=1)
    #eye = np.eye(A.shape[0])
    denominator = np.log10(degrees+1)
    for non_neighbor_node in non_neighbors_set:
        aa_coefficient = (A[node, :] * A[non_neighbor_node, :])/denominator
        non_neighbors_lp_scores.append((node, non_neighbor_node, aa_coefficient.sum()))
    if top_k != -1:
        #take the top-k in n*log(n)
        return heapq.nlargest(top_k, non_neighbors_lp_scores, key=lambda x: x[2])
    return non_neighbors_lp_scores


def directed_adamic_adar_top_k_edges(A, node, top_k, non_neighbors_set):
    """This method aims to predict the @top-k edges to recommend to @node according
        to the adamic adar coefficient for directed graphs.

        A: numpy array, it is the adjacency matrix of the graph;
        node: int, it is the node ID that will get the recommendations;
        top_k: int, it is the number of edges to recommend to the node.
    """
    #compute adamic adar coefficient for all non edges
    non_neighbors_lp_scores = list()
    #compute degrees of each node
    out_degrees = A.sum(axis=1)
    in_degrees = A.sum(axis=0)
    for non_neighbor_node in non_neighbors_set:
        aa_coefficient = 0
        for i in range(A.shape[0]):
            #if node i is neighbor of both node and non_neighbor_node then...
            if A[node][i]*A.T[non_neighbor_node][i] != 0:
                #compute aa numerator
                aa_numerator = A[node, :].dot(A.T[non_neighbor_node, :])
                #update aa score
                aa_coefficient += aa_numerator/((np.log10(out_degrees[node]) + np.log10(in_degrees[non_neighbor_node]))+1)
        non_neighbors_lp_scores.append((node, non_neighbor_node, aa_coefficient))
    #take the top-k in n*log(n)
    return heapq.nlargest(top_k, non_neighbors_lp_scores, key=lambda x: x[2])
