import peakutils
import numpy as np
import networkx as nx
from tqdm import tqdm
from scipy.stats import entropy, kurtosis, skew, pearsonr, kde



def compute_std_dev(G, x):
    return np.std(x)

def compute_avg_op(x, G):
    return np.mean(x)

def compute_entropy_opinions(x, G):
    return entropy(x, base=2)

def compute_BC(x, G):
    n = len(x)
    return (skew(x)**2 + 1) / (kurtosis(x) + (3*(n-1)**2)/((n-2)*(n-3)))


def compute_echo_chamber_index(x, G, delta=0.5, num_samples=25000, max_path_length=10, verbose=False):
    results = []
    iterable = tqdm(range(num_samples), desc="Computing ECI") if verbose else range(num_samples)
    for _ in iterable:
        v0 = np.random.randint(0, G.number_of_nodes())
        x0 = x[v0]
        v = v0
        sum_diff = 0
        path_length = 0
        neighbors = True
        while path_length < max_path_length and neighbors:
            path_length += 1
            neighbors = list(G.neighbors(v))
            if neighbors:
                v = np.random.choice(neighbors)
                sum_diff += (abs(x0 - x[v]) < delta)

            results.append(sum_diff / path_length)
    return np.mean(results)


def compute_network_disagreement(x, G=None, A=None):
    if G is not None:
        return np.mean(np.abs(np.subtract(*x[np.array(G.edges)].T)))
    return np.mean(np.abs(np.subtract(*x[np.array(list(zip(*np.where(A >= 1))))].T)))




def compute_clustering_coeff(x, G):
    return nx.average_clustering(G)

def compute_opinion_modularity(opinions, G):
    if not nx.is_directed(G):
        G_internal = G.to_directed().copy()
    else:
        G_internal = G.copy()

    Q = 0
    numb_edges = G_internal.number_of_edges()
    for u, v in G_internal.edges():
        delta = 1-np.abs(opinions[u] - opinions[v])
        Q += (1/numb_edges - G_internal.in_degree(v)*G_internal.out_degree(u)/(numb_edges**2))*delta
    return Q

def compute_cont_correlation_neighbors(opinions, G):
    avg_neigh_opinions = [np.mean([opinions[v] for v in G.neighbors(u)]) for u in G.nodes()]
    return pearsonr(avg_neigh_opinions, opinions)[0]


def compute_discr_correlation_funct(opinions, G):
    curr_opinions = [1 if x>.5 else -1 for x in opinions]
    opinions0, opinions1 = ([],[])
    for u, v in G.edges():
        opinions0.append(curr_opinions[u])
        opinions1.append(curr_opinions[v])
    cross_product = np.array(opinions0)*np.array(opinions1)
    same_product = np.array(opinions0)**2
    return np.mean(cross_product) - np.mean(same_product)


def compute_num_opinion_peaks(opinions, _G, only_num=True):
    nparam_density = kde.gaussian_kde(opinions)
    x = np.linspace(0, 1, 100)
    density = nparam_density(x)
    indexes = peakutils.indexes(density, thres=0, min_dist=10)
    if not only_num:
        return x, density, indexes
    return len(indexes)

def compute_numb_weakly_cc(_opinions, G):
    return len(list(nx.weakly_connected_components(G)))

def compute_numb_strongly_cc(_opinions, G):
    return len(list(nx.strongly_connected_components(G)))


def rw(G, u, comm_assgn, steps):
    curr = u
    for _ in range(steps):
        curr = np.random.choice(list(G.successors(curr)))
    return comm_assgn[curr]

def modify_graph(G, percentile=95):
    inner_graph = G.copy()
    degree_max = np.percentile(list(dict(inner_graph.degree).values()), percentile)
    hubs = {node for node, degree in inner_graph.degree if degree > degree_max}
    for node in hubs:
        out_edges = list(inner_graph.out_edges(node)).copy()
        inner_graph.remove_edges_from(out_edges)
    return inner_graph

def compute_RWC(init_opinions1, G):
    G1 = modify_graph(G)

    X = np.where(init_opinions1<.5)[0]
    Y = np.where(init_opinions1>=.5)[0]
    if len(X) == 0 or len(Y) == 0:
        return None

    nstart_X = {x:(1 if x in X else 0) for x in G1.nodes()}
    pr_X = nx.pagerank(G1, alpha=0.85, nstart=nstart_X, personalization=nstart_X)
    nstart_Y = {x:(1 if x in Y else 0) for x in G1.nodes()}
    pr_Y = nx.pagerank(G1, alpha=0.85, nstart=nstart_Y, personalization=nstart_Y)

    P_XX = sum([pr_X[u] for u in X])
    P_YY = sum([pr_Y[u] for u in Y])
    P_XY = sum([pr_X[u] for u in Y])
    P_YX = sum([pr_Y[u] for u in X])

    RWC = P_XX*P_YY - P_YX*P_XY
    return RWC
