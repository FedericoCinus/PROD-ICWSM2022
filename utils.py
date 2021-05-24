from scipy.stats import binom_test
import numpy as np
from itertools import cycle, product

def binomial_test_p_value(data, attribute):
    n = data.opinion_seed.nunique()
    convergence_prob_with_no_recommender = data[data.recommender=="None"].mean()[attribute]
    convergence_by_recommender_and_rewiring = data.groupby(["recommender", "rewiring"]).sum()[attribute]
    p_val_df = convergence_by_recommender_and_rewiring.apply(lambda x: binom_test(x, n=n, p=convergence_prob_with_no_recommender, alternative='less'))
    return p_val_df

def select_p_value_of_recommender(p_value_df, recommender):
    return p_value_df.xs(recommender, level="recommender")

def bootstrap(empirical_distro, B=10000, percentile = 5):
    predicted_boostrapped_values = []
    value = np.mean(empirical_distro)
    for _ in range(B):
        bootstrap_sample = np.random.choice(empirical_distro, replace=True, size=len(empirical_distro))
        predicted_boostrapped_values.append(value-np.mean(bootstrap_sample))
    return np.percentile(predicted_boostrapped_values, q=percentile), np.percentile(predicted_boostrapped_values, q=100-percentile)

def confidence_intervals_for_recommender(data, recommender, attr, B=10000, percentile=5):
    df_for_recommender = data[data.recommender==recommender]
    low_ci_for_each_rewiring = []
    up_ci_for_each_rewiring = []
    for r in df_for_recommender.rewiring.unique():
        df = df_for_recommender[df_for_recommender.rewiring==r]
        bootstrap_ci = bootstrap(df[attr].values, B, percentile)
        low_ci_for_each_rewiring.append(bootstrap_ci[0])
        up_ci_for_each_rewiring.append(bootstrap_ci[1])
    return low_ci_for_each_rewiring, up_ci_for_each_rewiring
        
def filter_data_by_seed(data, attrs):
    
    seeds = []
    
    attr_values = [data[attr].unique() for attr in attrs]
    all_grid_values = list(product(*attr_values))
    
    for i in range(len(all_grid_values)):
        filtered_df = data
        for j in range(len(attrs)):
            filtered_df = filtered_df[ (filtered_df[attrs[j]]==all_grid_values[i][j]) ]
        seeds.append(filtered_df.opinion_seed.unique())
        
    min_length = min(map(lambda x:len(x), seeds))
    seeds_list = [seeds_list for seeds_list in seeds if len(seeds_list)==min_length]
    return data[data.opinion_seed.isin(seeds_list[0])]

def unbalancing_in_opinion_spectrum(opinions, th=0.5):
    num_greater_than_th = (opinions >= th).sum()
    num_less_than_th = (opinions < th).sum()
    
    return num_less_than_th/len(opinions)
