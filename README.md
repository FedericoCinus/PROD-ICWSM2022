# PROD - People Recommenders on Opinion Dynamics

Reference code for "The Effect of People Recommenders on Echo Chambers and Polarization" accepted for publication@ICWSM'22.

The repository structure has three main components:

  1. the folder ```src``` where you can find the .py scripts that implement all the ingredients of the paper (e.g.: random network model, recommenders, etc.). 
  2. the folder ```Simulation_scripts``` contains a unique .py file that you need to invoke to run all the experiments.
  3. the folder ```notebooks``` where you can find the .ipynb notebooks to reproduce the visualizations shown in the paper.

If your goal is to reproduce the paper experiments you already know everything. Follow these two steps:
  1. run ```python Simulation_scripts/run_simulation.py```
  2. run the jupyter notebook ```notebooks/Network-visualizations.ipynb```


However PROD is an highly-customizable procedure with several reusable components:
  - At ```src/recommenders.py``` you can find the implementation of the recommenders.
    - ```twitter_wtf()``` is the implementation of what in the paper is called _SALSA_.
    - ```personalized_pagerank()``` is the implementation of _Personalized PageRank_ algorithm.
    - ```calculate_DJ_row()``` is the implementation of _Directed Jaccard_ heuristic.
    - ```biased_link_predictor()``` is the implementation of what in the paper is called _OBA_.
    - All the other recommenders are either unefficient versions of previous algorithms or other algorithms similar to the ones used (i.e.: Adamic-Adar with Directed Jaccard).
 
  - At ```src/synthetic_generator.py``` you can find the implementation of the random network model.
    -  the main function is ```generate_G_and_opinions()``` which returns the networkX graph, the opinions vector and the community assignments. If you want to reuse this module, we strongly warn that the LFR benchmark (the base algorithm to generate graphs with communities) is highly unstable, so stick with our values for the parameters ```avg_deg``` and ```power_law_coef```. The parameters ```mu``` and ```conformism``` tune the amount of modularity and initial homophily.

  -  At ```src/measures.py``` you can find the implementation of the echo chambers/polarization measures adopted in the paper.
    - The functions ```compute_cont_correlation_neighbors()``` and ```compute_RWC()``` allows to compute the _NCI_ and the _RWC_ measures.
    - We left other implemented measures for those interested.
    
  - At ```src/utils.py``` you can find auxiliary functions.
    - You can have a look at the ```binomial_test_p_value()``` implementation to check how we have assessed the statistical significance of the outcomes.
 
  - At ```src/PROD_EPISTEMIC.py``` you can find the main class that allows to run our _PROD_ procedure under the Epistemological opinion update rule.
    - in the ```compute_metrics()``` you can decide which metric you want to measure at the end of each round. **Here, you could add your own metric. Implement your function in measures.py, add it to the list and you are done!**
    - in the ```recommender_sys()``` you can find how the various implemented algorithms are chosen in a _PROD_ run. **Here, you could add your own recommender algorithm. Implements your function in recommenders.py, then add a specific if statement and you are done!**
    - the main function is ```simulate()``` that you can use to execute one single _PROD_ run and allows you to input a random seed to assess the robustness of your results.
 
  - At ```src/BCM.py``` you can find the extension of PROD with the bounded confidence opinion update rule. 
    - **You can clone this script to use your own opinion update rule**. It is quite simple, (i) extend the class contained in ```src/PROD_EPISTEMIC.py``` and (ii) override the method ```update_node_opinion()```.


