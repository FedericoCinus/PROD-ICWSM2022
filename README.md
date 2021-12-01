# PROD - People Recommenders on Opinion Dynamics

[![Python 3.8.3](https://img.shields.io/badge/python-3.8.3-blue.svg)](https://www.python.org/downloads/release/python-383/)

Reference code for "The Effect of People Recommenders on Echo Chambers and Polarization" accepted for publication @ICWSM'22.

The repository structure has three main components:

  1. the folder `src` where you can find the .py scripts that implement all the ingredients of the paper (e.g.: random network model, recommenders, etc.). 
  2. The folder `Simulation_scripts` contains a unique .py file that you need to invoke to run all the experiments.
  3. The folder `notebooks` where you can find the .ipynb notebooks to reproduce the visualizations shown in the paper.

### Reproducibility

If your goal is to reproduce the paper experiments, just follow these steps.

  1. Install the conda environment: `conda env create --file environment.yml` (issues with this step? go at the end of the page)
  2. Run `cd Simulation_scripts && python run_simulation.py`.
  3. Run the Jupyter notebook `notebooks/Network-visualizations.ipynb`.

### Re-usable components

However, PROD is a highly-customizable procedure with several reusable components:
  - At `src/recommenders.py` you can find the implementation of the people-recommender algorithms.
    - `twitter_wtf()` is the implementation of what in the paper is called _SALSA_.
    - `personalized_pagerank()` is the implementation of _Personalized PageRank_ algorithm.
    - `calculate_DJ_row()` is the implementation of _Directed Jaccard_ heuristic.
    - `biased_link_predictor()` is the implementation of what in the paper is called _OBA_.
    - All the other recommenders are either inefficient versions of previous algorithms (used for debug purposes) or other algorithms similar to the ones used (e.g.: Adamic-Adar with Directed Jaccard).
 
  - At `src/synthetic_generator.py` you can find the implementation of the random network model.
    -  the main function is `generate_G_and_opinions()` which returns the networkX graph, the opinions vector, and the community assignments. If you want to reuse this module, we strongly warn that the LFR benchmark (the base algorithm to generate graphs with communities) is highly unstable, so stick with our values for the parameters `avg_deg` and `power_law_coef`. The parameters `mu` and `conformism` tune the amount of modularity and initial homophily.
 
  -  At `src/measures.py` you can find the implementation of the echo chambers/polarization measures adopted in the paper.
      -  The functions `compute_cont_correlation_neighbors()` and `compute_RWC()` allows to compute the _NCI_ and the _RWC_ measures.
      - We left other implemented measures for those interested.
    
  - At `src/utils.py` you can find auxiliary functions.
    - You can have a look at the `binomial_test_p_value()` method to check how we have assessed the statistical significance of the outcomes.
 
  - At `src/PROD_EPISTEMIC.py` you can find the main class that allows running our _PROD_ procedure under the Epistemological opinion update rule.
    - in `compute_metrics()` you can decide which metric you want to measure at the end of each round. **Here, you could add your metric. Implement your function in measures.py, add it to the list, and you are done!**
    - in `recommender_sys()` you can find how the various implemented algorithms are chosen in a _PROD_ run. **Here, you could add your recommender algorithm. Implement your function in recommenders.py, then add a specific `if` statement, and you are done!**
    - the main function is `simulate()` that you can use to execute one single _PROD_ run and allows you to input a random seed to assess the robustness of your results.
 
  - At `src/BCM.py` you can find the extension of PROD with the bounded confidence opinion update rule. 
    - **You can clone this module to realize your own opinion update rule**. It is quite simple, (i) extend the class contained in `src/PROD_EPISTEMIC.py` and (ii) override the method `update_node_opinion()` as done for _BCM_.


### Issues with conda environment

We have noticed that, in particular cases, the conda environment seems unresolvably uninstallable. Despite encouraging you to upgrade your system to the latest version - especially `conda` and `python` - we have attached an environment version without build specifications that should run smoothly also in more extreme cases. Install the required environment using the command `CONDA_RESTORE_FREE_CHANNEL=1 conda env create -f environment_alt.yml`.
