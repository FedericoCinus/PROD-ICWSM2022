# PROD - People Recommenders on Opinion Dynamics

Reference code for "The Effect of People Recommenders on Echo Chambers and Polarization" accepted for publication@ICWSM'22.

The repository structure has three main components:

  1. the folder ```src``` where you can find the .py scripts that implement all the ingredients of the paper (e.g.: random network model, recommenders, etc.). 
  2. the folder ```Simulation_scripts``` contains a unique .py file that you need to invoke to run all the experiments.
  3. the jupyter notebook ```ANALYSIS.ipynb``` in the root folder reproduces all the visualizations shown in the paper. Run it only after you have run correctly all the needed experiments.

If your goal is to reproduce the paper experiments you already know everything. Follow these two steps (in strict chronological order):
  1. run ```python Simulation_scripts/simulation_script.py```
  2. run the jupyter notebook ```ANALYSIS.ipynb```


However PROD is an highly-customizable procedure with several reusable components...


