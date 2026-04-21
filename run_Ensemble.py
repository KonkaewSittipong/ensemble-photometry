import sys
import os
import importlib

module_path = '/lustre/MSSP/sittipong/buildmodule/ensemble'

# Prioritize your module path
if module_path not in sys.path:
    sys.path.insert(0, module_path)

os.chdir(module_path)

# Clear the cached module from Jupyter memory
if 'Ensemble' in sys.modules:
    del sys.modules['Ensemble']

# Import the module file itself, force a reload, then import the class
import Ensemble
importlib.reload(Ensemble)
from Ensemble import Ensemble

print("Module and class reloaded successfully.")


pi = Ensemble(config=None,  diagnostics = True)
pi.run(logfile='/lustre/MSSP/sittipong/reduce/2023-11-07/g/all.log', 
       ignor_stars = None,
       target_rms=0.02, 
       numstars=12,)
pi.plot_all_comparison_lr(save_folder='.')