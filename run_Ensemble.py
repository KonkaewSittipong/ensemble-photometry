import subprocess
import os

df = [ # logfile, save_path, target_id, ref_star_id, catalog_mag
    ['/lustre/MSSP/sittipong/reduce/2023-11-06/g/all.log', ' /lustre/MSSP/sittipong/reduce/2023-11-06/g/', 16, 17, 17.41 ],
    ['/lustre/MSSP/sittipong/reduce/2023-11-07/g/all.log', ' /lustre/MSSP/sittipong/reduce/2023-11-07/g/', 17, 16, 17.41 ],
    ['/lustre/MSSP/sittipong/reduce/2023-11-08/g/all.log', ' /lustre/MSSP/sittipong/reduce/2023-11-08/g/', 17, 15, 17.41 ],
    ['/lustre/MSSP/sittipong/reduce/2023-11-09/g/all.log', ' /lustre/MSSP/sittipong/reduce/2023-11-09/g/', 17, 15, 17.41 ],
    # 
    ['/lustre/MSSP/sittipong/reduce/2023-11-06/r/all.log', ' /lustre/MSSP/sittipong/reduce/2023-11-06/r/', 12, 18, 16.93 ],
    ['/lustre/MSSP/sittipong/reduce/2023-11-07/r/all.log', ' /lustre/MSSP/sittipong/reduce/2023-11-07/r/', 18, 16, 16.93 ],
    ['/lustre/MSSP/sittipong/reduce/2023-11-08/r/all.log', ' /lustre/MSSP/sittipong/reduce/2023-11-08/r/', 17, 18, 16.93 ],
    ['/lustre/MSSP/sittipong/reduce/2023-11-09/r/all.log', ' /lustre/MSSP/sittipong/reduce/2023-11-09/r/', 17, 16, 16.93 ],
    # 
    ['/lustre/MSSP/sittipong/reduce/2023-11-06/i/all.log', ' /lustre/MSSP/sittipong/reduce/2023-11-06/i/', 12, 19, 16.77 ],
    ['/lustre/MSSP/sittipong/reduce/2023-11-07/i/all.log', ' /lustre/MSSP/sittipong/reduce/2023-11-07/i/', 13, 19, 16.77 ],
    ['/lustre/MSSP/sittipong/reduce/2023-11-08/i/all.log', ' /lustre/MSSP/sittipong/reduce/2023-11-08/i/', 12, 19, 16.77 ],
    ['/lustre/MSSP/sittipong/reduce/2023-11-09/i/all.log', ' /lustre/MSSP/sittipong/reduce/2023-11-09/i/', 15, 20, 16.77 ],
]

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

for i in 
    pi = Ensemble(config=None,  diagnostics = True, save_path = '/lustre/MSSP/sittipong/reduce/2023-11-07/')
    pi.run(logfile='/lustre/MSSP/sittipong/reduce/2023-11-07/g/all.log', 
           ignor_stars = None,
           target_rms=0.02, 
           numstars=12,)
    pi.plot_all_comparison_lr(all_stars = True, xlim=None)
