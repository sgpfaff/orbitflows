# Exemplary Model for the Vertical Milky Way Potential

This directory holds the code for training an optimal model for the vertical MW potential (galpy's MWPotential2014) and analyzing it's performance.

## Runs


|  | [best_mw.py](best_mw.py) | [best_mw_2.py](best_mw_2.py) | [best_mw_3.py](best_mw_3.py) |
| :--- | :--- | :--- | :--- |
| **SECTION 1: COMPUTING** | | |
| *Device* | CPU | CPU | CUDA |
| *Precision* | float64 | float64 | float32 |
| **SECTION 2: MODEL ARCHITECTURE** | | |
| *Flow Layers* | 256 | 64 | 256 |
| *Conditioner Layers* | 2 | 2 | 2 | 
| *Conditioner Projections Dimensions* | 64 | 64 | 64 |
| **SECTION 3: TRAINING DATA** | | |
| *# of Orbits* | 16 | 16 | 16 |
| *Points per Orbit* | 256 | 256 | 256 | 
| *r_bounds* | 0.15-0.75 | 0.15-0.75 | 0.01-0.75 |
| **SECTION 4: TRAINING DETAILS** | | 
| *training steps* | 100k | 100k | 100k |
| *scheduler threshold* | 1e-7 | 1e-7 | 1e-15 |

## Results & Analysis

Summary plots are created in [analysis.ipynb](analysis.ipynb). To change the run, just change the string in the cell that loads the run.
