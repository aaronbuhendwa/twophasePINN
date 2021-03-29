# Physics-informed neural networks for two-phase flow problems
Physics-informed neural networks (PINN) give rise to a new approach for the quantification of flow fields
by combining available data of auxiliary variables with the underlying physical laws. This way, the
velocity and pressure of entire flow fields may be inferred, for which a direct measurement is usually
impracticle.

This repository contains the code for the rising bubble case in the paper "Inferring incompressible two-phase flow fields from the interface
motion using physics-informed neural networks" by Aaron B. Buhendwa, Stefan Adami and Nikolaus A. Adams.

If there are any questions regarding the code please contact us by [mail](mailto:aaron.buhendwa@tum.de).
# Prerequisites
Before running the scripts, the file containing the [CFD result](https://syncandshare.lrz.de/getlink/fi2mzU79pAJa8LXFtCgFsozR/rising_bubble.h5) has to be downloaded and put into the folder `cfd_data`. Furthermore, your python environment must have the following packages installed:
* numpy, scipy, pandas, mat4py, tensorflow==1.15, h5py==2.10, matplotlib==3.3.3 

# Running the scripts
We provide two scripts that are running out of the box located in `src`:

* `train_model.py` contains the implementation of the PINN class and the training routine. When running this script, the point distribution is generated and displayed for multiple time snapshots. Subsequently, the PINN is instantiated and trained using the (default) hyperparameters as described in the paper. Note that when using the default hyperparameters and amount of training points, a single epoch takes about 4 seconds on a GeForce RTX 2080Ti and thus may take substantially longer when running on a CPU. In this case the network size and/or amount of training points should be reduced by the user by setting the corresponding variables within the `main` function. During training, this script will generate a new folder called `checkpoints`, where, at user defined epoch intervals, the model is saved. Please refer to the respective function descriptions for further details.

* `test_model.py` contains a test environment that loads both the CFD result and a PINN. The prediction is then compared to the cfd by 
animating contourplots of the velocity and pressure. By default, this script will load the PINN that is located in the directory `trained_model`. We provide an already trained model in this directory, so that this script may be run out of the box.

# Citation
DOI: 10.1016/j.mlwa.2021.100029
