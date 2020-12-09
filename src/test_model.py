import sys
sys.path.append("../utilities")
from utilities import *
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'               
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def main():

    # LOAD CFD SOLUTION
    pressure_cfd, velocityX_cfd, velocityY_cfd, levelset_cfd, x, y, t = load_cfd(start_index=0, end_index=151,
        temporal_step_size=10, spatial_step_size=2)

    # REFERENCE PARAMETERS FOR NON-DIMENSIONALIZATION
    L_ref = 0.25
    rho_ref = 1000

    # NON-DIMENSIONALIZATION 
    x /= L_ref 
    y /= L_ref 
    t /= L_ref 
    pressure_cfd /= rho_ref
    
    # LOAD NN MODEL
    path = "../trained_model"
    model = load_nn_model(path, plot_loss_history=False)

    # PREPARE PREDICTION DATA - WE COMPUTE THE PREDICTION AT THE SAME SPATIAL/TEMPORAL COORDS WHERE THE CFD SOLUTION IS PROVIDED 
    test_data = reshape_test_data(x, y, t)

    # PREDICT AND RESHAPE SOLUTION
    print("\nPredicting nn solution") 
    velocityX_nn, velocityY_nn, pressure_nn, volume_fraction_nn = model.predict(test_data, batch_size=int(1e6), verbose=1)
    velocityX_nn = reshape_prediction(x, y, t, velocityX_nn)
    velocityY_nn = reshape_prediction(x, y, t, velocityY_nn)
    pressure_nn = reshape_prediction(x, y, t, pressure_nn) 
    volume_fraction_nn = reshape_prediction(x, y, t, volume_fraction_nn)

    # CONTOURPLOT PARAMETERS
    data = [pressure_nn, velocityX_nn, velocityY_nn, pressure_cfd, velocityX_cfd, velocityY_cfd]
    titles = ["p_pred", "u_pred", "v_pred", "p_cfd", "u_cfd", "v_cfd"]
    nrows_ncols = (2,3)

    # CREATE FIGURE 
    fig, grid, pcfsets, kwargs = grid_contour_plots(data, nrows_ncols, titles, x, y) 

    # ANIMATE 
    ani = FuncAnimation(fig, update_contourf, frames=len(t), fargs=([x]*np.prod(nrows_ncols), [y]*np.prod(nrows_ncols),
        data, [ax for ax in grid], pcfsets, kwargs), interval=50, blit=True, repeat=True)

    plt.show()

if __name__ == "__main__":
    main()
