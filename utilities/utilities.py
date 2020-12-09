import os
import tensorflow as tf
import numpy as np
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
mpl.rcParams["mpl_toolkits.legacy_colorbar"] = False
import h5py
import json
import mat4py

def reshape_test_data(x, y, t):

    ''' Function reshaping the test data to feed to neural network for prediction '''

    Y, T, X = np.meshgrid(y, t, x)
    X = X.reshape(X.size, 1)
    Y = Y.reshape(Y.size, 1)
    T = T.reshape(T.size, 1)
    test_data = np.concatenate((X, Y, T), axis=1)
    return test_data

def reshape_prediction(x, y, t, u):

    ''' Function reshaping the predictions into 2D arrays '''

    return u.reshape(len(t), len(y), len(x), order="C")

def load_nn_model(path, plot_loss_history=True):

    ''' Loads the keras model from using the files that are located in path. 

    In particular, 3 files are needed:

    1) .json containing the architecture
    2) .h5 containing the weights
    3) .mat containing loss history, adaptive activation history and optimizer state.

    Note that this function needs to be changed when using a model that was trained with
    adaptive activation functions, as these have to be provided as custom items when loading 
    the model from json.

    Args:
     path: absolute path to where the 3 required files are located
     plot_loss_history: bool indicating whether to plot the loss history when loading the model '''

    for file in os.listdir(path):
        if file.endswith("mat") and "weights" not in file:
            matfile = mat4py.loadmat(os.path.join(path, file))
        if file.endswith("json"):
            with open(os.path.join(path, file)) as json_data:
                architecture = json.load(json_data)
        if file.endswith("h5"):
            weights = os.path.join(path, file)

    print("\nLoading nn model: %s" % weights)

    # LOAD LOSS HISTORY AND ADAPTIVE ACTIVATION COEFFICIENT
    loss_history = matfile["loss_history"]

    if plot_loss_history:
        fig, ax = plt.subplots()
        fig.set_size_inches([15,8])
        for key in loss_history:
            print("Final loss %s: %e" % (key, loss_history[key][-1]))
            ax.semilogy(loss_history[key], label=key)
        ax.set_xlabel("epochs", fontsize=15)
        ax.set_ylabel("loss", fontsize=15)
        ax.tick_params(labelsize=15)
        ax.legend()
        plt.show()

    model = tf.keras.models.model_from_json(architecture)
    model.load_weights(weights)

    return model

def load_cfd(start_index, end_index, temporal_step_size, spatial_step_size):

    ''' Loads the cfd results and returns them in the respective numpy array.
    The cfd results contains 151 time snapshots from t = 0.0 until t = 3.0 with a resolution of 512x256
    The returned resolution may be coarsened using the input arguments of this function.
    If e.g. spatial_step_size = 2, then the spatial resolution is divided by 2.

    Args:
        start_index = start index of returned time snapshots
        end_index = end index of returned time snapshots
        temporal_step_size = index temporal resolution
        spatial_step_size = index spatial resolution '''

    # PATH TO CFD SOLUTION
    path = "../cfd_data/rising_bubble.h5"

    # OPEN AND ASSIGN TO NUMPY ARRAYS
    with h5py.File(path, "r") as data:
        
        X = np.array(data["X"])[::spatial_step_size]
        Y = np.array(data["Y"])[::spatial_step_size]
        time = np.array(data["time"])[start_index:end_index:temporal_step_size]
        levelset = np.array(data["levelset"])[start_index:end_index:temporal_step_size,::spatial_step_size,::spatial_step_size]
        pressure = np.array(data["pressure"])[start_index:end_index:temporal_step_size,::spatial_step_size,::spatial_step_size]
        velocityX = np.array(data["velocityX"])[start_index:end_index:temporal_step_size,::spatial_step_size,::spatial_step_size]
        velocityY = np.array(data["velocityY"])[start_index:end_index:temporal_step_size,::spatial_step_size,::spatial_step_size]
    
    print("Shape of cfd results:", pressure.shape)
    print("Loaded cfd time snapshots:\n", time)

    return pressure, velocityX, velocityY, levelset, X, Y, time


def update_contourf(i, xs, ys, data, axis, pcfsets, kwargs):

    ''' This function updates the contourplots '''

    list_of_collections = []

    for x, y, z, ax, pcfset, kw in zip(xs, ys, data, axis, pcfsets, kwargs):
        
        for tp in pcfset[0].collections:
            tp.remove()

        pcfset[0] = ax.contourf(x, y, z[i,:,:], **kw)
        list_of_collections += pcfset[0].collections

    return list_of_collections


def grid_contour_plots(data, nrows_ncols, titles, x, y, fontsize=15, labelsize=10):
    
    '''Creates contourplots using an ImageGrid

    Args:
        data: list of numpy arrays containing values to plot
        nrows_ncols: amount of rows and columns of the ImageGrid
        titles: list of titles - needs to be same length as data
        x - numpy array containing spatial coordinates in x-direction
        y - numpy array containing spatial coordinates in y-direction '''

    # CREATE FIGURE AND AXIS
    fig = plt.figure()
    grid = ImageGrid(fig, 111, direction="row", nrows_ncols=nrows_ncols, label_mode="1", axes_pad=0.8, share_all=False, cbar_mode="each",
        cbar_location="right", cbar_size="5%", cbar_pad=0.0)

    # CREATE ARGUMENTS DICT FOR CONTOURPLOTS
    minmax_list = []
    kwargs_list = []
    for d in data:
        minmax_list.append([np.min(d), np.max(d)])
        kwargs_list.append(dict(levels=np.linspace(minmax_list[-1][0],minmax_list[-1][1], 60),
            cmap="seismic", vmin=minmax_list[-1][0], vmax=minmax_list[-1][1]))

    # CREATE PLOTS
    pcfsets = []
    for ax, z, kwargs, minmax, title in zip(grid, data, kwargs_list, minmax_list, titles):

        pcf = [ax.contourf(x, y, z[0,:,:], **kwargs)]
        pcfsets.append(pcf)

        cb = ax.cax.colorbar(pcf[0], ticks=np.linspace(minmax[0],minmax[1],5))
        ax.cax.tick_params(labelsize=labelsize)
        ax.set_title(title, fontsize=fontsize, pad=10)
        ax.set_ylabel("y/R", labelpad=15, fontsize=fontsize, rotation="horizontal")
        ax.set_xlabel("x/R", fontsize=fontsize)
        ax.tick_params(labelsize=fontsize)
        ax.set_aspect("equal")

    fig.set_size_inches(20,10,True)

    return fig, grid, pcfsets, kwargs_list


def writeToJSONFile(path, fileName, data):
        filePathNameWExt = path + '/' + fileName + '.json'
        with open(filePathNameWExt, 'w') as fp:
                json.dump(data, fp)


class NNCreator:
    
    ''' Class implementing a keras dnn with optional adaptive activation coefficients '''

    def __init__(self, dtype):
        self.dtype = dtype

    def get_model_dnn(self, input_dim, hidden_layers, output_layer, activation_functions_dict, use_ad_act):
            input_tensor = tf.keras.layers.Input(shape=(input_dim,), name="input_tensor", dtype=self.dtype)
            x = input_tensor
            for i, nodes in enumerate(hidden_layers):
                x = tf.keras.layers.Dense(nodes, activation=self.get_activation_function(activation_functions_dict[i+1], use_ad_act), kernel_initializer=tf.keras.initializers.glorot_normal())(x)
            outputs = []
            for output_name, activation in output_layer :
                outputs.append(tf.keras.layers.Dense(1, activation=activation, name=output_name, kernel_initializer=tf.keras.initializers.glorot_normal())(x))
            model = tf.keras.models.Model(inputs=input_tensor, outputs=outputs)
            return model

    def get_activation_function(self, name_coeff_n, use_ad_act):
        function_name = name_coeff_n[0]
        ad_act_coeff = name_coeff_n[1]
        n = name_coeff_n[2]
        if use_ad_act == True:
            if function_name == "tanh":
                return lambda x: tf.keras.activations.tanh(ad_act_coeff * n * x)
            elif function_name == "sin":
                return lambda x: K.sin(ad_act_coeff * n * x)
            elif function_name == "logistic":
                return lambda x: tf.keras.activations.sigmoid(ad_act_coeff * n * x)
            elif function_name == "exponential":
                return lambda x: tf.keras.activations.exponential(ad_act_coeff * n * x)
        else:
            if function_name == "tanh":
                return tf.keras.activations.tanh
            elif function_name == "sin":
                return lambda x: K.sin(x)
            elif function_name == "logistic":
                return tf.keras.activations.sigmoid
            elif function_name == "exponentail":
                return tf.keras.activations.exponential
