import sys
sys.path.append("../utilities")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'               
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
GPU_ID = "0"                                                       
os.environ["CUDA_VISIBLE_DEVICES"]= GPU_ID                
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from tensorflow.keras import backend as K
import numpy as np
import pandas as pd
import scipy.io
from generate_points import *
from utilities import *
import time
import math
import glob
from datetime import datetime
import shutil
import logging

np.random.seed(1234)
tf.set_random_seed(1234)

class TwoPhasePinn:

    ''' This class implements a physics-informed neural network. It approximates the incompressible two-phase Navier-Stokes equations in 2D
    using a Volume-of-Fluid approach. Thus, the neural network maps (x, y, t) -> (u, v, p, a) where a is the volume fraction field. The placeholders and
    losses have to be constructed for each case individually as they depend on the boundary conditions. The present implementation corresponds to the
    rising bubble case, see paper.
    
    Args:
        sess: tensorflow session
        dtype: data type
        hidden_layers: list containing number of nodes for each hidden layer
        activation_functions: dictionary assigning layers to activation function
        adaptive_activation_coeff: dictionary assigning layers to adaptive activation coeff
        adaptive_activation_init: dictionary assigning initial value to adaptive activation coeff
        adaptive_activation_n: list containing the scale factor of the adapative activation coeff for each layer - must have same length as hidden_layers
        use_ad_act: bool indicating whether to use adaptive activation coeff
        loss_weights_A: loss weight for volume fraction loss
        loss_weights_PDE: loss weights for PDEs
        checkpoint_interval: interval in epochs indicating when to save model
        epochs: list of epochs
        batch_sizes: list of batch sizes - should have same length as epochs
        learning_rates: list of learning rates - should have same length as epochs
    '''

    def __init__(self, sess, dtype, hidden_layers, activation_functions, adaptive_activation_coeff, adaptive_activation_n, 
        adaptive_activation_init, use_ad_act, loss_weights_A, loss_weights_PDE, mu, sigma, g, rho, u_ref, L_ref, checkpoint_interval, epochs, batch_sizes,
        learning_rates):
        
        # CREATE OUTPUT FOLDER AND GET LOGGER
        self.dirname, logpath = self.make_output_dir()
        self.logger = self.get_logger(logpath)     

        # PHYSICAL PARAMETERS
        self.mu1 = mu[0]
        self.mu2 = mu[1]
        self.sigma = sigma
        self.g = g
        self.rho1 = rho[0]
        self.rho2 = rho[1]
        self.U_ref = u_ref
        self.L_ref = L_ref

        # MEMBERS FOR SAVING CHECKPOINTS AND TRACKING 
        self.epoch_loss_checkpoints = 1e10
        self.checkpoint_interval = checkpoint_interval
        self.mean_epoch_time = 0

        # SGD OPT MEMBERS
        self.learning_rates = learning_rates
        self.epochs = epochs
        self.batch_sizes = batch_sizes

        # TENSORFLOW SESSION
        self.sess = sess
        K.set_session(self.sess)

        self.print("Building Computational Graph")
        
        # PLACEHOLDERS
        x_A = tf.placeholder(dtype=dtype, shape=[None, 1], name="x_A")
        y_A = tf.placeholder(dtype=dtype, shape=[None, 1], name="y_A")
        t_A = tf.placeholder(dtype=dtype, shape=[None, 1], name="t_A")
        a_A = tf.placeholder(dtype=dtype, shape=[None, 1], name="a_A")

        x_N = tf.placeholder(dtype=dtype, shape=[None, 1], name="x_N")
        y_N = tf.placeholder(dtype=dtype, shape=[None, 1], name="y_N")
        t_N = tf.placeholder(dtype=dtype, shape=[None, 1], name="t_N")
        p_N = tf.placeholder(dtype=dtype, shape=[None, 1], name="p_N")

        x_E = tf.placeholder(dtype=dtype, shape=[None, 1], name="x_E")
        y_E = tf.placeholder(dtype=dtype, shape=[None, 1], name="y_E")
        x_W = tf.placeholder(dtype=dtype, shape=[None, 1], name="x_W")
        y_W = tf.placeholder(dtype=dtype, shape=[None, 1], name="y_W")
        t_EW = tf.placeholder(dtype=dtype, shape=[None, 1], name="t_EW")
 
        x_NSEW = tf.placeholder(dtype=dtype, shape=[None, 1], name="x_NSEW")
        y_NSEW = tf.placeholder(dtype=dtype, shape=[None, 1], name="y_NSEW")
        t_NSEW = tf.placeholder(dtype=dtype, shape=[None, 1], name="t_NSEW")
        u_NSEW = tf.placeholder(dtype=dtype, shape=[None, 1], name="u_NSEW")
        v_NSEW = tf.placeholder(dtype=dtype, shape=[None, 1], name="v_NSEW")
        
        x_PDE = tf.placeholder(dtype=dtype, shape=[None, 1], name="x_PDE")
        y_PDE = tf.placeholder(dtype=dtype, shape=[None, 1], name="y_PDE")
        t_PDE = tf.placeholder(dtype=dtype, shape=[None, 1], name="t_PDE")
        f_PDE = tf.placeholder(dtype=dtype, shape=[None, 1], name="f_PDE")

        self.learning_rate_opt = tf.placeholder(dtype=dtype, shape=[], name="learning_rate")

        data_set_names = ["A", "PDE", "N", "EW", "NSEW"]
        self.placeholders = dict((name, []) for name in data_set_names)
        self.placeholders["A"].extend([x_A, y_A, t_A, a_A])
        self.placeholders["PDE"].extend([x_PDE, y_PDE, t_PDE, f_PDE])
        self.placeholders["N"].extend([x_N, y_N, t_N, p_N])
        self.placeholders["EW"].extend([x_E, y_E, x_W, y_W, t_EW])
        self.placeholders["NSEW"].extend([x_NSEW, y_NSEW, t_NSEW, u_NSEW, v_NSEW])

        # VARIABLES ADAPTIVE ACTIVATION FOR HIDDEN LAYERS
        self.sanity_check_activation_functions(activation_functions, adaptive_activation_coeff, adaptive_activation_n, adaptive_activation_init, hidden_layers)
        self.ad_act_coeff = {}
        if use_ad_act:
            for key in adaptive_activation_coeff:
                initial_value = adaptive_activation_init[key]
                self.ad_act_coeff[key] = tf.Variable(initial_value, name=key)
        activation_functions_dict = self.get_activation_function_dict(activation_functions, adaptive_activation_coeff, adaptive_activation_n, hidden_layers, use_ad_act)

        # NETWORK ARCHITECTURE
        outputs = ["output_u", "output_v", "output_p", "output_a"]
        activations_output = [None, None, "exponential", "sigmoid"]
        output_layer = list(zip(outputs, activations_output))
        nn = NNCreator(dtype)
        self.model = nn.get_model_dnn(3, hidden_layers, output_layer, activation_functions_dict, use_ad_act)

        # LOSSES ASSOCIATED WITH A
        output_tensors = self.model(tf.concat([x_A, y_A, t_A], 1))
        loss_a_A = tf.reduce_mean(tf.square(a_A - output_tensors[3]))

        # LOSSES ASSOCIATED WITH FIXED VALUE NORTH SOUTH EAST WEST
        start = time.time()
        output_tensors = self.model(tf.concat([x_NSEW, y_NSEW, t_NSEW], 1))
        loss_u_NSEW = tf.reduce_mean(tf.square(u_NSEW - output_tensors[0]))
        loss_v_NSEW = tf.reduce_mean(tf.square(v_NSEW - output_tensors[1]))
        loss_NSEW = tf.reduce_sum(tf.stack([loss_u_NSEW, loss_v_NSEW]))
        self.print(time.time()-start, "s")
        
        # LOSSES ASSOCIATED WITH FIXED PRESSURE NORTH
        start = time.time()
        output_tensors = self.model(tf.concat([x_N, y_N, t_N], 1))
        loss_p_N = tf.reduce_mean(tf.square(p_N - output_tensors[2]))
        self.print(time.time()-start, "s")

        # LOSSES ASSOCIATED WITH PERIODIC BOUNDARY EAST WEST
        start = time.time()
        output_east = self.model(tf.concat([x_E, y_E, t_EW], 1))
        output_west = self.model(tf.concat([x_W, y_W, t_EW], 1))
        loss_u_EW = tf.reduce_mean(tf.square(output_east[0] - output_west[0]))
        loss_v_EW = tf.reduce_mean(tf.square(output_east[1] - output_west[1]))
        loss_p_EW = tf.reduce_mean(tf.square(output_east[2] - output_west[2]))
        loss_EW = tf.reduce_sum(tf.stack([loss_u_EW, loss_v_EW, loss_p_EW])) 
        self.print(time.time()-start, "s")        

        loss_NSEW = tf.reduce_sum(tf.stack([loss_p_N, loss_EW, loss_NSEW]))
        
        # LOSSES ASSOCIATED WITH PDEs -> PHYSICS INFORMED NEURAL NETS
        start = time.time()
        PDE_tensors = self.PDE_caller(x_PDE, y_PDE, t_PDE)
        loss_PDE_m = tf.losses.mean_squared_error(f_PDE, PDE_tensors[0])
        loss_PDE_u = tf.losses.mean_squared_error(f_PDE, PDE_tensors[1])
        loss_PDE_v = tf.losses.mean_squared_error(f_PDE, PDE_tensors[2])
        loss_PDE_a = tf.losses.mean_squared_error(f_PDE, PDE_tensors[3])
        self.print(time.time()-start, "s")
        
        loss_PDE = tf.tensordot(tf.stack([loss_PDE_m, loss_PDE_u, loss_PDE_v, loss_PDE_a]), np.array(loss_weights_PDE).astype("float32"), 1)

        # TOTAL LOSS
        loss_complete = loss_a_A + loss_NSEW + loss_PDE

        # OPTIMIZERS
        start = time.time()
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate_opt)
        self.minimize_op = self.optimizer.minimize(loss_complete)
        self.print(time.time()-start, "s")

        # DEFINING LISTS AND DICTIONARIES FOR TRACKING LOSSES AND SPECIFIC TENSORS
        self.loss_tensor_list = [loss_complete, loss_a_A, loss_NSEW, loss_PDE_m, loss_PDE_u, loss_PDE_v, loss_PDE_a] 
        self.loss_list = ["l", "a", "NSEW", "m", "u", "v", "PDE_a"]
        self.epoch_loss = dict.fromkeys(self.loss_list, 0)
        self.loss_history = dict((loss, []) for loss in self.loss_list)
        self.ad_act_coeff_history = dict((key, []) for key in self.ad_act_coeff)

        # INITIALIZING VARIABLES
        self.sess.run(tf.global_variables_initializer())

        # SET WEIGHTS AND OPTIMIZER STATE IF AVAILABLE
        self.set_variables()
        
        # FINALIZING
        self.model.save_weights(os.path.join(self.dirname, "Weights_loss_%.4e.h5" % (self.epoch_loss_checkpoints)))
        self.sess.graph.finalize()

    def make_output_dir(self):
        if not os.path.exists("checkpoints"):
            os.mkdir("checkpoints")
        dirname = os.path.abspath(os.path.join("checkpoints", datetime.now().strftime("%b-%d-%Y_%H-%M-%S")))
        os.mkdir(dirname)
        shutil.copyfile(__file__, os.path.join(dirname, __file__))
        shutil.copyfile("generate_points.py", os.path.join(dirname, "generate_points.py"))
        logpath = os.path.join(dirname, "output.log")
        return dirname, logpath

    def get_logger(self, logpath):
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)
        sh = logging.StreamHandler()
        sh.setLevel(logging.DEBUG)        
        sh.setFormatter(logging.Formatter('%(message)s'))
        fh = logging.FileHandler(logpath)
        logger.addHandler(sh)
        logger.addHandler(fh)
        return logger

    def sanity_check_activation_functions(self, activation_functions, adaptive_activations, adaptive_activation_n, adaptive_activation_init, hidden_layers):
        no_layers = len(hidden_layers)
        check = 0
        for key, value in list(adaptive_activations.items()):                  
            check += sum(value)
        assert no_layers*(no_layers+1)/2 == check, "Not every layer has been assigned with an adaptive activation coefficient unambiguously"
        check = 0
        for key, value in list(activation_functions.items()):                  
            check += sum(value)
        assert no_layers*(no_layers+1)/2 == check, "Not every layer has been assigned with an activation function unambiguously"
        assert no_layers == len(adaptive_activation_n), "Not every layer has an adaptive activation precoefficient"
        assert adaptive_activation_init.keys() == adaptive_activations.keys(), "Not every adaptive activation coefficient has been assigned an initial value"

    def get_activation_function_dict(self, activation_functions, adaptive_activation_coeff, adaptive_activation_n, hidden_layers, use_ad_act):
        activation_functions_dict = dict((key, [0, 0, 0]) for key in range(1, len(hidden_layers) + 1))
        for layer_no in activation_functions_dict:
            activation_functions_dict[layer_no][2] = adaptive_activation_n[layer_no-1]
            for func_name, layers in activation_functions.items():
                if layer_no in layers:
                    activation_functions_dict[layer_no][0] = func_name
            if use_ad_act:                                                  # if use_ad_act is False, self.ad_act_coeff is empty!
                for coeff_name, layers in adaptive_activation_coeff.items():
                    if layer_no in layers:
                        activation_functions_dict[layer_no][1] = self.ad_act_coeff[coeff_name]
        return activation_functions_dict

    def compute_gradients(self, x, y, t):
        u, v, p, a = self.model(tf.concat([x, y, t], 1))

        u_x = tf.gradients(u, x)[0]
        u_y = tf.gradients(u, y)[0]
        u_t = tf.gradients(u, t)[0]
        u_xx = tf.gradients(u_x, x)[0]
        u_yy = tf.gradients(u_y, y)[0]

        v_x = tf.gradients(v, x)[0]
        v_y = tf.gradients(v, y)[0]
        v_t = tf.gradients(v, t)[0]
        v_xx = tf.gradients(v_x, x)[0]
        v_yy = tf.gradients(v_y, y)[0]

        p_x = tf.gradients(p, x)[0]
        p_y = tf.gradients(p, y)[0]

        a_x = tf.gradients(a, x)[0]
        a_y = tf.gradients(a, y)[0]
        a_t = tf.gradients(a, t)[0]
        a_xx = tf.gradients(a_x, x)[0]
        a_yy = tf.gradients(a_y, y)[0]
        a_xy = tf.gradients(a_x, y)[0]

        return [u, u_x, u_y, u_t, u_xx, u_yy], [v, v_x, v_y, v_t, v_xx, v_yy], [p, p_x, p_y], [a, a_x, a_y, a_t, a_xx, a_yy, a_xy]

    def PDE_caller(self, x, y, t):
        u_gradients, v_gradients, p_gradients, a_gradients = self.compute_gradients(x, y, t)
        u, u_x, u_y, u_t, u_xx, u_yy = u_gradients[:]
        v, v_x, v_y, v_t, v_xx, v_yy = v_gradients[:]
        p, p_x, p_y = p_gradients[:]
        a, a_x, a_y, a_t, a_xx, a_yy, a_xy = a_gradients[:]

        mu = self.mu2 + (self.mu1 - self.mu2) * a
        mu_x = (self.mu1 - self.mu2) * a_x
        mu_y = (self.mu1 - self.mu2) * a_y
        rho = self.rho2 + (self.rho1 - self.rho2) * a

        abs_interface_grad = tf.sqrt(tf.square(a_x) + tf.square(a_y) + np.finfo(float).eps)

        curvature = - ( (a_xx + a_yy)/abs_interface_grad - (a_x**2*a_xx + a_y**2*a_yy + 2*a_x*a_y*a_xy)/tf.pow(abs_interface_grad, 3) )

        rho_ref = self.rho2

        one_Re = mu/(rho_ref*self.U_ref*self.L_ref)
        one_Re_x = mu_x/(rho_ref*self.U_ref*self.L_ref)
        one_Re_y = mu_y/(rho_ref*self.U_ref*self.L_ref)
        one_We = self.sigma/(rho_ref*self.U_ref**2*self.L_ref)
        one_Fr = self.g*self.L_ref/self.U_ref**2 

        PDE_m = u_x + v_y
        PDE_a = a_t + u*a_x + v*a_y
        PDE_u = (u_t + u*u_x + v*u_y)*rho/rho_ref + p_x - one_We*curvature*a_x - one_Re*(u_xx + u_yy) - 2.0*one_Re_x*u_x - one_Re_y*(u_y + v_x) 
        PDE_v = (v_t + u*v_x + v*v_y)*rho/rho_ref + p_y - one_We*curvature*a_y - one_Re*(v_xx + v_yy) - rho/rho_ref*one_Fr - 2.0*one_Re_y*v_y - one_Re_x*(u_y + v_x) 

        return PDE_m, PDE_u, PDE_v, PDE_a

    def set_variables(self):

        ''' Implements functionality to continue training from checkpoint. Loads the weights and optimizer state
        from the .h5 file and the .mat file, respectively. This is only done if the necessary files are located in
        the same folder as this script '''

        for file in glob.glob("*loss*"):
            if file.endswith("h5"):
                self.model.load_weights(file)
                self.print("Loading weights from file", file)
            if file.endswith("mat"):
                matfile = scipy.io.loadmat(file, squeeze_me=True)
                self.print("Setting optimizer variables according to file", file)
                optimizer_state = matfile["optimizer_state"]
                optimizer_variables = self.optimizer.variables()
                assert len(optimizer_variables) == len(optimizer_state), "Loading optimizer state failed: Not as many optimizer states saved as required, check architecture/aac compatibility!"
                for i in range(0, len(optimizer_variables)):
                    if optimizer_variables[i].shape == (1,):                # Shapes that require (1,) are loaded as floats from .mat file, thus have to be converted to np.array
                        optimizer_state[i] = np.array([optimizer_state[i]])
                    if len(optimizer_variables[i].shape) == 2:
                        if optimizer_variables[i].shape[1] == 1:            # Shapes that require (?,1) are loaded as (?,) from .mat file, thus need reshaping
                            optimizer_state[i] = optimizer_state[i].reshape(len(optimizer_state[i]),1)
                    self.sess.run(optimizer_variables[i].assign(optimizer_state[i]))
                self.print("Setting adaptive activation coefficients according to file", file)
                ad_act_coeff = matfile["ad_act_coeff"]
                if len(self.ad_act_coeff) > 0:
                    assert list(self.ad_act_coeff.keys()) == list(ad_act_coeff.dtype.names), "Loading adaptive activation coefficients failed: Restart coefficients %s do not match input %s" %(list(ad_act_coeff.dtype.names), list(self.ad_act_coeff.keys()))
                    for key in self.ad_act_coeff:
                        self.sess.run(self.ad_act_coeff[key].assign(float(ad_act_coeff[key])))

    def train(self, data_sets):

        ''' Implements the training loop 
        
        Args:
            data_sets: Dictionary assigning a pandas dataframe to each loss '''

        self.check_matching_keys(data_sets)
        self.print_point_distribution(data_sets)
        self.print("\nEPOCHS: ", self.epochs, " BATCH SIZES: ", self.batch_sizes, " LEARNING RATES: ", self.learning_rates)
        start_total = time.time() 
        for counter, epoch_value in enumerate(self.epochs):
            batch_sizes, number_of_batches = self.get_batch_sizes(counter, data_sets)
            for e in range(1, epoch_value + 1):
                start_epoch = time.time()
                data_sets = self.shuffle_data_and_reset_epoch_losses(data_sets)
                for b in range(number_of_batches):
                    batches = self.get_batches(data_sets, b, batch_sizes)
                    tf_dict = self.get_feed_dict(batches, counter)
                    _, batch_losses = self.sess.run([self.minimize_op, self.loss_tensor_list], tf_dict)
                    self.assign_batch_losses(batch_losses)
                self.append_loss_and_activation_coeff_history()
                self.save_model_checkpoint(self.epoch_loss[self.loss_list[0]], e, counter)
                self.print_info(e, self.epochs[counter], time.time() - start_epoch)
        self.print("\nTotal training time: %5.3fs" % (time.time() - start_total))
        self.logger.handlers[1].close()

    def check_matching_keys(self, data_sets):
        for key1, key2 in zip(data_sets, self.placeholders):
            assert key1 == key2, "Data set key %s does not match placeholder key %s" % (key1, key2)

    def print_point_distribution(self, data_sets):
        no_points = 0
        for key in data_sets:
            no_points += data_sets[key].shape[0]
            self.print("Training data %10s shape: %s" %(key, data_sets[key].shape))
        self.print("Total number of points %d" % no_points)

    def shuffle_data_and_reset_epoch_losses(self, data_sets):
        for key in data_sets:
            length = len(data_sets[key])
            shuffled_indices = np.random.choice(length, length, replace=False) 
            data_sets[key] = pd.DataFrame(data=data_sets[key].to_numpy()[shuffled_indices,:], columns=data_sets[key].columns)
        for key in self.epoch_loss:
            self.epoch_loss[key] = 0
        return data_sets

    def get_batches(self, data, b, batch_sizes):
        batches = dict.fromkeys(data.keys(), 0)
        for key in data:
            batches[key] = data[key][b*batch_sizes[key]:(b+1)*batch_sizes[key]]
        return batches

    def assign_batch_losses(self, batch_losses):
        for loss_values, key in zip(batch_losses, self.epoch_loss):
            self.epoch_loss[key] += loss_values

    def append_loss_and_activation_coeff_history(self):
        for key in self.loss_history:
            self.loss_history[key].append(self.epoch_loss[key])
        for key, value in self.ad_act_coeff.items():
            self.ad_act_coeff_history[key].append(self.sess.run(value))

    def get_feed_dict(self, batches , counter):
        tf_dict = {self.learning_rate_opt: self.learning_rates[counter]}
        feed_dicts = []
        for i, key in enumerate(self.placeholders):
            feed_dicts.append(dict.fromkeys(self.placeholders[key], 0))
            for placeholder, column_name in zip(self.placeholders[key], batches[key].columns):
                assert placeholder.name[:-2] == column_name, "Placeholder %s does not match column %s in data %s!" % (placeholder.name[:-2], column_name, key)
                feed_dicts[i][placeholder] = np.transpose(np.atleast_2d(batches[key][column_name].to_numpy()))
        for dicts in feed_dicts:
            tf_dict.update(dicts)
        return tf_dict              
        
    def save_model_checkpoint(self, loss, epoch, counter):

        ''' Saves the following files in self.dirname when a checkpoint epoch is reached:
        
        1) architecture (.json)
        2) weights (.h5)
        3) optimizer state, loss history, adaptive activation coefficient history (.mat) 

        These files may be used to restart a training run from checkpoint '''

        if loss < self.epoch_loss_checkpoints and not (epoch)%self.checkpoint_interval:
            for file in glob.glob(os.path.join(self.dirname, "*")):
                if file.endswith("json") or file.endswith("h5") or file.endswith("mat"):
                    os.remove(file)
            writeToJSONFile(self.dirname, "loss_%.4e_architecture" % (loss), self.model.to_json())
            data = dict(loss_history=self.loss_history, ad_act_coeff_history=self.ad_act_coeff_history, optimizer_state=self.sess.run(self.optimizer.variables()), 
                ad_act_coeff=self.sess.run(self.ad_act_coeff), epoch=epoch, learning_rate=self.learning_rates[counter])
            scipy.io.savemat(os.path.join(self.dirname, "loss_%.4e_variables.mat") % (loss), data)
            self.model.save_weights(os.path.join(self.dirname, "loss_%.4e_weights.h5" % (loss)))
            self.epoch_loss_checkpoints = loss

    def print_info(self, current_epoch, epochs, time_for_epoch):
        if current_epoch == 1:                                      # skipping first epoch, because it takes way longer
            self.mean_epoch_time = 0
        else:
            self.mean_epoch_time = self.mean_epoch_time*(current_epoch-2)/(current_epoch-1) + time_for_epoch/(current_epoch-1)       
        string = ["Epoch: %5d/%d - %7.2fms - avg: %7.2fms" % (current_epoch, epochs, time_for_epoch*1e3, self.mean_epoch_time*1e3)]
        for key, value in self.epoch_loss.items():
            string.append(" - %s: %.4e" % (key, value))
        for key, act_coeff in self.ad_act_coeff.items():
            string.append(" - %s: %.4e" % (key, self.sess.run(act_coeff)))
        self.print(*string)

    def get_batch_sizes(self, counter, data_sets):
        number_of_samples = sum([len(data_sets[key]) for key in data_sets])
        batch_sizes_datasets = dict.fromkeys(data_sets.keys(), 0)
        if self.batch_sizes[counter] >= number_of_samples:
            number_of_batches = 1
            for key in data_sets:
                batch_sizes_datasets[key] = len(data_sets[key])
            self.print("Batch size is larger equal the amount of training samples, thus going full batch mode")
            self.print("Total batch size: ", number_of_samples, " - ", "Batch sizes: ", batch_sizes_datasets, " - ", "learning rate: ", self.learning_rates[counter], "\n")
        else:
            number_of_batches = math.ceil(number_of_samples/self.batch_sizes[counter])
            batch_percentages = dict.fromkeys(data_sets.keys(), 0)
            print_batches = dict.fromkeys(data_sets.keys(), "")
            for key in data_sets:
                batch_percentages[key] = len(data_sets[key])/number_of_samples
                batch_sizes_datasets[key] = math.ceil(self.batch_sizes[counter]*batch_percentages[key])
                print_batches[key] = "%d/%d" % (batch_sizes_datasets[key], 0 if batch_sizes_datasets[key] == 0 else len(data_sets[key])%batch_sizes_datasets[key])
            total_batch_size = sum([batch_sizes_datasets[key] for key in batch_sizes_datasets])
            self.print("\nTotal batch size: ", total_batch_size, " - ", "number of batches: ", number_of_batches, " - ", "Batch sizes: ", print_batches, " - ", "learning rate: ", self.learning_rates[counter])
            for key in data_sets:
                if len(data_sets[key]) == 0:
                    continue
                assert (number_of_batches - 1) * batch_sizes_datasets[key] < len(data_sets[key]), "The specified batch size of %d will lead to empty batches with the present batch ratio, increase the batch size!" % (self.batch_sizes[counter])
        return batch_sizes_datasets, number_of_batches

    def print(self, *args):
        for word in args:
            if len(args) == 1:
                self.logger.info(word)
            elif word != args[-1]:
                for handler in self.logger.handlers:
                    handler.terminator = ""
                if type(word) == float or type(word) == np.float64 or type(word) == np.float32: 
                    self.logger.info("%.4e" % (word))
                else:
                    self.logger.info(word)
            else:
                for handler in self.logger.handlers:
                    handler.terminator = "\n"
                if type(word) == float or type(word) == np.float64 or type(word) == np.float32:
                    self.logger.info("%.4e" % (word))
                else:
                    self.logger.info(word)

                        
def compute_batch_size(training_data, number_of_batches):

    ''' Computes the batch size from number of batches and amount of training samples '''

    number_of_samples = sum([len(training_data[key]) for key in training_data])
    return math.ceil(number_of_samples/number_of_batches)

def main():
    ''' This scripts trains a PINN for the rising bubble case in <paper_cite_TBA>. The user may define the following:
    1) Number of points for various losses (check function description)
    2) The neural network architecture, i.e. number of hidden layers and the nodes in each hidden layer
    3) The training hyperparameters, i.e. number of epochs, batch size and learning rates
    '''

    # SETTING UP SESSION
    sess = tf.Session()
    
    # PARAMETRS FOR THE TRAINING DATA - NUMBER OF POINTS (NOP) FOR VARIOUS LOSSES
    NOP_a = (500, 400)                                                      
    NOP_PDE = (400, 2000, 3000)                                             
    NOP_north = (20, 20)                         
    NOP_south = (20, 20)
    NOP_east = (20, 20)    
    NOP_west = (20, 20)
    
    training_data = get_training_data(NOP_a, NOP_PDE, NOP_north, NOP_south, NOP_east, NOP_west)

    # NEURAL NETWORK ARCHITECTURE
    dtype = tf.float32
    no_layers = 8
    hidden_layers = [350]*no_layers
    activation_functions = dict(tanh = range(1,no_layers+1))                # dict assigning layer activation function to layer number
    
    # ADAPIVE ACTIVATION COEFFICIENTS SETUP
    adaptive_activation_coeff = {"aac_1": range(1,no_layers+1)}             # list shows corresponding layer numbers
    adaptive_activation_init = {"aac_1": 0.1}
    adaptive_activation_n = [10]*no_layers                                  # prefactor for activation function
    use_ad_act = False

    # PHYSICAL PARAMETERS
    mu = [1.0, 10.0]
    sigma = 24.5
    g = -0.98
    rho = [100, 1000]
    u_ref = 1.0
    L_ref = 0.25

    # HYPERPARAMETERS FOR TRAINING
    loss_weights_A = [1.0]                                      
    loss_weights_PDE = [1.0, 10.0, 10.0, 1.0]                   
    epochs = [5000]*5
    number_of_batches = 20
    batch_sizes = [compute_batch_size(training_data, number_of_batches)]*5
    learning_rates = [1e-4, 5e-5, 1e-5, 5e-6, 1e-6]
    checkpoint_interval = 100
    
    # INSTANTIATE PINN
    PINN = TwoPhasePinn(sess, dtype, hidden_layers, activation_functions, adaptive_activation_coeff, adaptive_activation_n, 
        adaptive_activation_init, use_ad_act, loss_weights_A, loss_weights_PDE, mu, sigma, g, rho, u_ref, L_ref, checkpoint_interval, epochs,
        batch_sizes, learning_rates)
    
    # TRAINING
    PINN.train(training_data)
        

if __name__ == "__main__":
        main()
