import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
import h5py

def get_points_for_interface(NOP, t_value, interface, normal, tangent, cell_size, refine_start, refine_end):
    no_points = math.ceil(NOP/(2*len(interface)))
    points_inward = []
    points_outward = []
    for i, [x_p, n_p, t_p] in enumerate(zip(interface, normal, tangent)):
        x_p = np.array(x_p)
        n_p = np.array(n_p)
        t_p = np.array(t_p)
        inward_n = np.random.uniform(refine_start, refine_end, no_points)
        outward_n = np.random.uniform(refine_start, refine_end, no_points)
        inward_t1 = np.random.uniform(refine_start, refine_end, no_points)
        inward_t2 = np.random.uniform(refine_start, refine_end, no_points)
        outward_t1 = np.random.uniform(refine_start, refine_end, no_points)
        outward_t2 = np.random.uniform(refine_start, refine_end, no_points)
        for in_n, in_t1, in_t2, out_n, out_t1, out_t2 in zip(inward_n, inward_t1, inward_t2, outward_n, outward_t1, outward_t2):
            points_inward.append(x_p + in_n*n_p)
            points_inward.append(x_p + cell_size/3*t_p + in_t1*n_p)
            points_inward.append(x_p + 2*cell_size/3*t_p + in_t2*n_p)
            points_outward.append(x_p - out_n*n_p)
            points_outward.append(x_p + cell_size/3*t_p - out_t1*n_p)
            points_outward.append(x_p + 2*cell_size/3*t_p - out_t2*n_p)
    interface_inward = np.hstack([np.array(points_inward), t_value*np.ones((len(points_inward), 1)), np.ones((len(points_inward), 1))])
    interface_outward = np.hstack([np.array(points_outward), t_value*np.ones((len(points_outward), 1)), np.zeros((len(points_outward), 1))])
    interface_data = np.vstack([interface_inward, interface_outward])
    indices_to_keep = np.random.choice(len(interface_data), NOP, replace=False)
    interface_data = interface_data[indices_to_keep, :]
    return interface_data

def get_points_for_domain(NOP, t_value, X, Y, levelset, max_levelset):
    no_y = math.ceil(NOP/len(X))
    domain_data = np.empty((0, 4), float)
    for index_x, x_value in enumerate(X):
        t_domain = t_value*np.ones((no_y, 1))
        x_domain = x_value*np.ones((no_y, 1))
        y_domain = []
        a_domain = []
        remaining_points_in_y = no_y
        while remaining_points_in_y != 0:
            indices_y = np.random.choice(np.arange(len(Y)), remaining_points_in_y, replace=False)
            y_temp = Y[indices_y]
            a_temp = (levelset[indices_y, index_x] > 0).astype(float)
            indices_to_keep = np.where((np.abs(levelset[indices_y, index_x]) >= max_levelset) | (np.abs(levelset[indices_y, index_x]) == 8))[0]
            y_temp = y_temp[indices_to_keep]
            a_temp = a_temp[indices_to_keep]
            a_domain.append(a_temp)
            y_domain.append(y_temp)
            remaining_points_in_y = remaining_points_in_y - len(indices_to_keep)
        a_domain = np.hstack(a_domain).reshape(no_y, 1)
        y_domain = np.hstack(y_domain).reshape(no_y, 1)
        domain_data = np.vstack([domain_data, np.hstack([x_domain, y_domain, t_domain, a_domain])])
    indices_to_keep = np.random.choice(len(domain_data), NOP, replace=False)
    domain_data = domain_data[indices_to_keep, :]
    return domain_data

def get_points_a(NOP, times, X, Y, cell_size, interface_all, normal_all, tangent_all, levelset_all):

    ''' Generates the points for the interface. The width of the refinement regions may be set with the respective variable within this function '''
    
    # PARAMETERS FOR INTERFACE REFINEMENT
    refine_start = 0.004
    refine_end = 0.008

    domain_start_levelset = 4
    data = np.empty((0, 4), float)
    for t_value, interface, normal, tangent, levelset in zip(times, interface_all, normal_all, tangent_all, levelset_all):
        interface_data = get_points_for_interface(NOP[0], t_value, interface, normal, tangent, cell_size, refine_start, refine_end)
        domain_data = get_points_for_domain(NOP[1], t_value, X, Y, levelset, domain_start_levelset)
        data = np.vstack([data, interface_data, domain_data])
    return data

def get_points_pde(NOP, times, X, Y, cell_size, interface_all, normal_all, tangent_all, levelset_all):

    ''' Generates the residual points. The width of the refinement regions may be set with the respective variable within this function '''

    # PARAMETERS FOR INTERFACE AND NEARFIELD REFINEMENT
    refine_interface_start = 0.0
    refine_interface_end = 0.001
    refine_nearfield_end = 0.1

    domain_start_levelset = 4
    data = np.empty((0, 4), float)
    for t_value, interface, normal, tangent, levelset in zip(times, interface_all, normal_all, tangent_all, levelset_all):
        interface_temp = get_points_for_interface(NOP[0], t_value, interface, normal, tangent, cell_size, refine_interface_start, refine_interface_end)
        nearfield_temp = get_points_for_interface(NOP[1], t_value, interface, normal, tangent, cell_size, refine_interface_end, refine_nearfield_end)
        domain_temp = get_points_for_domain(NOP[2], t_value, X, Y, levelset, domain_start_levelset)
        data = np.vstack([data, interface_temp, nearfield_temp, domain_temp])
    data[:,3] = 0.0
    return data

def get_north(NOP, x, y, t):
    low_t = t[0] + np.finfo(float).eps
    boundary_north_time = np.hstack([0.0, np.random.uniform(low_t, t[1], NOP[1] - 2), t[1]])
    boundary_north = np.empty((0, 6), float)
    for time_value in boundary_north_time:
        boundary_north_x = np.random.uniform(x[0], x[1], (NOP[0], 1))
        boundary_north_y = y[1]*np.ones((NOP[0], 1))
        boundary_north_p = np.ones((NOP[0], 1))
        boundary_north_v = np.zeros((NOP[0], 1))
        boundary_north_u = np.zeros((NOP[0], 1))
        boundary_north = np.vstack([boundary_north, np.hstack([boundary_north_x, boundary_north_y, time_value*np.ones((NOP[0], 1)), boundary_north_u, boundary_north_v, boundary_north_p])])
    return boundary_north

def get_south(NOP, x, y, t):
    low_t = t[0] + np.finfo(float).eps
    boundary_south_time = np.hstack([0.0, np.random.uniform(low_t, t[1], NOP[1] - 2), t[1]])
    boundary_south = np.empty((0, 5), float)
    for time_value in boundary_south_time:
        boundary_south_x = np.random.uniform(x[0], x[1], (NOP[0], 1))
        boundary_south_y = y[0]*np.ones((NOP[0], 1))
        boundary_south_v = np.zeros((NOP[0], 1))
        boundary_south_u = np.zeros((NOP[0], 1))
        boundary_south = np.vstack([boundary_south, np.hstack([boundary_south_x, boundary_south_y, time_value*np.ones((NOP[0], 1)), boundary_south_u, boundary_south_v])])
    return boundary_south

def get_east(NOP, x, y, t):
    low_t = t[0] + np.finfo(float).eps
    boundary_east_time = np.linspace(t[0], t[1], NOP[1])
    boundary_east = np.empty((0, 5), float)
    for time_value in boundary_east_time:
        boundary_east_y = np.linspace(y[0], y[1], NOP[0]).reshape(-1, 1)
        boundary_east_x = x[1]*np.ones((NOP[0], 1))
        boundary_east_p = np.ones((NOP[0], 1))
        boundary_east_v = np.zeros((NOP[0], 1))
        boundary_east_u = np.zeros((NOP[0], 1))
        boundary_east = np.vstack([boundary_east, np.hstack([boundary_east_x, boundary_east_y, time_value*np.ones((NOP[0], 1)), boundary_east_u, boundary_east_v])])
    return boundary_east

def get_west(NOP, x, y, t):
    low_t = t[0] + np.finfo(float).eps
    boundary_west_time = np.linspace(t[0], t[1], NOP[1])        
    boundary_west = np.empty((0, 5), float)
    for time_value in boundary_west_time:
        boundary_west_y = np.linspace(y[0], y[1], NOP[0]).reshape(-1, 1)        
        boundary_west_x = x[0]*np.ones((NOP[0], 1))
        boundary_west_v = np.zeros((NOP[0], 1))
        boundary_west_u = np.zeros((NOP[0], 1))
        boundary_west = np.vstack([boundary_west, np.hstack([boundary_west_x, boundary_west_y, time_value*np.ones((NOP[0], 1)), boundary_west_u, boundary_west_v])])
    return boundary_west

def compute_normals(X, Y, levelset, cell_size):

    ''' Gets the coordinates of all interface cells and computes the interface normal and tangent at those points from the levelset.
    These quantities are used to distribute the cells for the interface and nearfield refinement '''

    interface = []
    normal = []
    tangent = []

    for levelset_time_snapshot in levelset:

        indices_interface_y, indices_interface_x = np.where(np.abs(levelset_time_snapshot) <= 0.75) # levelset threshold for interface cell
        indices_interface = list(zip(indices_interface_x, indices_interface_y))

        interface_x = X[indices_interface_x]
        interface_y = Y[indices_interface_y]
        interface.append(list(zip(interface_x, interface_y)))

        levelset_x = np.zeros(len(indices_interface))
        levelset_y = np.zeros(len(indices_interface))
        for k, (i_x, i_y) in enumerate(indices_interface):
            levelset_x[k] = (levelset_time_snapshot[i_y, i_x+1] - levelset_time_snapshot[i_y, i_x-1])/(2*cell_size)
            levelset_y[k] = (levelset_time_snapshot[i_y+1, i_x] - levelset_time_snapshot[i_y-1, i_x])/(2*cell_size)

        grad_abs = np.sqrt(levelset_x**2 + levelset_y**2)
        normal_x = levelset_x/grad_abs
        normal_y = levelset_y/grad_abs
        normal.append(list(zip(normal_x, normal_y)))
 
        tangent.append(list(zip(-normal_y, normal_x)))

    return interface, normal, tangent

def get_training_data(NOP_A, NOP_PDE, NOP_north, NOP_south, NOP_east, NOP_west):

    ''' Generates the training points for all losses. To distribute the points for the volume fraction and the PDE residual,
    the time snapshots from the cfd are used. Note that only a smaller subset of all time snapshots provided by the cfd are
    used to distribute the points. They are manually selected within this function.

    Args:
        NOP_A: Tuple containig amount of training points for the volume fraction for the interface refinement (index 0)
            and the rest of the domain (index 1)
        NOP_PDE: Tuple containig amount of training points for PDEs for the interface refinement (index 0)
            the nearfield refinement (index 1) and the rest of the domain (index 2)
        NOP_boundaries: Tuple containig amount of points for the boundary condition. (index 0) amount spatial points
            (index 1) amount time snapshots '''

    path = "../cfd_data/rising_bubble.h5"

    with h5py.File(path, "r") as data:
        X = np.array(data["X"])
        Y = np.array(data["Y"])
        times = np.array(data["time"])
        levelset = np.array(data["levelset"])
    

    # TIME SNAPSHOT SELECTION
    indices = np.arange(len(times))
    indices = np.sort(np.concatenate([indices[0:30:30], indices[30:100:10], indices[100::10], indices[1:3]], axis=0))
    times = times[indices]
    levelset = levelset[indices]

    # DOMAIN PARAMETERS PROVIDED BY CFD
    t_bounds = [times[0], times[-1]]
    x_bounds = [X[0], X[-1]]
    y_bounds = [Y[0], Y[-1]]
    cell_size = np.diff(X)[0]

    print("\nDistributing points for time snapshots:\n", times)
    print("Number of time snapshots: ", len(times))
    print("Time bounds: ", t_bounds, "\n")

    interface, normal, tangent = compute_normals(X, Y, levelset, cell_size)
    print("Generating points for A")
    data_A = get_points_a(NOP_A, times, X, Y, cell_size, interface, normal, tangent, levelset)
    print("Generating points for PDE")
    data_PDE = get_points_pde(NOP_PDE, times, X, Y, cell_size, interface, normal, tangent, levelset)
    print("Generating points for NSEW")
    data_north = get_north(NOP_north, x_bounds, y_bounds, t_bounds)
    data_south = get_south(NOP_south, x_bounds, y_bounds, t_bounds)
    data_east = get_east(NOP_east, x_bounds, y_bounds, t_bounds)
    data_west = get_west(NOP_west, x_bounds, y_bounds, t_bounds)
    data_nsew = np.vstack([data_north[:,0:3], data_south[:,0:3], data_east[:,0:3], data_west[:,0:3]])

    # NONDIMENSIONALIZE SPACE AND TIME
    L_ref = 0.25
    data_A[:,:3] /= L_ref
    data_PDE[:,:3] /= L_ref
    data_north[:,:3] /= L_ref
    data_south[:,:3] /= L_ref
    data_east[:,:3] /= L_ref
    data_west[:,:3] /= L_ref
    data_nsew[:,:3] /= L_ref
    
    # PLOTTING SOME TIME SNAPSHOTS
    for i in range(0, len(times), math.ceil(len(times)/3)):
        fig, ax = plt.subplots(1,3)
        fig.set_size_inches(15,12,True)
        ax[0].scatter(data_A[i*sum(NOP_A):(i+1)*sum(NOP_A),0], data_A[i*sum(NOP_A):(i+1)*sum(NOP_A),1], c=data_A[i*sum(NOP_A):(i+1)*sum(NOP_A),3], s=1)
        ax[1].scatter(data_PDE[i*sum(NOP_PDE):(i+1)*sum(NOP_PDE),0], data_PDE[i*sum(NOP_PDE):(i+1)*sum(NOP_PDE), 1], s=1)
        ax[2].scatter(data_A[i*sum(NOP_A):i*sum(NOP_A)+NOP_A[0],0], data_A[i*sum(NOP_A):i*sum(NOP_A)+NOP_A[0],1], c=data_A[i*sum(NOP_A):i*sum(NOP_A)+NOP_A[0],3], s=1)
        ax[2].scatter(data_PDE[i*sum(NOP_PDE):i*sum(NOP_PDE)+NOP_PDE[0],0], data_PDE[i*sum(NOP_PDE):i*sum(NOP_PDE)+NOP_PDE[0], 1], s=1)
        ax[0].scatter(data_north[i*NOP_north[0]:(i+1)*NOP_north[0],0], data_north[i*NOP_north[0]:(i+1)*NOP_north[0],1], s=1, color="red")
        ax[0].scatter(data_south[i*NOP_south[0]:(i+1)*NOP_south[0],0], data_south[i*NOP_south[0]:(i+1)*NOP_south[0],1], s=1, color="red")
        ax[0].scatter(data_east[i*NOP_east[0]:(i+1)*NOP_east[0],0], data_east[i*NOP_east[0]:(i+1)*NOP_east[0],1], s=1, color="red")
        ax[0].scatter(data_west[i*NOP_west[0]:(i+1)*NOP_west[0],0], data_west[i*NOP_west[0]:(i+1)*NOP_west[0],1], s=1, color="red")
        ax[0].set_aspect("equal")
        ax[1].set_aspect("equal")
        ax[2].set_aspect("equal")
        # plt.show()
    
    print("Assembling data frames\n")
    data_EW = pd.DataFrame(data=np.hstack([data_east[:,0:2], data_west[:,0:3]]), columns=["x_E", "y_E", "x_W", "y_W", "t_EW"])
    data_N = pd.DataFrame(data=data_north[:,[0,1,2,5]], columns=["x_N", "y_N", "t_N", "p_N"])
    data_NSEW = pd.DataFrame(data=np.vstack([data_north[:,0:5], data_south, data_east[:NOP_east[0]], data_west[:NOP_west[0]]]), columns=["x_NSEW", "y_NSEW", "t_NSEW", "u_NSEW", "v_NSEW"])
    data_A = pd.DataFrame(data=data_A, columns=["x_A", "y_A", "t_A", "a_A"])
    data_PDE = pd.DataFrame(data=np.vstack([data_PDE, np.hstack([data_nsew, np.zeros((len(data_nsew), 1))])]), columns=["x_PDE", "y_PDE", "t_PDE", "f_PDE"])
    return dict(A=data_A, PDE=data_PDE, N=data_N, EW=data_EW, NSEW=data_NSEW)
