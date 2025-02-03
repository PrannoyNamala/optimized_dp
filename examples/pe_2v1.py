import imp
import numpy as np
# Utility functions to initialize the problem
from odp.Grid import Grid
from odp.Shapes import *

# Specify the  file that includes dynamic systems
from odp.dynamics import PursuitEvasion2v1
# Plot options
from odp.Plots import PlotOptions
from odp.Plots import plot_isosurface, plot_valuefunction

# Solver core
from odp.solver import HJSolver, computeSpatDerivArray

import math

def signed_distance_obs(x, y, collision_R=0.2, width=0.4, length=0.4):
    """
    Compute the signed distance from a point (x, y) to a rectangle obstacle centered at (0, 0)
    using torch tensors, with specified width and length, considering the collision radius.
    
    Parameters:
    x, y: torch tensor coordinates of the point
    collision_R: radius of the point
    width: width of the rectangle (default is 0.4)
    length: length of the rectangle (default is 0.6)

    Returns:
    torch tensor: Signed distance
    """
    half_width = width / 2
    half_length = length / 2
    adjusted_width = half_width + collision_R
    adjusted_length = half_length + collision_R

    # Compute outside distances
    dx = np.maximum(np.abs(x) - adjusted_width, 0.0)
    dy = np.maximum(np.abs(y) - adjusted_length, 0.0)
    outside_distance = np.sqrt(dx**2 + dy**2)

    # Compute inside distances
    inside_distance = np.minimum(np.maximum(np.abs(x) - adjusted_width, np.abs(y) - adjusted_length), 0.0)

    # Combine outside and inside distances
    return outside_distance + inside_distance

def signed_distance_circle(x, y, center_x=0.0, center_y=0.0, radius=0.5, collision_R=0.2):
    """
    Compute the signed distance from a point (x, y) to a circle with a given radius and center,
    while considering a collision radius.

    Parameters:
    x, y: numpy array or float, coordinates of the point
    center_x, center_y: float, coordinates of the circle's center (default is (0, 0))
    radius: float, radius of the circle (default is 0.5)
    collision_R: float, radius of the point (default is 0.2)

    Returns:
    float or numpy array: Signed distance
    """
    # Compute the Euclidean distance to the circle's center
    distance_to_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)

    # Compute the signed distance
    signed_distance = distance_to_center - (radius + collision_R)

    return signed_distance

def goal_obs(g, my_car, grid_size):
    ###### Reachable set ######
    ### Collision inter agent
    # Calculate distances in a vectorized way
    dist_1_e = np.sqrt((g.vs[0] - g.vs[4]) ** 2 + (g.vs[1] - g.vs[5]) ** 2)
    dist_2_e = np.sqrt((g.vs[2] - g.vs[4]) ** 2 + (g.vs[3] - g.vs[5]) ** 2)

    # Find the minimum distance and assign to collision_dist
    collision_dist = np.minimum(dist_1_e, dist_2_e) - my_car.collison_R

    # Initialize the sdf_array with the desired shape
    sdf_array = np.zeros((grid_size, grid_size, grid_size, grid_size, grid_size, grid_size))

    # Retrieve grid points for dimensions 5 and 6
    k_vals = g.vs[4][0, 0, 0, 0, :, 0]  # Shape: (grid_size,)
    l_vals = g.vs[5][0, 0, 0, 0, 0, :]  # Shape: (grid_size,)

    # Calculate boundary distances for dimension 2
    dist_bdry_l_e = k_vals - (-1)
    dist_bdry_r_e = 1 - k_vals

    # Calculate boundary distances for dimension 3
    dist_bdry_up_e = 1 - l_vals
    dist_bdry_dn_e = l_vals - (-1)

    # Calculate the minimum boundary distance at each (k, l) location
    boundary_distance = np.minimum(
        np.minimum(dist_bdry_l_e[:, None], dist_bdry_r_e[:, None]),
        np.minimum(dist_bdry_up_e[None, :], dist_bdry_dn_e[None, :])
    ) - my_car.evader_R

    # Calculate signed distance using the provided function, vectorized over (k, l) grid
    # signed_distances = signed_distance_obs(k_vals[:, None], l_vals[None, :], my_car.evader_R, my_car.obs_width, my_car.obs_length)
    signed_distances = signed_distance_circle(k_vals[:, None], l_vals[None, :], collision_R=my_car.evader_R, radius=my_car.obs_width)

    # Combine both arrays by broadcasting to match the shape (grid_size, grid_size, grid_size, grid_size)
    sdf_array = np.minimum(boundary_distance[None, None, None, None, :, :], signed_distances[None, None, None, None, :, :])

    # Expand and broadcast to fit the desired shape (grid_size, grid_size, grid_size, grid_size)
    sdf_array = np.broadcast_to(sdf_array, (grid_size, grid_size, grid_size, grid_size, grid_size, grid_size))

    goal = np.minimum(collision_dist, sdf_array)

    #####################

    ###### Avoid set ######
    # Calculate dist_p1_p2 using vectorized operations
    dist_p1_p2 = np.sqrt((g.vs[0] - g.vs[2]) ** 2 + (g.vs[1] - g.vs[3]) ** 2) - my_car.collison_R

    # Calculate boundary distances for p1
    dist_bdry_l_p1 = g.vs[0] - (-1)
    dist_bdry_r_p1 = 1 - g.vs[1]
    dist_bdry_up_p1 = 1 - g.vs[0]
    dist_bdry_dn_p1 = g.vs[1] - (-1)
    dist_b_p1 = np.minimum(
        np.minimum(dist_bdry_l_p1, dist_bdry_r_p1),
        np.minimum(dist_bdry_up_p1, dist_bdry_dn_p1)
    )

    # Calculate boundary distances for p2
    dist_bdry_l_p2 = g.vs[2] - (-1)
    dist_bdry_r_p2 = 1 - g.vs[3]
    dist_bdry_up_p2 = 1 - g.vs[2]
    dist_bdry_dn_p2 = g.vs[3] - (-1)
    dist_b_p2 = np.minimum(
        np.minimum(dist_bdry_l_p2, dist_bdry_r_p2),
        np.minimum(dist_bdry_up_p2, dist_bdry_dn_p2)
    )

    # Calculate the minimum boundary distance
    dist_b = np.minimum(dist_b_p1, dist_b_p2) - my_car.pursuer_R

    # Calculate the obstacle distance using signed_distance_obs function in a vectorized way
    # dist_obs_p1 = signed_distance_obs(g.vs[0], g.vs[1])
    # dist_obs_p2 = signed_distance_obs(g.vs[2], g.vs[3])
    dist_obs_p1 = signed_distance_circle(g.vs[0], g.vs[1], collision_R=my_car.evader_R, radius=my_car.obs_width)
    dist_obs_p2 = signed_distance_circle(g.vs[0], g.vs[1], collision_R=my_car.evader_R, radius=my_car.obs_width)
    dist_obs = np.minimum(dist_obs_p1, dist_obs_p2) - my_car.pursuer_R

    # Calculate the minimum of all distances and store in collision_dist with the correct shape
    obstacle = np.minimum(dist_p1_p2, np.minimum(dist_b, dist_obs))

    obstacle = np.broadcast_to(obstacle, (grid_size, grid_size, grid_size, grid_size, grid_size, grid_size))

    return goal, obstacle

grid_size  = 20

g = Grid(np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0]), np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]), 6, np.array([grid_size, grid_size, grid_size, grid_size, grid_size, grid_size]), [])

my_car = PursuitEvasion2v1(pursuer_R = 0.05, evader_R = 0.05)

goal, obstacle = goal_obs(g, my_car, grid_size)

# Look-back length and time step
lookback_length = 10
t_step = 0.025

small_number = 1e-5
tau = np.arange(start=0, stop=lookback_length + small_number, step=t_step)

# while plotting make sure the len(slicesCut) + len(plotDims) = grid.dims
po = PlotOptions(do_plot=True, plot_type="set", plotDims=[0, 1], slicesCut=[5, 5, 10, 10], 
                 colorscale="Hot", save_fig=True, filename="eg2v1", interactive_html=True)


compMethods = { "TargetSetMode": "minVWithVTarget",
                "ObstacleSetMode": "maxVWithObstacle"}
# HJSolver(dynamics object, grid, initial value function, time length, system objectives, plotting options)
result = HJSolver(my_car, g, [goal, obstacle], tau, compMethods, po, saveAllTimeSteps=False)

print(result.shape)

np.save(f'2v1pe_g_{grid_size}_speed{my_car.vel_e}_circle{my_car.obs_width}.npy', result)  # grid = grid_size
