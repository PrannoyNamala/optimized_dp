import imp
import numpy as np
# Utility functions to initialize the problem
from odp.Grid import Grid
from odp.Shapes import *

# Specify the  file that includes dynamic systems
from odp.dynamics import PursuitEvasion1v1
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
    collision_dist = np.sqrt((g.vs[0] - g.vs[2]) ** 2 + (g.vs[1] - g.vs[3]) ** 2) - my_car.collison_R

    ### Evader to boundary and obstacle
    # Initialize the sdf_array with the desired shape
    sdf_array = np.zeros((grid_size, grid_size, grid_size, grid_size))

    # Retrieve grid points for dimensions 2 and 3
    k_vals = g.vs[2][0, 0, :, 0]  # Shape: (grid_size,)
    l_vals = g.vs[3][0, 0, 0, :]  # Shape: (grid_size,)

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
    sdf_array = np.minimum(boundary_distance[None, None, :, :], signed_distances[None, None, :, :])

    # Expand and broadcast to fit the desired shape (grid_size, grid_size, grid_size, grid_size)
    sdf_array = np.broadcast_to(sdf_array, (grid_size, grid_size, grid_size, grid_size))

    goal = Union(sdf_array, collision_dist)

    #####################

    ###### Avoid set ######
    # Initialize the sdf_array with the desired shape
    obstacle = np.zeros((grid_size, grid_size, grid_size, grid_size))

    # Extract grid values for dimensions 0 and 1
    i_vals = g.vs[0][:, 0, 0, 0]  # Shape: (grid_size,)
    j_vals = g.vs[1][0, :, 0, 0]  # Shape: (grid_size,)

    # Calculate boundary distances for dimension 0 (i-axis)
    dist_bdry_l_e = i_vals - (-1)
    dist_bdry_r_e = 1 - i_vals

    # Calculate boundary distances for dimension 1 (j-axis)
    dist_bdry_up_e = 1 - j_vals
    dist_bdry_dn_e = j_vals - (-1)

    # Calculate the minimum boundary distance for each (i, j) location and subtract 0.1
    bdry_distances = np.minimum(
        np.minimum(dist_bdry_l_e[:, None], dist_bdry_r_e[:, None]),
        np.minimum(dist_bdry_up_e[None, :], dist_bdry_dn_e[None, :])
    ) - my_car.pursuer_R

    # Calculate signed distances using the `signed_distance_obs` function, vectorized over (i, j) grid
    # signed_distances = signed_distance_obs(i_vals[:, None], j_vals[None, :], my_car.pursuer_R, my_car.obs_width, my_car.obs_length)
    signed_distances = signed_distance_circle(i_vals[:, None], j_vals[None, :], collision_R=my_car.pursuer_R, radius=my_car.obs_width)
    
    # Combine both results by taking the maximum between -bdry_distances and -signed_distances
    sdf_combined = np.minimum(bdry_distances, signed_distances)

    # Expand and broadcast to match the full (grid_size, grid_size, grid_size, grid_size) shape
    obstacle = sdf_combined[:, :, None, None]
    obstacle = np.broadcast_to(obstacle, (grid_size, grid_size, grid_size, grid_size))

    return goal, obstacle

grid_size  = 40

g = Grid(np.array([-1.0, -1.0, -1.0, -1.0]), np.array([1.0, 1.0, 1.0, 1.0]), 4, np.array([grid_size, grid_size, grid_size, grid_size]), [])
# print(g.vs[2][0,0,24,0], g.vs[3][0,0,0,11])
# raise SystemExit

my_car = PursuitEvasion1v1(pursuer_R = 0.05, evader_R = 0.05)

goal, obstacle = goal_obs(g, my_car, grid_size)

# Look-back length and time step
lookback_length = 10
t_step = 0.025

small_number = 1e-5
tau = np.arange(start=0, stop=lookback_length + small_number, step=t_step)

# while plotting make sure the len(slicesCut) + len(plotDims) = grid.dims
po = PlotOptions(do_plot=True, plot_type="set", plotDims=[0, 1], slicesCut=[5, 5], 
                 colorscale="Hot", save_fig=True, filename="eg1v1", interactive_html=True)

# Define initial 2D matrix size and center
size = 40
center = size // 2

# Initialize a 4D matrix with zeros
matrix_4d = np.zeros((size, size, size, size))

# Define square boundaries
outer_radius = 2
inner_radius = 0.4

# Populate the first 2D slice (first two dimensions) of the 4D matrix
for i in range(size):
    for j in range(size):
        # Calculate the Manhattan distance from the center
        distance = max(abs(i - center), abs(j - center))
        
        # Assign values based on distance in the first 2D slice
        if distance < inner_radius:
            matrix_4d[i, j, :, :] = 1  # Inside the inner square
        elif inner_radius <= distance < outer_radius:
            matrix_4d[i, j, :, :] = -1  # Between the inner and outer squares
        elif distance == outer_radius:
            matrix_4d[i, j, :, :] = 0  # Outer square boundary
        elif distance > outer_radius:
            matrix_4d[i, j, :, :] = 1  # Outside the outer square boundary


# plot_isosurface(g, matrix_4d, po)

compMethods = { "TargetSetMode": "minVWithVTarget",
                "ObstacleSetMode": "maxVWithObstacle"}
# HJSolver(dynamics object, grid, initial value function, time length, system objectives, plotting options)
result = HJSolver(my_car, g, [goal, obstacle], tau, compMethods, po, saveAllTimeSteps=False)

print(result.shape)

np.save(f'1v1pe_g_{grid_size}_speed{my_car.vel_e}_circle{my_car.obs_width}.npy', result)  # grid = grid_size


