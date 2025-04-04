import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
import numpy as np
import nibabel as nib
from scipy.ndimage import binary_dilation, generate_binary_structure
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
import plotly.colors as pc
import matplotlib.pyplot as plt
from tabulate import tabulate
from scipy.spatial import ConvexHull  # pip install -U concave_hull

from concave_hull import (  # noqa: F401
    concave_hull,
    concave_hull_indexes,
    convex_hull_indexes,
)

def run_model(mask,verbose=False):
    
    if verbose:
        print(f"Verbose was set to {verbose}, printing progress statements...")

    def count_valid_pairs(pairs, neck_width_tolerance=0.07):
        """
        Counts the number of (angle, length) pairs where the length is within
        neck_width_tolerance of the smallest length in the sorted list.

        :param pairs: List of (angle, length) tuples sorted by length.
        :param neck_width_tolerance: Tolerance factor (default is 0.07 or 7%).
        :return: Number of pairs satisfying the condition.
        """
        if not pairs:
            return 0

        min_length = pairs[0][1]  # The first entry has the smallest length
        threshold = min_length * (1 + neck_width_tolerance)

        count = sum(1 for _, length in pairs if length <= threshold)
        return count
    
    # Function to project points onto a 1D line at a given angle
    def project_onto_line(points, angle):
        # Convert angle to radians
        angle_rad = np.radians(angle)

        # Define the unit vector for the projection direction
        direction = np.array([np.cos(angle_rad), np.sin(angle_rad)])

        # Project each point of the concave hull onto the direction (dot product)
        projection = np.dot(points, direction)

        return projection
    
    # Find valid plane length
    def find_plane_endpoints(center, direction, image_shape):
        return min(min(abs((0 - center[i]) / direction[i]), abs((image_shape[i] - center[i]) / direction[i])) if direction[i] != 0 else float('inf') for i in range(3))

    # Generate points on the plane
    def generate_plane_points(left_start, left_end, right_start, right_end, num_points=100):
        # Create a grid of points on the plane using the two directions
        vector1 = left_end - left_start
        vector2 = right_start - left_start
        points = []
        for i in np.linspace(0, 1, num_points):
            for j in np.linspace(0, 1, num_points):
                point = left_start + i * vector1 + j * vector2
                points.append(point)
        return np.array(points)

    # Count red pixels (vessel or aneurysm points)
    def count_red_pixels_on_plane(points, seg_data):
        count = 0
        for point in points:
            # Round the point to nearest voxel indices (ensure they are within bounds)
            x, y, z = np.round(point).astype(int)
            if 0 <= x < seg_data.shape[0] and 0 <= y < seg_data.shape[1] and 0 <= z < seg_data.shape[2]:
                if seg_data[x, y, z] == 1:  # Assuming '1' corresponds to the red pixels (vessel)
                    count += 1
        return count
    

    #=================================================================================================================================
    #RUNNING THE ACTUAL MODEL
    #=================================================================================================================================
    
    matrix_corr_mask = mask
    image_shape = mask.shape  # Image dimensions

    # Keep only the '2' values
    matrix_aneurysm = np.zeros_like(matrix_corr_mask)
    matrix_aneurysm[matrix_corr_mask == 2] = 2
    
    aneurysm_size = np.sum(matrix_corr_mask == 2)

    matrix_vessels = np.zeros_like(matrix_corr_mask)
    matrix_vessels[matrix_corr_mask == 1] = 1
    

    # Define 26-connectivity structure in 3D (one pixel away in any direction)
    structure = generate_binary_structure(3, 3)  # 3D, full 26-neighborhood

    # Find coordinates for aneurysm points (value = 2)
    x, y, z = np.where(matrix_corr_mask == 2)

    # Find coordinates for correlation mask points (value = 1)
    x1, y1, z1 = np.where(matrix_corr_mask == 1)

    # Create a mask of neighboring points that touch value = 1
    dilated_mask = binary_dilation(matrix_corr_mask == 1, structure=structure)

    if verbose:
        print("=============================Mask analysis=====================================")
        print(f"Unique values in the aneurysm matrix: ",np.unique(matrix_aneurysm))
        print(f"Unique values in the corrected mask matrix: ",np.unique(matrix_corr_mask))
        print(f"Unique values in the surface boundary green matrix: ",np.unique(matrix_vessels))
        print("Unique values in dilated_mask:", np.unique(dilated_mask))

    # Find aneurysm points (value = 2) that have at least one 1-neighbor
    green_mask = (matrix_corr_mask == 2) & dilated_mask
    x_green, y_green, z_green = np.where(green_mask)

    #Create matrix with surface points on the aneurysm-vessel boundary
    surface_matrix = np.zeros_like(matrix_aneurysm)
    surface_matrix = green_mask

    # Get surface points
    surface_points = np.argwhere(surface_matrix == 1)
    #print(surface_points)

    # Compute PCA
    pca = PCA(n_components=3)
    pca.fit(surface_points)

    # The normal is the eigenvector corresponding to the smallest eigenvalue
    normal_vector = pca.components_[-1]  # Smallest component gives normal

    # Ensure it's a unit vector
    normal_vector /= np.linalg.norm(normal_vector)
    #print('Normal vector Ina: ',normal_vector)

    # Choose a point to originate the normal (e.g., mean of surface)
    origin = np.mean(surface_points, axis=0)
    #print('Origin Ina: ',origin)

    # Step 1: Compute local coordinate system
    normal_vector /= np.linalg.norm(normal_vector)  # Normalize normal  #TODO: no need again?

    # Pick a reference vector for basis computation   #TODO: no need?
    if abs(normal_vector[0]) < 0.9:
        reference_vector = np.array([1, 0, 0])
    else:
        reference_vector = np.array([0, 1, 0])

    # Compute basis vectors
    basis_vector_1 = np.cross(normal_vector, reference_vector)
    basis_vector_1 /= np.linalg.norm(basis_vector_1)

    basis_vector_2 = np.cross(normal_vector, basis_vector_1)
    basis_vector_2 /= np.linalg.norm(basis_vector_2)

    # Construct the transformation matrix (Global -> Local)
    T = np.column_stack([basis_vector_1, basis_vector_2, normal_vector]).T  # Transpose to map to local
    

    # Step 2: Transform all surface points into the local coordinate system
    origin = np.mean(surface_points, axis=0)
    centered_points = surface_points - origin
    local_points = (T @ centered_points.T).T  # Transform each point to local [x', y', n']
    #print(local_points)

    # Extract x' and y' coordinates for the scatter plot
    x_prime = local_points[:, 0]  # x' coordinates
    y_prime = local_points[:, 1]  # y' coordinates

    projected_2d = local_points[:, :2]  # Extract x and y coordinates

    # Your 2D points (make sure 'projected_2d' is already defined) #TODO: fix comment
    points = np.array(projected_2d)

    # Add the first point to the end of the array to close the loop
    points = np.vstack([points, points[0]])

    # Compute Convex Hull using scipy
    convex_hull = ConvexHull(points[:, :2])  # points are already 2D (N-by-2)

    # Compute the Concave Hull and plot it
    idxes = concave_hull_indexes(
        points[:, :2],
        length_threshold=0,
        # convex_hull_indexes=convex_hull.vertices.astype(np.int32),
    )
    #print(idxes)

    # Fill the area inside the Concave Hull with black color
    concave_x = points[idxes, 0]  # X coordinates of the concave hull points
    concave_y = points[idxes, 1]  # Y coordinates of the concave hull points

    # Get the concave hull points
    concave_hull_points = points[idxes]
    #print(concave_hull_points)

    # Define angles to project the concave hull onto (0 to 180 degrees)
    angles = np.linspace(0, 360, 361)  # From 0 to 180 degrees
    #print(angles)
    projections = []
    projection_ranges = []

    # Plot projections for each angle and find the angle with the shortest projection range
    for angle in angles:
        # Project the concave hull onto the 1D line at each angle
        projection = project_onto_line(concave_hull_points, angle)

        # Calculate the range of the projection (max - min)
        projection_range = np.max(projection) - np.min(projection)
        projection_ranges.append(projection_range)

    #print(projection_ranges)
    angles_ranges = [(angles[i], projection_ranges[i]) for i in range (0, 360)]
    angles_ranges_sorted = sorted(angles_ranges, key=lambda x: x[1])
    #print('-----')
    #print(angles_ranges_sorted)

    # Find the angle with the shortest 1D projection
    shortest_projection_angle = angles[np.argmin(projection_ranges)]
    #print(shortest_projection_angle)

    # Find the angle with the longest 1D projection
    longest_projection_angle = angles[np.argmax(projection_ranges)]
    #print(longest_projection_angle)

    # Step 1: Calculate the unit vector in the 2D plane corresponding to the projection angle
    shortest_projection_angle_rad = np.deg2rad(angles_ranges_sorted[0][0])  # Convert theta to radians
    u_2d_local = np.array([np.cos(shortest_projection_angle_rad), np.sin(shortest_projection_angle_rad)])

    # Transformation matrix (local to global)
    T_inv = np.linalg.inv(T)

    # Step 2: Transform the 2D local vector back to the global coordinate system
    u_2d_global = T_inv @ np.append(u_2d_local, 0)  # Append 0 for the normal component (since u_2d_local is 2D)

    # make a list with 3D projection vectors sorted from most optimal projection vector to least optimal

    shortest_projection_angle_rad_list = np.deg2rad(angles_ranges_sorted)

    u_2d_local_list = []
    for angle in shortest_projection_angle_rad_list:
        u_2d_local_list.append(np.array([np.cos(angle[0]), np.sin(angle[0])]))
    #print(u_2d_local_list)

    u_2d_global_list = []
    for u_2d_local in u_2d_local_list:
        u_2d_global_list.append(T_inv @ np.append(u_2d_local, 0))
    #print(u_2d_global_list)

    # Compute the list with global projection directions
    projection_direction_list = []
    for u_2d_global in u_2d_global_list:
        projection_direction_list.append(np.cross(normal_vector, u_2d_global))
        projection_direction_list[-1] /= np.linalg.norm(projection_direction_list[-1])  # Normalize

    #print(projection_direction_list)

    how_many_planes = count_valid_pairs(angles_ranges_sorted)
    #print(how_many_planes)

    # Downsampling function
    def downsample_coordinates(x, y, z, factor=2):
        indices = np.arange(0, len(x), factor)
        return x[indices], y[indices], z[indices]

    # Find vessel and aneurysm coordinates
    x1, y1, z1 = downsample_coordinates(*np.where(mask == 1), factor=2)
    x2, y2, z2 = downsample_coordinates(*np.where(mask == 2), factor=2)

    #print(normal_vector)
    center_x,center_y,center_z = origin[0], origin[1], origin[2]
    #print(center_x,center_y,center_z)

    # Compute two perpendicular vectors
    v1, v2 = basis_vector_1, basis_vector_2
    #print(v1,v2)

    # Initialize an empty list to store the values
    angles_and_pixel_counts = []

    # Loop through angles_ranges_sorted
    for i, (angle, max_width) in enumerate(angles_ranges_sorted[0:how_many_planes]):

        #print(angle, max_width)
        angle_rad = np.deg2rad(angle)
        rotated_vector = np.cos(angle_rad) * v1 + np.sin(angle_rad) * v2

        # Calculate the max length of the plane
        max_length = find_plane_endpoints([center_x, center_y, center_z], rotated_vector, image_shape)

        half_width = max_width / 2
        start = np.array([center_x, center_y, center_z])

        # Calculate the left and right start points
        left_start = start + np.cos(angle_rad) * v1 * -half_width + np.sin(angle_rad) * v2 * -half_width
        right_start = start + np.cos(angle_rad) * v1 * half_width + np.sin(angle_rad) * v2 * half_width

        # Compute direction vectors and normalize
        left_vector = left_start - start
        right_vector = right_start - start
        left_vector /= np.linalg.norm(left_vector)
        right_vector /= np.linalg.norm(right_vector)

        # Adjust left and right starts to be exactly max_width apart
        left_start = start + left_vector * (-max_width / 2)
        right_start = start + right_vector * (max_width / 2)

        # Compute the perpendicular direction (cross product with normal_vector)
        perpendicular_direction = np.cross(normal_vector, left_vector)
        perpendicular_direction /= np.linalg.norm(perpendicular_direction)

        # Move the start points along the perpendicular direction
        left_start += perpendicular_direction * half_width
        right_start -= perpendicular_direction * half_width

        # Compute the end points for left and right
        left_end = left_start + rotated_vector * max_length
        right_end = right_start + rotated_vector * max_length

        # Count red pixels on the plane
        red_pixel_count = count_red_pixels_on_plane(generate_plane_points(left_start, left_end, right_start, right_end, num_points=100), seg_data=mask)

        # Add (angle, max_width, red_pixel_count) as a tuple to the list
        angles_and_pixel_counts.append((angle, max_width, red_pixel_count))

        # Output the resulting list
        
    if verbose:
        print("=============================Angle analysis=====================================")
        print(tabulate(angles_and_pixel_counts,headers=["Angle [deg]","Max_width [pixels]","Red pixel count [pixels]"],tablefmt="grid"))



    # Initialize an empty list to store the values
    both_sides_info = []
    processed = set()

    # Loop through the list of angles_and_pixel_counts
    for i, (angle, max_width, red_pixels) in enumerate(angles_and_pixel_counts):
        if angle in processed:
            continue  # Skip angles already processed

        # Find opposite angle (angle + 180 or angle - 180)
        opposite_angle = angle + 180 if any(angle + 180 == other_angle for other_angle, _, _ in angles_and_pixel_counts) else angle - 180

        # Check if opposite angle exists in angles_and_pixel_counts
        if any(opposite_angle == other_angle for other_angle, _, _ in angles_and_pixel_counts):
            # Find red pixels count for opposite angle
            opposite_red_pixels = next((other_red_pixels for other_angle, _, other_red_pixels in angles_and_pixel_counts if other_angle == opposite_angle), 0)

            total_red_pixels = red_pixels + opposite_red_pixels

            # Ensure the smaller angle is always the key
            chosen_angle = min(angle, opposite_angle)
            paired_angle = max(angle, opposite_angle)

            # Store the tuple in the list (chosen_angle, paired_angle, max_width, total_red_pixels)
            both_sides_info.append((chosen_angle, paired_angle, max_width, total_red_pixels))
            processed.add(angle)
            processed.add(opposite_angle)

    # Sort list based on the total red pixel values (the 4th element of each tuple)
    sorted_both_sides_info = sorted(both_sides_info, key=lambda x: x[3])

    # Output the sorted list
    if verbose:
        print("=============================Sorted Angles=====================================")
        print(tabulate(sorted_both_sides_info,headers=["Chosen angle [°]","Paired angle [°]","Max width [pixels]","Amount of red pixels [-]"],tablefmt="grid"))

    # Calculate the unit vector in the 2D plane corresponding to the projection angle
    optimal_projection_angle_rad = np.deg2rad(sorted_both_sides_info[0][0])  # Convert theta to radians
    optimal_u_2d_local = np.array([np.cos(optimal_projection_angle_rad), np.sin(optimal_projection_angle_rad)])

    # Transform the 2D local vector back to the global coordinate system
    optimal_u_2d_global = T_inv @ np.append(optimal_u_2d_local, 0)  # Append 0 for the normal component (since u_2d_local is 2D)

    # Find the final optimal projection direction in the global coordinate system
    optimal_projection_direction = np.cross(normal_vector, u_2d_global)
    optimal_projection_direction /= np.linalg.norm(optimal_projection_direction)  # Normalize
    
    if verbose:
        print("=============================optimal projection direction=====================================")
        print("=====",optimal_projection_direction,"=====")
        print("==============================================================================================")

    # Convert to DSA angles
    w_x, w_y, w_z = optimal_projection_direction # get the x,y,z components

    if w_y >= 0:    #fluoroscopic rotation angle rho
        rho = -np.arcsin(w_x)
    else:
        rho = np.arcsin(w_x)

    denominator = np.sqrt(1 - w_x**2)
    if w_y >= 0:  #fluoroscopic angulation angle psi
        psi = np.arcsin(w_z / denominator)
    else:
        psi = -np.arcsin(w_z / denominator)

    # Convert to degrees for easier interpretation
    rho_deg = np.degrees(rho)
    psi_deg = np.degrees(psi)

    if verbose:
        print("=============================Translated optimal projeciton angles=====================================")
        print(f"Final optimal DSA Rotation Angle (ρ): {rho_deg:.2f}°")
        print(f"Final optimal DSA Angulation Angle (ψ): {psi_deg:.2f}°")

    return rho_deg, psi_deg

