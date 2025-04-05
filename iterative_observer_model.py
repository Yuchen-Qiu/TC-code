import dash
from dash import dcc, html, Input, Output
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import numpy as np
import nibabel as nib
from scipy.ndimage import binary_dilation, generate_binary_structure
from sklearn.decomposition import PCA
from skimage.measure import marching_cubes
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.spatial import cKDTree
from collections import defaultdict
from PIL import Image
import os
import numpy as np
from matplotlib import image as mpimg
import matplotlib.pyplot as plt

def run_model(mask,input_path,showimgs=False,its=36,verbose=False):
    
    if verbose:
        print(f"Verbose was set to {verbose}, printing progress statements...")

    seg_data = mask
    
    # Get the coordinates of voxels with values 1 and 2
    x1, y1, z1 = np.where(seg_data == 1)  # Cerebral vessels
    x2, y2, z2 = np.where(seg_data == 2)  # Aneurysms

    x_max, y_max, z_max = seg_data.shape

    # Downsampling function
    def downsample_coordinates(x, y, z, factor=2):
        indices = np.arange(0, len(x), factor)
        return x[indices], y[indices], z[indices]

    # Downsample coordinates
    x1, y1, z1 = downsample_coordinates(x1, y1, z1, factor=1)
    x2, y2, z2 = downsample_coordinates(x2, y2, z2, factor=1)

    # Perform additional dilation on the connection region to make the connection surface thicker
    def find_thick_connection_region(seg_data, iterations=1):
        # Create masks for vessels and aneurysms
        vessels_mask = (seg_data == 1)
        aneurysms_mask = (seg_data == 2)
        
        # Basic dilation to get the initial connection region
        structure = generate_binary_structure(3, 1)  # 3D connectivity structure
        dilated_vessels = binary_dilation(vessels_mask, structure=structure)
        dilated_aneurysms = binary_dilation(aneurysms_mask, structure=structure)
        connection_region = np.logical_and(dilated_vessels, dilated_aneurysms)
        
        x, y, z = np.where(connection_region)
        return x, y, z

    # Use the further dilated connection region
    x, y, z = find_thick_connection_region(seg_data, iterations=1)

    # Calculate the normal vector of the connected region
    def calculate_normal_vector(x, y, z):
        # Combine the coordinates into a set of points
        points = np.vstack((x, y, z)).T
        
        # Use PCA to fit a plane
        pca = PCA(n_components=3)
        pca.fit(points)
        
        # The normal vector is the eigenvector corresponding to the smallest eigenvalue of PCA
        normal_vector = pca.components_[2]
        return normal_vector

    # Calculate the normal vector and center point
    if len(x) > 0:
        normal_vector = calculate_normal_vector(x, y, z)
        center_x, center_y, center_z = np.mean(x), np.mean(y), np.mean(z)
    else:
        normal_vector = np.array([0, 0, 1])  # Default normal vector
        center_x, center_y, center_z = 0, 0, 0  # Default center point

    # Generate mesh for cerebral vessels and aneurysms
    def generate_mesh(seg_data, label):
        mask = (seg_data == label)
        if np.any(mask):
            verts, faces, _, _ = marching_cubes(mask, level=0.5)
            return verts, faces
        return None, None

    # Generate meshes for vessels (label 1) and aneurysms (label 2)
    verts1, faces1 = generate_mesh(seg_data, 1)  # Cerebral vessels
    verts2, faces2 = generate_mesh(seg_data, 2)  # Aneurysms

    # Generate mesh for the (thickened) connected region
    def generate_connection_mesh(x, y, z):
        if len(x) > 0:
            # Create a binary mask for the connected region
            connection_mask = np.zeros_like(seg_data, dtype=bool)
            connection_mask[x, y, z] = True
            
            # Generate mesh using marching cubes
            verts, faces, _, _ = marching_cubes(connection_mask, level=0.5)
            return verts, faces
        return None, None

    # Generate mesh for the connected region
    verts_conn, faces_conn = generate_connection_mesh(x, y, z)

    # Calculate the two basis vectors of the orthogonal plane
    def calculate_orthogonal_plane(normal_vector):
        # Choose a vector that is not parallel to the normal vector
        if normal_vector[0] != 0 or normal_vector[1] != 0:
            base_vector = np.array([0, 0, 1])
        else:
            base_vector = np.array([1, 0, 0])
        
        # Calculate the first orthogonal vector
        u = np.cross(normal_vector, base_vector)
        u /= np.linalg.norm(u)
        
        # Calculate the second orthogonal vector
        v = np.cross(normal_vector, u)
        v /= np.linalg.norm(v)
        
        return u, v

    # Calculate the two basis vectors of the orthogonal plane
    u, v = calculate_orthogonal_plane(normal_vector)

    # Generate points on the plane
    plane_size = 50  # Size of the plane
    plane_points = np.array([[center_x + u[0] * i + v[0] * j,
                              center_y + u[1] * i + v[1] * j,
                              center_z + u[2] * i + v[2] * j]
                             for i in range(-plane_size, plane_size + 1, 10)
                             for j in range(-plane_size, plane_size + 1, 10)])

    # Generate plane indices for Mesh3d
    def generate_plane_indices(plane_size):
        indices = []
        for i in range(plane_size * 2):
            for j in range(plane_size * 2):
                indices.append([i, i + 1, j])
                indices.append([i + 1, j, j + 1])
        return np.array(indices)

    plane_indices = generate_plane_indices(plane_size)

    # Modified camera view calculation - camera is on the orthogonal plane facing toward the center
    def calculate_camera_view(normal_vector, angle, zoom_factor=2):
        angle_rad = np.deg2rad(angle)
        normal_vector = normal_vector / np.linalg.norm(normal_vector)
        u, v = calculate_orthogonal_plane(normal_vector)
        
        radius = 512 # Distance from the camera to the center
        plane_center=np.array([center_x, center_y, center_z])
        eye_pos = plane_center + radius * (np.cos(angle_rad) * u + np.sin(angle_rad) * v)

        # Project to the plane to eliminate numerical errors 
        projection_distance = np.dot(eye_pos - np.array([center_x, center_y, center_z]), normal_vector) 
        eye_pos = eye_pos - projection_distance * (normal_vector) 
        if projection_distance> 1e-6: 
            eye_pos = plane_center + (eye_pos - plane_center) - np.dot(eye_pos - plane_center, normal_vector) * normal_vector 

        # Set camera parameters 
        camera_eye = dict(x=float(eye_pos[0]), y=float(eye_pos[1]), z=float(eye_pos[2])) 
        camera_center = dict(x=float(center_x), y=float(center_y), z=float(center_z)) 
        camera_up = dict(x=float(normal_vector[0]), y=float(normal_vector[1]), z=float(normal_vector[2])) 
        
        return camera_eye, camera_center, camera_up


    # Project 3D points to 2D
    def project_to_2d(x, y, z, camera_eye, camera_center, camera_up):
        camera_eye = np.array([camera_eye['x'], camera_eye['y'], camera_eye['z']])
        camera_center = np.array([camera_center['x'], camera_center['y'], camera_center['z']])
        camera_up = np.array([camera_up['x'], camera_up['y'], camera_up['z']])

        # Calculate the view direction (from eye to center)
        view_dir = camera_eye - camera_center
        view_dir = view_dir / np.linalg.norm(view_dir)

        # Calculate the right vector (orthogonal to camera_up and view_dir)
        right = np.cross(camera_up, view_dir)
        right = right / np.linalg.norm(right)

        # Recalculate the up vector to ensure orthogonality
        up = np.cross(view_dir, right)
        up = up / np.linalg.norm(up)

        # Convert to camera coordinate system
        points = np.column_stack([x, y, z])
        translated_points = points - camera_eye

        # Calculate the projected coordinates
        projected_x = np.dot(translated_points, right)
        projected_y = np.dot(translated_points, up)
        depth = np.dot(translated_points, view_dir) # Keep depth information

        # Calculate the projected coordinates of the center point (should be close to (0,0))
        center_proj_x = np.dot(camera_center - camera_eye, right)
        center_proj_y = np.dot(camera_center - camera_eye, up)

        # Translate all points so that the center point is (0,0)
        projected_x -= center_proj_x
        projected_y -= center_proj_y

        # Handle NaN values
        projected_x = np.nan_to_num(projected_x, nan=0)
        projected_y = np.nan_to_num(projected_y, nan=0)
        depth = np.nan_to_num(depth, nan=0)

        return projected_x, projected_y, depth # Return depth information
    
    def convert_angle_to_DSA(projection_vector):
            # Convert to DSA angles
        w_x, w_y, w_z = projection_vector # get the x,y,z components

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

        return rho_deg, psi_deg
     
    def calculate_azimuth_elevation(camera_eye, camera_center):
        # Calculate vector from camera to center
        direction = np.array([camera_center['x'] - camera_eye['x'],
                              camera_center['y'] - camera_eye['y'],
                              camera_center['z'] - camera_eye['z']])
        direction = direction / np.linalg.norm(direction)
        
        azimuth = np.arctan2(direction[1], direction[0]) * 180 / np.pi
        elevation = np.arctan2(direction[2], np.sqrt(direction[0]**2 + direction[1]**2)) * 180 / np.pi
        return azimuth, elevation


    def save_2d_projection_matplotlib(angle, azimuth, elevation, camera_eye, save_path):
        # Generate the figures - now expecting 4 return values
        fig_3d, fig_2d_overlap, fig_2d_no_overlap, _ = update_plots(angle, ['vessels', 'aneurysms', 'connection', 'plane'])
        
        # Rest of the function remains the same...
        # Extract projection data for the overlap plot (first trace in fig_2d_overlap)
        data_overlap = fig_2d_overlap.data[0]  # The first trace is the projection data with overlap
        x_overlap = data_overlap['x']
        y_overlap = data_overlap['y']
        colors_overlap = data_overlap['marker']['color']
                

        # Create a matplotlib image for the overlap plot
        plt.figure(figsize=(8, 8),facecolor='white')  # Increased figure size for clarity
        plt.scatter(x_overlap, y_overlap, c=colors_overlap, s=8, alpha=1.0)  # Fully opaque
        plt.gca().set_aspect('equal', adjustable='box')  # Maintain aspect ratio

        fixed_range = 120 
        plt.xlim(-fixed_range, fixed_range)
        plt.ylim(-fixed_range, fixed_range)

        plt.axis('off')  # Disable axes
        plt.tight_layout()

        # Save the overlap plot
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        filename = f"2d_projection_overlap_angle_{angle}.png"
        plt.savefig(os.path.join(save_path, filename), bbox_inches='tight', pad_inches=0, dpi=700,facecolor='white')
        plt.close()

        # Extract projection data for the no-overlap plot (second trace in fig_2d_no_overlap)
        data_no_overlap = fig_2d_no_overlap.data[0]  # The first trace is the projection data without overlap
        x_no_overlap = data_no_overlap['x']
        y_no_overlap = data_no_overlap['y']
        colors_no_overlap = data_no_overlap['marker']['color']
        
        # Create a matplotlib image for the no-overlap plot
        plt.figure(figsize=(8, 8))  # Increased figure size for clarity
        plt.scatter(x_no_overlap, y_no_overlap, c=colors_no_overlap, s=8, alpha=1.0)  # Fully opaque
        plt.gca().set_aspect('equal', adjustable='box')  # Maintain aspect ratio

        # Ensure axis limits are equal to prevent stretching
        plt.xlim(-fixed_range, fixed_range)
        plt.ylim(-fixed_range, fixed_range)
        plt.axis('off')  # Disable axes
        plt.tight_layout()

        # Save the no-overlap plot
        filename_no_overlap = f"2d_projection_no_overlap_angle_{angle}.png"
        plt.savefig(os.path.join(save_path, filename_no_overlap), bbox_inches=None, pad_inches=0, dpi=700, facecolor='white')
        plt.close()

        # Save camera information to text file
        camera_info = {
            'angle': angle,
            'azimuth': azimuth,
            'elevation': elevation,
            'camera_eye_x': camera_eye['x'],
            'camera_eye_y': camera_eye['y'],
            'camera_eye_z': camera_eye['z']
        }
        
        # Write to text file
        info_file_path = os.path.join(save_path, "camera_info.txt")
        with open(info_file_path, 'a') as f:
            f.write(f"Angle: {angle}°, Azimuth: {azimuth:.2f}°, Elevation: {elevation:.2f}°, "
                    f"Camera Eye: ({camera_eye['x']:.2f}, {camera_eye['y']:.2f}, {camera_eye['z']:.2f})\n")
        
        if verbose:
            print(f"Angle: {angle}°, Azimuth: {azimuth:.2f}°, Elevation: {elevation:.2f}°, "
                        f"Camera Eye: ({camera_eye['x']:.2f}, {camera_eye['y']:.2f}, {camera_eye['z']:.2f})\n")

    def update_plots(angle, visible_structures):
        fig_3d = go.Figure()
        fig_2d_overlap = go.Figure()
        fig_2d_no_overlap = go.Figure()

        # Calculate the radius of the connection region
        if len(x) > 0:
            # Calculate distances from center to all connection points
            connection_points = np.column_stack([x, y, z])
            center_point = np.array([center_x, center_y, center_z])
            distances = np.linalg.norm(connection_points - center_point, axis=1)
            max_radius = np.max(distances)
        else:
            max_radius = 0

        # Add cerebral vessels
        if 'vessels' in visible_structures and verts1 is not None and faces1 is not None:
            fig_3d.add_trace(go.Mesh3d(
                x=verts1[:, 0], y=verts1[:, 1], z=verts1[:, 2],
                i=faces1[:, 0], j=faces1[:, 1], k=faces1[:, 2],
                color='red',
                opacity=0.5,
                name="Cerebral Vessels (Label 1)"
            ))

        # Add aneurysms
        if 'aneurysms' in visible_structures and verts2 is not None and faces2 is not None:
            fig_3d.add_trace(go.Mesh3d(
                x=verts2[:, 0], y=verts2[:, 1], z=verts2[:, 2],
                i=faces2[:, 0], j=faces2[:, 1], k=faces2[:, 2],
                color='blue',
                opacity=0.5,
                name="Aneurysms (Label 2)"
            ))

        # Add connection region
        if 'connection' in visible_structures and verts_conn is not None and faces_conn is not None:
            fig_3d.add_trace(go.Mesh3d(
                x=verts_conn[:, 0], y=verts_conn[:, 1], z=verts_conn[:, 2],
                i=faces_conn[:, 0], j=faces_conn[:, 1], k=faces_conn[:, 2],
                color='green',
                opacity=0.5,
                name="Connection Region"
            ))

        # Add spherical area display (only if connected areas exist)
        if len(x) > 0 and 'plane' in visible_structures:
            # Create a spherical mesh
            phi, theta = np.mgrid[0:np.pi:20j, 0:2*np.pi:20j]
            sphere_radius = max_radius * 1.1
            sphere_x = center_x + sphere_radius * np.sin(phi) * np.cos(theta)
            sphere_y = center_y + sphere_radius * np.sin(phi) * np.sin(theta)
            sphere_z = center_z + sphere_radius * np.cos(phi)

            fig_3d.add_trace(go.Surface(
                x=sphere_x,
                y=sphere_y,
                z=sphere_z,
                opacity=0.3, # Set transparency
                colorscale='Blues',
                showscale=False,
                name="Ignore area (sphere)",
                hoverinfo='name'
            ))
            
        # Add normal vector
        if len(x) > 0:
            start_point = np.array([center_x, center_y, center_z])
            end_point = start_point + normal_vector * 50
            fig_3d.add_trace(go.Scatter3d(
                x=[start_point[0], end_point[0]],
                y=[start_point[1], end_point[1]],
                z=[start_point[2], end_point[2]],
                mode='lines',
                line=dict(color='purple', width=3),
                name="Normal Vector"
            ))

        # Calculate camera view
        camera_eye, camera_center, camera_up = calculate_camera_view(normal_vector, angle, zoom_factor=2)

        # Project points to 2D
        projected_x1, projected_y1, depth1 = project_to_2d(x1, y1, z1, camera_eye, camera_center, camera_up)
        projected_x2, projected_y2, depth2 = project_to_2d(x2, y2, z2, camera_eye, camera_center, camera_up)

        if verts_conn is not None:
            projected_x_conn, projected_y_conn, depth_conn = project_to_2d(verts_conn[:, 0], verts_conn[:, 1], verts_conn[:, 2], camera_eye, camera_center, camera_up)
        else:
            projected_x_conn, projected_y_conn, depth_conn = np.array([]), np.array([]), np.array([])



        # Combine all points with their labels and depths
        all_points = np.concatenate([
            np.column_stack((projected_x1, projected_y1, depth1, np.zeros_like(depth1), x1, y1, z1)),  # Label 0: Cerebral vessels + original 3D coords
            np.column_stack((projected_x2, projected_y2, depth2, np.ones_like(depth2), x2, y2, z2)),   # Label 1: Aneurysms + original 3D coords
            np.column_stack((projected_x_conn, projected_y_conn, depth_conn, 2 * np.ones_like(depth_conn), 
                            verts_conn[:, 0], verts_conn[:, 1], verts_conn[:, 2]))  # Label 2: Connection region + original 3D coords
        ])
        
        # Sort by depth (descending order)
        sorted_indices = np.argsort(all_points[:, 2])
        sorted_points = all_points[sorted_indices]
        
        # Identify vessel points that are within the spherical region around the connection surface
        vessel_mask = (sorted_points[:, 3] == 0)  # Points labeled as vessels
        vessel_3d_coords = sorted_points[vessel_mask, 4:7]  # Original 3D coordinates of vessels
        
        if len(x) > 0:
            # Calculate distances from center to each vessel point
            vessel_distances = np.linalg.norm(vessel_3d_coords - np.array([center_x, center_y, center_z]), axis=1)
            # Mark vessels within the radius as to be ignored for overlap
            ignore_vessels = vessel_distances <= max_radius * 1.1  # Using 1.1*radius to be slightly more inclusive
        else:
            ignore_vessels = np.zeros_like(vessel_mask[vessel_mask], dtype=bool)
        
        # Create a mask for all points that should be considered for overlap detection
        consider_for_overlap = np.ones(len(sorted_points), dtype=bool)
        # Find the indices of the vessel points in the sorted_points array
        vessel_indices = np.where(vessel_mask)[0]
        # Mark the vessels within the sphere to be ignored
        consider_for_overlap[vessel_indices[ignore_vessels]] = False
        
        # Use KDTree for faster overlap detection (only considering points marked in consider_for_overlap)
        points_to_consider = sorted_points[consider_for_overlap]
        tree = cKDTree(points_to_consider[:, :2])
        overlapping_indices_vessels_aneurysms = set()  # For vessels + aneurysms overlap (pink)
        overlapping_indices_vessels_connection = set()  # For vessels + connection overlap (gold)

        for i, point in enumerate(points_to_consider[:, :2]):
            neighbors = tree.query_ball_point(point, r=0.3)  # Adjust radius as needed
            if len(neighbors) > 1:
                # Get labels of neighboring points (from the points_to_consider subset)
                labels = points_to_consider[neighbors, 3]
                unique_labels = np.unique(labels)
                
                # Check for vessels + aneurysms overlap
                if 0 in unique_labels and 1 in unique_labels:
                    # Need to map back to original indices
                    original_indices = np.where(consider_for_overlap)[0][neighbors]
                    overlapping_indices_vessels_aneurysms.update(original_indices)
                
                # Check for vessels + connection overlap
                if 0 in unique_labels and 2 in unique_labels:
                    original_indices = np.where(consider_for_overlap)[0][neighbors]
                    overlapping_indices_vessels_connection.update(original_indices)

        # Assign colors
        color_map = {0: 'red', 1: 'blue', 2: 'green'}
        colors = np.array([color_map[label] for label in sorted_points[:, 3]])

        # Mark overlapping points (only those not ignored)
        if len(overlapping_indices_vessels_aneurysms) > 0:
            colors[list(overlapping_indices_vessels_aneurysms)] = 'pink'  # Vessels + aneurysms overlap
        if len(overlapping_indices_vessels_connection) > 0:
            colors[list(overlapping_indices_vessels_connection)] = 'gold'  # Vessels + connection overlap

        # For ignored vessel points, make them more transparent
        alpha_values = np.ones(len(sorted_points))
        if len(x) > 0:
            alpha_values[vessel_indices[ignore_vessels]] = 0.3  # Make ignored vessels semi-transparent

        # Add 2D projection with overlap after sorting
        fig_2d_overlap.add_trace(go.Scatter(
            x=sorted_points[:, 0],
            y=sorted_points[:, 1],
            mode='markers',
            marker=dict(
                size=6, 
                color=colors,
                opacity=alpha_values  # Apply transparency
            ),
            name='2D Projection (With Overlap)'
        ))

        # Add legend traces for the overlap plot
        fig_2d_overlap.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(size=2, color='red'),
            name="Cerebral Vessels"
        ))
        fig_2d_overlap.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(size=2, color='blue'),
            name="Aneurysms"
        ))
        fig_2d_overlap.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(size=2, color='green'),
            name="Connection Region"
        ))
        fig_2d_overlap.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(size=2, color='pink'),
            name="Overlap (Vessels + Aneurysms)"
        ))
        fig_2d_overlap.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(size=2, color='gold'),
            name="Overlap (Vessels + Connection)"
        ))
        fig_2d_overlap.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(size=2, color='red', opacity=0.3),
            name="Ignored Vessels (Near Connection)"
        ))
        
        # Set aspect ratio to equal for the overlap plot
        fig_2d_overlap.update_layout(
            xaxis=dict(scaleanchor="y"),
            yaxis=dict(scaleanchor="x"),
        )

        # For the no-overlap plot, we'll still show all points but mark ignored vessels as transparent
        colors_sorted = np.array([color_map[label] for label in sorted_points[:, 3]])
        alpha_values_no_overlap = np.ones(len(sorted_points))
        if len(x) > 0:
            alpha_values_no_overlap[vessel_indices[ignore_vessels]] = 0.3

        # Add 2D projection without overlap
        fig_2d_no_overlap.add_trace(go.Scatter(
            x=sorted_points[:, 0],
            y=sorted_points[:, 1],
            mode='markers',
            marker=dict(
                size=6, 
                color=colors_sorted,
                opacity=alpha_values_no_overlap
            ),
            name='2D Projection (No Overlap, Sorted by Depth)'
        ))

        # Add legend traces for the no-overlap plot
        fig_2d_no_overlap.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(size=2, color='red'),
            name="Cerebral Vessels"
        ))
        fig_2d_no_overlap.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(size=2, color='blue'),
            name="Aneurysms"
        ))
        fig_2d_no_overlap.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(size=2, color='green'),
            name="Connection Region"
        ))
        fig_2d_no_overlap.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(size=2, color='red', opacity=0.3),
            name="Ignored Vessels (Near Connection)"
        ))

        # Set aspect ratio to equal for the no-overlap plot
        fig_2d_no_overlap.update_layout(
            xaxis=dict(scaleanchor="y"),
            yaxis=dict(scaleanchor="x"),
        )

        # Update 3D layout with proper camera settings
        fig_3d.update_layout(
            scene=dict(
                xaxis=dict(title='X'),
                yaxis=dict(title='Y'),
                zaxis=dict(title='Z'),
                camera=dict(
                    eye=dict( 
                        x=camera_eye['x']/x_max, 
                        y=camera_eye['y']/y_max, 
                        z=camera_eye['z']/z_max 
                    ),
                    center=dict( 
                        x=camera_center['x']/x_max, 
                        y=camera_center['y']/y_max, 
                        z=camera_center['z']/x_max 
                    ),
                    up=dict(
                        x=camera_up['x'],
                        y=camera_up['y'],
                        z=camera_up['z']
                    )
                )
            ),
            title="3D Projection"
        )

        return fig_3d, fig_2d_overlap, fig_2d_no_overlap, camera_eye

    # Process each angle for this folder
    save_path = os.path.join(os.path.dirname(input_path),"saved_images")
    
    # Create or clear the camera info file at the beginning
    info_file_path = os.path.join(save_path, "camera_info.txt")
    if os.path.exists(info_file_path):
        os.remove(info_file_path)
    
    direction_vector_dict = {}

    increments= 360//its

    if verbose:
        print("=============================saved 2d projections=====================================")

    for nr, angle in enumerate(range(0, 361, increments)):
        if verbose:
            print(f"saving projection {nr} out of {len(range(0, 361, increments))}")

        # Calculate camera view - now returns camera_center as well
        camera_eye, camera_center, _ = calculate_camera_view(normal_vector, angle, zoom_factor=2)
        azimuth, elevation = calculate_azimuth_elevation(camera_eye, camera_center)
        
        dir_vector = np.array([camera_eye['x'], camera_eye['y'], camera_eye['z']]) - np.array([camera_center['x'], camera_center['y'], camera_center['z']])
        unit_dir_vector = dir_vector/np.linalg.norm(dir_vector)
        direction_vector_dict[angle] = unit_dir_vector

        # Save the 2D projections and camera info
        save_2d_projection_matplotlib(angle, azimuth, elevation, camera_eye, save_path)

    # Define color ranges (in RGB format)
    red_color = np.array([255, 0, 0])  # Blood vessels
    green_color = np.array([0, 128, 0])  # Connection region
    blue_color = np.array([0, 0, 255])  # Aneurysm
    pink_color = np.array([255, 192, 203])  # Overlap between blood vessels and aneurysm
    purple_color = np.array([128, 0, 128])  # Overlap between blood vessels and connection region

    # Define color tolerance
    tolerance = 50

    def count_pixels_by_color(image, target_color, tolerance):
        """Count the pixels of a specific color in the image."""
        image_array = np.array(image)
        color_diff = np.abs(image_array - target_color)
        within_tolerance = np.all(color_diff <= tolerance, axis=-1)
        return np.sum(within_tolerance)

    def evaluate_overlap_and_connection(image, connection_color, blood_vessel_color, tolerance=20):
        """Evaluate the overlap of connection region and blood vessel areas."""
        connection_pixels = count_pixels_by_color(image, connection_color, tolerance)
        blood_vessel_pixels = count_pixels_by_color(image, blood_vessel_color, tolerance)
        connection_blood_vessel_overlap = count_pixels_by_color(image, purple_color, tolerance)  # Overlap area
        
        # Handle the special case when connection_pixels is zero
        if connection_pixels == 0:
            if connection_blood_vessel_overlap > 0:
                connection_overlap_percentage = float('inf')  # Infinite overlap ratio
            else:
                connection_overlap_percentage = 0  # No overlap, no connection area
        else:
            connection_overlap_percentage = connection_blood_vessel_overlap / connection_pixels

        return connection_pixels, connection_blood_vessel_overlap, connection_overlap_percentage, blood_vessel_pixels

    def is_connection_visible(no_overlap_image, connection_color, tolerance):
        """Check if connection region is visible in the no-overlap image."""
        connection_pixels = count_pixels_by_color(no_overlap_image, connection_color, tolerance)
        return connection_pixels > 0

    def process_images(folder_path, result_file):
        """Process images to identify the top 10 with minimal connection overlap, then calculate the optimal angle by different criterions."""
        no_overlap_files = [f for f in os.listdir(folder_path) if '2d_projection_no_overlap_' in f and f.endswith('.png')]
        overlap_files = [f for f in os.listdir(folder_path) if '2d_projection_overlap_' in f and f.endswith('.png')]
        
        no_overlap_dict = {f.split('_')[-1].replace('.png', ''): f for f in no_overlap_files}
        overlap_dict = {f.split('_')[-1].replace('.png', ''): f for f in overlap_files}

        # Step 1: Calculate neck (green) pixels for all overlap images and find max neck size
        neck_pixels_list = []
        for angle, overlap_filename in overlap_dict.items():
            if angle in no_overlap_dict:
                overlap_image = Image.open(os.path.join(folder_path, overlap_filename)).convert('RGB')
                neck_pixels = count_pixels_by_color(overlap_image, green_color, tolerance)
                neck_pixels_list.append((neck_pixels, angle))

        if not neck_pixels_list:
            return None, None, None  # No valid images

        # Find maximum neck size
        max_neck_pixels = max(neck_pixels_list, key=lambda x: x[0])[0]
        threshold = 0.1 * max_neck_pixels

        # Filter angles: keep those with neck >= 10% of max neck
        filtered_angles = [angle for (neck_pixels, angle) in neck_pixels_list if neck_pixels >= threshold]

        # If all angles are filtered out, keep top 5 angles with largest necks
        if not filtered_angles:
            neck_pixels_list.sort(reverse=True, key=lambda x: x[0])
            filtered_angles = [angle for (neck_pixels, angle) in neck_pixels_list[:5]]

        # Step 2: Proceed with filtered angles (min connection overlap selection)
        connection_overlap_list = []
        for nr, angle in enumerate(filtered_angles):
    
            if verbose:
                print(f"Processing angle [{nr+1}/{len(filtered_angles)}]")

            no_overlap_filename = no_overlap_dict[angle]
            overlap_filename = overlap_dict[angle]

            no_overlap_image = Image.open(os.path.join(folder_path, no_overlap_filename)).convert('RGB')
            overlap_image = Image.open(os.path.join(folder_path, overlap_filename)).convert('RGB')

            # Skip if connection region is not visible in no-overlap image
            if not is_connection_visible(no_overlap_image, green_color, tolerance):
                result_file.write(f"Skipping {no_overlap_filename} - connection region completely blocked\n")
                continue

            # Evaluate connection overlap
            connection_pixels, connection_blood_vessel_overlap, connection_overlap_percentage, blood_vessel_pixels = evaluate_overlap_and_connection(
                overlap_image, green_color, red_color)
            
            # Log the results
            result_file.write(f"File (no_overlap): {no_overlap_filename}\n")
            result_file.write(f"File (overlap): {overlap_filename}\n")
            result_file.write(f"connection_overlap_percentage: {connection_overlap_percentage}\n")
            result_file.write(f"connection_pixels: {connection_pixels}\n")
            result_file.write(f"connection_blood_vessel_overlap: {connection_blood_vessel_overlap}\n")
            result_file.write("-" * 30 + "\n")

            if verbose:
                print(f"File (no_overlap): {no_overlap_filename}")
                print(f"File (overlap): {overlap_filename}")
                print(f"connection_overlap_percentage: {connection_overlap_percentage}")
                print(f"connection_pixels: {connection_pixels}")
                print(f"connection_blood_vessel_overlap: {connection_blood_vessel_overlap}")
                print("-" * 30)

            
            # Store connection overlap percentage and filenames
            connection_overlap_list.append((connection_overlap_percentage, no_overlap_filename, overlap_filename))

        # If no valid angles found, return None for all selections
        if not connection_overlap_list:
            return None, None, None

        # Sort by connection overlap percentage (ascending order)
        connection_overlap_list.sort(key=lambda x: x[0])
        
        # Take up to 10 images (fewer if not enough available)
        top_n_min_connection_overlap = connection_overlap_list[:min(10, len(connection_overlap_list))]

        # Step 3: Apply final selection criteria
        if not top_n_min_connection_overlap:
            return None, None, None

        # Criterion 1: Shortest connection region (smallest connection_pixels)
        shortest_connection = min(top_n_min_connection_overlap, key=lambda x: count_pixels_by_color(
            Image.open(os.path.join(folder_path, x[1])).convert('RGB'), green_color, tolerance))

        # Criterion 2: Largest connection region (largest connection_pixels)
        longest_connection = max(top_n_min_connection_overlap, key=lambda x: count_pixels_by_color(
            Image.open(os.path.join(folder_path, x[1])).convert('RGB'), green_color, tolerance))
    
        # Criterion 3: Largest aneurysm (largest blue pixels)
        largest_aneurysm = max(top_n_min_connection_overlap, key=lambda x: count_pixels_by_color(
            Image.open(os.path.join(folder_path, x[2])).convert('RGB'), blue_color, tolerance))

        return shortest_connection, longest_connection, largest_aneurysm

    def save_and_display_image(image, title, save_path, original_filename):
        """Save and display the image."""
        safe_title = title.replace(' ', '_').replace(':', '').replace('/', '_')
        save_filename = original_filename.replace('2d_projection', safe_title)
        save_full_path = os.path.join(save_path, save_filename)

        if verbose:
            print(f"Saving {title} to: {save_full_path}")

        plt.imsave(save_full_path, image, dpi=700)

        if verbose:
            print(f"Saved {title} successfully.")


    #RUNS THE MODEL THINGY
    folder_path = os.path.join(os.path.dirname(input_path),"saved_images")
    if not os.path.exists(folder_path):
        print(f"Skipping: {folder_path} does not exist.")
        return
    
    save_optimal_path = os.path.join(folder_path, "optimal")
    if not os.path.exists(save_optimal_path):
        os.makedirs(save_optimal_path)

    if verbose:
        print("=============================Processed images=====================================")
    with open(os.path.join(folder_path, 'results.txt'), 'w') as result_file:
        shortest_connection, longest_connection, largest_aneurysm = process_images(folder_path, result_file)
    
    shortest_connection_angle = int(shortest_connection[1].split('_')[-1].replace('.png', ''))
    longest_connection_angle = int(longest_connection[1].split('_')[-1].replace('.png', ''))
    largest_aneurysm_angle = int(largest_aneurysm[2].split('_')[-1].replace('.png', ''))

    shortest_connection_rho_deg, shortest_connection_psi_deg = convert_angle_to_DSA(direction_vector_dict[shortest_connection_angle])
    longest_connection_rho_deg, longest_connection_psi_deg = convert_angle_to_DSA(direction_vector_dict[longest_connection_angle])
    largest_aneurysm_rho_deg, largest_aneurysm_psi_deg = convert_angle_to_DSA(direction_vector_dict[largest_aneurysm_angle])

    output_dict = {}
    output_dict["shortest_connection"] = (shortest_connection_rho_deg, shortest_connection_psi_deg)
    output_dict["longest_connection"] = (longest_connection_rho_deg, longest_connection_psi_deg)
    output_dict["largest_aneurysm"] = (largest_aneurysm_rho_deg, largest_aneurysm_psi_deg)

    # If no valid images found, skip this folder
    if shortest_connection is None:
        print(f"No valid images found for this image (all connection regions were blocked)")
        return

    # Dictionary of the best selections
    selections = {
        "Shortest Connection": shortest_connection,
        "Longest Connection": longest_connection,
        "Largest Aneurysm": largest_aneurysm
    }

    # Display and save optimal images
    for title, selection in selections.items():
        if selection:
            total_overlap, no_overlap_filename, overlap_filename = selection
            
            if verbose:
                print(f"\n{title}: {no_overlap_filename}")

            # Load and display the no-overlap image
            no_overlap_image = mpimg.imread(os.path.join(folder_path, no_overlap_filename))
            save_and_display_image(no_overlap_image, f"No Overlap ({title})", save_optimal_path, no_overlap_filename)

            # Load and display the overlap image
            overlap_image = mpimg.imread(os.path.join(folder_path, overlap_filename))
            save_and_display_image(overlap_image, f"Overlap ({title})", save_optimal_path, overlap_filename)
    
    return output_dict
