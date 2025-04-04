import numpy as np
import SimpleITK as sitk
from scipy import ndimage
import os
from plotly.subplots import make_subplots
import plotly.graph_objects as go

class AneurysmIsolation:
    def __init__(self, ct_path, mask_path, output_dir):
        # Load CT and segmentation mask
        self.ct = sitk.ReadImage(ct_path)
        self.mask = sitk.ReadImage(mask_path)
        
        # Convert to NumPy arrays
        self.ct_array = sitk.GetArrayFromImage(self.ct)
        self.mask_array = sitk.GetArrayFromImage(self.mask)
        
        # Normalize CT intensity
        self.ct_array = (self.ct_array - np.min(self.ct_array)) / (np.max(self.ct_array) - np.min(self.ct_array))

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir

    def isolate_aneurysms(self, verbose=False):
        """
        Isolate aneurysms in the segmentation mask and generate new images with each aneurysm isolated.
        """
        # Isolate vasculature
        vasculature_array = np.where(self.mask_array == 1, self.mask_array, 0)

        # Label aneurysms
        labeled_array, nr_of_aneurysms = ndimage.label(self.mask_array == 2)

        if verbose:
            print(f'Found {nr_of_aneurysms} aneurysms.')

        isolated_aneurysms = []

        # Process each aneurysm
        for aneurysm_nr in range(nr_of_aneurysms):
            isolated_aneurysm_arr = np.where(labeled_array == aneurysm_nr + 1, 2, np.where(labeled_array != 0, 1, 0))

            # Combine aneurysm with vasculature
            total_array = np.add(isolated_aneurysm_arr.astype(int), vasculature_array.astype(int))
            isolated_aneurysms.append(total_array)

            # Save as a new NIfTI file
            output_filename = os.path.join(self.output_dir, f"isolated_aneurysm_{aneurysm_nr + 1}.nii.gz")
            aneurysm_image = sitk.GetImageFromArray(total_array)
            aneurysm_image.CopyInformation(self.mask)
            sitk.WriteImage(aneurysm_image, output_filename)

            if verbose:
                print(f'Saved: {output_filename}')

        return isolated_aneurysms
    


def get_aneurysm_paths(output_dir):
    """
    Retrieves all aneurysm file paths from the specified directory.

    Parameters:
    - output_dir (str): Path to the results directory.

    Returns:
    - list of str: Full paths to aneurysm files.
    """
    aneurysm_paths = []  # List to store paths

    # Loop through all files in the directory
    for file in os.listdir(output_dir):
        if file.startswith("isolated_aneurysm_") and file.endswith(".nii.gz"):
            aneurysm_paths.append(os.path.join(output_dir, file))

    return aneurysm_paths





def plot_3d_scatter(mask_path, aneurysm_paths=None):
    """
    Plots a 3D scatter visualization for a mask and multiple aneurysms.
    
    Parameters:
    - mask_path (str): Path to the segmentation mask file.
    - aneurysm_paths (list of str, optional): Paths to aneurysm segmentation files.
    """
    # Load segmentation mask
    mask = sitk.ReadImage(mask_path)
    mask_array = sitk.GetArrayFromImage(mask)

    # Get 3D coordinates of non-zero mask values
    mask_indices = np.where(mask_array > 0)
    mask_x, mask_y, mask_z = mask_indices[2], mask_indices[1], mask_indices[0]

    # Ensure aneurysm_paths is a list (handle cases where it's None or a single string)
    if aneurysm_paths is None:
        aneurysm_paths = []
    elif isinstance(aneurysm_paths, str):
        aneurysm_paths = [aneurysm_paths]

    # Load aneurysm masks (if any)
    aneurysm_arrays = [sitk.GetArrayFromImage(sitk.ReadImage(path)) for path in aneurysm_paths]
    aneurysm_indices = [np.where(arr > 0) for arr in aneurysm_arrays]

    # Create subplots (mask + each aneurysm)
    num_plots = 1 + len(aneurysm_paths)  # 1 for mask + aneurysms
    fig = make_subplots(
        rows=num_plots, cols=1,
        specs=[[{'type': 'scatter3d'}]] * num_plots,
        subplot_titles=["Mask"] + [f"Aneurysm {i+1}" for i in range(len(aneurysm_paths))]
    )

    # Plot mask
    fig.add_trace(go.Scatter3d(
        x=mask_x, y=mask_y, z=mask_z,
        mode="markers",
        marker=dict(size=2, color=mask_array[mask_indices], colorscale="jet"),
    ), row=1, col=1)

    # Plot each aneurysm
    for i, (aneurysm_array, indices) in enumerate(zip(aneurysm_arrays, aneurysm_indices)):
        aneurysm_x, aneurysm_y, aneurysm_z = indices[2], indices[1], indices[0]
        fig.add_trace(go.Scatter3d(
            x=aneurysm_x, y=aneurysm_y, z=aneurysm_z,
            mode="markers",
            marker=dict(size=2, color=aneurysm_array[indices], colorscale="jet"),
        ), row=i+2, col=1)

    # Update layout
    fig.update_layout(
        height=400 * num_plots,
        title="3D Scatter Plots",
    )

    fig.show()

