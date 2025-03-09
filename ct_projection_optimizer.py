#Imports
import os
import numpy as np
import nibabel as nib
import scipy.ndimage as ndimage
import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import json
import SimpleITK as sitk

class OptimalViewFinder:
    """
    OptimalViewFinder helps in determining the best projection angle for brain aneurysm CT images.
    
    Features:
    Loads a NIfTI file.
    Isolates aneurysms in the segmentation mask.
    Exports isolated aneurysms to new NIFTI files.
    Supports automatic output directory creation.
    
    Attributes:
    input_path (str):           Path to the input NIfTI file.
    output_path (str):          Path to the output directory.
    maskimg (numpy array):      3D segmentation mask loaded from the file.
    metadata ():                Header of the input NIFTI file.
    vasculature_array:          Numpy array that contains the segmentation mask of the vasculature
    self.isolated_anuerysms:    List of numpy arrays that contains the segmentation of isolated aneurysms.

    Methods:
    isolate_aneurysms(verbose=False): Identifies and isolates aneurysms in the mask.
    """
    
    def __init__(self,input_path):
        
        #Setting up input path
        self.input_path = input_path


        #Loading niftii file.
        nii_file = nib.load(input_path)
        self.maskimg = nii_file.get_fdata()
        self.metadata = nii_file.header

        print(f"OptimalviewFinder succesfully initialized, using input: {self.input_path}")

    def isolate_aneurysms(self, verbose=False):
        """
        Function that determines the amount of aneurysms present in the segmentation mask, n, and generates n new images 
        where one of the aneurysms is isolated, other aneurysms are set to the same segmentation label as surrounding vasculature.
        Returns a list that contains all the generated segmentation masks.
        
        Input variables:
        verbose:            Boolean operator that determines if progress print statements are shown in the kernel or not.

        Output variables:
        isolated_aneurysms: A list that contains all 3d image arrays where one aneurysm has label 2, 
                            while all other aneurysms and vasculature have label 1.
        """

        self.vasculature_array = np.where(self.maskimg ==1, self.maskimg, 0) #Isolating vasculature.

        labeled_array, self.nr_of_aneurysms = ndimage.label(self.maskimg == 2) #Finding number of aneurysms and assigning each one a different label.

        #Progress print statement.
        if verbose:
            print(f'Found {self.nr_of_aneurysms} different aneurysms.')

        self.isolated_aneurysms = []

        #Going through every aneurysm present in the image and creating a new image where only this aneurysm has label 2.
        for aneurysm_nr in range(self.nr_of_aneurysms):
            isolated_aneurysm_arr = np.where(labeled_array == aneurysm_nr + 1, 2, np.where(labeled_array != 0, 1, 0))
            
            total_array = np.add(isolated_aneurysm_arr.astype(np.int32),self.vasculature_array.astype(np.int32)) #Adding aneurysm and vasculature together.

            self.isolated_aneurysms.append(total_array)

            if verbose:
                print(f'Isolated aneurysm {aneurysm_nr+1}')
        
        return self.isolated_aneurysms 

    def export_data(self,output_path=None):
        """
        Funcion that exports the isolated aneurysms as a new niftii file, while adding relevant metadata.
        """

        #Setting output paths.
        if output_path is None:
            output_path = os.path.join(os.path.dirname(self.input_path), 'results') #If no output path is specified it defaults to a results folder in the input folder.
        
        #Ensure output path exists.
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        self.output_path = output_path      

        self.aneurysm_paths = []

        #Exporting and saving actual data:
        for aneurysm_nr, aneurysm_img_data in enumerate(self.isolated_aneurysms):

            # Save as a new NIfTI file
            output_filename = os.path.join(self.output_path, f"isolated_aneurysm_{aneurysm_nr + 1}_out_of_{len(self.isolated_aneurysms)}.nii.gz")
            self.aneurysm_paths.append(output_filename)

            # Load the stored metadata (header)
            new_header = self.metadata.copy()  # Ensure we don't modify the original header

            additional_metadata = {}
            
            additional_metadata["Number of aneurysms"] = self.nr_of_aneurysms
            additional_metadata["Isolated aneurysm"] = f"{aneurysm_nr+1} out of {self.nr_of_aneurysms}"

            # Convert dictionary to JSON and encode as bytes
            metadata_json = json.dumps(additional_metadata).encode('utf-8')

            # Create a NIfTI extension (code 40 for JSON)
            header_extension = nib.nifti1.Nifti1Extension(40, metadata_json)

            # Add the extension to the header
            new_header.extensions.append(header_extension)

            final_nii_file = nib.Nifti1Image(aneurysm_img_data, affine=None, header=new_header)
            nib.save(final_nii_file,output_filename)
            
            print(f"Saved isolated aneurysm {aneurysm_nr+1} at {output_filename}")



    def plot_3d_scatter(self):
        """
        Plots a 3D scatter visualization for a mask and multiple aneurysms.
        """
        # Load segmentation mask
        mask = sitk.ReadImage(self.input_path)
        mask_array = sitk.GetArrayFromImage(mask)

        # Get 3D coordinates of non-zero mask values
        mask_indices = np.where(mask_array > 0)
        mask_x, mask_y, mask_z = mask_indices[2], mask_indices[1], mask_indices[0]

        # Load aneurysm masks (if any)
        aneurysm_arrays = self.isolated_aneurysms
        aneurysm_indices = [np.where(arr > 0) for arr in aneurysm_arrays]

        # Create subplots (mask + each aneurysm)
        num_plots = 1 + len(self.isolated_aneurysms)  # 1 for mask + aneurysms

        fig = make_subplots(
            rows=num_plots, cols=1,
            specs=[[{'type': 'scatter3d'}]] * num_plots,
            subplot_titles=["Mask"] + [f"Aneurysm {i+1}" for i in range(len(self.isolated_aneurysms))]
        )

        # Plot mask
        fig.add_trace(go.Scatter3d(
            x=mask_x, y=mask_y, z=mask_z,
            mode="markers",
            marker=dict(size=2, color=mask_array[mask_indices], colorscale="jet"),
        ), row=1, col=1)

        # Plot each aneurysm
        for i, (aneurysm_array, indices) in enumerate(zip(aneurysm_arrays, aneurysm_indices)):
            aneurysm_x, aneurysm_y, aneurysm_z = indices[0], indices[1], indices[2]
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