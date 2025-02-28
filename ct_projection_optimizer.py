#Imports
import os
import numpy as np
import nibabel as nib
import scipy.ndimage as ndimage
import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Patch


class OptimalViewFinder:
    """
    OptimalViewFinder helps in determining the best projection angle for brain aneurysm CT images.
    
    Features:
    Loads a NIfTI file.
    Isolates aneurysms in the segmentation mask.
    Supports automatic output directory creation.
    
    Attributes:
    input_path (str): Path to the input NIfTI file.
    output_path (str): Path to the output directory.
    maskimg (numpy array): 3D segmentation mask loaded from the file.

    Methods:
    isolate_aneurysms(verbose=False): Identifies and isolates aneurysms in the mask.
    """
    
    def __init__(self,input_path,output_path=None):
        
        #Setting up input and output paths.
        if output_path is None:
            output_path = os.path.join(os.path.dirname(input_path), 'results') #If no output path is specified it defaults to a results folder in the input folder.
        
        #Ensure output path exists.
        if not os.path.exists(output_path):
            os.makedirs(output_path)        

        self.input_path = input_path
        self.output_path = output_path


        #Loading niftii file.
        nii_file = nib.load(input_path)
        self.maskimg = nii_file.get_fdata()
        


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
            
            total_array = np.add(isolated_aneurysm_arr.astype(int),self.vasculature_array.astype(int)) #Adding aneurysm and vasculature together.

            self.isolated_aneurysms.append(total_array)

            if verbose:
                print(f'Isolated aneurysm {aneurysm_nr+1}')
        
        return self.isolated_aneurysms 

    #I chatGPT'ed the FUCK out of this hahaha
    def visualize_3d_voxels(self,downsample_factor=2):
        """
        Visualizes 3D labeled voxel data using Matplotlib's voxel plot.
        
        Parameters:
        plot_data (numpy array): 3D array with labeled voxel data (e.g., 0 for background, 1 and 2 for aneurysms).
        downsample_factor (int): Factor to reduce data size for faster rendering (optional).
        
        Returns:
        - None (displays the plot directly)
        """
        # Convert to NumPy array and validate
        plot_data = np.asarray(self.isolated_aneurysms[0])
        if plot_data.ndim != 3:
            raise ValueError(f"Expected a 3D array, got shape {plot_data.shape}")
        
        # Downsample the data if needed (reduces memory usage and speeds up plotting)
        if downsample_factor > 1:
            plot_data = plot_data[::downsample_factor, ::downsample_factor, ::downsample_factor]
        
        print(f"Visualizing data with shape: {plot_data.shape}")
        print(f"Unique values in data: {np.unique(plot_data)}")
        
        # Create boolean arrays for each aneurysm
        aneurysm1 = plot_data == 1  # Where value is 1
        aneurysm2 = plot_data == 2  # Where value is 2
        
        # Set up the 3D plot
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Define colors matching the shape of aneurysm1 and aneurysm2
        colors1 = np.zeros(aneurysm1.shape + (4,))  # RGBA array
        colors1[..., 0] = 1  # Red channel
        colors1[..., 3] = 0.5  # Alpha channel (transparency)

        colors2 = np.zeros(aneurysm2.shape + (4,))
        colors2[..., 2] = 1  # Blue channel
        colors2[..., 3] = 0.5  # Alpha channel

        # Plot with color arrays
        if np.any(aneurysm1):
            ax.voxels(aneurysm1, facecolors=colors1, edgecolor='k')
        else:
            print("No voxels found for Aneurysm 1")

        if np.any(aneurysm2):
            ax.voxels(aneurysm2, facecolors=colors2, edgecolor='k')
        else:
            print("No voxels found for Aneurysm 2")
        
        # Customize the plot
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('3D Visualization of Aneurysms')
        legend_elements = [
            Patch(facecolor='red', edgecolor='k', label='Aneurysm 1'),
            Patch(facecolor='blue', edgecolor='k', label='Aneurysm 2')
        ]

        ax.legend(handles=legend_elements)
        # Adjust the view angle (optional: tweak these values)
        ax.view_init(elev=20, azim=45)
        
        # Show the plot
        plt.show()