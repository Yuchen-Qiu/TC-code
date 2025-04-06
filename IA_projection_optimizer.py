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
import SimpleITK as sitk
import Virtual_Aneurysm_Spectator_Model
import iterative_observer_model

class OptimalViewFinder:
    """
    OptimalViewFinder helps in determining the best projection angle for intercranial aneurysms using DSA.
    
    Features:
    Loads a NIfTI file.
    Isolates aneurysms in the segmentation mask.
    Determines optimal projection angle of intercranial aneurysm via DSA.
    Supports two models (virtual aneurysm spectator (VAS) + iterative observer).
    Writes results to output file
    
    Attributes:
    input_path (str):                           Path to the input NIfTI file.
    output_path (str):                          Path to the output directory.
    maskimg (np.array):                         3D segmentation mask loaded from the file.
    metadata (str):                             Header of the input NIFTI file.
    vasculature_array (np.array):               Numpy array that contains the segmentation mask of the vasculature.
    isolated_anuerysms:                         List of numpy arrays that contains the segmentation of isolated aneurysms.
    self.ran_VAS (boolean):                     Checks if VAS model has been used.
    self.ran_itobs (boolean):                   Checks if iterative observer model has been used.
    self.VAS_output_text (str):                 String variable that contains all results from the VAS model.
    self.iterative_observer_output_text (str:)  String variable that contains all results from the iterative observer model.   

    Methods:
    isolate_aneurysms(verbose=False):           Identifies and isolates aneurysms in the mask.
    run_modified_VAS_model:                     Runs modified VAS model on all isolated aneurysms.
    run_iterative_observer_model:               Runs iterative observer model on all isolated aneurysms.
    write_outputfile:                           Writes all generated results to output txt file.

    """
    
    def __init__(self,input_path):
        
        #Setting up input path
        self.input_path = input_path

        #Loading niftii file.
        nii_file = nib.load(input_path)
        self.maskimg = nii_file.get_fdata()
        self.metadata = nii_file.header

        self.ran_VAS = False
        self.ran_itobs = False

        print(f"OptimalviewFinder succesfully initialized, using input: {self.input_path}")

    def isolate_aneurysms(self):
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

        self.vasculature_array = np.where(self.maskimg == 1, self.maskimg, 0)  #Isolating vasculature.

        labeled_array, self.nr_of_aneurysms = ndimage.label(self.maskimg == 2)  #Finding number of aneurysms and assigning each one a different label.

        print(f'Found {self.nr_of_aneurysms} different aneurysms.')

        self.isolated_aneurysms = []

        aneurysm_sizes = []

        #Going through every aneurysm present in the image and creating a new image where only this aneurysm has label 2.
        for aneurysm_nr in range(self.nr_of_aneurysms):
            isolated_aneurysm_arr = np.where(labeled_array == aneurysm_nr + 1, 2, np.where(labeled_array != 0, 1, 0))

            total_array = np.add(isolated_aneurysm_arr.astype(np.int32), self.vasculature_array.astype(np.int32))  #Adding aneurysm and vasculature together.

            self.isolated_aneurysms.append(total_array)

            #Calculate the size of the aneurysm by counting the number of voxels labeled as part of the aneurysm
            aneurysm_size = np.sum(isolated_aneurysm_arr == 2)
            aneurysm_sizes.append((aneurysm_size, total_array))  #Store size and corresponding image

            print(f'Isolated aneurysm {aneurysm_nr + 1}')

        #Sort aneurysms by size (from largest to smallest)
        self.isolated_aneurysms = [x[1] for x in sorted(aneurysm_sizes, key=lambda x: x[0], reverse=True)]

        return self.isolated_aneurysms

    def run_VAS_model(self, verbose=False):
        """
        Function that implements the virtual aneurysm spectator (VAS) model found in Virtual_Aneurysm_Spectator_Model.py, 
        to find optimal projection angles of an intercranial aneurysm by analysis of a segmented 3D CTA image. Returns 
        both rotation angle ρ and angulation angle ψ to be used inside a DSA machine. Exporting results via write_outputfile function.
        
        Input variables:
        verbose:            Boolean operator that determines if progress print statements are shown in the kernel or not.

        Output variables:
        None
        """
        
        self.VAS_output_text = f"RESULT OF AUTOMATIC PROJECTION ANGLE DETERMINATION VIA VIRTUAL ANEURYSM SPECTATOR (VAS) MODEL.\n"
        self.VAS_output_text += f"===================================================================================\n"
        
        for nr, segmentation in enumerate(self.isolated_aneurysms):
            print(f"Running model for Aneurysm [{nr+1}/{len(self.isolated_aneurysms)}]")
            
            #Running the VAS model for every isolated aneurysm in self.isolated aneurysms
            rho_deg, psi_deg = Virtual_Aneurysm_Spectator_Model.run_model(segmentation,verbose=verbose)    
            
            #Adds results to result string variable
            self.VAS_output_text += f"=Aneurysm [{nr+1}/{len(self.isolated_aneurysms)}]=\n"
            self.VAS_output_text += f"Final optimal DSA Rotation Angle (ρ): {rho_deg}°\n"
            self.VAS_output_text += f"Final optimal DSA Angulation Angle (ψ): {psi_deg}°\n"
            self.VAS_output_text += f"===================================================================================\n"
        
        print("===================================================================================")
        print(self.VAS_output_text)
        
        #Updates variable if model has been used
        self.ran_VAS = True

    def run_iterative_observer_model(self,showimgs=False,its=36,verbose=False):
        """
        Function that implements the iterative_observer_model found in iterative_observer_model.py, to find optimal projection angles of 
        an intercranial aneurysm by analysis of a segmented 3D CTA image. Returns both rotation angle ρ and angulation angle ψ to be 
        used inside a DSA machine. Exporting results via write_outputfile function.
        
        Input variables:
        showimgs:           Boolean operator that determines if the 2D projections of optimal angles are printed in the kernel.
        its:                Amount of iterations the model performed (default: 36 means one iteration per 10 degrees)
        verbose:            Boolean operator that determines if progress print statements are shown in the kernel or not.

        Output variables:
        None
        """

        self.iterative_observer_output_text = f"RESULT OF AUTOMATIC PROJECTION ANGLE DETERMINATION VIA ITERATIVE OBSERVER MODEL.\n"
        self.iterative_observer_output_text += f"===================================================================================\n"

        for nr, segmentation in enumerate(self.isolated_aneurysms):
            print(f"Running model for Aneurysm [{nr+1}/{len(self.isolated_aneurysms)}]")
            
            #Runs the iterative observer model for every segmentation in isolated aneurysms
            output_dict = iterative_observer_model.run_model(segmentation,input_path=self.input_path,showimgs=showimgs,its=its,verbose=verbose)    
            
            #Adds results to result string variable
            self.iterative_observer_output_text += f"=Aneurysm [{nr+1}/{len(self.isolated_aneurysms)}]=\n"
            for key, value in output_dict.items():
                self.iterative_observer_output_text += f"===================={key}====================\n"
                self.iterative_observer_output_text += f"Final optimal DSA Rotation Angle (ρ): {value[0]}°\n"
                self.iterative_observer_output_text += f"Final optimal DSA Angulation Angle (ψ): {value[1]}°\n"
            
            self.iterative_observer_output_text += f"===================================================================================\n"
        
        print(self.iterative_observer_output_text)
        
        #Updates variable if model has been used
        self.ran_itobs = True
    
    def write_outputfile(self,output_folder=None):
        """
        Function that writes the results of the VAS model and iterative observer model inside of a txt file.
        Note: If no models were ran before executing this function, an empty results file will be generated.
        
        Input variables:
        output_folder:      Str variable that contains the path to the folder in which the output should be saved.

        Output variables:
        file:               .txt file that contains the outputstrings of both the VAS model as the iterative observer model
                            (only when applicable)
        """
       
        #Checks if output_folder is specified, otherwise defaults to input folder and creates new folder "optimal_angle_results"
        if not output_folder:
            self.output_folder = os.path.join(os.path.dirname(self.input_path),"Optimal_angle_results")
        else:
            self.output_folder = output_folder
        
        #If specified output_folder does not exist, it creates it.
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
        
        #Determines final output path
        self.output_path = self.output_folder+f"/Optimal_angle_ results_{os.path.basename(self.input_path)}.txt"

        #Writes results to file
        with open(self.output_path, 'w', encoding='utf-8') as file:
            if self.ran_VAS:
                file.write(self.VAS_output_text)
            
            if self.ran_itobs:
                file.write(self.iterative_observer_output_text)
            
            if not (self.ran_VAS and self.ran_itobs):
                file.write("No models were used on this data.")
        
        print(f"Saved txt file at {self.output_path}")
        with open(self.output_path, 'r') as file:
            filecontent = file.read()

        # Print the contents
        print(f"File content [{filecontent}]")