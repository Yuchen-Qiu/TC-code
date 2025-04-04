import tkinter as tk
from tkinter import filedialog

def select_input_path():
    """
    Function that opens a windows explorer instance and allows for input file selection. Checks if a file is of NIFTI format.

    Output parameters:
    file_path:      String variable that contains the path to the file selected.
    """

    # Open the file dialog to select a file
    file_path = filedialog.askopenfilename(title="Select a file")

    assert file_path.endswith('.nii'), "Input file must be of NIFTI (.nii) format!"

    return file_path

def select_output_folder():
    """
    Function that opens a windows explorer instance and allows for folder selection.

    Output parameters:
    folder_path:    String variable that contains the path to the folder selected.
    """

    # Open the file dialog to select a folder
    folder_path = filedialog.askdirectory(title="Select a folder")

    # Check if a folder is selected
    if not folder_path:
        print("No folder selected, using default output")
        folder_path = None
        
    return folder_path