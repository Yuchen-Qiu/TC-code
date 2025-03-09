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