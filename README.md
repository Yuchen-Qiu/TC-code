🧠 Optimal Projection Angle Finder for Brain Aneurysms
This project automatically determines the optimal projection angle for visualizing brain aneurysms in CT images. It leverages a combination of geometric analysis, image processing, and model-based scoring — including a Visual Attention Simulation (VAS) model and an Iterative Observer model — to find the angle that yields the clearest and most informative view.

🚀 Features
🔍 Automated best-angle detection from 3D aneurysm segmentations.

🧪 Supports both VAS model and Iterative Observer model for evaluation.

🧼 Includes preprocessing steps to align and prepare images and masks.

📊 Exports results to neatly formatted .txt files.

🖼️ Visualizes 2D projections at optimal angles.

📂 Folder Structure
css
Kopiëren
Bewerken
.
├── main.py
├── aneurysm_angle_finder/
│   ├── angle_optimizer.py
│   ├── vas_model.py
│   ├── observer_model.py
│   └── utils/
├── data/
│   └── sample_case/
│       ├── image.nii.gz
│       └── mask.nii.gz
├── output/
│   └── Optimal_angle_results_*.txt
└── README.md
🧠 How It Works
Input: A CT or CTA scan + binary mask (2-label mask: aneurysm and vessel).

Preprocessing: Aligns mask, removes noise, extracts ROI.

Angle Optimization: Projects the mask at multiple angles, evaluates each one.

Model Evaluation: Scores each angle using VAS / Iterative Observer (or both).

Output: Best-view angle and scoring written to .txt file.

🛠️ Installation
Clone the repo:

bash
Kopiëren
Bewerken
git clone https://github.com/yourusername/aneurysm-angle-finder.git
cd aneurysm-angle-finder
Set up a virtual environment (recommended):

bash
Kopiëren
Bewerken
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Install dependencies:

bash
Kopiëren
Bewerken
pip install -r requirements.txt
(Optional) If using Elastix/Transformix for registration, make sure they’re installed and accessible in your PATH.

🧪 Usage
You can run the angle optimization through a script or Jupyter notebook:

python
Kopiëren
Bewerken
from aneurysm_angle_finder.angle_optimizer import AngleOptimizer

optimizer = AngleOptimizer(
    input_image="data/sample_case/image.nii.gz",
    input_mask="data/sample_case/mask.nii.gz"
)

optimizer.run_VAS_model()
optimizer.run_iterative_observer()
optimizer.write_outputfile()
📝 Output Format
The write_outputfile() method saves a .txt file like:

sql
Kopiëren
Bewerken
--- VAS Model Results ---
Best angle: 125°
Score: 0.87

--- Iterative Observer Results ---
Best angle: 130°
Score: 0.85
🧩 Dependencies
numpy

nibabel

matplotlib

scipy

scikit-image

SimpleITK (optional, if using Elastix/Transformix)

🧠 Notes
Make sure your mask has distinct labels for vessel and aneurysm (e.g., 1 = vessel, 2 = aneurysm).

The pipeline assumes the input is already roughly aligned; no full affine registration is performed unless added.

📸 Example Output
(insert example 2D projection images or plots here if desired)

📄 License
MIT License


