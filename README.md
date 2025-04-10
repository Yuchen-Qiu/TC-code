# Optimal Projection Angle Finder for Brain Aneurysms

This project automatically determines the **optimal projection angle** for visualizing brain aneurysms in CT images. It leverages a combination of geometric analysis, image processing, and model-based scoring — including a **Visual Attention Simulation (VAS)** model and an **Iterative Observer** model — to find the angle that yields the clearest and most informative view.

## 🧠 How It Works

1. **Input:** A CTA scan segmentation file that contains a mask where all vasculature is represented by a 1 and all aneurysm are represented by 2.
2. **Preprocessing:** Checks for multiple aneurysms in one mask, isolates these aneurysm for individual analysis.
3. **Angle Optimization:** Uses the models (VAS/Iterative Observer) to determine the optimal projection angle of intercranial aneurysm for DSA.
4. **Output:** Best-view angle and scoring written to `.txt` file.

## 🛠️ Installation

1. Clone the repo:
   ```bash
   git clone https://github.com/yourusername/aneurysm-angle-finder.git
   cd aneurysm-angle-finder

2. Install required dependencies:
   ```bash
   pip install -r requirements.txt

## 🧪 Usage
You can run the angle optimization through a script or the provided Jupyter notebook. Implementing this software through a script could look like this:

from aneurysm_angle_finder.angle_optimizer import AngleOptimizer

```bash
optimizer = AngleOptimizer(
    input_path="data/sample_case/mask.nii.gz"
)

optimizer.isolate_aneurysms()
optimizer.run_VAS_model()
optimizer.run_iterative_observer()
optimizer.write_outputfile()
```

## 📝 Output Format

Executing write_outputfile() generates a .txt file in a specified results folder using the following format:

```bash
--- VAS Model Results ---
=Aneurysm [1/1]=
  - Final optimal DSA Rotation Angle (ρ): X°
  - Final optimal DSA Angulation Angle (ψ): Y°

--- Iterative Observer Model Results ---
=Aneurysm [1/1]=
  Shortest connection:
    - Final optimal DSA Rotation Angle (ρ): X°
    - Final optimal DSA Angulation Angle (ψ): Y°
  Longest connection:
    - Final optimal DSA Rotation Angle (ρ): X°
    - Final optimal DSA Angulation Angle (ψ): Y°
  Largest aneurysm:
    - Final optimal DSA Rotation Angle (ρ): X°
    - Final optimal DSA Angulation Angle (ψ): Y°
```

