<p align="center">
  <img src="Resources/AAL3BrainLabeling.png" alt="AAL3BrainLabeling Logo" width="450"/>
</p>

# AAL3BrainLabeling: Automated Anatomical Labeling 3 and Distance-Connectome Pipeline for 3D Slicer

**AAL3BrainLabeling** is a robust, high-fidelity neuroimaging pipeline designed as an extension for **3D Slicer**. Developed for publication-quality neuroanatomy research, it seamlessly maps the **AAL3** (Automated Anatomical Labeling 3) atlas to patient-specific MRI data.

The pipeline handles bias field correction, multimodal image registration, automated anatomical segmentation, morphometric statistics extraction, hemispheric asymmetry calculation, and structural connectome generation.

## 🌟 Key Features

1. **N4ITK Bias Field Correction:** Harmonizes MRI intensities to remove radio-frequency (RF) coil inhomogeneities.
2. **High-Fidelity Elastix Registration:** Employs an optimized, multi-stage registration approach (Rigid → Affine → B-Spline) mapping the MNI152 template to the patient's native anatomical space with 10,000 spatial samples for peak cortical precision.
3. **Automated Atlas Warping:** Accurately deforms the AAL3 atlas to fit the patient's specific cortical and subcortical geometry.
4. **Morphometric Extraction:** Calculates the volume (mm<sup>3</sup>) and mean intensity for every predefined anatomical region.
5. **Hemispheric Asymmetry Analysis:** Automatically computes the Asymmetry Index (AI) for corresponding left and right hemispheric structures.
6. **Centroid-Based Connectomics:** Generates a complete Euclidean distance-based structural connectome matrix using exact regional centroids.
7. **Batch Processing:** Includes a fully automated batch processing module for cohort studies.
8. **Segment Editor Integration:** Automatically renders 3D cortical surfaces with standard AAL3 native nomenclature upon completion.

## 🧮 Methodology & Formulations


### ⚖️ Asymmetry Index (AI)

The pipeline automatically computes the Asymmetry Index for laterally paired anatomical structures to evaluate morphological dominance:

$$AI = \frac{Volume_L - Volume_R}{Volume_L + Volume_R + 10^{-6}}$$

*(A positive AI indicates leftward asymmetry, whereas a negative AI indicates rightward asymmetry).*


### 🕸️ Distance Connectome

The structural distance matrix is constructed by calculating the Euclidean distance between the geometric centroids of every region $i$ and $j$:

$$d(i, j) = \sqrt{(x_i - x_j)^2 + (y_i - y_j)^2 + (z_i - z_j)^2}$$


## 🖼️ Visual Validation and Output Example

Below is a demonstration of the AAL3BrainLabeling pipeline output as rendered in 3D Slicer's Segment Editor module.

<p align="center">
  <img src="Screenshots/Screenshot.png" alt="AAL3BrainLabeling Pipeline Output in 3D Slicer" width="800"/>
  <br>
  <i><b>Figure 1:</b> High-resolution 3D surface rendering generated automatically from a patient's T1 MRI. Note the smooth non-linear conformity to cortical geometry (preventing tissue 'melting') and the preservation of native AAL3 nomenclature in the segment list.</i>
</p>

## 📂 Repository Structure

```text
SlicerAAL3BrainLabeling/
├── AAL3BrainLabeling.json
├── AAL3BrainLabeling.py
├── CMakeLists.txt
├── LICENSE.txt
├── README.md
├── Screenshots/
│   └── Screenshot.png
└── Resources/
    ├── AAL3BrainLabeling.png
    ├── Templates/
    │   └── MNI152_T1_1mm.nii.gz
    ├── Atlas/
    │   ├── AAL3v1_1mm.nii.gz
    │   └── AAL3_ColorTable.ctbl
    └── Labels/
        └── AAL3_labels.csv

## ⚙️ Requirements \& Installation
1. **3D Slicer:** Version 5.10.0 or higher is recommended.
2. **SlicerElastix:** You must install the SlicerElastix extension via the 3D Slicer Extension Manager.

### 🛠️ Installation Steps
1. Clone this repository to your local machine.
2. Open 3D Slicer.
3. Navigate to **Edit** → **Application Settings** → **Modules**.
4. Add the cloned `SlicerAAL3BrainLabeling` folder to your **Additional Module Paths**.
5. Restart 3D Slicer. The module will be available under the **Neuroimaging** category.

## 🚀 Usage
### 👤 Single Patient Analysis
1. Select the patient's T1-weighted MRI volume from the **Input MRI** dropdown.
2. Choose an **Output Folder** for the resulting CSV files.
3. Click **Run FULL Pipeline**.
4. Upon completion, Slicer will automatically switch to the Segment Editor and display the 3D brain map.

### 📂 Batch Processing
1. Place all T1-weighted MRI files (`.nii` or `.nii.gz`) into a single directory.
2. Select your designated **Output Folder**.
3. Click **Batch Process Folder** and select the directory containing your MRI files.

## 👨‍🔬 Contributors
Dr. Mustafa Sakci
Prof. Dr. Niyazi Acer 

## 📚 Acknowledgments \& Citations
If you use this pipeline in your research, please ensure you cite the foundational AAL3 atlas paper:
Rolls, E. T., Huang, C. C., Lin, C. P., Feng, J., \& Joliot, M. (2020). Automated anatomical labelling atlas 3. Neuroimage, 206, 116189.
