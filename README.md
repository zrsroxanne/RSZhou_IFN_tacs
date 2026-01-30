# Functional Hierarchy Disruption: A Mechanistic Link to Cognitive Decline and Treatment Targets in Alzheimer’s Disease

This repository contains the official implementation for the study investigating how macroscale cortical gradients and functional hierarchy are altered in Alzheimer's Disease (AD).

## 📌 Project Overview
The project explores the disruption of functional brain hierarchy as a core mechanism in AD, linking these neural changes to cognitive decline and identifying potential neurostimulation (e.g., tACS) or pharmacological treatment targets.

---

## 📂 Repository Structure

The codebase is organized into functional modules corresponding to different stages of the neuroimaging analysis pipeline:

| Folder / File | Description |
| :--- | :--- |
| **`postpreprocess/`** | **Data Refinement.** Post-processing scripts for fMRI data. |
| └── `postProcess.py` | Main script for denoising, filtering, and preparing BOLD signals. |
| **`individualized_network/`** | **Brain Mapping.** Implementation of individualized brain network construction. |
| └── `my_iterative_code.py` | Core iterative algorithm to define person-specific functional boundaries. |
| **`Analysis/`** | **Statistical Analysis.** Main results and figure generation. |
| └── `main5.ipynb` | Jupyter Notebook for hierarchy calculations and cognitive correlation analysis. |
| **`Plos_ref/`** | **Toolbox & Utilities.** Essential functions for data computation and handling. |

---

## 🚀 Getting Started

### 1. Prerequisites
Ensure you have the following environments set up:
* Python 3.x
* Required libraries: `numpy`, `scipy`, `pandas`, `nibabel`, `nilearn`, `scikit-learn`
* Jupyter Notebook (to run `.ipynb` files)

### 2. Workflow Execution
To replicate the results, follow this order:

1.  **Preprocessing**: Run `postpreprocess/postProcess.py` to clean and refine the imaging data.
2.  **Network Individualization**: Use `my_iterative_code.py` to generate individualized functional maps. This step is crucial for capturing person-specific variations in AD.
3.  **Hierarchy Analysis**: Execute `main5.ipynb` to perform the functional gradient analysis and link neural changes to clinical cognitive scores.

> **Note**: All scripts rely on the utility functions in `Plos_ref/`. Ensure this directory is in your Python path or maintained in the root structure.

---

## 📊 Methodology Highlights

The code implements a multi-step framework:
* **Gradient Mapping**: Quantifying the macroscale organization of the human cortex.
* **Hierarchy Modeling**: Measuring the "distance" between primary sensory areas and transmodal regions.
* **Clinical Correlation**: Identifying how hierarchy disruption serves as a link to cognitive decline.



---

## 📜 Citation
If you use this code or find our research helpful, please cite our work:

## ✉️ Contact
For questions regarding the code or data, please open an issue or contact the project lead.
