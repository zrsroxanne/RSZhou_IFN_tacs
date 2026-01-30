# Functional Hierarchy Disruption in Alzheimer’s Disease

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository provides the official Python implementation for the study:  
**"Functional Hierarchy Disruption: A Mechanistic Link to Cognitive Decline and Treatment Targets in Alzheimer’s Disease"**.
This repository is currently being populated with the scripts and data processing pipelines used in our study. We are performing a final cleanup of the code (e.g., removing absolute paths, improving documentation) to ensure full reproducibility for the peer-review process.

All core algorithms and analysis scripts will be fully uploaded shortly.

Status: Repository initialization and code sanitization in progress.

## 📌 Project Overview
This project characterizes the disruption of macroscale functional hierarchy as a central mechanism in Alzheimer's Disease (AD). By linking neural hierarchy alterations to cognitive decline, the codebase provides tools to identify potential neurostimulation (e.g., tACS) and pharmacological treatment targets.

---

## 📂 Repository Structure

The pipeline is organized into modular directories corresponding to the neuroimaging analysis workflow:

| Directory / File | Description |
| :--- | :--- |
| **`postpreprocess/`** | **Data Refinement.** Scripts for post-processing fMRI data (denoising, filtering). |
| └── `postProcess.py` | Main script for preparing BOLD time-series. |
| **`individualized_network/`** | **Individualized Mapping.** Precision mapping of functional brain networks. |
| └── `my_iterative_code.py` | Core iterative algorithm for subject-specific functional boundaries. |
| **`Analysis/`** | **Statistical Analysis.** Main computational pipeline and figure generation. |
| └── `main5.ipynb` | Notebook for hierarchy calculation and clinical correlation analysis. |
| **`Plos_ref/`** | **Dependency Library.** Custom functions for data handling and matrix operations. |

---

## 🚀 Getting Started

### 1. Prerequisites
Ensure your environment meets the following requirements:
* **Python**: 3.8 or higher
* **Core Libraries**: `numpy`, `scipy`, `pandas`, `nibabel`, `nilearn`, `scikit-learn`
* **Interface**: Jupyter Notebook (for `.ipynb` execution)

### 2. Execution Workflow
To replicate the findings, execute the scripts in the following sequence:

1. **Preprocessing**:  
   Run `postpreprocess/postProcess.py` to refine raw imaging data into analysis-ready BOLD signals.
2. **Network Individualization**:  
   Execute `my_iterative_code.py` to generate individualized functional maps, capturing subject-specific topography in AD patients.
3. **Hierarchy Analysis**:  
   Open `Analysis/main5.ipynb` to calculate functional gradients and perform cognitive correlation analyses.

> [!IMPORTANT]  
> All scripts depend on the **`Plos_ref/`** utility package. Ensure this directory is added to your Python path or kept in the root folder.

---

## 📊 Methodology Highlights
The implementation follows a multi-scale framework:
* **Manifold Learning**: Applying gradient mapping to quantify cortical organization.
* **Hierarchy Modeling**: Calculating the geodesic distance between primary and transmodal regions.
* **Target Identification**: Linking neural disruptions to therapeutic intervention points.

---

## 📜 Citation
If you find this work or the code helpful for your research, please cite our preprint:

```bibtex
@article{zhao2025functional,
  title={Functional Hierarchy Disruption: A Mechanistic Link to Cognitive Decline and Treatment Targets in Alzheimer’s Disease},
  author={Zhao, et al.},
  year={2025},
  doi={10.21203/rs.3.rs-8184263/v1},
  journal={Research Square},
  note={Preprint}
}
