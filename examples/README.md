# Individualized Functional Network Example

This directory provides a minimal example for constructing individualized
functional networks from postprocessed resting-state fMRI data.

## Contents

- `IFN_example.py`: example analysis script.
- `data/dataset/mcad/`: example cortical time-series inputs.
- `data/dataset/mcad_tsnr/`: example temporal signal-to-noise ratio inputs.
- `data/outputs/`: output directory created when the example is executed.

## Requirements

- Python 3.x
- NumPy
- SciPy
- NiBabel

Install the required packages with:

```bash
pip install numpy scipy nibabel
```

## Input data

Two example participants, `sub-0001` and `sub-0002`, are included.

Each participant requires:

1. `{subject}_timeframes_fs4.mat`
   - Contains `lhData` and `rhData`.
   - Each array has shape `(2562, n_timepoints)`.

2. `{subject}_tsnr_fs4.mat`
   - Contains the variable `data`.
   - The array contains 5124 values ordered as left-hemisphere vertices
     followed by right-hemisphere vertices.

## Running the example

From the repository root, run:

```bash
python examples/IFN_example.py
```

## Outputs

The script creates participant-specific output directories under:

```text
examples/data/outputs/
```

The generated files contain individualized network parcellations,
confidence values, and correlation values.

## Notes

Large intermediate neuroimaging files are not included in this repository.
The included MATLAB files provide the minimal inputs required by this example.