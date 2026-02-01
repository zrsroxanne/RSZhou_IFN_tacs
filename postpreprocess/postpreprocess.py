# =============================================================================
# Post-fMRIPrep preprocessing
#
# All fMRI data were first preprocessed with fMRIPrep v23.1.4.
#
# Before running:
# 1) fMRIPrep output folder (BIDS derivatives).
# 2) fMRIPrep confounds, loaded via nilearn.interfaces.fmriprep.load_confounds.
# 3) If you run CIFTI/surface smoothing or resampling: Workbench (wb_command) and
#    FreeSurfer (mri_surf2surf, mri_convert).
#
# What this script does:
# - Loads nuisance regressors from fMRIPrep (motion + WM/CSF), optionally adds global signal.
# - Adds simple trends (linear, quadratic).
# - Applies nuisance regression + band-pass filtering.
# - Saves cleaned outputs for downstream analyses.
# =============================================================================

from nilearn.interfaces.fmriprep import load_confounds
from nilearn import image as nimg
import glob
import pandas as pd
import numpy as np
import os
import nibabel as nib
from nilearn.signal import clean
import tqdm
from neuromaps.utils import run
import json

# =============================================================================
# User config
# =============================================================================

# fMRIPrep derivatives directory, containing sub-*/ses-*/func/*
FMRIPREP_DIR = "/path/to/fmriprep/derivatives/fmriprep"
BIDS_DIR = "/path/to/BIDS_dataset_root"
# Output directory
OUT_DIR = "/path/to/output/clean_imgs"

# Optional: a template NIfTI used to make volume outputs have the same shape
VOLUME_TEMPLATE = None  # e.g. "/path/to/tpl-MNI152NLin2009cAsym_res-02_desc-brain_T1w.nii.gz"

# Optional: a standard mask used for volume resampling target
STANDARD_MASK_3MM = None  # e.g. "/path/to/standard_mesh_atlases/mni_wb_mask_3mm.nii"

# Workbench surfaces for CIFTI smoothing (only used in smooth_cifti)
WB_LEFT_SURF = "/path/to/fsaverage_LR32k/fsaverage.L.midthickness.32k_fs_LR.surf.gii"
WB_RIGHT_SURF = "/path/to/fsaverage_LR32k/fsaverage.R.midthickness.32k_fs_LR.surf.gii"

HIGH_PASS = 0.01
LOW_PASS = 0.08
DEFAULT_TR = 2.0
SMOOTH_FWHM_CIFTI = 4
SMOOTH_FWHM_VOLUME = 6

# =============================================================================
# Core functions
# =============================================================================

def get_confound(image_name):
    confounds_no_gs, _ = load_confounds(
        image_name,
        strategy=['motion', 'wm_csf'],
        motion='derivatives',
        wm_csf='derivatives',
        demean=False
    )
    confounds_no_gs['trend'] = np.linspace(1, confounds_no_gs.shape[0], confounds_no_gs.shape[0])
    confounds_no_gs['trend2'] = confounds_no_gs['trend'] ** 2

    confounds_gs, _ = load_confounds(
        image_name,
        strategy=['motion', 'wm_csf', 'global_signal'],
        motion='derivatives',
        wm_csf='derivatives',
        global_signal='derivatives',
        demean=False
    )
    confounds_gs['trend'] = np.linspace(1, confounds_gs.shape[0], confounds_gs.shape[0])
    confounds_gs['trend2'] = confounds_gs['trend'] ** 2

    return confounds_no_gs.values, confounds_gs.values


def GSR_filter(image, save_dir, confound_no_gs, confound_gs, sub_name_idx=7,
               high_pass=HIGH_PASS, low_pass=LOW_PASS, t_r=DEFAULT_TR):

    save_img_dir = os.path.join(save_dir, image.split('/')[sub_name_idx])
    os.makedirs(save_img_dir, exist_ok=True)

    filename = os.path.basename(image).split('.')[0]
    file_no_gs = os.path.join(save_img_dir, filename + '_filter.dtseries.nii')
    file_gsr = os.path.join(save_img_dir, filename + '_filter_gsr.dtseries.nii')

    if os.path.exists(file_gsr):
        return [file_no_gs, file_gsr]

    img = nib.load(image)
    image_data = np.asarray(img.get_fdata())

    clean_data = clean(
        image_data, confounds=confound_no_gs, detrend=False, standardize=True,
        low_pass=low_pass, high_pass=high_pass, t_r=t_r
    )
    clean_gsr_data = clean(
        image_data, confounds=confound_gs, detrend=False, standardize=True,
        low_pass=low_pass, high_pass=high_pass, t_r=t_r
    )

    nib.Cifti2Image(clean_data, img.header).to_filename(file_no_gs)
    nib.Cifti2Image(clean_gsr_data, img.header).to_filename(file_gsr)
    return [file_no_gs, file_gsr]


def _read_tr_from_bids_json(bids_dir, sub, run_id, ses="ses-1", task="rest", tr_default=DEFAULT_TR):
    if bids_dir is None:
        return tr_default
    json_path = os.path.join(bids_dir, sub, ses, "func", f"{sub}_{ses}_task-{task}_{run_id}_bold.json")
    if not os.path.exists(json_path):
        return tr_default
    with open(json_path, "r") as f:
        jinfo = json.load(f)
    return float(jinfo.get("RepetitionTime", tr_default))


def GSR_filter_gii(image, save_dir, confound_no_gs, confound_gs, sub_name_idx=7,
                   bids_dir=BIDS_DIR, high_pass=HIGH_PASS, low_pass=LOW_PASS,
                   tr_default=DEFAULT_TR):

    save_img_dir = os.path.join(save_dir, image.split('/')[sub_name_idx])
    os.makedirs(save_img_dir, exist_ok=True)

    filename = os.path.basename(image).split('.')[0]
    file_gsr = os.path.join(save_img_dir, filename + '_filter_gsr.func.gii')

    img = nib.load(image)
    image_data = np.asarray(img.agg_data()).squeeze().T  # (timepoints, vertices)

    run_id = os.path.basename(image).split('_')[3]  # e.g. "run-0"
    sub = image.split('/')[sub_name_idx]
    t_r = _read_tr_from_bids_json(bids_dir=bids_dir, sub=sub, run_id=run_id, tr_default=tr_default)

    clean_gsr_data = clean(
        image_data, confounds=confound_gs, detrend=False, standardize=True,
        low_pass=low_pass, high_pass=high_pass, t_r=t_r
    )

    simg = nib.gifti.GiftiImage(header=img.header)
    for i in range(clean_gsr_data.shape[0]):
        data = nib.gifti.GiftiDataArray(clean_gsr_data[i], intent='NIFTI_INTENT_TIME_SERIES')
        simg.add_gifti_data_array(data)
    simg.to_filename(file_gsr)

    return [file_gsr]


def get_nifti_mask(image=None, res=3, standard_mask_path=STANDARD_MASK_3MM):
    """
    Prefer using the fMRIPrep brain mask next to the BOLD file.
    If standard_mask_path is provided, use it as the mask (useful for resampling target).
    """
    if standard_mask_path is not None:
        return standard_mask_path

    if image is None:
        raise ValueError("Either provide image or set STANDARD_MASK_3MM.")

    dirname = os.path.dirname(image)
    filename = os.path.basename(image)

    if "_desc-preproc_bold.nii.gz" in filename:
        mask_name = filename.replace("_desc-preproc_bold.nii.gz", "_desc-brain_mask.nii.gz")
        mask = os.path.join(dirname, mask_name)
        return mask

    filenames = filename.split('_')
    mask = os.path.join(
        dirname,
        "%s_%s_%s_%s_%s_%s_desc-brain_mask.nii.gz" % (
            filenames[0], filenames[1], filenames[2], filenames[3], filenames[4], filenames[5]
        )
    )
    return mask


def Regress_Resample_Filter_nifti(image, save_dir, confound_no_gs, confound_gs,
                                 sub_name_idx=7, target_res=3,
                                 high_pass=HIGH_PASS, low_pass=LOW_PASS,
                                 t_r=DEFAULT_TR, smooth_fwhm=SMOOTH_FWHM_VOLUME,
                                 template_path=VOLUME_TEMPLATE,
                                 standard_mask_path=STANDARD_MASK_3MM):

    save_img_dir = os.path.join(save_dir, image.split('/')[sub_name_idx])
    os.makedirs(save_img_dir, exist_ok=True)

    raw_func_img = nimg.load_img(image)
    func_img = raw_func_img

    if np.sum(np.isnan(confound_gs)) > 0:
        with open('./postProcess_Log_confound_error_fs.txt', 'a') as f:
            f.write('Image: {} \n'.format(image))
        return None

    # Decide mask
    mask_img_path = get_nifti_mask(image=image, res=target_res, standard_mask_path=standard_mask_path)
    if mask_img_path is not None and (not os.path.exists(mask_img_path)):
        raise FileNotFoundError(f"Mask not found: {mask_img_path}")

    # Optional: resample to standard mask or template to make shapes consistent
    # If template_path is given, we resample to template (recommended for "anyone can run").
    if template_path is not None:
        if not os.path.exists(template_path):
            raise FileNotFoundError(f"Template not found: {template_path}")
        func_img = nimg.resample_to_img(func_img, template_path)
        mask_for_clean = None if mask_img_path is None else nimg.resample_to_img(mask_img_path, template_path)
    else:
        # old behavior: resample to mask image if provided
        if mask_img_path is not None:
            func_img = nimg.resample_to_img(func_img, mask_img_path)
        mask_for_clean = mask_img_path

    clean_img = nimg.clean_img(
        func_img, confounds=confound_no_gs, detrend=False, standardize=True,
        low_pass=low_pass, high_pass=high_pass, t_r=t_r, mask_img=mask_for_clean
    )
    clean_gsr_img = nimg.clean_img(
        func_img, confounds=confound_gs, detrend=False, standardize=True,
        low_pass=low_pass, high_pass=high_pass, t_r=t_r, mask_img=mask_for_clean
    )

    filename = os.path.basename(image).split('.')[0]
    file_no_gs = os.path.join(save_img_dir, filename + f'_res-{target_res}_filter_s{smooth_fwhm}.nii.gz')
    file_gsr = os.path.join(save_img_dir, filename + f'_res-{target_res}_filter_gsr_s{smooth_fwhm}.nii.gz')

    if smooth_fwhm is not None and smooth_fwhm > 0:
        smooth_img = nimg.smooth_img(clean_img, [smooth_fwhm, smooth_fwhm, smooth_fwhm])
        smooth_img.to_filename(file_no_gs)

        smooth_img = nimg.smooth_img(clean_gsr_img, [smooth_fwhm, smooth_fwhm, smooth_fwhm])
        smooth_img.to_filename(file_gsr)
    else:
        clean_img.to_filename(file_no_gs)
        clean_gsr_img.to_filename(file_gsr)

    return [file_no_gs, file_gsr]


def smooth_gii(images, save_dir, hemi, name_idx=7, FWHM=4):
    new_images = []
    command = (
        "mri_surf2surf --srcsubject fsaverage5 --trgsubject fsaverage5 "
        "--hemi {hemi} --sval {image} --tval {out_image_file_name} --fwhm {fwhm}"
    )

    for image in images:
        subj_name = image.split('/')[name_idx]
        image_file_name = os.path.basename(image).split('.')[0]
        save_img_dir = os.path.join(save_dir, subj_name)
        os.makedirs(save_img_dir, exist_ok=True)
        out_image_file_name = os.path.join(save_img_dir, image_file_name + f"_fwmh{FWHM}.func.gii")
        new_images.append(out_image_file_name)

        run(command.format(image=image, hemi=hemi, fwhm=FWHM, out_image_file_name=out_image_file_name))

    return new_images


def smooth_cifti(images, save_dir, name_idx=7, FWHM=4,
                 left_surf=WB_LEFT_SURF, right_surf=WB_RIGHT_SURF):

    import math
    sigma = float(FWHM) / (2 * math.sqrt(2 * math.log(2)))

    command = (
        "wb_command -cifti-smoothing {image} {sigma} {sigma} COLUMN {out_image} "
        "-left-surface {left_surf} -right-surface {right_surf} -merged-volume"
    )

    new_images = []
    for image in images:
        subj_name = image.split('/')[name_idx]
        image_file_name = os.path.basename(image).split('.')[0]
        save_img_dir = os.path.join(save_dir, subj_name)
        os.makedirs(save_img_dir, exist_ok=True)

        out_image_file_name = os.path.join(save_img_dir, image_file_name + f"_fwmh{FWHM}.dtseries.nii")
        new_images.append(out_image_file_name)

        run(command.format(
            image=image, sigma=sigma, out_image=out_image_file_name,
            left_surf=left_surf, right_surf=right_surf
        ))

    return new_images

def cifti(fmriprep_dir=FMRIPREP_DIR, save_dir=OUT_DIR,
          pattern="sub-*/ses-1/func/sub-*_ses-1_task-rest_run-0_space-fsLR_den-91k_bold.dtseries.nii",
          sub_name_idx=6):
    image_names = sorted(glob.glob(os.path.join(fmriprep_dir, pattern)))
    for image_name in tqdm.tqdm(image_names, desc="cifti"):
        confound_no_gs, confound_gs = get_confound(image_name)
        names = GSR_filter(image_name, confound_no_gs=confound_no_gs, confound_gs=confound_gs,
                           save_dir=save_dir, sub_name_idx=sub_name_idx)
        names = smooth_cifti(names, save_dir=save_dir, name_idx=sub_name_idx + 1, FWHM=SMOOTH_FWHM_CIFTI)
        # fslr_to_fsa(names, save_dir=save_dir, name_idx=sub_name_idx + 1)  # keep your original call if needed


def nifti(info_csv_path,
          fmriprep_dir=FMRIPREP_DIR,
          save_dir=OUT_DIR,
          sub_name_idx=7,
          target_res=3):
    infotable = pd.read_csv(info_csv_path)
    image_name_par = os.path.join(
        fmriprep_dir, "{sub}", "ses-1", "func",
        "{sub}_ses-1_task-rest_run-0_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
    )

    for _, row in tqdm.tqdm(infotable.iterrows(), total=infotable.shape[0], desc="volume"):
        sub = row['sub']
        image_name = image_name_par.format(sub=sub)
        if not os.path.exists(image_name):
            continue
        confound_no_gs, confound_gs = get_confound(image_name)
        Regress_Resample_Filter_nifti(
            image_name, confound_no_gs=confound_no_gs, confound_gs=confound_gs,
            save_dir=save_dir, sub_name_idx=sub_name_idx,
            target_res=target_res,
            template_path=VOLUME_TEMPLATE,
            standard_mask_path=STANDARD_MASK_3MM
        )


if __name__ == "__main__":

    cifti(
        fmriprep_dir=FMRIPREP_DIR,
        save_dir=OUT_DIR,
        pattern="sub-*/ses-1/func/sub-*_ses-1_task-rest_run-0_space-fsLR_den-91k_bold.dtseries.nii",
        sub_name_idx=6
    )