import os
from nilearn import image as nimg
from nilearn import plotting as nplot
import numpy as np
import nibabel as nib
import pandas as pd
import glob
import tqdm

images = glob.glob('./fmriprep/sub*/ses-1/func/*MNI152NLin6Asym_res-2_*_bold.nii.gz')
#images = [
    #'./fmriprep/sub-0130/ses-1/func/sub-0130_ses-1_task-rest_run-0_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz',
    #'./fmriprep/sub-0231/ses-1/func/sub-0231_ses-1_task-rest_run-0_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz',
    #'./fmriprep/sub-0881/ses-1/func/sub-0881_ses-1_task-rest_run-0_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz',
    
    
    #]
print('Image number:',str(len(images)))
for image in tqdm.tqdm(images):
    #print(image)
    dirname = os.path.dirname(image)
    filename = os.path.basename(image)
    filenames = filename.split('_')
    saveDir = './post-process/clean_imgs/%s/' % filenames[0]
    if(not os.path.exists(saveDir)):
        os.makedirs(saveDir)
    else:
        with open('./postProcess_Log_exist_fs.txt' ,'a') as f:
            f.write('Image: {} \n'.format(image))
        #continue
    with open('./postProcess_Log_fs.txt' ,'a') as f:
        f.write('Start : Image: {} \n'.format(image))
    #sub-0001_ses-1_task-rest_run-0_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz
    #sub-0001_ses-1_task-rest_run-0_desc-confounds_timeseries.tsv
    #sub-0001_ses-1_task-rest_run-0_space-MNI152NLin6Asym_res-2_desc-brain_mask.nii.gz
    confound = os.path.join(dirname,"%s_%s_%s_%s_desc-confounds_timeseries.tsv" % (filenames[0],filenames[1],filenames[2],filenames[3]))
    #print(confound)
    mask = os.path.join(dirname,"%s_%s_%s_%s_%s_%s_desc-brain_mask.nii.gz" % (filenames[0],filenames[1],filenames[2],filenames[3],filenames[4],filenames[5]))
    #print(mask)
    confound_df = pd.read_csv(confound,delimiter='\t')
    confound_vars = ['trans_x','trans_y','trans_z','rot_x','rot_y','rot_z','global_signal','a_comp_cor_01','a_comp_cor_02']
    confound_vars = ['csf','white_matter','trans_x','trans_y','trans_z','rot_x','rot_y','rot_z']
    confound_gsr_vars = ['csf','white_matter','trans_x','trans_y','trans_z','rot_x','rot_y','rot_z', 'global_signal',] 
    confound_gsr_df = confound_df[confound_gsr_vars]
    confound_df = confound_df[confound_vars]
    
    for col in confound_gsr_df.columns:
        new_name = '{}_dt'.format(col)
        new_col = confound_gsr_df[col].diff()
        confound_gsr_df[new_name] = new_col
        confound_gsr_df.head()

    for col in confound_df.columns:
        new_name = '{}_dt'.format(col)
        new_col = confound_df[col].diff()
        confound_df[new_name] = new_col
        confound_df.head()

    raw_func_img = nimg.load_img(image)
    func_img = raw_func_img#.slicer[:,:,:,10:]
    drop_confound_df = confound_df#.loc[10:]
    drop_confound_gsr_df = confound_gsr_df#.loc[10:]
    
    confounds_matrix = drop_confound_df.values
    confounds_matrix[0,8:16] = 0
    if(np.sum(np.isnan(confounds_matrix))>0):
        with open('./postProcess_Log_confound_error_fs.txt' ,'a') as f:
            f.write('Image: {} \n'.format(image))
        continue

    confounds_gsr_matrix = drop_confound_gsr_df.values
    confounds_gsr_matrix[0,8:18] = 0

    high_pass= 0.01
    low_pass = 0.08
    t_r = 2
    clean_img = nimg.clean_img(func_img,confounds=confounds_matrix,detrend=True,standardize=True,
                            low_pass=low_pass,high_pass=high_pass,t_r=t_r, mask_img=mask)
    clean_gsr_img = nimg.clean_img(func_img,confounds=confounds_gsr_matrix,detrend=True,standardize=True,
                            low_pass=low_pass,high_pass=high_pass,t_r=t_r, mask_img=mask)

    
    
    cleanName = os.path.join(saveDir,"%s_%s_%s_%s_%s_%s_desc-noGSR_fs.nii.gz" % (filenames[0],filenames[1],filenames[2],filenames[3],filenames[4],filenames[5]))
    clean_img.to_filename(cleanName)

    cleanName = os.path.join(saveDir,"%s_%s_%s_%s_%s_%s_desc-GSR_fs.nii.gz" % (filenames[0],filenames[1],filenames[2],filenames[3],filenames[4],filenames[5]))
    clean_gsr_img.to_filename(cleanName)

    
    #smooth_img = nimg.smooth_img(clean_img,[6,6,6])
    #cleanName = os.path.join(saveDir,"%s_%s_%s_%s_%s_%s_desc-S6noGSR.nii.gz" % (filenames[0],filenames[1],filenames[2],filenames[3],filenames[4],filenames[5]))
    #smooth_img.to_filename(cleanName)

    #smooth_img = nimg.smooth_img(clean_gsr_img,[6,6,6])
    #cleanName = os.path.join(saveDir,"%s_%s_%s_%s_%s_%s_desc-S6GSR.nii.gz" % (filenames[0],filenames[1],filenames[2],filenames[3],filenames[4],filenames[5]))
    #smooth_img.to_filename(cleanName)
    
    with open('./postProcess_Log_fs.txt' ,'a') as f:
        f.write('End : Image: {} \n'.format(image))
        f.write('Confounds: {} \n'.format(confound))
        f.write('mask: {} \n'.format(mask))
        f.write('\n')
    
    


