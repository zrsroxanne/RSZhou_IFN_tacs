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
def get_confound(image_name):
    
    confounds_no_gs, sample_mask = load_confounds(
        image_name,
        strategy = ['motion','wm_csf'],
        motion = 'derivatives',
        wm_csf = 'derivatives',
        demean = False
    )
    confounds_no_gs['trend'] = np.linspace(1,confounds_no_gs.shape[0],confounds_no_gs.shape[0])
    confounds_no_gs['trend2'] = np.linspace(1,confounds_no_gs.shape[0],confounds_no_gs.shape[0])**2
    
    confounds_gs, sample_mask = load_confounds(
        image_name,
        strategy = ['motion','wm_csf','global_signal'],
        motion = 'derivatives',
        wm_csf = 'derivatives',
        global_signal= 'derivatives',
        demean = False
    )
    confounds_gs['trend'] = np.linspace(1,confounds_gs.shape[0],confounds_gs.shape[0])
    confounds_gs['trend2'] = np.linspace(1,confounds_gs.shape[0],confounds_gs.shape[0])**2

    return confounds_no_gs.values,confounds_gs.values

def GSR_filter(image,save_dir,confound_no_gs,confound_gs,sub_name_idx=7):

    save_img_dir = save_dir + image.split('/')[sub_name_idx] + '/'
    print(save_img_dir)
    if(not os.path.exists(save_img_dir)):
        os.mkdir(save_img_dir)
    filename = os.path.basename(image).split('.')[0]
    file_no_gs = save_img_dir+ filename + '_filter.dtseries.nii'
    file_gsr = save_img_dir+ filename +  '_filter_gsr.dtseries.nii'
    if(os.path.exists(file_gsr)):
        return [file_no_gs,file_gsr]
        
    img = nib.load(image)
    image_data = nib.load(image).get_fdata()
    image_data = np.array(image_data)
    
    high_pass= 0.01
    low_pass = 0.08
    t_r = 2

    clean_data = clean(image_data,confounds=confound_no_gs,detrend=False,standardize=True,
                            low_pass=low_pass,high_pass=high_pass,t_r=t_r)
    
    clean_gsr_data = clean(image_data,confounds=confound_gs,detrend=False,standardize=True,
                            low_pass=low_pass,high_pass=high_pass,t_r=t_r)
    
    nib.Cifti2Image(clean_data,img.header).to_filename(file_no_gs)
    nib.Cifti2Image(clean_gsr_data,img.header).to_filename(file_gsr)

    return [file_no_gs,file_gsr]

def GSR_filter_gii(image,save_dir,confound_no_gs,confound_gs,sub_name_idx=7):

    save_img_dir = save_dir + image.split('/')[sub_name_idx] + '/'
    print(save_img_dir)
    if(not os.path.exists(save_img_dir)):
        os.mkdir(save_img_dir)
    filename = os.path.basename(image).split('.')[0]
    file_no_gs = save_img_dir+ filename + '_filter.func.gii'
    file_gsr = save_img_dir+ filename +  '_filter_gsr.func.gii'
    # if(os.path.exists(file_gsr)):
    #     pass
    #     return [file_no_gs,file_gsr]
    img = nib.load(image)
    image_data = nib.load(image).agg_data()
    image_data = np.array(image_data).squeeze().T
    
    high_pass= 0.01
    low_pass = 0.08
    dataset = save_dir.split('/')[4]
    run = os.path.basename(image).split('_')[3]
    with open('/n01dat01/pdchen/Dataset/{dataset}/BIDS/{sub}/ses-1/func/{sub}_ses-1_task-rest_{run}_bold.json'.format(dataset=dataset,sub=image.split('/')[sub_name_idx],run=run)) as f:
        jinfo = json.load(f)
    t_r = jinfo['RepetitionTime']

    clean_data = clean(image_data,confounds=confound_no_gs,detrend=False,standardize=True,
                            low_pass=low_pass,high_pass=high_pass,t_r=t_r)
    
    clean_gsr_data = clean(image_data,confounds=confound_gs,detrend=False,standardize=True,
                            low_pass=low_pass,high_pass=high_pass,t_r=t_r)
    
    simg = nib.gifti.GiftiImage(header=img.header)
    data = nib.gifti.GiftiDataArray(clean_data.T)
    simg.add_gifti_data_array(data)
    #simg.to_filename(file_no_gs)

    simg = nib.gifti.GiftiImage(header=img.header)
    for i in range(clean_gsr_data.shape[0]):
        data = nib.gifti.GiftiDataArray(clean_gsr_data[i],intent='NIFTI_INTENT_TIME_SERIES')
        simg.add_gifti_data_array(data)
    simg.to_filename(file_gsr)

    return [file_gsr]

def get_nifti_mask(image=None,res=2):

    if image:
        dirname = os.path.dirname(image)
        filename = os.path.basename(image)
        filenames = filename.split('_')
        mask = os.path.join(dirname,"%s_%s_%s_%s_%s_%s_desc-brain_mask.nii.gz" % (filenames[0],filenames[1],filenames[2],filenames[3],filenames[4],filenames[5]))
        return mask
    else:
        return "standard_mesh_atlases/mni_wb_mask_%dmm.nii" %res


def Regress_Resample_Filter_nifti(image,save_dir,confound_no_gs,confound_gs,sub_name_idx=7,target_res=3):

    save_img_dir = save_dir + image.split('/')[sub_name_idx] + '/'
    if(not os.path.exists(save_img_dir)):
        os.mkdir(save_img_dir)

    raw_func_img = nimg.load_img(image)
    func_img = raw_func_img#.slicer[:,:,:,10:]
    confound_no_gs = confound_no_gs#[10:]
    confound_gs = confound_gs#[10:]
    
    if(np.sum(np.isnan(confound_gs))>0):
        with open('./postProcess_Log_confound_error_fs.txt' ,'a') as f:
            f.write('Image: {} \n'.format(image))
        return

    high_pass= 0.01
    low_pass = 0.08
    t_r = 2

    mask_img = get_nifti_mask(res=target_res)
    func_img = nimg.resample_to_img(func_img,mask_img)

    clean_img = nimg.clean_img(func_img,confounds=confound_no_gs,detrend=False,standardize=True,
                            low_pass=low_pass,high_pass=high_pass,t_r=t_r, mask_img=mask_img)
    
    clean_gsr_img = nimg.clean_img(func_img,confounds=confound_gs,detrend=False,standardize=True,
                            low_pass=low_pass,high_pass=high_pass,t_r=t_r, mask_img=mask_img)

    filename = os.path.basename(image).split('.')[0]
    file_no_gs = save_img_dir+ filename + '_res-3_filter_s6.nii.gz'
    file_gsr = save_img_dir+ filename +  '_res-3_filter_gsr_s6.nii.gz'

    # clean_img.to_filename(cleanName)
    # clean_gsr_img.to_filename(cleanName)
    
    smooth_img = nimg.smooth_img(clean_img,[6,6,6])
    smooth_img.to_filename(file_no_gs)

    smooth_img = nimg.smooth_img(clean_gsr_img,[6,6,6])
    smooth_img.to_filename(file_gsr)
    
    return [file_no_gs,file_gsr]

def smooth_gii(images,save_dir,hemi,name_idx=7,FWHM=4):
    import time

    new_images = []
    command = "mri_surf2surf --srcsubject fsaverage5 --trgsubject fsaverage5 \
               --hemi {hemi} \
               --sval {image}  \
               --tval {out_image_file_name} \
               --fwhm {fwhm}"
    
    for image in images:

        subj_name = image.split('/')[name_idx]
        image_file_name = os.path.basename(image).split('.')[0]
        save_img_dir = save_dir + subj_name
        out_image_file_name = save_img_dir + '/' +  image_file_name + '_fwmh%d.func.gii' % FWHM
        new_images.append(out_image_file_name)
        
        run(command.format(image=image,hemi=hemi,fwhm=FWHM,out_image_file_name=out_image_file_name))
    print('smooth------------------------------')
    print(new_images)
    return new_images

def smooth_cifti(images,save_dir,name_idx=7,FWHM=4):
    
    new_images = []
    import math
    sigma = float(FWHM) / (2 * math.sqrt(2*math.log(2)))

    command = "wb_command -cifti-smoothing {image} {sigma} {sigma} COLUMN  {out_image}  \
              -left-surface  /n01dat01/pdchen/Dataset/code/standard_mesh_atlases/fsaverage_LR32k/fsaverage.L.midthickness.32k_fs_LR.surf.gii  \
              -right-surface /n01dat01/pdchen/Dataset/code/standard_mesh_atlases/fsaverage_LR32k/fsaverage.R.midthickness.32k_fs_LR.surf.gii  \
              -merged-volume"
    
    for image in images:

        subj_name = image.split('/')[name_idx]
        image_file_name = os.path.basename(image).split('.')[0]
        save_img_dir = save_dir + subj_name
        out_image_file_name = save_img_dir + '/' +  image_file_name + '_fwmh%d.dtseries.nii' % FWHM
        new_images.append(out_image_file_name)
        # if(os.path.exists(out_image_file_name)):
        #     continue
    
        run(command.format(image=image,sigma=sigma,out_image=out_image_file_name))
    print('smooth------------------------------')
    print(new_images)
    return new_images

def resample_fsaverage_to_fs5(func_l,func_r):
    from neuromaps.transforms import fsaverage_to_fsaverage
    import os
    #orl = /n01dat01/pdchen/Dataset/MCAD/fmriprep_2314/sub-0001/ses-1/func/sub-0001_ses-1_task-rest_run-0_hemi-L_space-fsaverage_bold.func.gii
    newl = func_l.replace("fsaverage","fsaverage5")
    newr = func_r.replace("fsaverage","fsaverage5")

    if(not os.path.exists(newl)):
        cmd = "mri_surf2surf --srcsubject fsaverage --srcval {func_src} --trgsubject fsaverage5 --trgval {func_target} --hemi {hemi}"
        for func,func_target,hemi in zip([func_l,func_r],[newl,newr],['lh','rh']):
            os.system(cmd.format(func_src=func,func_target=func_target,hemi=hemi))
    return newl,newr
def resample_fsaverage5_to_fs4(func_l,func_r,if_path=False,temp_name='temp_fsa5_to_fsa4'):
    import os   
    newl = temp_name + ".L.func.gii"
    newr = temp_name + ".R.func.gii"
    cmd = "mri_surf2surf --srcsubject fsaverage5 --sval {func_src} --trgsubject fsaverage4 --tval {func_target} --hemi {hemi}"
    for func,func_target,hemi in zip([func_l,func_r],[newl,newr],['lh','rh']):
        os.system(cmd.format(func_src=func,func_target=func_target,hemi=hemi))

    if(if_path):
        return newl,newr
    else:
        newl = nib.load(newl).agg_data()
        newr = nib.load(newr).agg_data()
        return newl,newr
    
def process_fsaverage():
    func_ls = sorted(glob.glob("/n01dat01/pdchen/Dataset/MCAD/fmriprep_2314/sub-*/ses-1/func/sub-*_ses-1_task-rest_run-0_hemi-L_space-fsaverage_bold.func.gii"))
    func_rs = sorted(glob.glob("/n01dat01/pdchen/Dataset/MCAD/fmriprep_2314/sub-*/ses-1/func/sub-*_ses-1_task-rest_run-0_hemi-R_space-fsaverage_bold.func.gii"))
    image_name_par = "/n01dat01/pdchen/Dataset/MCAD/fmriprep_2314/{sub}/ses-1/func/{sub}_ses-1_task-rest_run-0_space-fsLR_den-91k_bold.dtseries.nii"
    save_dir = "/n01dat01/pdchen/Dataset/MCAD/processed/clean_imgs/"
    for func_l,func_r in tqdm.tqdm(zip(func_ls,func_rs)):
        sub_l = func_l.split('/')[6]
        sub_r = func_r.split('/')[6]

        assert sub_l == sub_r

        if(os.path.exists("/n01dat01/pdchen/Dataset/MCAD/processed/clean_imgs/{sub}/{sub}_ses-1_task-rest_run-0_hemi-R_space-fsaverage5_bold_filter_gsr_fwmh4.func.gii".format(sub=sub_l))):
            continue
        image_name = image_name_par.format(sub=sub_l)
        func5_l,func5_r = resample_fsaverage_to_fs5(func_l,func_r)
        confound_no_gs, confound_gs = get_confound(image_name)
        names = GSR_filter_gii(func5_l,confound_no_gs=confound_no_gs,confound_gs=confound_gs,save_dir=save_dir,sub_name_idx=6)
        smooth_gii(names,save_dir=save_dir,name_idx=7,hemi='lh',FWHM=4)

        names = GSR_filter_gii(func5_r,confound_no_gs=confound_no_gs,confound_gs=confound_gs,save_dir=save_dir,sub_name_idx=6)
        smooth_gii(names,save_dir=save_dir,name_idx=7,hemi='rh',FWHM=4)
    

fsLR_L_sphere = "~/neuromaps-data/atlases/fsLR/tpl-fsLR_space-fsaverage_den-32k_hemi-L_sphere.surf.gii"
fsLR_R_sphere = "~/neuromaps-data/atlases/fsLR/tpl-fsLR_space-fsaverage_den-32k_hemi-R_sphere.surf.gii"
fsLR_L_va = "standard_mesh_atlases/resample_fsaverage/fs_LR.L.midthickness_va_avg.32k_fs_LR.shape.gii"
fsLR_R_va = "standard_mesh_atlases/resample_fsaverage/fs_LR.R.midthickness_va_avg.32k_fs_LR.shape.gii"
fsLR_L_mask = "standard_mesh_atlases/tpl-fsLR_den-32k_hemi-L_desc-nomedialwall_dparc.label.gii"
fsLR_R_mask = "standard_mesh_atlases/tpl-fsLR_den-32k_hemi-R_desc-nomedialwall_dparc.label.gii"

fs_L_sphere = {"fs5":"standard_mesh_atlases/resample_fsaverage/fsaverage5_std_sphere.L.10k_fsavg_L.surf.gii",
               "fs4":"standard_mesh_atlases/resample_fsaverage/fsaverage4_std_sphere.L.3k_fsavg_L.surf.gii"}

fs_R_sphere = {"fs5":"standard_mesh_atlases/resample_fsaverage/fsaverage5_std_sphere.R.10k_fsavg_R.surf.gii",
               "fs4":"standard_mesh_atlases/resample_fsaverage/fsaverage4_std_sphere.R.3k_fsavg_R.surf.gii"}

fs_L_va = {"fs5":"standard_mesh_atlases/resample_fsaverage/fsaverage5.L.midthickness_va_avg.10k_fsavg_L.shape.gii",
           "fs4":"standard_mesh_atlases/resample_fsaverage/fsaverage4.L.midthickness_va_avg.3k_fsavg_L.shape.gii"}

fs_R_va = {"fs5":"standard_mesh_atlases/resample_fsaverage/fsaverage5.R.midthickness_va_avg.10k_fsavg_R.shape.gii",
           "fs4":"standard_mesh_atlases/resample_fsaverage/fsaverage4.R.midthickness_va_avg.3k_fsavg_R.shape.gii"}

fs_L_mask = {"fs5":"standard_mesh_atlases/resample_fsaverage/tpl-fsaverage_den-10k_hemi-L_desc-nomedialwall_dparc.label.gii",
             "fs4":"standard_mesh_atlases/resample_fsaverage/tpl-fsaverage_den-3k_hemi-L_desc-nomedialwall_dparc.label.gii"}

fs_R_mask = {"fs5":"standard_mesh_atlases/resample_fsaverage/tpl-fsaverage_den-10k_hemi-R_desc-nomedialwall_dparc.label.gii",
             "fs4":"standard_mesh_atlases/resample_fsaverage/tpl-fsaverage_den-3k_hemi-R_desc-nomedialwall_dparc.label.gii"}

def fslr_to_fsa(images,save_dir,name_idx=6,save_format='mgh',temp_name='temp',target_space='fs5'):

    new_images = []
    wb_command = 'wb_command'


    temp_split_cortex_left = save_dir + temp_name + '_left.func.gii'
    temp_split_cortex_right = save_dir + temp_name + '_right.func.gii'

    separate_command = '{wb_command} -cifti-separate {dtseries} COLUMN -metric CORTEX_LEFT {cortex_left} -metric CORTEX_RIGHT {cortex_right}'
    resample_command_L = '{wb_command} -metric-resample {cortex_left} {current_L_sphere} {fs5_L_sphere} ADAP_BARY_AREA {fs5_func_L} -area-metrics {current_L_va} {fs5_L_va} -current-roi {srcmask}'
    resample_command_R = '{wb_command} -metric-resample {cortex_right} {current_R_sphere} {fs5_R_sphere} ADAP_BARY_AREA {fs5_func_R} -area-metrics {current_R_va} {fs5_R_va} -current-roi {srcmask}'
    mask_command = 'wb_command -metric-mask {out} {trgmask} {out}'

    for image in images:

        subj_name = image.split('/')[name_idx]
        image_file_name = os.path.basename(image).split('.')[0]
        save_img_dir = save_dir + subj_name
        if(not os.path.exists(save_img_dir)):
            os.mkdir(save_img_dir)
        fs_func_L = save_img_dir + '/' +  image_file_name + '_{}.L.func.gii'.format(target_space)
        fs_func_R = save_img_dir + '/' +  image_file_name + '_{}.R.func.gii'.format(target_space)

        if(save_format):

            if(save_format  == 'mgh'):
                n_fs_func_L = save_img_dir + '/' +  image_file_name + '_{}.L.mgh'.format(target_space)
                n_fs_func_R = save_img_dir + '/' +  image_file_name + '_{}.R.mgh'.format(target_space)
                new_images.append({'lh':n_fs_func_L,'rh':n_fs_func_R})
            elif(save_format == 'gii'):
                new_images.append({'lh':fs_func_L,'rh':fs_func_R})
    
        run(separate_command.format(wb_command=wb_command,dtseries=image,cortex_left=temp_split_cortex_left,cortex_right=temp_split_cortex_right))

        run(resample_command_L.format(wb_command=wb_command,cortex_left=temp_split_cortex_left,
                        current_L_sphere=fsLR_L_sphere,fs5_L_sphere=fs_L_sphere[target_space],fs5_func_L=fs_func_L,
                        current_L_va=fsLR_L_va,fs5_L_va=fs_L_va[target_space],srcmask=fsLR_L_mask))
        run(mask_command.format(out=fs_func_L,trgmask=fs_L_mask[target_space]))
        
        run(resample_command_R.format(wb_command=wb_command,cortex_right=temp_split_cortex_right,
                        current_R_sphere=fsLR_R_sphere,fs5_R_sphere=fs_R_sphere[target_space],fs5_func_R=fs_func_R,
                        current_R_va=fsLR_R_va,fs5_R_va=fs_R_va[target_space],srcmask=fsLR_R_mask))
        run(mask_command.format(out=fs_func_R,trgmask=fs_R_mask[target_space]))

        if(save_format  == 'mgh'):

            run('mri_convert {fs_func_L} {n_fs_func_L}'.format(fs_func_L = fs_func_L,n_fs_func_L = n_fs_func_L))
            run('mri_convert {fs_func_R} {n_fs_func_R}'.format(fs_func_R = fs_func_R,n_fs_func_R = n_fs_func_R))
           
        
        if(save_format is None):
            assert len(images) == 1
            data_l = nib.load(fs_func_L).agg_data()
            data_r = nib.load(fs_func_R).agg_data()
            
            os.remove(fs_func_L)
            os.remove(fs_func_R)
            return np.array(data_l),np.array(data_r)
        
    return new_images

def fslr_to_fsa4(name_idx,save_dir):
    from neuromaps.transforms import fslr_to_fsaverage
    import scipy.io as scio

    image_names = glob.glob("/n01dat01/pdchen/Dataset/MCAD/processed/clean_imgs/sub-*/sub-*_ses-1_task-rest_run-0_space-fsLR_den-91k_bold_filter_gsr_fwmh4.dtseries.nii")
    image_names = sorted(image_names)
    
    for image_name in tqdm.tqdm(image_names):
        subj_name = image_name.split('/')[name_idx]

        fs4_l,fs4_r = fslr_to_fsa([image_name],save_dir="/n01dat01/pdchen/Dataset/MCAD/processed/clean_imgs/",name_idx=name_idx,target_space='fs4',save_format=None)
        # 2562*timepoint
        scio.savemat(save_dir+'{}_timeframes_fs4.mat'.format(subj_name), mdict={'lhData': fs4_l.T,'rhData':fs4_r.T})
def fsa5_to_fsa4(infotable,save_dir):
    import scipy.io as scio

    funcl_path_par = "/n01dat01/pdchen/Dataset/MCAD/processed/clean_imgs/{sub}/{sub}_ses-1_task-rest_run-0_hemi-L_space-fsaverage5_bold_filter_gsr_fwmh4.func.gii"
    funcr_path_par = "/n01dat01/pdchen/Dataset/MCAD/processed/clean_imgs/{sub}/{sub}_ses-1_task-rest_run-0_hemi-R_space-fsaverage5_bold_filter_gsr_fwmh4.func.gii"
    
    for index,row in tqdm.tqdm(infotable.iterrows(),total=infotable.shape[0]):
        subj_name = row['sub']
        fs4_l,fs4_r = resample_fsaverage5_to_fs4(funcl_path_par.format(sub=subj_name),funcr_path_par.format(sub=subj_name),if_path=False)
       
        # 2562*timepoint
        scio.savemat(save_dir+'{}_timeframes_fs4_from_fsa5.mat'.format(subj_name), mdict={'lhData': fs4_l,'rhData':fs4_r})
    
def cifti():       
    image_names = glob.glob("/n01dat01/pdchen/Dataset/MCAD/fmriprep_2314/sub-*/ses-1/func/sub-*_ses-1_task-rest_run-0_space-fsLR_den-91k_bold.dtseries.nii")
    #image_names = glob.glob("/n01dat01/pdchen/Dataset/MCAD/processed_fs/post-process/clean_imgs/sub-*/sub-*_ses-1_task-rest_run-0_space-fsLR_den-91k_bold_filter*fwmh4.dtseries.nii")
    
    image_names = sorted(image_names)
    save_dir = "/n01dat01/pdchen/Dataset/MCAD/processed/clean_imgs/"
    for image_name in tqdm.tqdm(image_names):
        confound_no_gs, confound_gs = get_confound(image_name)
        names = GSR_filter(image_name,confound_no_gs=confound_no_gs,confound_gs=confound_gs,save_dir=save_dir,sub_name_idx=6)
        names = smooth_cifti(names,save_dir=save_dir,name_idx=7)
        fslr_to_fsa(names,save_dir=save_dir,name_idx=7)

def re_smooth_cifti():
    # 第一次平滑的时候 皮下没有跨结构去平滑
    infotable = pd.read_csv("/n01dat01/pdchen/pin_ind_net/infoTable/MCAD_info_809_fmriprep_withTIV_ts_length.csv")
    image_name_par = "/n01dat01/pdchen/Dataset/MCAD/processed/clean_imgs/{sub}/{sub}_ses-1_task-rest_run-0_space-fsLR_den-91k_bold_filter_gsr.dtseries.nii"
    image_name_par2 = "/n01dat01/pdchen/Dataset/MCAD/processed/clean_imgs/{sub}/{sub}_ses-1_task-rest_run-0_space-fsLR_den-91k_bold_filter.dtseries.nii"
    
    save_dir = "/n01dat01/pdchen/Dataset/MCAD/processed/clean_imgs/"
    for index,row in infotable.iterrows():
        names = [image_name_par.format(sub=row['sub']),image_name_par2.format(sub=row['sub'])]
        smooth_cifti(names,save_dir=save_dir,name_idx=7)

def nifti():

    infotable = pd.read_csv("/n01dat01/pdchen/pin_ind_net/infoTable/MCAD_info_809_fmriprep_withTIV_ts_length.csv")
    image_name_par = "/n01dat01/pdchen/Dataset/MCAD/processed_fs/fmriprep/{sub}/ses-1/func/{sub}_ses-1_task-rest_run-0_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
    save_dir = "/n01dat01/pdchen/Dataset/MCAD/processed_fs/post-process/clean_imgs/"
    for index,row in tqdm.tqdm(infotable.iterrows(),total=infotable.shape[0]):
        image_name = image_name_par.format(sub=row['sub'])
        confound_no_gs, confound_gs = get_confound(image_name)
        names = Regress_Resample_Filter_nifti(image_name,confound_no_gs=confound_no_gs,confound_gs=confound_gs,save_dir=save_dir,sub_name_idx=7)

def seperate_sub_from_cifti():
    image_names = glob.glob("/n01dat01/pdchen/Dataset/MCAD/processed/clean_imgs/sub-*/sub-*_ses-1_task-rest_run-0_space-fsLR_den-91k_bold_filter_gsr_fwmh4.dtseries.nii")    
    image_names = sorted(image_names)
    save_dir = "/n01dat01/pdchen/Dataset/MCAD/processed/clean_imgs/"
    name_idx = 7
    separate_command = '{wb_command} -cifti-separate {dtseries} COLUMN -volume-all {output}'
    wb_command = 'wb_command'
    for image in tqdm.tqdm(image_names):
        
        subj_name = image.split('/')[name_idx]
        image_file_name = os.path.basename(image).split('.')[0]
        save_img_dir = save_dir + subj_name
        if(not os.path.exists(save_img_dir)):
            os.mkdir(save_img_dir)

        out_image_file_name = save_img_dir + '/' +  image_file_name + '_subcortical.nii.gz'
        run(separate_command.format(wb_command=wb_command,dtseries=image,output=out_image_file_name))

# 用于体空间处理，用于计算ReHo/ALFF/fALFF，fmriprep处理时为了节省空间会将0值省略，导致每个被试MNI下尺寸不一致，所以在这里做一个resample
def preprocess_volume_for_alff_reho(
    fmriprep_dir,
    output_dir,
    info_table_path,
    sessions=['ses-01', 'ses-02', 'ses-03'],  # 新增：支持多session
    template_path='/data1/rszhou/templateflow/tpl-MNI152NLin2009cAsym/tpl-MNI152NLin2009cAsym_res-02_desc-brain_T1w.nii.gz',
    use_gsr=True,
    sub_name_idx=8
):
    """
    预处理多session的volume数据用于ALFF/ReHo计算
    
    Parameters:
    -----------
    sessions : list
        要处理的session列表，如 ['ses-01', 'ses-02', 'ses-03']
    """
    from nilearn import image as nimg
    from nilearn.interfaces.fmriprep import load_confounds
    import nibabel as nib
    import numpy as np
    import pandas as pd
    import os
    import json
    from tqdm import tqdm

    template = nib.load(template_path)
    info_df = pd.read_csv(info_table_path)
    subject_list = info_df['sub'].tolist()
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 记录处理结果
    processing_log = {
        'processed': [],
        'skipped': [],
        'errors': []
    }
    
    # 遍历每个session
    for session in sessions:
        session_output_dir = os.path.join(output_dir, session)
        os.makedirs(session_output_dir, exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"Processing {session}")
        print(f"{'='*60}")
        
        for subject in tqdm(subject_list, desc=f"Processing {session}"):
            try:
                # 构建文件路径
                bold_file = os.path.join(
                    fmriprep_dir, subject, session, 'func',
                    f'{subject}_{session}_task-rest_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz'
                )
                mask_file = bold_file.replace('_desc-preproc_bold.nii.gz', '_desc-brain_mask.nii.gz')
                json_file = bold_file.replace('.nii.gz', '.json')
                
                # 检查文件是否存在
                if not os.path.exists(bold_file):
                    processing_log['errors'].append(f"{subject}/{session}: BOLD file not found")
                    continue
                    
                if not os.path.exists(mask_file):
                    processing_log['errors'].append(f"{subject}/{session}: Mask file not found")
                    continue
                
                # 创建被试的session输出目录
                sub_session_dir = os.path.join(session_output_dir, subject)
                os.makedirs(sub_session_dir, exist_ok=True)
                
                # 输出文件名
                gsr_suffix = 'gsr' if use_gsr else 'nogsr'
                output_file = os.path.join(
                    sub_session_dir, 
                    f'{subject}_{session}_preprocessed_{gsr_suffix}_MNI2009_2mm.nii.gz'
                )
                
                # 检查是否已处理
                if os.path.exists(output_file):
                    processing_log['skipped'].append(f"{subject}/{session}")
                    continue
                
                # 加载confounds
                if use_gsr:
                    confounds, _ = load_confounds(
                        bold_file,
                        strategy=['motion', 'wm_csf', 'global_signal'],
                        motion='derivatives', 
                        wm_csf='derivatives', 
                        global_signal='derivatives', 
                        demean=False
                    )
                else:
                    confounds, _ = load_confounds(
                        bold_file,
                        strategy=['motion', 'wm_csf'],
                        motion='derivatives', 
                        wm_csf='derivatives', 
                        demean=False
                    )
                
                # 添加趋势项
                n = confounds.shape[0]
                confounds['trend'] = np.linspace(1, n, n)
                confounds['trend2'] = confounds['trend'] ** 2
                
                # 检查NaN
                if np.any(np.isnan(confounds.values)):
                    processing_log['errors'].append(f"{subject}/{session}: Confounds contain NaN")
                    continue
                
                # 获取TR
                tr = 2.0
                if os.path.exists(json_file):
                    with open(json_file, 'r') as f:
                        tr = json.load(f).get('RepetitionTime', 2.0)
                
                # 清理数据
                cleaned_img = nimg.clean_img(
                    bold_file,
                    confounds=confounds.values,
                    detrend=True,
                    standardize=False,  
                    low_pass=0.1,
                    high_pass=0.01,
                    t_r=tr,
                    mask_img=mask_file
                )
                
                # 重采样到模板空间
                resampled_img = nimg.resample_to_img(
                    cleaned_img, 
                    template, 
                    interpolation='continuous'
                )
                
                # 保存
                resampled_img.to_filename(output_file)
                processing_log['processed'].append(f"{subject}/{session}")
                
            except Exception as e:
                error_msg = f"{subject}/{session}: {str(e)}"
                processing_log['errors'].append(error_msg)
                print(f"\n❌ Error: {error_msg}")
    
    # 保存处理日志
    log_file = os.path.join(output_dir, 'processing_log.txt')
    with open(log_file, 'w') as f:
        f.write("="*60 + "\n")
        f.write("PROCESSING SUMMARY\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"Successfully processed: {len(processing_log['processed'])}\n")
        f.write(f"Skipped (already exist): {len(processing_log['skipped'])}\n")
        f.write(f"Errors: {len(processing_log['errors'])}\n\n")
        
        if processing_log['errors']:
            f.write("="*60 + "\n")
            f.write("ERRORS:\n")
            f.write("="*60 + "\n")
            for error in processing_log['errors']:
                f.write(f"  - {error}\n")
        
        if processing_log['skipped']:
            f.write("\n" + "="*60 + "\n")
            f.write("SKIPPED:\n")
            f.write("="*60 + "\n")
            for skipped in processing_log['skipped'][:20]:  # 只显示前20个
                f.write(f"  - {skipped}\n")
            if len(processing_log['skipped']) > 20:
                f.write(f"  ... and {len(processing_log['skipped']) - 20} more\n")
    
    # 打印总结
    print(f"\n{'='*60}")
    print("PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"✅ Successfully processed: {len(processing_log['processed'])}")
    print(f"⏭️  Skipped (already exist): {len(processing_log['skipped'])}")
    print(f"❌ Errors: {len(processing_log['errors'])}")
    print(f"\nDetailed log saved to: {log_file}")
    
    return processing_log

def save_gifti(data, header, filename):
    gii_img = nib.gifti.GiftiImage(header=header)
    for t in range(data.shape[0]):
        darray = nib.gifti.GiftiDataArray(
            data[t, :].astype(np.float32),
            intent='NIFTI_INTENT_NONE', 
            datatype='NIFTI_TYPE_FLOAT32'
        )
        gii_img.add_gifti_data_array(darray)
    nib.save(gii_img, filename)

# fsaverage5下计算alff和reho的预处理
def postprocess_surface_for_metrics(
    fmriprep_dir,
    save_dir,
    sub_pattern='sub-*',
    use_gsr=True,
    smooth_fwhm=4,
    sub_name_idx=8
):
   
    os.makedirs(save_dir, exist_ok=True)
    
    func_pattern = os.path.join(fmriprep_dir, sub_pattern, 'ses-*/func',
                                '*_hemi-L_space-fsaverage5_bold.func.gii')
    func_files_L = sorted(glob.glob(func_pattern))
    
    print(f"Found {len(func_files_L)} subjects")
    
    for func_L in tqdm.tqdm(func_files_L, desc="Processing surface"):
        try:
            func_R = func_L.replace('hemi-L', 'hemi-R')
            cifti_ref = func_L.replace('_hemi-L_space-fsaverage5_bold.func.gii',
                                      '_space-fsLR_den-91k_bold.dtseries.nii')
            
            sub_name = func_L.split('/')[sub_name_idx]
            sub_dir = os.path.join(save_dir, sub_name)
            os.makedirs(sub_dir, exist_ok=True)
            
            gsr_suffix = '_gsr' if use_gsr else ''
        
            check_files = [
                os.path.join(sub_dir, 
                    f"{sub_name}_ses-1_task-rest_run-0_hemi-L_desc-regressed{gsr_suffix}_detrended_smoothed{smooth_fwhm}mm_bold.func.gii"),
                os.path.join(sub_dir, 
                    f"{sub_name}_ses-1_task-rest_run-0_hemi-R_desc-regressed{gsr_suffix}_filtered_standardized_bold.func.gii")
            ]
            if all(os.path.exists(f) for f in check_files):
                print(f"  Skipping {sub_name}")
                continue
            
            confound_no_gs, confound_gs = get_confound(cifti_ref)
            confounds = confound_gs if use_gsr else confound_no_gs
            
            json_file = cifti_ref.replace('.dtseries.nii', '.json')
            if os.path.exists(json_file):
                with open(json_file, 'r') as f:
                    tr = json.load(f).get('RepetitionTime', 2.0)
            else:
                tr = 2.0
            for hemi, func_file in [('L', func_L), ('R', func_R)]:
                fs_hemi = 'lh' if hemi == 'L' else 'rh'
                
                img = nib.load(func_file)
                raw_data = img.agg_data().T  # (timepoints, vertices)
                
                # === ALFF数据：去混淆+去趋势+平滑 ===
                alff_data = clean(raw_data, confounds=confounds, detrend=True,
                                 standardize=False, low_pass=None, high_pass=None, t_r=tr)
                temp_mgh = os.path.join(sub_dir, f"{sub_name}_hemi-{hemi}_temp.mgh")
                alff_file = os.path.join(sub_dir,
                    f"{sub_name}_ses-1_task-rest_run-0_hemi-{hemi}_desc-regressed{gsr_suffix}_detrended_smoothed{smooth_fwhm}mm_bold.func.gii")
                
                # shape: (n_vertices, 1, 1, n_timepoints)
                mgh_data = alff_data.T[:, np.newaxis, np.newaxis, :]
                mgh_img = nib.freesurfer.mghformat.MGHImage(mgh_data, np.eye(4))
                nib.save(mgh_img, temp_mgh)
                
                # 使用mri_surf2surf平滑
                smooth_mgh = temp_mgh.replace('_temp.mgh', '_smooth.mgh')
                cmd = (f"mri_surf2surf --srcsubject fsaverage5 --trgsubject fsaverage5 "
                      f"--hemi {fs_hemi} --sval {temp_mgh} --tval {smooth_mgh} "
                      f"--fwhm {smooth_fwhm}")
                
                ret = os.system(cmd)
                if ret != 0:
                    raise RuntimeError(f"mri_surf2surf failed with code {ret}")
                smooth_data = nib.load(smooth_mgh).get_fdata().squeeze()  # (n_vertices, n_timepoints)
                save_gifti(smooth_data.T, img.header, alff_file)
                os.remove(temp_mgh)
                os.remove(smooth_mgh)
                
                # === ReHo数据：去混淆+去趋势+滤波+标准化 ===
                reho_data = clean(raw_data, confounds=confounds, detrend=False,
                                standardize=False, 
                                low_pass=0.1, high_pass=0.01, t_r=tr)

                reho_file = os.path.join(sub_dir,
                    f"{sub_name}_ses-1_task-rest_run-0_hemi-{hemi}_desc-regressed{gsr_suffix}_filtered_nostd_bold.func.gii")
                # save_gifti(reho_data.T, img.header, reho_file) 
                save_gifti(reho_data, img.header, reho_file)
            
            print(f"✅ {sub_name}")
            
        except Exception as e:
            print(f"❌ Error {func_L}: {e}")
            import traceback
            with open(os.path.join(save_dir, 'error_log.txt'), 'a') as f:
                f.write(f"{func_L}: {e}\n")
                f.write(traceback.format_exc() + "\n")


if __name__ == "__main__":
    
    # 直接处理fsaverage，因为fsLR的media wall太大，fsaverage上会有空洞
    #process_fsaverage()

    # re_smooth_cifti()

    # seperate_sub_from_cifti()

    # infotable = pd.read_csv("/n01dat01/pdchen/pin_ind_net/infoTable/MCAD_info_809_fmriprep_withTIV_ts_length.csv")
    # fsa5_to_fsa4(infotable,save_dir="/n01dat01/pdchen/pin_ind_net/Plos_ref/dataset/mcad/")
    #fslr_to_fsa4(name_idx=7,save_dir = "/n01dat01/pdchen/pin_ind_net/Plos_ref/dataset/mcad/")
    
    processing_log = preprocess_volume_for_alff_reho(
        fmriprep_dir='/data1/rszhou/xwtacs/fmriprep_2314/fMRIprep_processed',
        output_dir='/data1/rszhou/xwtacs/fmriprep_2314/processed/clean_imgs/volume',
        info_table_path='/data/home/rszhou/xw_project_zhou/infoTable/xwtacs_info_all_clinic.csv',
        sessions=['ses-01', 'ses-02', 'ses-03'],  # 指定要处理的session
        use_gsr=True,
        sub_name_idx=8
    )


