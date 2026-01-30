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
from postpreprocess_mcad import get_confound,smooth_cifti,fslr_to_fsa,Regress_Resample_Filter_nifti
import sys
def GSR_filter(image,save_dir,confound_no_gs,confound_gs,sub_name_idx=7,cut_timepoint=0):

    save_img_dir = save_dir + image.split('/')[sub_name_idx] + '/'
    print(save_img_dir)
    if(not os.path.exists(save_img_dir)):
        os.mkdir(save_img_dir)
    filename = os.path.basename(image).split('.')[0]
    file_no_gs = save_img_dir+ filename + '_filter.dtseries.nii'
    file_gsr = save_img_dir+ filename +  '_filter_gsr.dtseries.nii'

    # if(os.path.exists(file_gsr)):
    #     return [file_no_gs,file_gsr]
        
    img = nib.load(image)
    image_data = nib.load(image).get_fdata()
    header = img.header

    image_data = np.array(image_data)[cut_timepoint:,:]
    confound_no_gs = confound_gs[cut_timepoint:,:]
    confound_gs = confound_gs[cut_timepoint:,:]

    high_pass= 0.01
    low_pass = 0.08

    with open('/n01dat01/pdchen/Dataset/OASIS/BIDS/{sub}/ses-1/func/{sub}_ses-1_task-rest_run-0_bold.json'.format(sub=image.split('/')[sub_name_idx])) as f:
        jinfo = json.load(f)
    t_r = jinfo['RepetitionTime']
    clean_data = clean(image_data,confounds=confound_no_gs,detrend=False,standardize=True,
                            low_pass=low_pass,high_pass=high_pass,t_r=t_r)
    
    clean_gsr_data = clean(image_data,confounds=confound_gs,detrend=False,standardize=True,
                            low_pass=low_pass,high_pass=high_pass,t_r=t_r)
    
    header.get_index_map(0).number_of_series_points = header.get_index_map(0).number_of_series_points - cut_timepoint
    nib.Cifti2Image(clean_data,header).to_filename(file_no_gs)
    nib.Cifti2Image(clean_gsr_data,header).to_filename(file_gsr)

    return [file_no_gs,file_gsr]

def cifti():       
    image_names = glob.glob("/n01dat01/pdchen/Dataset/OASIS/fmriprep_2314/*/ses-1/func/*_ses-1_task-rest_run-0_space-fsLR_den-91k_bold.dtseries.nii")
    image_names = sorted(image_names)
    save_dir = "/n01dat01/pdchen/Dataset/OASIS/processed/clean_imgs/"
    
    new_imgs = []
    for image_name in tqdm.tqdm(image_names):
        sub = image_name.split('/')[6]
        if(os.path.exists(save_dir+'{sub}/{sub}_ses-1_task-rest_run-0_space-fsLR_den-91k_bold_filter_gsr_fwmh4_fs5.R.mgh'.format(sub=sub))):
            continue
        new_imgs.append(image_name)
    image_names = new_imgs

    split_id = int(sys.argv[1])
    print(split_id)
    if(split_id == 1):
        image_names = image_names[:500]
    elif(split_id == 2):
        image_names = image_names[500:1000]
    elif(split_id == 3):
        image_names = image_names[1000:1500]
    else:
        image_names = image_names[1500:]

    print('需处理的总数：{}'.format(len(image_names)))
    print(image_names)

    for image_name in tqdm.tqdm(image_names):

        confound_no_gs, confound_gs = get_confound(image_name)
        names = GSR_filter(image_name,confound_no_gs=confound_no_gs,confound_gs=confound_gs, 
                           save_dir=save_dir,sub_name_idx=6,cut_timepoint=10)
        names = smooth_cifti(names,save_dir=save_dir,name_idx=7)
        fslr_to_fsa(names,save_dir=save_dir,name_idx=7,temp_name='temp'+str(split_id))

def nifti():
    infotable = pd.read_csv("/n01dat01/pdchen/pin_ind_net/infoTable/MCAD_info_809_fmriprep_withTIV_ts_length.csv")
    image_name_par = "/n01dat01/pdchen/Dataset/MCAD/processed_fs/fmriprep/{sub}/ses-1/func/{sub}_ses-1_task-rest_run-0_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
    save_dir = "/n01dat01/pdchen/Dataset/MCAD/processed_fs/post-process/clean_imgs/"
    for index,row in tqdm.tqdm(infotable.iterrows(),total=infotable.shape[0]):
        image_name = image_name_par.format(sub=row['sub'])
        confound_no_gs, confound_gs = get_confound(image_name)
        names = Regress_Resample_Filter_nifti(image_name,confound_no_gs=confound_no_gs,confound_gs=confound_gs,save_dir=save_dir,sub_name_idx=7)
def cut_timepoint():
    image_names = glob.glob("/n01dat01/pdchen/Dataset/OASIS/processed/clean_imgs/sub-*/sub-*_ses-1_task-rest_run-0_space-fsLR_den-91k_bold_filter_gsr_fwmh4_fs5.*.mgh")    
    image_names = sorted(image_names)
    for image in tqdm.tqdm(image_names):
        img = nib.load(image)
        img_data = img.get_fdata()
        print(img_data.shape)
        return
def seperate_sub_from_cifti():
    image_names = glob.glob("/n01dat01/pdchen/Dataset/OASIS/processed/clean_imgs/sub-*/sub-*_ses-1_task-rest_run-0_space-fsLR_den-91k_bold_filter_gsr_fwmh4.dtseries.nii")    
    image_names = sorted(image_names)
    save_dir = "/n01dat01/pdchen/Dataset/OASIS/processed/clean_imgs/"
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

def fslr_to_fsa4(name_idx,save_dir):
    from neuromaps.transforms import fslr_to_fsaverage
    import scipy.io as scio

    image_names = glob.glob("/n01dat01/pdchen/Dataset/OASIS/processed/clean_imgs/sub-*/sub-*_ses-1_task-rest_run-0_space-fsLR_den-91k_bold_filter_gsr_fwmh4.dtseries.nii")
    image_names = sorted(image_names)
    
    for image_name in tqdm.tqdm(image_names):
        subj_name = image_name.split('/')[name_idx]

        fs4_l,fs4_r = fslr_to_fsa([image_name],save_dir="/n01dat01/pdchen/Dataset/OASIS/processed/clean_imgs/",name_idx=name_idx,target_space='fs4',save_format=None)
        # 2562*timepoint
        scio.savemat(save_dir+'{}_timeframes_fs4.mat'.format(subj_name), mdict={'lhData': fs4_l.T,'rhData':fs4_r.T})
def fsa5_to_fsa4(infotable,save_dir):
    import scipy.io as scio
    from postpreprocess_mcad import resample_fsaverage5_to_fs4

    funcl_path_par = "/n01dat01/pdchen/Dataset/OASIS/processed/clean_imgs/{sub}/{sub}_ses-1_task-rest_{run}_hemi-L_space-fsaverage5_bold_filter_gsr_fwmh4.func.gii"
    funcr_path_par = "/n01dat01/pdchen/Dataset/OASIS/processed/clean_imgs/{sub}/{sub}_ses-1_task-rest_{run}_hemi-R_space-fsaverage5_bold_filter_gsr_fwmh4.func.gii"
    
    for index,row in tqdm.tqdm(infotable.iterrows(),total=infotable.shape[0]):
        subj_name = row['sub']
        for i in range(int(row['run_number'])):
            run = row['run_%d' % (i+1)]
            funcl_path, funcr_path = funcl_path_par.format(sub=subj_name,run=run),funcr_path_par.format(sub=subj_name,run=run)
            fs4_l,fs4_r = resample_fsaverage5_to_fs4(funcl_path,funcr_path,if_path=False,temp_name='oasis_fsa5_to_fsa4')
            scio.savemat(save_dir+'{}_{}_timeframes_fs4.mat'.format(subj_name,run), mdict={'lhData': fs4_l,'rhData':fs4_r})

def process_fsaverage(infotable):
    from postpreprocess_mcad import GSR_filter_gii,smooth_gii
    func_ls = sorted(glob.glob("/n01dat01/pdchen/Dataset/OASIS/fmriprep_2314/*/ses-1/func/*_ses-1_task-rest_run-*_hemi-L_space-fsaverage5_bold.func.gii"))
    func_rs = sorted(glob.glob("/n01dat01/pdchen/Dataset/OASIS/fmriprep_2314/*/ses-1/func/*_ses-1_task-rest_run-*_hemi-R_space-fsaverage5_bold.func.gii"))
    image_name_par = "/n01dat01/pdchen/Dataset/OASIS/fmriprep_2314/{sub}/ses-1/func/{sub}_ses-1_task-rest_{run}_space-fsLR_den-91k_bold.dtseries.nii"
    save_dir = "/n01dat01/pdchen/Dataset/OASIS/processed/clean_imgs/"

    for func_l,func_r in tqdm.tqdm(zip(func_ls,func_rs)):

        sub_l = func_l.split('/')[6]
        sub_r = func_r.split('/')[6]

        assert sub_l == sub_r

        run = os.path.basename(func_l).split('_')[3]
        
        json_file = "/n01dat01/pdchen/Dataset/OASIS/fmriprep_2314/{sub}/ses-1/func/{sub}_ses-1_task-rest_{run}_hemi-L_space-fsaverage5_bold.json".format(sub=sub_l,run=run)
        if(not os.path.exists(json_file)):
            continue
        with open(json_file) as f:
            jinfo = json.load(f)
        res = jinfo['SliceTimingCorrected']
        print(res)
        if(res is False):
            print(sub_l,run)

            # 删除那些之前处理过的不合法的数据
            if(os.path.exists("/n01dat01/pdchen/Dataset/OASIS/processed/clean_imgs/{sub}/{sub}_ses-1_task-rest_{run}_hemi-R_space-fsaverage5_bold_filter_gsr_fwmh4.func.gii".format(sub=sub_l,run=run))):
                os.remove("/n01dat01/pdchen/Dataset/OASIS/processed/clean_imgs/{sub}/{sub}_ses-1_task-rest_{run}_hemi-L_space-fsaverage5_bold_filter_gsr_fwmh4.func.gii".format(sub=sub_l,run=run))
                os.remove("/n01dat01/pdchen/Dataset/OASIS/processed/clean_imgs/{sub}/{sub}_ses-1_task-rest_{run}_hemi-R_space-fsaverage5_bold_filter_gsr_fwmh4.func.gii".format(sub=sub_l,run=run))
        continue
        image_name = image_name_par.format(sub=sub_l,run=run)
        confound_no_gs, confound_gs = get_confound(image_name)
        if(confound_gs.shape[0] < 50):
            print('So short:',sub_l,run)
            continue
        names = GSR_filter_gii(func_l,confound_no_gs=confound_no_gs,confound_gs=confound_gs,save_dir=save_dir,sub_name_idx=6)
        smooth_gii(names,save_dir=save_dir,name_idx=7,hemi='lh',FWHM=4)

        names = GSR_filter_gii(func_r,confound_no_gs=confound_no_gs,confound_gs=confound_gs,save_dir=save_dir,sub_name_idx=6)
        smooth_gii(names,save_dir=save_dir,name_idx=7,hemi='rh',FWHM=4)
def produce_new_infotable_by_screen_the_results(infotable):
    # 一个是生成sub列
    # 第二个是通过判断最终预处理文件是否存在，来对其进行标记，增加4列， run_number,  run_1, run_2, run_3
    infotable['sub'] = ' '
    infotable['run_number'] = 0
    infotable['run_1'] = 0
    infotable['run_2'] = 0
    infotable['run_3'] = 0
    for index,row in infotable.iterrows():
        sub = 'sub-' + row['Subject'] + row['VSCODE_fmri']
        
        target_par = "/n01dat01/pdchen/Dataset/OASIS/processed/clean_imgs/{sub}/{sub}_ses-1_task-rest_*_hemi-R_space-fsaverage5_bold_filter_gsr_fwmh4.func.gii".format(sub=sub,run=run)
        runs = glob.glob(target_par)
        infotable.loc[index,'run_number'] = len(runs)
        for i in range(len(runs)):
            run_name = os.path.basename(runs[i]).split('_')[3]
            infotable.loc[index,'run_%d' % (i+1)] = run_name
        infotable.loc[index,'sub'] = sub
    return infotable
def produce_new_infotable_by_screen_the_results_add_subname(infotable):
    for index,row in infotable.iterrows():
        sub = 'sub-' + row['Subject'] + row['VSCODE_fmri']
        infotable.loc[index,'sub'] = sub
    return infotable

# 用于体空间处理，用于计算ReHo/ALFF/fALFF，fmriprep处理时为了节省空间会将0值省略，导致每个被试MNI下尺寸不一致，所以在这里做一个resample
def preprocess_volume_for_alff_reho(
    fmriprep_dir,
    output_dir,
    info_table_path=None,
    template_path='/data1/rszhou/templateflow/tpl-MNI152NLin2009cAsym/tpl-MNI152NLin2009cAsym_res-02_desc-brain_T1w.nii.gz',
    use_gsr=True,
    task_name='rest',
    run_id=None  # 改为None，自动查找
):
    """
    体积空间预处理：为ALFF和ReHo计算准备数据
    
    参数:
    ----------
    run_id : str, optional
        run编号。如果为None，自动查找第一个匹配的run
        MCAD/ADNI: 'run-0'
        OASIS: 'run-01' 或 'run-02'
    """
    from nilearn import image as nimg
    import nibabel as nib
    import numpy as np
    import pandas as pd
    import os
    import json
    import glob
    from tqdm import tqdm
    
    template = nib.load(template_path)
    
    # 获取被试列表
    if info_table_path is not None:
        info_df = pd.read_csv(info_table_path)
        subject_list = info_df['sub'].tolist()
        print(f"从信息表读取 {len(subject_list)} 个被试")
    else:
        subject_dirs = sorted(glob.glob(os.path.join(fmriprep_dir, 'sub-*')))
        subject_list = [os.path.basename(d) for d in subject_dirs]
        print(f"扫描目录找到 {len(subject_list)} 个被试")
    
    os.makedirs(output_dir, exist_ok=True)
    processed_files = []
    errors = []
    
    for subject in tqdm(subject_list, desc="预处理"):
        try:
            # 构建查找模式（不指定run，查找所有）
            if run_id is not None:
                # 指定了run编号
                bold_pattern = os.path.join(
                    fmriprep_dir, subject, 'ses-*/func',
                    f'{subject}_ses-*_task-{task_name}_run-{run_id}_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz'
                )
            else:
                # 自动查找任意run
                bold_pattern = os.path.join(
                    fmriprep_dir, subject, 'ses-*/func',
                    f'{subject}_ses-*_task-{task_name}_run-*_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz'
                )
            
            bold_files = sorted(glob.glob(bold_pattern))
            
            if len(bold_files) == 0:
                errors.append(f"{subject}: 未找到BOLD文件")
                continue
            
            # 取第一个run（如果有多个run，只处理第一个）
            bold_file = bold_files[0]
            mask_file = bold_file.replace('_desc-preproc_bold.nii.gz', '_desc-brain_mask.nii.gz')
            confounds_file = bold_file.replace('_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz', 
                                               '_desc-confounds_timeseries.tsv')
            
            if not os.path.exists(mask_file) or not os.path.exists(confounds_file):
                errors.append(f"{subject}: 文件不完整")
                continue
            
            # 输出路径
            sub_dir = os.path.join(output_dir, subject)
            os.makedirs(sub_dir, exist_ok=True)
            gsr_suffix = '_gsr' if use_gsr else '_nogsr'
            output_file = os.path.join(sub_dir, f'{subject}_preprocessed{gsr_suffix}_MNI2009_2mm.nii.gz')
            
            if os.path.exists(output_file):
                processed_files.append(output_file)
                continue
            
            # 读取confounds
            confounds_df = pd.read_csv(confounds_file, sep='\t')
            confound_cols = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z',
                           'trans_x_derivative1', 'trans_y_derivative1', 'trans_z_derivative1',
                           'rot_x_derivative1', 'rot_y_derivative1', 'rot_z_derivative1',
                           'csf', 'white_matter', 'csf_derivative1', 'white_matter_derivative1']
            
            if use_gsr:
                confound_cols.extend(['global_signal', 'global_signal_derivative1'])
            
            missing_cols = [col for col in confound_cols if col not in confounds_df.columns]
            if missing_cols:
                errors.append(f"{subject}: 缺少列: {missing_cols}")
                continue
            
            confounds = confounds_df[confound_cols].values
            
            # 添加趋势项
            n = confounds.shape[0]
            trend = np.linspace(1, n, n).reshape(-1, 1)
            confounds = np.hstack([confounds, trend, trend**2])
            confounds = np.nan_to_num(confounds, nan=0.0)
            
            # 获取TR
            json_file = bold_file.replace('.nii.gz', '.json')
            tr = 2.0
            if os.path.exists(json_file):
                with open(json_file, 'r') as f:
                    tr = json.load(f).get('RepetitionTime', 2.0)
            
            # 预处理（关键：standardize=False）
            cleaned_img = nimg.clean_img(
                bold_file,
                confounds=confounds,
                detrend=True,
                standardize=False,  # 不标准化！
                low_pass=0.1,
                high_pass=0.01,
                t_r=tr,
                mask_img=mask_file
            )
            
            # 重采样
            resampled_img = nimg.resample_to_img(cleaned_img, template, interpolation='continuous')
            resampled_img.to_filename(output_file)
            processed_files.append(output_file)
            
        except Exception as e:
            errors.append(f"{subject}: {str(e)}")
    
    if errors:
        with open(os.path.join(output_dir, 'error_log.txt'), 'w') as f:
            f.write('\n'.join(errors))
    
    print(f"\n完成: {len(processed_files)}/{len(subject_list)} 个被试")
    if errors:
        print(f"失败: {len(errors)} 个")
    
    return processed_files

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

    #cifti()
    # fslr_to_fsa4(name_idx=7,save_dir = "/n01dat01/pdchen/pin_ind_net/Plos_ref/dataset/adni/")
    # infotable = pd.read_csv("/n01dat01/pdchen/Dataset/OASIS/OASIS_table_all_fmri.csv")
    #process_fsaverage(infotable)

    # fs5 重映射到 fsa4
    # fsa5_to_fsa4(infotable=infotable,save_dir = "/n01dat01/pdchen/pin_ind_net/Plos_ref/dataset/oasis/")
    # fsa5_to_fsa4(pd.read_csv("/n01dat01/pdchen/Dataset/OASIS/OASIS_table_all_fmri_screen_by_post.csv"),
     #             save_dir = "/n01dat01/pdchen/pin_ind_net/Plos_ref/dataset/oasis/")

    # nn_infotable = produce_new_infotable_by_screen_the_results(infotable)
    # nn_infotable.to_csv("/n01dat01/pdchen/Dataset/OASIS/OASIS_table_all_fmri_screen_by_post.csv")

    # infotable = pd.read_csv("/n01dat01/pdchen/Dataset/OASIS/OASIS_table_all_fmri_bl.csv")
    # nn_infotable = produce_new_infotable_by_screen_the_results(infotable)
    # nn_infotable.to_csv("/n01dat01/pdchen/Dataset/OASIS/OASIS_table_all_fmri_screen_by_post_bl.csv")

    # infotable = pd.read_csv("/n01dat01/pdchen/Dataset/OASIS/OASIS_table_all_fmri_screen_by_post.csv")
    # os.rename("/n01dat01/pdchen/Dataset/OASIS/OASIS_table_all_fmri_screen_by_post.csv","/n01dat01/pdchen/Dataset/OASIS/bak/OASIS_table_all_fmri_screen_by_post.csv")
    # info = produce_new_infotable_by_screen_the_results_add_subname(infotable)
    # info.to_csv("/n01dat01/pdchen/Dataset/OASIS/OASIS_table_all_fmri_screen_by_post.csv")

    # infotable = pd.read_csv("/n01dat01/pdchen/Dataset/OASIS/OASIS_table_all_fmri_screen_by_post_bl.csv")
    # os.rename("/n01dat01/pdchen/Dataset/OASIS/OASIS_table_all_fmri_screen_by_post_bl.csv","/n01dat01/pdchen/Dataset/OASIS/bak/OASIS_table_all_fmri_screen_by_post_bl.csv")
    # info = produce_new_infotable_by_screen_the_results_add_subname(infotable)
    # info.to_csv("/n01dat01/pdchen/Dataset/OASIS/OASIS_table_all_fmri_screen_by_post_bl.csv")


    preprocess_volume_for_alff_reho(
        fmriprep_dir='/data1/temp_data/disk2/Dataset/OASIS/fmriprep_2314',
        output_dir='/data1/rszhou/OASIS/fmriprep/processed/clean_imags/volume',
        info_table_path='/data/home/rszhou/xw_project_zhou/infoTable/OASIS_table_all_fmri_screen_by_post.csv',
        use_gsr=True
    )
    # postprocess_surface_for_metrics(
    #     fmriprep_dir='/data1/temp_data/disk2/Dataset/OASIS/fmriprep_2314',
    #     save_dir='/data1/rszhou/OASIS/fmriprep/processed/clean_imags/postprocess_fs5',
    #     sub_pattern='sub-*',
    #     use_gsr=True,
    #     smooth_fwhm=4,
    #     sub_name_idx=7
    # )
    