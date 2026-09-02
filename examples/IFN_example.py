#!/usr/bin/env python3
"""Example: individualized functional network construction in MCAD.

Resting-state fMRI data were preprocessed with fMRIPrep v23.1.4 and further
processed with ``postpreprocess.py``. The fMRIPrep fsaverage surface
time series were resampled to fsaverage5. Nuisance regression included
head-motion, white-matter, cerebrospinal-fluid, and global-signal regressors
and their temporal derivatives, together with linear and quadratic trends.
The time series were band-pass filtered at 0.01--0.08 Hz, standardized,
smoothed on the fsaverage5 surface with a 4-mm full-width at half-maximum
kernel, and resampled to fsaverage4.

Postprocessing generated one MATLAB file per participant in
``Plos_ref/dataset/mcad``: ``{subject}_timeframes_fs4_from_fsa5.mat``. The
copy distributed with this example uses the abbreviated name
``{subject}_timeframes_fs4.mat``. Each file contains ``lhData`` and
``rhData``, with shape ``(2562, n_timepoints)`` (vertices by time points).

--------------------------------------------------
Individualized functional networks were generated using an iterative
parcellation approach based on:

Wang, D., et al. Parcellating cortical functional networks in individuals.
Nature Neuroscience 18, 1853-1860 (2015).
https://doi.org/10.1038/nn.4164

Inputs
1. ``{subject}_timeframes_fs4.mat`` 
   ``lhData`` and ``rhData``, each with shape ``(2562, n_timepoints)``.
2. ``{subject}_tsnr_fs4.mat``:
   ``data`` with 5124 values ordered as left-hemisphere vertices followed by
   right-hemisphere vertices. These files are stored in
   ``Plos_ref/dataset/mcad_tsnr``. 

Outputs
``Network_Par_[L|R]_18.npz`` and ``Network_Confi_[L|R]_18.npz`` contain all
iterations with shape ``(n_iterations, 18, 2562)``. 
"""

import scipy.io as scio
import numpy as np
import os
import nibabel as nib

base_dir = os.path.dirname(os.path.abspath(__file__)) + '/../HFR_ai'
def cal_corr(data1,data2):
    '''
    param: 
    data1 data2   timepoint * feature
    '''
    data1 = (data1 - np.mean(data1,axis=0)) / np.std(data1,axis=0)
    data2 = (data2 - np.mean(data2,axis=0)) / np.std(data2,axis=0)
    corr = data1.T @ data2 / (data1.shape[0])
    return corr

def normalize(data,vmin=0.4,vmax=1):
    # normalize the range to 0.4 ~1. Therefore the inv will be between 1~2.5.
    data = vmin + (vmax-vmin)*(data- np.min(data))/(np.max(data) - np.min(data)) 
    return data

def get_confi(CorrMatrix,mask):
    '''
    param: CorrMatrix : network * voxel
    return: 
    Nework: network * voxel
    Network_confi: network * voxel
    '''
    voxel_num = CorrMatrix.shape[1]
    parc_membership = np.zeros([voxel_num])
    parc_confidence = np.zeros([voxel_num])

    for v in range(voxel_num):

        idx = np.argmax(CorrMatrix[:,v])
        cor = np.sort(CorrMatrix[:,v])[::-1]
        parc_membership[v] = idx + 1
        parc_confidence[v] = cor[0]/(cor[1] + 1e-6)

    parc_membership[mask > 0] = 0
    Network_par = np.zeros_like(CorrMatrix)
    Network_confi = np.zeros_like(CorrMatrix)
    for n in range(1,CorrMatrix.shape[0]+1):
        network = np.zeros_like(parc_membership)
        confid = np.zeros_like(parc_membership)

        network[parc_membership==n]= 1
        confid[parc_membership==n] = parc_confidence[parc_membership==n]
        network = np.nan_to_num(network)

        Network_par[n-1] = network
        Network_confi[n-1] = network*confid

    return Network_par,Network_confi

def get_new_seed_ts_by_confi_tsnr(func_data, tsnr, confi,confidence_threshold):
    
    confi = np.nan_to_num(confi)
    idx = confi >= confidence_threshold
    if(np.sum(idx) == 0):
        idx = np.argsort(confi)[::-1][:1] # Change the highest one voxel in first 5 voxel to improve statbility by P.Chen
    seed_data = tsnr[idx] @ func_data[idx,:] #% weight the individual signal based on SNR
    return seed_data

def smooth_data(data,adj_matrix):
    
    smooth_data = np.zeros_like(data)
    for i in range(data.shape[1]):
        smooth_data[:,i] = np.nanmean(data[:,adj_matrix[i]],axis=1)
    return smooth_data

def get_yeo_17_fsa5(num=17):
    l_label, _ , _ = nib.freesurfer.read_annot("/data/fs5_data/lh.Yeo2011_%dNetworks_N1000.annot" % num)
    r_label, r_table, r_names = nib.freesurfer.read_annot("/data/fs5_data/rh.Yeo2011_%dNetworks_N1000.annot" % num)
    Yeo_label = np.concatenate([l_label,r_label])
    initial_atlas_l = []
    initial_atlas_r = []
    for i in range(18):
        vol = np.zeros_like(l_label)
        vor = np.zeros_like(r_label)
        vol[l_label == i] = 1
        vor[r_label == i] = 1
        initial_atlas_l.append(vol)
        initial_atlas_r.append(vor)
    return initial_atlas_l,initial_atlas_r

def get_adj_matrix_surf(face_data,vertex_num):
    adj = [[] for i in range(vertex_num)]
    for i in range(face_data.shape[0]):
        vertex_face = face_data[i]
        for j in range(vertex_face.shape[0]):
            for k in range(vertex_face.shape[0]):
                if((vertex_face[k] not in adj[vertex_face[j]])):
                    adj[vertex_face[j]].append(vertex_face[k])
    return adj

def get_adj_matrix_fsa5():
    import pickle
    lh_file = "/HFR_py/atlas/adj_fsa5_lh.dat"
    rh_file = "/HFR_py/atlas/adj_fsa5_rh.dat"
    if((os.path.exists(lh_file)) & (os.path.exists(rh_file))):
        with open(lh_file,'rb') as f:
            adj_matrix_lh = pickle.load(f)
        with open(rh_file,'rb') as f:
            adj_matrix_rh = pickle.load(f)
        return adj_matrix_lh,adj_matrix_rh
    
    lh = "/HFR_py/atlas/tpl-fsaverage_den-10k_hemi-L_pial.surf.gii"
    rh = "/HFR_py/atlas/tpl-fsaverage_den-10k_hemi-R_pial.surf.gii"
    face_data_l = nib.load(lh).agg_data()[1]
    face_data_r = nib.load(rh).agg_data()[1]

    adj_lh = get_adj_matrix_surf(face_data_l,10242)
    adj_rh = get_adj_matrix_surf(face_data_r,10242)

    with open(lh_file,'wb') as f:
        pickle.dump(adj_lh, f)
    with open(rh_file,'wb') as f:
        pickle.dump(adj_rh, f)

    return adj_lh,adj_rh

def iterative_parcellation_hemi(lhData,rhData,initial_atlas_l,initial_atlas_r,
                           tsnr_data_l,tsnr_data_r,
                           adj_l,adj_r,
                           variability_l,variability_r,
                           combineLeftRight,output_dir,base_dir,
                           confidence_threshold=3,num_iter=20,):
    '''
    Iteration based individualized functional parcellation
    Parameters
    ----------
    func_data : array_like, [timepoint * voxel]
    functional timeseries
    tsnr_data : array_like, [voxel]
    initial_atlas : array_like, [voxel]
    atlas for initialization
    '''
    vertex_num = int(variability_l.shape[0])
    roi_num = len(initial_atlas_l) - 1

    variability = np.concatenate([variability_l,variability_r])
    varInv_lh = 1 / normalize(variability)[:vertex_num]
    varInv_rh = 1 / normalize(variability)[vertex_num:]
    
    tsnr_data = np.concatenate([tsnr_data_l,tsnr_data_r],axis=0)
    SNR_lh = normalize(tsnr_data)[:vertex_num]
    SNR_rh = normalize(tsnr_data)[vertex_num:]

    seedDatalh = np.zeros([roi_num,lhData.shape[1]])
    seedDatarh = np.zeros([roi_num,rhData.shape[1]])

    NetworkConfidence_lh = []
    NetworkConfidence_rh = []

    Network_lh = []
    Network_rh = []
    # ---------------------------------------------------
    #  Iterative parcellation
    # ---------------------------------------------------
        
    GrpNetlh = np.zeros([roi_num+1,vertex_num])
    GrpNetrh = np.zeros([roi_num+1,vertex_num])
    for cnt in range(1,num_iter+1):
        
        if(not os.path.exists(output_dir)):
            os.makedirs(output_dir)

        if(cnt == 1):

            ventLh = initial_atlas_l[0] > 0
            GrpNetlh[0,:] = ventLh

            ventRh = initial_atlas_r[0] > 0
            GrpNetrh[0,:] = ventRh

            for i2 in range(1,roi_num+1):  # get the seed waveforms based on Thomas' parcellation, and weight it by inv(Variability)
            
                idx = initial_atlas_l[i2] > 0
                seedDatalh[i2-1,:]= varInv_lh[idx] @ lhData[idx,:] # weight the group map using the inverse of individual difference
                GrpNetlh[i2] = idx

                idx = initial_atlas_r[i2] > 0
                seedDatarh[i2-1,:]= varInv_rh[idx] @ rhData[idx,:] # weight the group map using the inverse of individual difference
                GrpNetrh[i2] = idx

            GrpSeedDatalh =seedDatalh.copy()
            GrpSeedDatarh =seedDatarh.copy()
        else:

            for i2 in range(1,roi_num+1): # get the seed waveforms based on the last parcellation
                confi = NetworkConfidence_lh[-1][i2-1]
                seedDatalh[i2-1,:] = get_new_seed_ts_by_confi_tsnr(lhData,SNR_lh,confi,confidence_threshold)

                confi = NetworkConfidence_rh[-1][i2-1]
                seedDatarh[i2-1,:] = get_new_seed_ts_by_confi_tsnr(rhData,SNR_rh,confi,confidence_threshold)

        # Weight in the group seed in each iteration, should throw in individual variability map as weight in the future
        if(cnt>1):
            seedDatalh = seedDatalh + GrpSeedDatalh/(cnt-1);
            seedDatarh = seedDatarh + GrpSeedDatarh/(cnt-1);

        if (combineLeftRight):
            tmp = seedDatalh.copy()
            seedDatalh = seedDatalh+seedDatarh/(cnt+2)
            seedDatarh = seedDatarh+tmp/(cnt+2)

        cValuelh  =  cal_corr(seedDatalh.T,lhData.T) # 2562 * 18
        cValuerh  =  cal_corr(seedDatarh.T,rhData.T)
        cValuelh = 0.5*np.log((1+cValuelh)/(1-cValuelh))
        cValuerh = 0.5*np.log((1+cValuerh)/(1-cValuerh))
        cValuelh = np.nan_to_num(cValuelh)
        cValuerh = np.nan_to_num(cValuerh)
        cValuelh = smooth_data(cValuelh,adj_l)
        cValuerh = smooth_data(cValuerh,adj_r)

        # Further weight in the group map * inv(Variability) by adding correlation coefficient of 0~ 0.5 according to inv(Variability).
        for i in range(roi_num):
            idx = GrpNetlh[i+1]
            cValuelh[i, idx > 0] = cValuelh[i, idx > 0] + (((varInv_lh[idx > 0]-1)/3)/cnt).T

            idx = GrpNetrh[i+1]
            cValuerh[i, idx > 0] = cValuerh[i, idx > 0] + (((varInv_rh[idx > 0]-1)/3)/cnt).T

        
        network_par, network_confi = get_confi(cValuelh,ventLh)
        Network_lh.append(network_par.copy())
        NetworkConfidence_lh.append(network_confi.copy())
        
        network_par, network_confi = get_confi(cValuerh,ventRh)
        Network_rh.append(network_par.copy())
        NetworkConfidence_rh.append(network_confi.copy())


    np.savez_compressed(output_dir + '/Network_Par_L_{}.npz'.format(roi_num),data=Network_lh)
    np.savez_compressed(output_dir + '/Network_Par_R_{}.npz'.format(roi_num),data=Network_rh)
    np.savez_compressed(output_dir + '/Network_Confi_L_{}.npz'.format(roi_num),data=NetworkConfidence_lh)
    np.savez_compressed(output_dir + '/Network_Confi_R_{}.npz'.format(roi_num),data=NetworkConfidence_rh)
    np.savez_compressed(output_dir + '/Network_Corr_L_{}.npz'.format(roi_num),data=cValuelh)
    np.savez_compressed(output_dir + '/Network_Corr_R_{}.npz'.format(roi_num),data=cValuerh)

def get_adj_matrix(combine_hemi=False):
    
    
    adj_fs4 = scio.loadmat(base_dir + "/Utilities/fs4_Firstadjacent_vertex.mat")
    adj_lh = adj_fs4['fs4_Firstadjacent_vertex_lh'].squeeze()
    new_adj_matrix_l = []
    for i in range(adj_lh.shape[0]):
        new_adj_matrix_l.append(adj_lh[i].squeeze() - 1)

    adj_rh = adj_fs4['fs4_Firstadjacent_vertex_rh'].squeeze()
    new_adj_matrix_r = []
    for i in range(adj_rh.shape[0]):
        new_adj_matrix_r.append(adj_rh[i].squeeze() - 1)

    return new_adj_matrix_l,new_adj_matrix_r

def get_yeo_18(surf='fsa4'):
    base_dir = os.path.dirname(os.path.abspath(__file__)) + '/../HFR_ai'
    initial_atlas_l = []
    initial_atlas_r = []
    for i in range(1,20): # 1 - 19
        if(surf=='fsa4'):
            vol = nib.load(base_dir + '/Templates/Parcellation_template/lh_network_{}_asym_fs4.mgh'.format(i)).get_fdata().squeeze()
            vor = nib.load(base_dir + '/Templates/Parcellation_template/rh_network_{}_asym_fs4.mgh'.format(i)).get_fdata().squeeze()
        else:
            from neuromaps.transforms import fsaverage_to_fsaverage
            from plos_utils import array_to_gii
            template_data = np.zeros([2562,2])
            for i,hemi in enumerate(['lh','rh']):
                for net_n in range(1,20):
                    data = nib.load("HFR_ai/Templates/Parcellation_template/{}_network_{}_asym_fs4.mgh".format(hemi,net_n)).get_fdata().squeeze()
                    template_data[data > 0, i] = net_n-1
            l,r = fsaverage_to_fsaverage(array_to_gii(template_data[:,0],template_data[:,1],if_path=True,surf='fsa4'),target_density='10k',method='nearest')
            
        initial_atlas_l.append(vol)
        initial_atlas_r.append(vor)
    return initial_atlas_l,initial_atlas_r

def get_indi_parcel_hemi(timeseries_path,tsnr_path,output_dir):

    base_dir = os.path.dirname(os.path.abspath(__file__)) + '/../HFR_ai'
    adj_matrix_l,adj_matrix_r = get_adj_matrix()

    variability = scio.loadmat(base_dir + "/Utilities/Variability_FS4.mat")
    variability_l = variability['lh'].squeeze()
    variability_r = variability['rh'].squeeze()
    
    func_data = scio.loadmat(timeseries_path)
    lhData = func_data['lhData']
    rhData = func_data['rhData']
    tsnr_data = scio.loadmat(tsnr_path)['data'].squeeze()
    
    initial_atlas_l,initial_atlas_r = get_yeo_18()
    iterative_parcellation_hemi(lhData = lhData,rhData = rhData, initial_atlas_l=initial_atlas_l,initial_atlas_r=initial_atlas_r, base_dir=base_dir,
                           tsnr_data_l = tsnr_data[:2562],tsnr_data_r = tsnr_data[2562:],
                           adj_l=adj_matrix_l,adj_r=adj_matrix_r,
                           variability_l=variability_l,variability_r=variability_r,
                           combineLeftRight=True,output_dir=output_dir,
                           confidence_threshold=3,num_iter=10,)

def get_indi_parcel_hemi_fsa5(timeseries_path_l,timeseries_path_r,tsnr_path,output_dir):

    base_dir = os.path.dirname(os.path.abspath(__file__)) + '/../HFR_ai'
    adj_matrix_l,adj_matrix_r = get_adj_matrix_fsa5()

    variability = scio.loadmat(base_dir + "/Utilities/Variability_FS5.mat")
    variability_l = variability['lh'].squeeze()
    variability_r = variability['rh'].squeeze()

    lhData = nib.load(timeseries_path_l).agg_data()
    rhData = nib.load(timeseries_path_r).agg_data()

    tsnr_data = scio.loadmat(tsnr_path)['data'].squeeze()
    
    initial_atlas_l,initial_atlas_r = get_yeo_17_fsa5()

    iterative_parcellation_hemi(lhData = lhData,rhData = rhData, initial_atlas_l=initial_atlas_l,initial_atlas_r=initial_atlas_r, base_dir=base_dir,
                           tsnr_data_l = tsnr_data[:10242],tsnr_data_r = tsnr_data[10242:],
                           adj_l=adj_matrix_l,adj_r=adj_matrix_r,
                           variability_l=variability_l,variability_r=variability_r,
                           combineLeftRight=True,output_dir=output_dir,
                           confidence_threshold=3,num_iter=10,)

def mcad_cortex():
    func_path_par = "/dataset/mcad/{}_timeframes_fs4.mat"
    tsnr_path_par = "dataset/mcad_tsnr/{}_tsnr_fs4.mat"
    output_dir_par = 'IndiPar/mcad/{}/'

    subjects = np.loadtxt("dataset/sub_list.txt",dtype='object').tolist()
    for sub in subjects:
        
        ts_path = func_path_par.format(sub)
        tsnr_path = tsnr_path_par.format(sub)
        output_dir = output_dir_par.format(sub)
        get_indi_parcel_hemi(ts_path,tsnr_path,output_dir)


if __name__ == "__main__":

    mcad_cortex()