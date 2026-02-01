"""
Iterative individualized functional parcellation (fsaverage4, left/right hemispheres).

---------------------
This script performs iterative, individual-specific functional network parcellation
on cortical surface time series, following the general approach described in:

Wang, D., Buckner, R.L., Fox, M.D., Holt, D.J., Holmes, A.J., Stoecklein, S., Langs, G.,
Pan, R., Qian, T., Li, K., et al. (2015).
Parcellating cortical functional networks in individuals. Nat Neurosci 18, 1853-1860.
https://doi.org/10.1038/nn.4164

----------------------
1) Loads fs4 left/right hemisphere time series (lhData, rhData).
2) Loads per-vertex tSNR weights.
3) Initializes 18 networks (plus medial wall) using a group atlas template.
4) Iteratively updates network seed time series using confidence and tSNR weighting.
5) Computes vertex-to-network correlations, applies local neighborhood smoothing,
   and updates parcellation and confidence maps across iterations.

------
Per subject:
1) Time series MAT file (fs4):
   - keys: 'lhData', 'rhData'
   - shape: (n_vertices, n_timepoints) for each hemisphere
   Example: sub-0001_timeframes_fs4.mat

2) tSNR MAT file (fs4):
   - key: 'data'
   - shape: (n_vertices_total,) where n_vertices_total = 2562*2 for fs4
   Example: sub-0001_tsnr_fs4.mat

Shared resources:
3) Group atlas templates (fs4, 19 files per hemi: 1..19)
   - Templates/Parcellation_template/lh_network_{i}_asym_fs4.mgh
   - Templates/Parcellation_template/rh_network_{i}_asym_fs4.mgh

4) Adjacency (first-order neighbor list on fs4 mesh)
   - Utilities/fs4_Firstadjacent_vertex.mat

5) Variability map on fs4 mesh
   - Utilities/Variability_FS4.mat

Outputs
-------
For each subject, the script writes NumPy arrays into the output directory:
- Network_Par_L.npy / Network_Par_R.npy
  list over iterations, each is (18, 2562) parcellation (binary masks per network)

- Network_Confi_L.npy / Network_Confi_R.npy
  list over iterations, each is (18, 2562) confidence-weighted masks

- Network_Corr_L.npy / Network_Corr_R.npy
  final iteration correlation maps after Fisher-z transform and smoothing

-----
- This implementation assumes fsaverage4 with 2562 vertices per hemisphere.
- The default number of networks is 18.
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
    corr = data1.T @ data2 / (data1.shape[0] - 1)
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
        smooth_data[:,i] = np.mean(data[:,adj_matrix[i]],axis=1)
    return smooth_data

def iterative_parcellation_hemi(lhData,rhData,initial_atlas_l,initial_atlas_r,
                           tsnr_data_l,tsnr_data_r,
                           adj_l,adj_r,
                           variability_l,variability_r,
                           combineLeftRight,output_dir,base_dir,
                           confidence_threshold=3,num_iter=10,):
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

    variability = np.concatenate([variability_l,variability_r],axis=0)
    varInv_lh = 1 / normalize(variability)[:2562]
    varInv_rh = 1 / normalize(variability)[2562:]
    
    tsnr_data = np.concatenate([tsnr_data_l,tsnr_data_r],axis=0)
    SNR_lh = normalize(tsnr_data)[:2562]
    SNR_rh = normalize(tsnr_data)[2562:]

    seedDatalh = np.zeros([18,lhData.shape[1]])
    seedDatarh = np.zeros([18,rhData.shape[1]])

    NetworkConfidence_lh = []
    NetworkConfidence_rh = []

    Network_lh = []
    Network_rh = []
    # ---------------------------------------------------
    #  Iterative parcellation
    # ---------------------------------------------------
        
    for cnt in range(1,num_iter+1):
        
        if(not os.path.exists(output_dir+ '/Iter_' + str(cnt))):
            os.makedirs(output_dir+ '/Iter_' + str(cnt))

        GrpNetlh = np.zeros([19,2562])
        GrpNetrh = np.zeros([19,2562])
        if(cnt == 1):

            ventLh = initial_atlas_l[0] > 0
            GrpNetlh[0,:] = ventLh

            ventRh = initial_atlas_r[0] > 0
            GrpNetrh[0,:] = ventRh

            for i2 in range(1,19):  # get the seed waveforms based on Thomas' parcellation, and weight it by inv(Variability)
            
                idx = initial_atlas_l[i2] > 0
                seedDatalh[i2-1,:]= varInv_lh[idx] @ lhData[idx,:] # weight the group map using the inverse of individual difference
                GrpNetlh[i2] = idx

                idx = initial_atlas_r[i2] > 0
                seedDatarh[i2-1,:]= varInv_rh[idx] @ rhData[idx,:] # weight the group map using the inverse of individual difference
                GrpNetrh[i2] = idx

            GrpSeedDatalh =seedDatalh.copy()
            GrpSeedDatarh =seedDatarh.copy()
        else:

            for i2 in range(1,19): # get the seed waveforms based on the last parcellation
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
        for i in range(18):
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

        
    np.save(output_dir + '/Network_Par_L.npy', Network_lh)
    np.save(output_dir + '/Network_Par_R.npy', Network_rh)

    np.save(output_dir + '/Network_Confi_L.npy', NetworkConfidence_lh)
    np.save(output_dir + '/Network_Confi_R.npy', NetworkConfidence_rh)

    np.save(output_dir + '/Network_Corr_L.npy',cValuelh)
    np.save(output_dir + '/Network_Corr_R.npy',cValuerh)

def iterative_parcellation_sub(func_data,
                           tsnr_data,
                           initial_atlas,
                           adj_matrix,
                           variability,
                           output_dir,
                           confidence_threshold=3,
                           num_iter=10,):
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
    

    varInv = 1 / normalize(variability)
    tsnr_data = normalize(tsnr_data)

    region_number = np.unique(initial_atlas).shape[0] # roi + 1, first one is 0
    voxel_number  = func_data.shape[0]
    timepoints = func_data.shape[1]
    seedData = np.zeros([region_number-1,timepoints])

    NetworkConfidence = []
    Network = []
    # ---------------------------------------------------
    #  Iterative parcellation
    # ---------------------------------------------------
        
    for cnt in range(1,num_iter+1):
        if(not os.path.exists(output_dir+ '/Iter_' + str(cnt))):
            os.makedirs(output_dir+ '/Iter_' + str(cnt))
        GrpNet = np.zeros([region_number,voxel_number],dtype=np.int16)
        # GrpNet 19 * 5124
        if(cnt == 1):
            vent = initial_atlas == 0 # medial wall
            GrpNet[0,:] = vent        # save in the group network

            for roi in range(1,19):  # get the seed waveforms based on Thomas' parcellation, and weight it by inv(Variability)
                idx = initial_atlas == (roi)
                seedData[roi-1,:]= varInv[idx] @ func_data[idx,:] # weight the group map using the inverse of individual difference
                GrpNet[roi] = idx
            GrpSeedData =seedData.copy()
        else:
            for roi in range(1,19): # get the seed waveforms based on the last parcellation
                confi = NetworkConfidence[-1][roi-1]
                seedData[roi-1,:] = get_new_seed_ts_by_confi_tsnr(func_data,tsnr_data,confi,confidence_threshold)

        # Weight in the group seed in each iteration, should throw in individual variability map as weight in the future
        if(cnt>1):
            seedData = seedData + GrpSeedData/(cnt-1);

        cValue = cal_corr(seedData.T,func_data.T) # 18 * 5124 
        cValue = 0.5*np.log((1+cValue)/(1-cValue))
        cValue = np.nan_to_num(cValue)
        cValueS = np.zeros_like(cValue) # create new, ensure the orginal values stay in the mean
        # Smooth, decrease the noise
        for i in range(voxel_number):
            cValueS[:,i] = np.mean(cValue[:,adj_matrix[i].squeeze() - 1],axis=1);
        cValue = cValueS
        # Further weight in the group map * inv(Variability) by adding correlation coefficient of 0~ 0.5 according to inv(Variability).
        for i in range(18):
            idx = GrpNet[i+1]
            cValue[i,idx > 0] = cValue[i,idx > 0] + (((varInv[idx > 0]-1)/3)/cnt)

        network_par, network_confi = get_confi(cValue,vent)
        NetworkConfidence.append(network_confi)
        Network.append(network_par)
        
    np.save(output_dir + '/Network_Par_sub.npy', Network)
    np.save(output_dir + '/Network_Confi_sub.npy', NetworkConfidence)
    np.save(output_dir + '/Network_Corr_sub.npy',cValue)

def get_yeo_18():
    base_dir = os.path.dirname(os.path.abspath(__file__)) + '/../HFR_ai'
    initial_atlas_l = []
    initial_atlas_r = []
    for i in range(1,20): # 1 - 19
        vol = nib.load(base_dir + '/Templates/Parcellation_template/lh_network_{}_asym_fs4.mgh'.format(i)).get_fdata().squeeze()
        vor = nib.load(base_dir + '/Templates/Parcellation_template/rh_network_{}_asym_fs4.mgh'.format(i)).get_fdata().squeeze()
        initial_atlas_l.append(vol)
        initial_atlas_r.append(vor)
    return initial_atlas_l,initial_atlas_r

def get_adj_matrix(combine_hemi=False):
    
    
    adj_fs4 = scio.loadmat(base_dir + "/Utilities/fs4_Firstadjacent_vertex.mat")
    adj_lh = adj_fs4['fs4_Firstadjacent_vertex_lh'].squeeze() # -1
    new_adj_matrix_l = []
    for i in range(adj_lh.shape[0]):
        new_adj_matrix_l.append(adj_lh[i].squeeze() - 1)

    adj_rh = adj_fs4['fs4_Firstadjacent_vertex_rh'].squeeze() # -1
    new_adj_matrix_r = []
    for i in range(adj_rh.shape[0]):
        new_adj_matrix_r.append(adj_rh[i].squeeze() - 1)

    return new_adj_matrix_l,new_adj_matrix_r

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

if __name__ == "__main__":

    # 1) Subject list
    subjects = ["sub-0001"]

    # 2) Input file patterns
    # timeseries .mat must contain: lhData, rhData
    func_path_par = "/path/to/dataset/fs4_timeseries/{sub}_timeframes_fs4.mat"

    # tsnr .mat must contain: data (length 5124 = 2562 LH + 2562 RH)
    tsnr_path_par = "/path/to/dataset/fs4_tsnr/{sub}_tsnr_fs4.mat"

    # 3) Output folder pattern (one folder per subject)
    output_dir_par = "/path/to/output/IndiPar/fs4/{sub}/"

    for sub in subjects:
        ts_path = func_path_par.format(sub=sub)
        tsnr_path = tsnr_path_par.format(sub=sub)
        out_dir = output_dir_par.format(sub=sub)

        os.makedirs(out_dir, exist_ok=True)
        get_indi_parcel_hemi(ts_path, tsnr_path, out_dir)
