import nibabel as nib
import numpy as np
from nilearn import surface
import os
import matplotlib as mpl
import time
import matplotlib.pyplot as plt
from nilearn import datasets
from nilearn import plotting
fsaverage = datasets.fetch_surf_fsaverage()
import seaborn as sns

def mappingParcellToVolume(data,atlas):
    if(isinstance(atlas,str)):
        atlas = nib.load(atlas)
    atlas_data = atlas.get_fdata()
    atlas.uncache()
    new_data = np.zeros_like(atlas_data)
    atlas_data = np.nan_to_num(atlas_data)
    rois = np.sort(np.unique(atlas_data[:]))[1:]
    print(rois.shape)
    for i,roi in enumerate(rois):
        new_data[atlas_data==roi] = data[i]
    newImage = nib.Nifti1Image(new_data,atlas.affine)
    return newImage
def mapping_gii_cifti(l_surf,r_surf,volume,Outfile):
    template = "E:\matTool\workbench\HCP_WB_Tutorial_1.0\HCP_Q1-Q6_R440_tfMRI_ALLTASKS_level3_zstat1_hp200_s2.dscalar.nii"
    cmd = "e:/matTool/workbench/bin_windows64/wb_command.exe -cifti-create-dense-from-template {template} {scalar_file} -volume-all {volume} -metric CORTEX_LEFT {l_surf} -metric CORTEX_RIGHT {r_surf}"
    print(cmd.format(template=template,scalar_file=Outfile,volume=volume,l_surf = l_surf,r_surf=r_surf))
    print(os.popen(cmd.format(template=template,scalar_file=Outfile,volume=volume,l_surf = l_surf,r_surf=r_surf)).read())
def mapping_volume_to_cifti(nifti,out_file):
    import tempfile
    dir = tempfile.gettempdir()
    surf_L = dir + './L.func.gii'
    surf_R = dir + './R.func.gii'
    volume = dir + './Volume.nii.gz'
    newL,newR = volumeToSurface(nifti)
    nifti.to_filename(volume)
    newL.to_filename(surf_L)
    newR.to_filename(surf_R)
    mapping_gii_cifti(surf_L,surf_R,volume,out_file)

def draw_surf(overlay,cmap='RdBu_r',surf='fslr',color_range=(-5,5),xlabel='Network Loading',add_wall=True,save_path=None,zero_transparent=True,img_size=(800,600),cbar_kws = dict(outer_labels_only=True, pad=.02, n_ticks=2, decimals=2,fontsize=20)):
    from surfplot import Plot
    from surfplot.utils import add_fslr_medial_wall
    from neuromaps.datasets import fetch_fslr, fetch_fsaverage
    import matplotlib.pyplot as plt
    from brainspace.plotting.base import Plotter
    if(surf=='fslr'):
        surfaces = fetch_fslr()
        lh, rh = surfaces['veryinflated']
        if(add_wall):
            overlay = add_fslr_medial_wall(overlay)

    elif(surf == 'fsaverage'):
        surfaces = fetch_fsaverage('10k')
        lh, rh = surfaces['inflated']

    p = Plot(lh, rh,size=img_size,zoom=1.6,brightness=0.6)
    p.add_layer(overlay, cmap=cmap,color_range=color_range,as_outline=False,zero_transparent=zero_transparent)    
    fig = p.build(cbar_kws=cbar_kws,figsize=(img_size[0]/100,img_size[0]/100))
    #fig.axes[1].set_xlabel(xlabel, labelpad=-11, fontstyle='italic',fontsize=15)
    if(save_path):
        fig.savefig(save_path,bbox_inches='tight',dpi=400)
        plt.close()
    Plotter.close_all()

def volumeToSurface(image,interpolation='linear'):
    fsaverageL = r"E:\brain\BNatlas\BN_Atlas_freesurfer\fsaverage\fsaverage_LR32k\fsaverage.L.inflated.32k_fs_LR.surf.gii"
    fsaverageR = r"E:\brain\BNatlas\BN_Atlas_freesurfer\fsaverage\fsaverage_LR32k\fsaverage.R.inflated.32k_fs_LR.surf.gii"
    textureL = surface.vol_to_surf(image,fsaverageL,inner_mesh=r"E:\brain\BNatlas\BN_Atlas_freesurfer\fsaverage\fsaverage_LR32k\fsaverage.L.white.32k_fs_LR.surf.gii")
    textureR = surface.vol_to_surf(image,fsaverageR,inner_mesh=r"E:\brain\BNatlas\BN_Atlas_freesurfer\fsaverage\fsaverage_LR32k\fsaverage.R.white.32k_fs_LR.surf.gii")
    
    #textureL = surface.vol_to_surf(image,fsaverageL,radius=4,interpolation=interpolation,kind='line',mask_img='Mask.nii')
    #textureR = surface.vol_to_surf(image,fsaverageR,radius=4,interpolation=interpolation,kind='line',mask_img='Mask.nii')
    
    newL = nib.GiftiImage(header=nib.load(r"E:\brain\GS\surface\NC_tmap.nii.L.func.gii").header)
    newR = nib.GiftiImage(header=nib.load(r"E:\brain\GS\surface\NC_tmap.nii.R.func.gii").header)

    dataArrayL = nib.gifti.gifti.GiftiDataArray(textureL.astype(np.float32))
    dataArrayR = nib.gifti.gifti.GiftiDataArray(textureR.astype(np.float32))

    newL.add_gifti_data_array(dataArrayL)
    newR.add_gifti_data_array(dataArrayR)

    return newL,newR

def GMV_Mapping(PATH,pos_max=None,pos_min=None,neg_max=None,neg_min=None,pcmap=None,ncmap=None,MAX=False):
    from matplotlib import cm
    cmd = 'E:/matTool/workbench/bin_windows64/wb_command.exe -volume-to-surface-mapping  %s \
            E:/brain/BNatlas/BN_Atlas_freesurfer/fsaverage/fsaverage_LR32k/fsaverage.L.inflated.32k_fs_LR.surf.gii %s -trilinear'
    cmd2 = 'E:/matTool/workbench/bin_windows64/wb_command.exe -volume-to-surface-mapping  %s \
            E:/brain/BNatlas/BN_Atlas_freesurfer/fsaverage/fsaverage_LR32k/fsaverage.R.inflated.32k_fs_LR.surf.gii %s -trilinear'
    
    
    for path in PATH:
        dir = os.path.dirname(path)
        name = os.path.basename(path)
        
        CMD = cmd % (path,'%s/' % dir+name+'.L.func.gii')
        print(CMD)
        print(os.system(CMD))
        CMD = cmd2 % (path,'%s/' % dir+name+'.R.func.gii')
        print(os.system(CMD))
    time.sleep(5)
    for path in PATH:
        dir = os.path.dirname(path)
        name = os.path.basename(path)
        L_data = nib.load('%s/' % dir+name+'.L.func.gii').agg_data()*100
        R_data = nib.load('%s/' % dir+name+'.R.func.gii').agg_data()*100

        Data = np.concatenate([L_data, R_data])
        if(MAX):
            if(np.sum(Data>0) > 0):
                print('Postive:',)
                print('min:',np.min(Data[Data>0]),'max:',np.max(Data[Data>0]))
            if(np.sum(Data<0) > 0):
                print('Negative:',)
                print('min:',np.min(Data[Data<0]),'max:',np.max(Data[Data<0]))
        Data[np.abs(Data)<1] = 0
        if(pos_max is None):
            pos_max = np.max(Data[Data>0])
        if(pos_min is None):
            pos_min = np.min(Data[Data>0])
        if(neg_max is None):
            neg_max = np.max(Data[Data<0])
        if(neg_min is None):
            neg_min = np.min(Data[Data<0])
        norm_p = mpl.colors.Normalize(vmin=pos_min, vmax=pos_max)
        norm_n = mpl.colors.Normalize(vmin=neg_min, vmax=neg_max)

        colorMapping_p = cm.ScalarMappable(norm=norm_p, cmap = pcmap)
        colorMapping_n = cm.ScalarMappable(norm=norm_n, cmap = ncmap)

        labelDict = {}
        for i, p in enumerate(Data):
            colorValue = (1,1,1,1)
            if(p>0):
                colorValue = colorMapping_p.to_rgba(p)
            elif(p<0):
                colorValue = colorMapping_n.to_rgba(p)
            labelDict[p] = (i, colorValue)

        names = ['CIFTI_STRUCTURE_CORTEX_LEFT' for i in range(L_data.shape[0])]
        names.extend(['CIFTI_STRUCTURE_CORTEX_RIGHT' for i in range(R_data.shape[0])])
        verteces = [i for i in range(L_data.shape[0])]
        verteces.extend([i for i in range(L_data.shape[0])])
        verteces = np.asarray(verteces)
        brainModelAxis = nib.cifti2.cifti2_axes.BrainModelAxis(name=names, vertex=np.asarray(verteces),
                                                               nvertices={'CIFTI_STRUCTURE_CORTEX_LEFT': 32492,
                                                                          'CIFTI_STRUCTURE_CORTEX_RIGHT': 32492}, )
        newLabelAxis = nib.cifti2.cifti2_axes.LabelAxis(['aaa'], labelDict)
        newheader = nib.cifti2.cifti2.Cifti2Header.from_axes((newLabelAxis, brainModelAxis))
        newImage = nib.cifti2.cifti2.Cifti2Image(dataobj=Data.reshape([1, -1]), header=newheader)
        newImage.to_filename('%s/' % dir+name+'.dlabel.nii')
        
def GMV_MappingAtlas(PATH, pos_max, pos_min, neg_max, neg_min, pcmap, ncmap, MAX=False,
                nii_path=r"E:\brain\BNatlas\BN_Atlas_274_with_cerebellum_without_255.nii",savePath = '',saveName=''):
    from nilearn import surface
    fsaverageL = r"E:\brain\BNatlas\BN_Atlas_freesurfer\fsaverage\fsaverage_LR32k\fsaverage.L.inflated.32k_fs_LR.surf.gii"
    fsaverageR = r"E:\brain\BNatlas\BN_Atlas_freesurfer\fsaverage\fsaverage_LR32k\fsaverage.R.inflated.32k_fs_LR.surf.gii"
    dlable = nib.load(
        r"E:\brain\BNatlas\BN_Atlas_freesurfer\fsaverage\fsaverage_LR32k\fsaverage.BN_Atlas.32k_fs_LR.dlabel.nii")
    llabel = nib.load(r"E:\brain\BNatlas\BN_Atlas_freesurfer\fsaverage\fsaverage_LR32k\fsaverage.L.BN_Atlas.32k_fs_LR.label.gii")
    rlabel = nib.load(r"E:\brain\BNatlas\BN_Atlas_freesurfer\fsaverage\fsaverage_LR32k\fsaverage.R.BN_Atlas.32k_fs_LR.label.gii")
    llable_data = llabel.agg_data()
    rlable_data = rlabel.agg_data()
    dlabel_data = np.asarray(dlable.dataobj)
    nii = nib.load(nii_path)
    #print(nii.affine)
    
    template = np.nan_to_num(nii.get_fdata())
    template = template[::-1,:,:]
    rois = np.unique(template[:])[1:]
    BrianModelAxis = dlable.header.get_axis(1)

    for path in PATH:
        if(isinstance(path,str)):
            dir = os.path.dirname(path)
            name = os.path.basename(path)
            image = nib.load(path)
            print(image.affine)
            imageData = image.get_fdata()
            image.uncache()
        else:
            image = path
            imageData = image.get_fdata()   
        #del image
        imageData = np.nan_to_num(imageData)
        if (MAX):
            if (np.sum(imageData > 0) > 0):
                print('Postive:', )
                print('min:', np.min(imageData[imageData > 0]), 'max:', np.max(imageData[imageData > 0]))
            if (np.sum(imageData < 0) > 0):
                print('Negative:', )
                print('min:', np.min(imageData[imageData < 0]), 'max:', np.max(imageData[imageData < 0]))

        norm_p = mpl.colors.Normalize(vmin=pos_min, vmax=pos_max)
        norm_n = mpl.colors.Normalize(vmin=neg_min, vmax=neg_max)
        colorMapping_p = mpl.cm.ScalarMappable(norm=norm_p, cmap=pcmap)
        colorMapping_n = mpl.cm.ScalarMappable(norm=norm_n, cmap=ncmap)
        labelDict = {}
        #for i in range(1,421):
        #    labelDict[i] = (labelDict[i][0],( 1, 1, 1, 1))
        for roi in range(1,211):
            fvalue = np.max(imageData[template == roi])
            #print(fvalue)
            if(fvalue == 0):
                colorValue = (1,1,1,1)
            elif(fvalue > 0):
                colorValue = colorMapping_p.to_rgba(fvalue)
            else:
                colorValue = colorMapping_n.to_rgba(fvalue)
            labelDict[roi] = (roi, colorValue)
            labelDict[roi+210] = (roi+210, colorValue)
     
        imageData[template<211] = 0
        NewImage = nib.Nifti1Image(imageData,image.affine)
        textureL = surface.vol_to_surf(NewImage, fsaverageL,
                                       inner_mesh=r"E:\brain\BNatlas\BN_Atlas_freesurfer\fsaverage\fsaverage_LR32k\fsaverage.L.white.32k_fs_LR.surf.gii")
        textureR = surface.vol_to_surf(NewImage, fsaverageR,
                                       inner_mesh=r"E:\brain\BNatlas\BN_Atlas_freesurfer\fsaverage\fsaverage_LR32k\fsaverage.R.white.32k_fs_LR.surf.gii")

        #Data[np.abs(Data) < 1] = 0

        textureL = textureL* 1000
        textureR = textureR* 1000
        textureL[np.abs(textureL) < 421] = 0
        textureR[np.abs(textureR) < 421] = 0
        Data = np.concatenate([textureL, textureR])


        norm_p = mpl.colors.Normalize(vmin=pos_min*1000, vmax=pos_max*1000)
        norm_n = mpl.colors.Normalize(vmin=neg_min*1000, vmax=neg_max*1000)
        colorMapping_p = mpl.cm.ScalarMappable(norm=norm_p, cmap=pcmap)
        colorMapping_n = mpl.cm.ScalarMappable(norm=norm_n, cmap=ncmap)
        for i, p in enumerate(Data):
            colorValue = (1, 1, 1, 1)
            if (p > 0):
                colorValue = colorMapping_p.to_rgba(p)
            elif (p < 0):
                colorValue = colorMapping_n.to_rgba(p)
            labelDict[p] = (p, colorValue)

        textureL[BrianModelAxis.vertex[:29696]] = dlabel_data[0,:29696]
        textureR[BrianModelAxis.vertex[29696:]] = dlabel_data[0,29696:]
        textureL[llable_data!=-1] = llable_data[llable_data!=-1]
        textureR[rlable_data!=-1] = rlable_data[rlable_data!=-1]
        Data = np.concatenate([textureL, textureR])

        names = ['CIFTI_STRUCTURE_CORTEX_LEFT' for i in range(textureL.shape[0])]
        names.extend(['CIFTI_STRUCTURE_CORTEX_RIGHT' for i in range(textureR.shape[0])])
        verteces = [i for i in range(textureL.shape[0])]
        verteces.extend([i for i in range(textureR.shape[0])])
        verteces = np.asarray(verteces)
        brainModelAxis = nib.cifti2.cifti2_axes.BrainModelAxis(name=names, vertex=np.asarray(verteces),
                                                               nvertices={'CIFTI_STRUCTURE_CORTEX_LEFT': 32492,
                                                                          'CIFTI_STRUCTURE_CORTEX_RIGHT': 32492}, )
        newLabelAxis = nib.cifti2.cifti2_axes.LabelAxis(['aaa'], labelDict)
        newheader = nib.cifti2.cifti2.Cifti2Header.from_axes((newLabelAxis, brainModelAxis))
        newImage = nib.cifti2.cifti2.Cifti2Image(dataobj=Data.reshape([1, -1]), header=newheader)
        newImage.to_filename(savePath+'/' +saveName+ '.dlabel.nii')


def ViewSurface(image,threshold=1.94):
    fsaverageL = r"E:\brain\atlas\fsaverage\pial_left.gii"
    fsaverageR = r"E:\brain\atlas\fsaverage\pial_right.gii"
    textureL = surface.vol_to_surf(image,fsaverage.pial_left,inner_mesh=fsaverage.white_left)
    textureR = surface.vol_to_surf(image,fsaverage.pial_right,inner_mesh=fsaverage.white_right)
    
    fig,axes = plt.subplots(1,2,figsize=[10,6],subplot_kw={'projection': '3d'})
    vmax = np.max([np.abs(textureL),np.abs(textureR)])
    
    plotting.plot_surf_stat_map(fsaverage.pial_left,textureL, hemi='left',
                            threshold=threshold,
                            bg_map=fsaverage.sulc_right,
                            cmap=sns.color_palette('cold_hot',as_cmap=True),
                            darkness=0.5,
                            colorbar=False,
                            #symmetric_cbar=True,
                            bg_on_data = True,
                            axes=axes[0],
                            vmax=vmax,
                            figure=fig
                           )
    
    plotting.plot_surf_stat_map(fsaverage.pial_right,textureR, hemi='right',
                        threshold=threshold,
                        bg_map=fsaverage.sulc_right,
                        cmap=sns.color_palette('cold_hot',as_cmap=True),
                        darkness=0.5,
                        colorbar=False,
                        #symmetric_cbar=True,
                       bg_on_data = True,
                        axes=axes[1],
                        vmax=vmax,
                            figure=fig
                       )
    plt.show()
    fig2,axes2 = plt.subplots(1,2,figsize=[10,6],subplot_kw={'projection': '3d'})
    plotting.plot_surf_stat_map(fsaverage.pial_left,textureL, hemi='right',
                        threshold=threshold,
                        bg_map=fsaverage.sulc_right,
                        cmap=sns.color_palette('cold_hot',as_cmap=True),
                        darkness=0.5,
                        colorbar=False,
                        #symmetric_cbar=True,
                       bg_on_data = True,
                        axes=axes2[0],
                        vmax=vmax,
                            figure=fig2
                       )
    plotting.plot_surf_stat_map(fsaverage.pial_right,textureR, hemi='left',
                        threshold=threshold,
                        bg_map=fsaverage.sulc_right,
                        cmap=sns.color_palette('cold_hot',as_cmap=True),
                        darkness=0.5,
                        colorbar=True,
                        #symmetric_cbar=True,
                       bg_on_data = True,
                        axes=axes2[1],
                        vmax=vmax,
                            figure=fig2
                       )
    plt.show()

def plot_surf_fsa4(left_data, right_data, if_roi=False,
                     hemispheres=['left', 'right'], bg_on_data=False,
                     inflate=False, views=['lateral', 'medial'],
                     output_file=None, title=None, colorbar=True, width = 5,
                     vmin=None, vmax=None, threshold=None,overlay_roi = None, darkness = 0.5,
                     symmetric_cbar='auto', cmap='cold_hot', **kwargs):
   
    from matplotlib import gridspec
    from nilearn.plotting import plot_surf_stat_map,plot_surf_roi
    from nilearn.plotting.surf_plotting import _check_views,_check_hemispheres,_get_colorbar_and_data_ranges,_colorbar_from_array
    import itertools
    for arg in ("figure", "axes", "engine"):
        if arg in kwargs:
            raise ValueError(
                'plot_img_on_surf does not accept '
                f'{arg} as an argument')
    base_dir = os.path.dirname(os.path.abspath(__file__))
    surf = {
        'left': base_dir + "/templates/fsaverage/fsaverage4/tpl-fsaverage_den-3k_hemi-L_inflated.surf.gii",
        'right': base_dir + "/templates/fsaverage/fsaverage4/tpl-fsaverage_den-3k_hemi-R_inflated.surf.gii",
    }
    surf_mesh = {}
    surf_mesh['curv_left'] = base_dir + "/templates/fsaverage/fsaverage4/tpl-fsaverage_den-3k_hemi-L_desc-sulc_midthickness.shape.gii"
    surf_mesh['curv_right'] = base_dir + "/templates/fsaverage/fsaverage4/tpl-fsaverage_den-3k_hemi-R_desc-sulc_midthickness.shape.gii"
    texture = {
        'left': left_data,
        'right': right_data
    }

    modes = _check_views(views)
    hemis = _check_hemispheres(hemispheres)


    if(overlay_roi is not None):
        overlay = {}
        overlay['left'] = overlay_roi[:2562]
        overlay['right'] = overlay_roi[2562:]

    cbar_h = .2
    title_h = .25 * (title is not None)
    #w, h = plt.figaspect()
    h = width * (len(modes) + cbar_h + title_h) / len(hemispheres)
    fig = plt.figure(figsize=(width, h), constrained_layout=False)
    height_ratios = [title_h] + [1.] * len(modes) + [cbar_h]
    grid = gridspec.GridSpec(
        len(modes) + 2, len(hemis),
        left=0., right=1., bottom=0., top=1.,
        height_ratios=height_ratios, hspace=0.0, wspace=0.0)
    axes = []

    # get vmin and vmax for entire data (all hemis)
    _, _, vmin, vmax = _get_colorbar_and_data_ranges(
        np.concatenate([left_data,right_data]),
        vmin=vmin,
        vmax=vmax,
        symmetric_cbar=symmetric_cbar,
        symmetric_data_range=False,
    )
    if(not if_roi):
        plot_func = plot_surf_stat_map
    else:
        plot_func = plot_surf_roi
    for i, (mode, hemi) in enumerate(itertools.product(modes, hemis)):

        curv_map = surface.load_surf_data(surf_mesh[f"curv_{hemi}"])
        bg_map = curv_map
        ax = fig.add_subplot(grid[i + len(hemis)], projection="3d")
        axes.append(ax)

        plot_func(
            surf[hemi],
            texture[hemi],
            view=mode,
            hemi=hemi,
            bg_map=bg_map,
            bg_on_data=bg_on_data,
            axes=ax,
            colorbar=False,  # Colorbar created externally.
            vmin=vmin,
            vmax=vmax,
            threshold=threshold,
            cmap=cmap,
            symmetric_cbar=symmetric_cbar,
            darkness=darkness,
            **kwargs,
        )
        if(overlay_roi is not None):
                plotting.plot_surf_contours(surf[hemi], overlay[hemi],
                            axes=ax, levels=[1], colors=['gray'])
        ax.set_box_aspect(None, zoom=1.5)

    if colorbar:
        sm = _colorbar_from_array(
            left_data,
            vmin,
            vmax,
            threshold,
            symmetric_cbar=symmetric_cbar,
            cmap=plt.get_cmap(cmap),
        )

        cbar_grid = gridspec.GridSpecFromSubplotSpec(3, 3, grid[-1, :])
        cbar_ax = fig.add_subplot(cbar_grid[1])
        axes.append(cbar_ax)
        fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    if title is not None:
        fig.suptitle(title, y=1. - title_h / sum(height_ratios), va="bottom")

    if output_file is not None:
        fig.savefig(output_file, bbox_inches="tight", dpi= 400)
        plt.close(fig)
    else:
        return fig, axes
def plot_surf_fsa4_2border(
    left_data, right_data, if_roi=False,
    hemispheres=['left', 'right'], bg_on_data=False,
    inflate=False, views=['lateral', 'medial'],
    output_file=None, title=None, colorbar=True, width=5,
    vmin=None, vmax=None, threshold=None, overlay_roi=None, overlay_roi2=None, darkness=0.5,
    symmetric_cbar='auto', cmap='cold_hot', **kwargs
):
    from matplotlib import gridspec
    from nilearn import plotting, surface
    import itertools
    import numpy as np
    import os
    for arg in ("figure", "axes", "engine"):
        if arg in kwargs:
            raise ValueError(
                'plot_img_on_surf does not accept '
                f'{arg} as an argument')
    base_dir = os.path.dirname(os.path.abspath(__file__))
    surf = {
        'left': base_dir + "/templates/fsaverage/fsaverage4/tpl-fsaverage_den-3k_hemi-L_inflated.surf.gii",
        'right': base_dir + "/templates/fsaverage/fsaverage4/tpl-fsaverage_den-3k_hemi-R_inflated.surf.gii",
    }
    surf_mesh = {}
    surf_mesh['curv_left'] = base_dir + "/templates/fsaverage/fsaverage4/tpl-fsaverage_den-3k_hemi-L_desc-sulc_midthickness.shape.gii"
    surf_mesh['curv_right'] = base_dir + "/templates/fsaverage/fsaverage4/tpl-fsaverage_den-3k_hemi-R_desc-sulc_midthickness.shape.gii"
    texture = {
        'left': left_data,
        'right': right_data
    }

    from nilearn.plotting.surf_plotting import _check_views, _check_hemispheres, _get_colorbar_and_data_ranges, _colorbar_from_array

    modes = _check_views(views)
    hemis = _check_hemispheres(hemispheres)

    if overlay_roi is not None:
        overlay = {}
        overlay['left'] = overlay_roi[:2562]
        overlay['right'] = overlay_roi[2562:]
    if overlay_roi2 is not None:
        overlay2 = {}
        overlay2['left'] = overlay_roi2[:2562]
        overlay2['right'] = overlay_roi2[2562:]

    cbar_h = .2
    title_h = .25 * (title is not None)
    h = width * (len(modes) + cbar_h + title_h) / len(hemispheres)
    fig = plt.figure(figsize=(width, h), constrained_layout=False)
    height_ratios = [title_h] + [1.] * len(modes) + [cbar_h]
    grid = gridspec.GridSpec(
        len(modes) + 2, len(hemis),
        left=0., right=1., bottom=0., top=1.,
        height_ratios=height_ratios, hspace=0.0, wspace=0.0)
    axes = []

    _, _, vmin, vmax = _get_colorbar_and_data_ranges(
        np.concatenate([left_data, right_data]),
        vmin=vmin,
        vmax=vmax,
        symmetric_cbar=symmetric_cbar,
        symmetric_data_range=False,
    )
    if not if_roi:
        plot_func = plotting.plot_surf_stat_map
    else:
        plot_func = plotting.plot_surf_roi
    for i, (mode, hemi) in enumerate(itertools.product(modes, hemis)):

        curv_map = surface.load_surf_data(surf_mesh[f"curv_{hemi}"])
        bg_map = curv_map
        ax = fig.add_subplot(grid[i + len(hemis)], projection="3d")
        axes.append(ax)

        plot_func(
            surf[hemi],
            texture[hemi],
            view=mode,
            hemi=hemi,
            bg_map=bg_map,
            bg_on_data=bg_on_data,
            axes=ax,
            colorbar=False,
            vmin=vmin,
            vmax=vmax,
            threshold=threshold,
            cmap=cmap,
            symmetric_cbar=symmetric_cbar,
            darkness=darkness,
            **kwargs,
        )
        if overlay_roi is not None:
            plotting.plot_surf_contours(
                surf[hemi], overlay[hemi],
                axes=ax, levels=[1], colors=['#5B7FA3'], linewidths=1
            )
        if overlay_roi2 is not None:
            plotting.plot_surf_contours(
                surf[hemi], overlay2[hemi],
                axes=ax, levels=[1], colors=['#D8A47F'], linewidths=1
            )
        ax.set_box_aspect(None, zoom=1.5)

    if colorbar:
        sm = _colorbar_from_array(
            left_data,
            vmin,
            vmax,
            threshold,
            symmetric_cbar=symmetric_cbar,
            cmap=plt.get_cmap(cmap),
        )
        cbar_grid = gridspec.GridSpecFromSubplotSpec(3, 3, grid[-1, :])
        cbar_ax = fig.add_subplot(cbar_grid[1])
        axes.append(cbar_ax)
        fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    if title is not None:
        fig.suptitle(title, y=1. - title_h / sum(height_ratios), va="bottom")

    if output_file is not None:
        fig.savefig(output_file, bbox_inches="tight", dpi=400)
        plt.close(fig)
    else:
        return fig, axes
def custom_function(vertices):
    '''
    For better show the surface parcellation
    '''
    if(np.sum(np.isnan(vertices)) > 0):
        return np.nan
    else:
        count = np.bincount([vertices[0],vertices[1],vertices[2]])
        value = np.argmax(count)
        return value.astype(np.float16)
def custom_function2(vertices):
    if(np.sum(np.isnan(vertices)) == 3):
        return np.nan
    elif((np.sum(np.isnan(vertices))) > 0 & (np.sum(np.isnan(vertices))) < 2):
        return np.nanmax(vertices).astype(np.float16)
    else:
        count = np.bincount([vertices[0],vertices[1],vertices[2]])
        value = np.argmax(count)
        return value.astype(np.float16)
    

    

def _get_surf_data(if_roi,surf_mesh,surf_type,left_data,right_data,bg_type,set_ven_nan=False,):
    from nilearn.plotting import plot_surf_stat_map,plot_surf_roi
    if(not if_roi):
        plot_func = plot_surf_stat_map
    else:
        plot_func = plot_surf_roi

    base_dir = os.path.dirname(os.path.abspath(__file__))
    surf = {
        'left': base_dir + "/templates/fsaverage/fsaverage4/tpl-fsaverage_den-3k_hemi-L_{surf_type}.surf.gii".format(surf_type=surf_type),
        'right': base_dir + "/templates/fsaverage/fsaverage4/tpl-fsaverage_den-3k_hemi-R_{surf_type}.surf.gii".format(surf_type=surf_type),
        'bg_left':base_dir + "/templates/fsaverage/fsaverage4/tpl-fsaverage_den-3k_hemi-L_desc-{bg_type}_midthickness.shape.gii".format(bg_type=bg_type),
        'bg_right':base_dir + "/templates/fsaverage/fsaverage4/tpl-fsaverage_den-3k_hemi-R_desc-{bg_type}_midthickness.shape.gii".format(bg_type=bg_type)
    }
    texture = {
        'left': left_data,
        'right': right_data
    }

    if(isinstance(left_data,list)):
        num_brain = len(left_data)
        texture = [{'left': left, 'right': right} for left,right in zip(left_data,right_data)]
    else:
        num_brain = 1
        texture = [{'left': left_data, 'right': right_data}]
    return plot_func, surf, texture, num_brain


def plot_surf_v(left_data, right_data, surf_mesh='fsaverage4', if_roi=False, width = 10, surf_type= 'inflated', bg_type= 'sulc',
                     hemispheres=['left','left','right','right'], bg_on_data=False, title_fontsize=18,
                     set_ven_nan=False, views=['lateral', 'medial','medial', 'lateral'],
                     output_file=None, title=None, colorbar=True,
                     vmin=None, vmax=None, threshold=None,
                     symmetric_cbar='auto', cmap='cold_hot', overlay_roi = None, **kwargs):

    from matplotlib import gridspec
    from nilearn.plotting.surf_plotting import _check_views,_check_hemispheres,_get_colorbar_and_data_ranges,_colorbar_from_array
    import itertools
    for arg in ("figure", "axes"):
        if arg in kwargs:
            raise ValueError(
                'plot_img_on_surf does not accept '
                f'{arg} as an argument')
        
    plot_func, surf, texture, num_brain = _get_surf_data(if_roi,surf_mesh,surf_type,left_data,right_data,bg_type,set_ven_nan=False)
    cbar_w = .1 * (colorbar)
    title_w = .2 * (title is not None)

    h = 1 / (cbar_w + title_w + 1.3 * len(hemispheres)) * width
    fig = plt.figure(figsize=(width, h * num_brain), constrained_layout=False)
    weight_ratios = [title_w] + [1.3]* len(hemispheres) + [cbar_w]
    grid = gridspec.GridSpec(1 * num_brain, len(hemispheres) + 2,
        left=0., right=1., bottom=-0.1, top=1.1,
        width_ratios=weight_ratios, hspace=0.0, wspace=0.0)
    axes = []
    for pt in range(num_brain):
        # _, _, vmin, vmax = _get_colorbar_and_data_ranges(
        #     np.concatenate([texture[pt]['left'],texture[pt]['right']]),
        #     vmin=vmin,
        #     vmax=vmax,
        #     symmetric_cbar=symmetric_cbar,
        #     symmetric_data_range=False,
        # )
        if title is not None:
            ax = fig.add_subplot(grid[pt,0])
            axes.append(ax)
            ax.set_title(title[pt], loc='left', y =0.5, verticalalignment='center', horizontalalignment= 'left', rotation = 'vertical', fontsize=title_fontsize)
            ax.axis('off')
        for i, (hemi,mode) in enumerate(zip(hemispheres,views)):         
            bg_map = surface.load_surf_data(surf[f"bg_{hemi}"])
            ax = fig.add_subplot(grid[pt,i + 1], projection="3d")
            axes.append(ax)
            plot_func(
                surf[hemi],
                texture[pt][hemi],
                view=mode,
                hemi=hemi,
                bg_map=bg_map,
                bg_on_data=bg_on_data,
                axes=ax,
                colorbar=False,
                # vmin=vmin,
                # vmax=vmax,
                threshold=threshold,
                cmap=cmap,
                avg_method=custom_function,
                # engine='matplotlib',
                # symmetric_cbar=symmetric_cbar,
                # avg_method='min',
                **kwargs,
            )
            if(overlay_roi):
                plotting.plot_surf_contours(surf[hemi], overlay_roi,
                            axes=ax)
            ax.set_box_aspect(None, zoom=1.5)
        if colorbar:
            sm = _colorbar_from_array(
                left_data,
                vmin,
                vmax,
                threshold,
                symmetric_cbar=symmetric_cbar,
                cmap=plt.get_cmap(cmap),
            )
            cbar_grid = gridspec.GridSpecFromSubplotSpec(3, 2, grid[pt,i + 2],height_ratios=[0.2,1,0.2])
            cbar_ax = fig.add_subplot(cbar_grid[1,1])
            axes.append(cbar_ax)
            fig.colorbar(sm, cax=cbar_ax, orientation='vertical')
    if output_file is not None:
        fig.savefig(output_file, bbox_inches="tight")
        plt.close(fig)
    else:
        return fig, axes


if __name__ == "__main__":

    #L,R = volumeToSurface(r"E:\zhoakun\node\AD.nii")
    #nib.save(L,r"E:\zhoakun\node\surface\AD.L.func.gii")
    #nib.save(R,r"E:\zhoakun\node\surface\AD.R.func.gii")
    b = ['adnc','admc','mcnc']
    a = ['AD','MCI','NC',]
    for name in b:
        data = nib.load("E:/zhoakun/node/%s.nii" % (name)).get_fdata()
        max1 = np.nanmax(data)
        data[data==0] = 10
        min1 = np.nanmin(data)
        print(min1,max1)
        GMV_MappingAtlas(["E:/zhoakun/node/%s.nii" % name],pcmap="YlOrRd",pos_max=16, pos_min=1, neg_max=-1, neg_min=0, ncmap='Oranges',
        savePath = 'E:/zhoakun/node/surface/',saveName=name,nii_path=r"E:\zhoakun\BN_Atlas_246_1mm.nii")