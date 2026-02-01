import numpy as np
from plos_utils import plt,pd,sns,mpl,plot_surf_v,custom_function,custom_function2,yeo_cmap_19,yeo_color_palette_19,ven_l,ven_r,yeo_color_palette

def show_average(indipa,info_df,dataset='mcad',hemispheres=['left','left','right','right'],
                 views=['lateral', 'medial','medial', 'lateral'],width = 10):
    
    mean_nc = indipa[info_df.Group == "NC"].sum(0).argmax(0).astype(np.float16)
    tt = mean_nc.copy() ; tt[tt == 0] = np.nan
    plot_surf_v(left_data=tt[:2562],right_data=tt[2562:],cmap=yeo_cmap_19,if_roi=True,bg_on_data=False,alpha=1,width = width,
                          avg_method=custom_function,title=['NC'],colorbar=False,hemispheres = hemispheres, views = views,
                          output_file='Plos_ref/results_figure/S2_mean_topo/{}/NC_Mean_Par.pdf'.format(dataset))
    
    if(dataset != 'oasis'):
        mean_ad = indipa[info_df.Group == 'MCI'].sum(0).argmax(0).astype(np.float16)
        tt = mean_ad.copy() ; tt[tt == 0] = np.nan
        plot_surf_v(left_data=tt[:2562],right_data=tt[2562:],cmap=yeo_cmap_19,if_roi=True,bg_on_data=False,alpha=1,width = width,
                            avg_method=custom_function,title=['MCI'],colorbar=False,hemispheres = hemispheres, views = views,
                            output_file='Plos_ref/results_figure/S2_mean_topo/{}/MCI_Mean_Par.pdf'.format(dataset))

    mean_ad = indipa[info_df.Group == 'AD'].sum(0).argmax(0).astype(np.float16)
    tt = mean_ad.copy() ; tt[tt == 0] = np.nan
    plot_surf_v(left_data=tt[:2562],right_data=tt[2562:],cmap=yeo_cmap_19,if_roi=True,bg_on_data=False,alpha=1,width = width,
                         avg_method=custom_function,title=['AD'],colorbar=False,hemispheres = hemispheres, views = views,
                         output_file='Plos_ref/results_figure/S2_mean_topo/{}/AD_Mean_Par.pdf'.format(dataset))
    idx = mean_ad != mean_nc
    idx = idx.astype(np.float16)
    idx[idx ==1 ] = mean_nc[idx == 1]
    tt = idx.copy(); tt[np.concatenate([ven_l,ven_r]) > 0] = np.nan
    tt[tt == 0] = np.nan
    ccc2 =  mpl.colors.ListedColormap(yeo_color_palette_19[1:-1].tolist(),name='my')
    plot_surf_v(left_data=tt[:2562],right_data=tt[2562:],cmap=ccc2,if_roi=True,bg_on_data=True,alpha=1,
                           avg_method=custom_function2,colorbar=False, hemispheres=['left', 'left',],views=['lateral', 'medial'],
                           title=['NC-AD'],threshold=0,darkness=0.5, width = 5,
                           output_file='Plos_ref/results_figure/S2_mean_topo/{}/AD-NC_Mean_Par.pdf'.format(dataset))

    tt[np.isnan(tt)] = 0
    tt = tt.astype(np.int16)
    df = pd.DataFrame(columns=['Area','Network'])
    df['Area'] = np.bincount(tt,minlength=19)[1:]
    df['Network'] = ['%d' % i for i in range(1,19)]
    plt.figure(figsize=[4,2])
    sns.barplot(data=df,x='Network',y='Area',palette=yeo_color_palette[1:])
    plt.savefig('Plos_ref/results_figure/S2_mean_topo/{}/AD-NC_Area_Bar.pdf'.format(dataset),bbox_inches='tight')
    return df