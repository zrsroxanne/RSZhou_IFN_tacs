import os
import sys
if(os.path.exists('I:/My_Work/pin_ind_net')):
    os.chdir('I:/My_Work/pin_ind_net')
    sys.path.append('I:/My_Work/nits')
    sys.path.append('I:/My_Work/pin_ind_net')
else:
    os.chdir('/home/zrs/My_Work/pin_ind_net')
    sys.path.append('/home/zrs/My_Work/nits')
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import nibabel as nib
import h5py
import scipy.io as scio
from scipy import stats
from utils import array_to_gii
from S5_analysis_function import draw_surf
import seaborn as sns
from neuromaps.transforms import fsaverage_to_fslr
import tqdm
from statannotations.Annotator import Annotator
from nilearn import plotting
import statsmodels.formula.api as smf
from IPython.display import display, Image
import matplotlib.pyplot as plt
from brainspace.plotting.base import Plotter
from enigmaView import plot_surface,load_customised,plot_cortical_v
from view import plot_surf_v, plot_surf_fsa4
from view import custom_function2,custom_function
os.environ['PATH'] = "/Users/warx/tools/workbench/bin_macosx64:" + os.environ['PATH']
from brainspace.plotting.surface_plotting import plot_hemispheres
from enigmatoolbox.plotting import plot_surf,plot_cortical
from statsmodels.stats.multitest import fdrcorrection
from PIL import Image
import matplotlib as mpl
cmp = sns.color_palette('tab20',as_cmap=True)
cmp = np.array(cmp.colors)[[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,16,17,18],:].tolist()
cmp = mpl.colors.ListedColormap(cmp[:17],"MY")

group_cmap = sns.color_palette(['#bababa','#404040','#e63946'])
group_cmap2 = sns.color_palette(['#bababa','#e63946'])


_, yeo_color_palette , _ = nib.freesurfer.read_annot("data/Yeo_JNeurophysiol11_FreeSurfer/fsaverage5/label/lh.Yeo2011_17Networks_N1000.annot")
yeo_color_palette = yeo_color_palette[:,:3] / 255
#yeo_color_palette = yeo_color_palette[1:]
yeo_cmap = mpl.colors.ListedColormap(yeo_color_palette.tolist(),name='my')

yeo_color_palette_18 = np.concatenate([yeo_color_palette,[[0.2,0.2,0.2]]],axis=0)
yeo_cmap_18 = mpl.colors.ListedColormap(yeo_color_palette_18.tolist(),name='yeo18')
yeo_color_palette_19 = np.concatenate([yeo_color_palette,[[0.86,0.71,0.56]]],axis=0)
yeo_cmap_19 = mpl.colors.ListedColormap(yeo_color_palette_19.tolist(),name='yeo18')
yeo_cmap_19_no_ven = mpl.colors.ListedColormap(yeo_color_palette_19[1:].tolist(),name='my')


cmap = sns.color_palette(['#303841','#0077b6','#e63946'])

box_pairs_m = [(("1","NC"),("1",'MCI')),(("1","NC"),("1",'AD')),
             (("2","NC"),("2",'MCI')),(("2","NC"),("2",'AD')),
             (("3","NC"),("3",'MCI')),(("3","NC"),("3",'AD')),
             (("4","NC"),("4",'MCI')),(("4","NC"),("4",'AD')),
             (("5","NC"),("5",'MCI')),(("5","NC"),("5",'AD')),
             (("6","NC"),("6",'MCI')),(("6","NC"),("6",'AD')),
             (("7","NC"),("7",'MCI')),(("7","NC"),("7",'AD')),
             (("8","NC"),("8",'MCI')),(("8","NC"),("8",'AD')),
             (("9","NC"),("9",'MCI')),(("9","NC"),("9",'AD')),
             (("10","NC"),("10",'MCI')),(("10","NC"),("10",'AD')),
            (("11","NC"),("11",'MCI')),(("11","NC"),("11",'AD')),
                (("12","NC"),("12",'MCI')),(("12","NC"),("12",'AD')),
                (("13","NC"),("13",'MCI')),(("13","NC"),("13",'AD')),
                    (("14","NC"),("14",'MCI')),(("14","NC"),("14",'AD')),
                    (("15","NC"),("15",'MCI')),(("15","NC"),("15",'AD')),
                        (("16","NC"),("16",'MCI')),(("16","NC"),("16",'AD')),
                        (("17","NC"),("17",'MCI')),(("17","NC"),("17",'AD')),
                            (("18","NC"),("18",'MCI')),(("18","NC"),("18",'AD')),  ]

netnames = ['Visual peripheral','Visual central','Somatomotor A','Somatomotor B',
            'Dorsal attention A','Dorsal attention B','Ventral attention','Salience',
            'Limbic A','Limbic B','Control C','Control A','Control B','Auditory',
            'Default C','Default A','Default B','Hand sensorimotor'            ]
def change_group_name(x):
    if(x == 1):
        return "NC"
    elif(x == 2):
        return "MCI"
    elif(x == 3):
        return "AD"

def show_mean_group_atlas(path_par="Plos_ref/results_py/S1_mean_group_fn/{}/mean_{}_fn_fsa4.npy",dataset='mcad',if_save=True):
    from IPython.display import display
    for group in ['nc','mci','ad']:
        Mean_Group_Par = np.load(path_par.format(dataset,group))
        Mean_Group_Par = np.argmax(Mean_Group_Par,axis=2).astype(np.int16)
        l,r = fsaverage_to_fslr(array_to_gii(Mean_Group_Par[0],Mean_Group_Par[1],if_path=True,surf='fsa4'),method='nearest')
        display(plot_surface(textureL=l.agg_data().astype(np.int16),textureR=r.agg_data().astype(np.int16),surface_name='fsLR', embed_nb=True,
                    interactive=False, transparent_bg=None, surface_type='very_inflated',cmap=yeo_cmap_19,zero_to_nan=False,color_bar=False,
                    share='both',size=(400,300),zoom=1.4,screenshot=if_save,dpi=800,
                    cb__labeltextproperty={'color': (0, 0, 0), 'italic': False, 'shadow': False, 'bold': True, 'fontFamily': 'Arial', 'fontSize': 1},
                    filename="Plos_ref/results_py/S1_mean_group_fn/{}/mean_{}_py.png".format(dataset,group)))
                
def show_second_group_atlas(path_par="Plos_ref/results_py/S1_mean_group_fn/{}/mean_{}_fn_fsa4.npy",dataset='mcad',order=2,if_save=True) -> (np.ndarray, np.ndarray):
    res_dict = {}
    for group in ['nc','mci','ad']:
        Mean_Group_Par = np.load(path_par.format(dataset,group)) # 2 2562 19       
        Mean_Group_num = np.sort(Mean_Group_Par,axis=2)[:,:,-order]
        Mean_Group_Par = np.argsort(Mean_Group_Par,axis=2)[:,:,-order].astype(np.int16)

        l,r = fsaverage_to_fslr(array_to_gii(Mean_Group_Par[0],Mean_Group_Par[1],if_path=True,surf='fsa4'),method='nearest')
        a = plot_surface(textureL=l.agg_data().astype(np.int16),textureR=r.agg_data().astype(np.int16),surface_name='fsLR', embed_nb=True,
                    interactive=False, transparent_bg=None, surface_type='very_inflated',cmap=yeo_cmap_19,zero_to_nan=False,color_bar=True,
                    share='both',size=(300,300),zoom=1.4,screenshot=if_save,dpi=800,
                    cb__labeltextproperty={'color': (0, 0, 0), 'italic': False, 'shadow': False, 'bold': True, 'fontFamily': 'Arial', 'fontSize': 1},
                    filename="Plos_ref/results_py/S1_mean_group_fn/{}/mean_{}_second_fn.png".format(dataset,group))
        display(a)
        l,r = fsaverage_to_fslr(array_to_gii(Mean_Group_num[0],Mean_Group_num[1],if_path=True,surf='fsa4'))
        a = plot_surface(textureL=l.agg_data().astype(np.int16),textureR=r.agg_data(),surface_name='fsLR', embed_nb=True,
                    interactive=False, transparent_bg=None, surface_type='very_inflated',cmap='jet',zero_to_nan=False,color_bar=True,
                    share='both',size=(300,300),zoom=1.4,screenshot=if_save,dpi=800,
                    cb__labeltextproperty={'color': (0, 0, 0), 'italic': False, 'shadow': False, 'bold': True, 'fontFamily': 'Arial', 'fontSize': 1},
                    filename="Plos_ref/results_py/S1_mean_group_fn/{}/mean_{}_second_fn.png".format(dataset,group))
        display(a)
        res_dict[group] = [Mean_Group_num, Mean_Group_Par]
    return res_dict

def regress_cov(features, covariance, center=True, keep_scale=True):
    """
    :param covarance: covariance
    :param center: Boolean, wether to add intercept
    """
    if features.ndim == 1:
        features = features.reshape(len(features), 1)
    if covariance.ndim == 1:
        covariance = covariance.reshape(len(features), 1)
    residuals = np.zeros_like(features)
    result = np.zeros_like(features)
    if center is True:
        b = covariance
    else:
        b = np.hstack([covariance, np.ones([covariance.shape[0], 1])])
    for f in range(features.shape[1]):
        w = np.linalg.lstsq(b, features[:, f])[0]
        if center is True:
            residuals[:, f] = features[:, f] - covariance.dot(w)
        else:
            residuals[:, f] = features[:, f] - covariance.dot(w[:-1])
    if keep_scale is True:
        for f in range(features.shape[1]):
            if np.min(features[:, f]) == np.max(features[:, f]):
                result[:, f] = features[:, f]
            else:
                result[:, f] = MinMaxScaler(feature_range=(np.min(features[:, f]), np.max(features[:, f]))). \
                    fit_transform(residuals[:, f].reshape(-1, 1)).flatten()
    else:
        result = residuals
    return result
def regress_cov_linear(table, expression, keep_scale = True):
    feature = table['feature'].values
    res = smf.ols(expression,data=table).fit()
    results = res.resid.values
    if keep_scale is True:        
        if np.min(feature) != np.max(feature):
            results = MinMaxScaler(feature_range=(np.min(feature), np.max(feature))). \
                fit_transform(results.reshape(-1, 1)).flatten()
    
    return results
def to_categorical(y, num_classes=None):
    """Converts a class vector (integers) to binary class matrix.

    E.g. for use with categorical_crossentropy.

    # Arguments
        y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: total number of classes.

    # Returns
        A binary matrix representation of the input.
    """
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes))
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical
def combat(data,infoTable):

    from neuroCombat import neuroCombat
    covars = pd.DataFrame()
    covars['gender'] = infoTable.Gender
    covars['age'] = infoTable.Age
    covars['batch'] = infoTable.center
    covars['group'] = infoTable.Group
    data_for_combat = data.T
    print('Nan:',np.sum(np.isnan(data_for_combat)))
    print('Nan:',np.sum(np.isinf(data_for_combat)))
    data_combat = neuroCombat(dat=data_for_combat,
        covars=covars,
        batch_col=['batch'],continuous_cols=['age'],
        categorical_cols=['gender','group'])["data"]
    data_combat = data_combat.T
    return data_combat
def pre_feature(data,infoTable):
    '''
    infoTable : Gender, Age, center, Group
    Combat + Regression
    '''
    #data_combat = combat(data,infoTable)
    data_combat = data
    if('tSNR' in infoTable.columns):
        covs = infoTable.loc[:,['Age','Gender','tSNR','center']]#.values
    else:
        covs = infoTable.loc[:,['Age','Gender','tSNR_run_1','center']]#.values
        covs['tSNR'] = covs['tSNR_run_1']
    #res = regress_cov(data_combat,covs,center=True,keep_scale=True)
    res = np.zeros_like(data_combat)
    for i in range(res.shape[1]):
        covs['feature'] = data_combat[:,i]
        res[:,i] = regress_cov_linear(table=covs,expression='feature ~ Age + C(Gender) + tSNR + C(center)')
    return res
def pre_feature_linear(features,infoTable,keep_scale = True, site_effect='linear'):
    '''
    infoTable : Gender, Age, center, Group
    Regression
    '''
    ndim = features.ndim
    n_feature = 1 if ndim == 1 else features.shape[1]
    features = features.reshape([-1,1]) if ndim == 1 else features
    result = np.zeros_like(features)
    residuals = np.zeros_like(features)
    if(site_effect == 'linear'):
        for n in range(n_feature):
            if('tSNR' in infoTable.columns):
                covs = infoTable.loc[:,['Age','Gender','tSNR','center']].copy()
                covs.loc[:,'feature'] = features[:,n]
                residuals[:,n] = regress_cov_linear(table=covs,expression='feature ~ Age + C(Gender) + tSNR + C(center)')
            else:
                covs = infoTable.loc[:,['Age','Gender','tSNR_run_1','center']].copy()
                covs.loc[:,'feature'] = features[:,n]
                residuals[:,n] = regress_cov_linear(table=covs,expression='feature ~ Age + C(Gender) + tSNR_run_1 + C(center)')    
        
        
    elif(site_effect == 'combat'):
        data_combat = combat(features,infoTable)
        for n in range(n_feature):
            if('tSNR' in infoTable.columns):
                covs = infoTable.loc[:,['Age','Gender','tSNR','center']].copy()
                covs.loc[:,'feature'] = data_combat[:,n]
                residuals[:,n] = regress_cov_linear(table=covs,expression='feature ~ Age + C(Gender) + tSNR')
            else:
                covs = infoTable.loc[:,['Age','Gender','tSNR_run_1','center']].copy()
                covs.loc[:,'feature'] = data_combat[:,n]
                residuals[:,n] = regress_cov_linear(table=covs,expression='feature ~ Age + C(Gender) + tSNR_run_1') 
    else:
        raise NameError
    
    if keep_scale is True:
            for f in range(features.shape[1]):
                if np.min(features[:, f]) == np.max(features[:, f]):
                    result[:, f] = features[:, f]
                else:
                    result[:, f] = MinMaxScaler(feature_range=(np.min(features[:, f]), np.max(features[:, f]))). \
                        fit_transform(residuals[:, f].reshape(-1, 1)).flatten()
    else:
        result = residuals
    result = result.flatten() if ndim == 1 else result
    return result

def screen_site_number(infotable):
    site_names = np.unique(infotable.center.values)
    site_numbers = []
    site_numbers_name = []
    for s in site_names:
        site_numbers.append(np.sum(infotable.center.values == s))
        site_numbers_name.append(s)
    site_numbers = np.array(site_numbers)
    site_numbers_name = np.array(site_numbers_name)
    return site_numbers,site_numbers_name

def screen_site(infotable,data, threshold = 3):
    site_number, site_name = screen_site_number(infotable)
    screen_site = site_name[site_number >=threshold ]
    data = data[infotable.center.isin(screen_site)]
    infotable = infotable.loc[infotable.center.isin(screen_site)].copy()
    return infotable, data
def show_surf(surf_data,surf = 'fsa4', target = 'fslr', astype=np.float32,cmap='jet',color_range=None):
    if(surf == 'fsa4'):
        vertex_num = 2562
    elif(surf == 'fsa'):
        vertex_num = 10242
    arrays = []
    if(np.ndim(surf_data) == 1):
        surf_data = [surf_data]
    for i in range(len(surf_data)):
        sub_1 = surf_data[i]
        l,r = fsaverage_to_fslr(array_to_gii(sub_1[0:vertex_num],sub_1[vertex_num:],if_path=True,surf=surf))
        arrays.append(np.concatenate([l.agg_data().squeeze().astype(astype),r.agg_data().squeeze().astype(astype)]))

    return plot_cortical_v(array_name=arrays, surface_name='conte69', color_bar=True, color_range=color_range, label_text=['Map %d' % i for i in range(len(arrays))],
                    cmap=cmap, nan_color=(1, 1, 1, 0), zoom=1.2,
                    background=(1, 1, 1), size=(800, int(200*len(arrays))), interactive=False,
                    embed_nb=True, screenshot=False, filename=None,
                    scale=(1, 1), transparent_bg=False)
def get_yeo():
    '''
    return Yeo 18 in fsa4 in [5124] 0,2,3,...,18
    '''
    networks_lh = np.zeros([2562])
    networks_rh = np.zeros([2562])
    for i in range(1,20):
        network_1_lh = nib.load("Plos_ref/HFR_ai/Templates/Parcellation_template/lh_network_%d_asym_fs4.mgh" % i).get_fdata().squeeze()
        network_1_rh = nib.load("Plos_ref/HFR_ai/Templates/Parcellation_template/rh_network_%d_asym_fs4.mgh" % i).get_fdata().squeeze()
        networks_lh[network_1_lh > 0] = i - 1
        networks_rh[network_1_rh > 0] = i - 1 
    networks = np.concatenate([networks_lh,networks_rh])
    return networks

from sklearn.svm import SVC,LinearSVC
from sklearn.model_selection import GridSearchCV, train_test_split,KFold
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc,accuracy_score,classification_report
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
def cog_ref(connectome,cog,fold_num = 5):
    accs = []
    kfold = KFold(fold_num,shuffle=True,random_state=None)
    for train_index,test_index in kfold.split(connectome):
        x_train,x_test,=connectome[train_index],connectome[test_index]
        y_train,y_test=cog[train_index],cog[test_index]
        corr, Y_predict, Y_   = svr_regression(x_train, y_train, x_test, y_test)
        accs.append(corr)
    return accs

def class_ref(connectome,cog,fold_num = 5):
    accs = []
    kfold = KFold(fold_num,shuffle=True,random_state=None)
    for train_index,test_index in kfold.split(connectome):
        x_train,x_test,=connectome[train_index],connectome[test_index]
        y_train,y_test=cog[train_index],cog[test_index]
        corr, Y_predict, Y_   = svr_regression(x_train, y_train, x_test, y_test)
        accs.append(corr)
    return accs

def class_ref(connectome,cog):
    accs = []
    kfold = KFold(5,shuffle=True,random_state=None)
    for train_index,test_index in kfold.split(connectome):
        x_train,x_test,=connectome[train_index],connectome[test_index]
        y_train,y_test=cog[train_index],cog[test_index]
        fpr,tpr,auc,acc, feature_importance   = logistic_regression(x_train, y_train, x_test, y_test)
        accs.append(acc)
    return accs

def svr_classification(X, Y, X_, Y_,kernel = 'rbf'):  
    from sklearn.model_selection import GridSearchCV
    from sklearn.svm import SVC
    from scipy.stats import spearmanr
    from sklearn.metrics import mean_squared_error
    model = SVC(kernel=kernel)
    if(kernel == 'rbf'):
        param_grid = {'C': [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000], 'gamma': [0.1, 0.01, 0.001, 0.0001]} 
    else:
        param_grid = {'C': [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000]} 
    grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs = -1)  
    grid_search.fit(X, Y)
    best_parameters = grid_search.best_estimator_.get_params()  

    model = SVC(kernel=kernel, C=best_parameters['C'], gamma=best_parameters['gamma']) 
    model.fit(X, Y)
    
    Y_predict = model.predict(X_)
    acc = accuracy_score(Y_,Y_predict)
    if(kernel == 'linear'):
        feature_idx = model.coef_.squeeze()
    else:
        feature_idx = np.ones(X.shape[1])

    return acc, Y_predict, Y_  
def elastic_net_classify(X, Y, X_, Y_,**kwgs):
    from sklearn.linear_model import LogisticRegression 
    from sklearn.model_selection import GridSearchCV
    
    model = LogisticRegression(penalty='elasticnet', solver='saga', max_iter=10000, class_weight='balanced',n_jobs = -1)
    param_grid = {'C': [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000], 'l1_ratio': [0, 0.2, 0.4, 0.4, 0.6,  0.8, 1.0]}
    grid_search = GridSearchCV(model, param_grid, n_jobs = -1, cv=5)  
    grid_search.fit(X, Y)  
    best_parameters = grid_search.best_estimator_.get_params() 
    print("best paramters:")
    print(best_parameters['C'])
    print(best_parameters['l1_ratio'])
    model = LogisticRegression(class_weight='balanced', penalty='elasticnet', solver='saga', max_iter=10000, C=best_parameters['C'], l1_ratio=best_parameters['l1_ratio'], n_jobs = -1)
    model.fit(X, Y)

    y_hat = model.predict(X_)
    y_score = model.decision_function(X_)
    fpr,tpr,roc_auc,acc = cal_score(y_hat,y_score,Y_)
    feature_wight = model.coef_.flatten()

    return fpr,tpr,roc_auc,acc,feature_wight
def svr_regression(X, Y, X_, Y_,kernel = 'rbf'):  
    from sklearn.model_selection import GridSearchCV
    from sklearn.svm import SVR
    from scipy.stats import spearmanr
    from sklearn.metrics import mean_squared_error
    model = SVR(kernel=kernel)
    if(kernel == 'rbf'):
        param_grid = {'C': [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000], 'gamma': [0.1, 0.01, 0.001, 0.0001]} 
    else:
        param_grid = {'C': [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000]} 
    grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs = -1)  
    grid_search.fit(X, Y)
    best_parameters = grid_search.best_estimator_.get_params()  

    model = SVR(kernel=kernel, C=best_parameters['C'], gamma=best_parameters['gamma']) 
    model.fit(X, Y)
    
    Y_predict = model.predict(X_)
    corr = spearmanr(Y_predict,Y_)[0]
    mse = mean_squared_error(Y_predict,Y_)
    if(kernel == 'linear'):
        feature_idx = model.coef_.squeeze()
    else:
        feature_idx = np.ones(X.shape[1])

    return corr, Y_predict, Y_  

def logistic_regression(x_train, y_train, x_test, y_test):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV
    model = LogisticRegression(penalty='l1', solver='saga', class_weight='balanced',max_iter=1000) # LogisticRegression(penalty='l2') C
    param_grid = {'C': [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000, 10000]}
    grid_search = GridSearchCV(model, param_grid, n_jobs = -1)  
    grid_search.fit(x_train, y_train)  
    best_parameters = grid_search.best_estimator_.get_params() 

    model = LogisticRegression(penalty='l1', solver='saga', C=best_parameters['C'], class_weight='balanced',max_iter=10000)
    model.fit(x_train, y_train)
    y_hat = model.predict(x_test)
    y_score = model.decision_function(x_test)
    acc = accuracy_score(y_test, y_hat)
    fpr,tpr,auc = roc_curve(y_test, y_score)
    feature_importance = model.coef_.squeeze()
    return fpr,tpr,auc,acc, feature_importance

def ridge_regression(X,Y,X_,Y_,**kwgs):
    '''
    return corr, Y_predict, Y_, coef_
    '''
    from sklearn.linear_model import RidgeCV
    from sklearn.linear_model import Ridge
    from scipy.stats import pearsonr,spearmanr
    Lambdas=np.logspace(-5,2,10)
    ridge_cv=RidgeCV(alphas=Lambdas,cv=5)
    ridge_cv.fit(X,Y)
    ridge=Ridge(alpha=ridge_cv.alpha_,max_iter=10000)
    ridge.fit(X, Y) 
    Y_predict = ridge.predict(X_)
    coef_ = ridge.coef_
    corr = spearmanr(Y_predict,Y_)[0]
    return corr, Y_predict, Y_, coef_
def dice_kernel(X,Y):
    #return 2* X @ Y.T / (np.sum(X) + np.sum(Y))
    print(X.shape)
    print(Y.shape)
    from scipy.spatial.distance import cdist
    dice_matrix = cdist(X, Y, metric='dice')
    print(dice_matrix.shape)
    return dice_matrix

def dice_kernel_krr(X,Y):
    return 2*  X @ Y.T / (np.sum(X) + np.sum(Y))

def my_kernel_ridge_regression(X,Y,X_,Y_,**kwgs):
    from sklearn.kernel_ridge import KernelRidge
    import numpy as np
    from scipy.stats import pearsonr,spearmanr
    rng = np.random.RandomState(0)
    krr = KernelRidge(alpha=1.0,kernel=dice_kernel_krr)
    krr.fit(X, Y)
    Y_predict = krr.predict(X_)
    coef_ = 1
    corr = spearmanr(Y_predict,Y_)[0]
    return corr, Y_predict, Y_, coef_


def svm_dice_kernel(x_train,y_train,x_test,y_test,kernel=dice_kernel):
    y_test = y_test.astype(np.int8)
    y_train = y_train.astype(np.int8)
    svc = SVC(kernel=kernel)
    c_range = np.logspace(-5, 5, 10, base=2)
    param_grid = [{'C': c_range}]
    grid = GridSearchCV(svc, param_grid, cv=5, n_jobs=-1)
    y_score = grid.fit(x_train, y_train).decision_function(x_test)
    y_hat = grid.predict(x_test)
    acc = accuracy_score(y_test, y_hat)
    fpr,tpr,threshold = roc_curve(y_test, y_score) ###计算真正率和假正率
    roc_auc = auc(fpr,tpr) ###计算auc的值
    return fpr,tpr,roc_auc,acc

# predict by topo


def prediction_topo(infoTable,ind_net_data,feature,specific_net=None,iteration_num = 100):
    '''
    使用地形数据进行回归
    '''
    networks = get_yeo()
    if(specific_net):
        ind_net_data = ind_net_data[:,networks == specific_net]
    print(ind_net_data.shape)
    train_data = ind_net_data
    if(np.sum(np.isnan(train_data))>0):
        print(infoTable.iloc[np.isnan(train_data.sum(1))])

    train_label = infoTable[feature].values
    res_list = []
    for i in tqdm.tqdm(range(iteration_num)):
        res = reg_cog_by_topo(train_data,train_label)
        res_list.append(res)
        
    return res_list
def regression_topo(infoTable,dataset,indi_label,feature,result='results_py',sub_col='sub',model=None,networks = get_yeo() - 1):
    '''
    使用地形数据进行回归
    '''
    from S5_prediction_function import ML,svmTrainTest,logistic_regression,ridge_classify
    model_name = model.__name__
    netname = indi_label if indi_label < 19 else 'all'
    if(feature == 'corr'):
        get_data = get_ind_par_corr
    elif(feature == 'confi'):
        get_data = get_ind_par_confi
    elif(feature == 'par'):
        get_data = get_ind_par_py
    if(indi_label == 19):
        indi_label = np.ones([19]).astype(np.bool_)
        indi_label[0] = False
    ind_net_data = []
    for index,row in infoTable.iterrows():
        sub = row[sub_col]
        sub_confi = get_data(sub,dataset)
        sub_confi = sub_confi[indi_label]
        if(sub_confi.ndim==1):
            sub_confi = sub_confi[np.newaxis,:]
        sub_confi = sub_confi[:,networks > 0].flatten()
        ind_net_data.append(sub_confi)
    ind_net_data = np.array(ind_net_data)
    print(ind_net_data.shape)
    train_data = ind_net_data
    if(np.sum(np.isnan(train_data))>0):
        print(infoTable.iloc[np.isnan(train_data.sum(1))])
    train_label = infoTable.Group.values
    train_label[train_label == 'NC'] = 0
    train_label[train_label == 'AD'] = 1
    train_label = train_label.astype(np.int16)
    accs = []
    for i in tqdm.tqdm(range(5)):
        classify_ml = ML(train_data,train_label,model)
        accs.append(classify_ml.kfoldML())
    print(accs)
    np.save('Plos_ref/{}/S3_prediction/{}/{}_Regression_AD_Net-{}_Feature-{}.npy'.format(result,dataset,model_name,netname, feature), accs)

def classify_topo(infoTable,dataset,indi_label,feature,result='results_py',sub_col='sub',model=None,networks = get_yeo() - 1,iteration=100):
    '''
    使用地形数据进行分类
    '''
    from S5_prediction_function import ML,svmTrainTest,logistic_regression,ridge_classify
    model_name = model.__name__
    netname = indi_label if indi_label < 19 else 'all'
    if(feature == 'corr'):
        get_data = get_ind_par_corr
    elif(feature == 'confi'):
        get_data = get_ind_par_confi
    elif(feature == 'par'):
        get_data = get_ind_par_py
    if(indi_label == 19):
        indi_label = np.ones([19]).astype(np.bool_)
        indi_label[0] = False
    ind_net_data = []
    for index,row in infoTable.iterrows():
        sub = row[sub_col]
        sub_confi = get_data(sub,dataset)
        sub_confi = sub_confi[indi_label]
        if(sub_confi.ndim==1):
            sub_confi = sub_confi[np.newaxis,:]
        sub_confi = sub_confi[:,networks > 0].flatten()
        ind_net_data.append(sub_confi)
    ind_net_data = np.array(ind_net_data)
    print(ind_net_data.shape)
    train_data = ind_net_data
    if(np.sum(np.isnan(train_data))>0):
        print(infoTable.iloc[np.isnan(train_data.sum(1))])
    train_label = infoTable.Group.values
    train_label[train_label == 'NC'] = 0
    train_label[train_label == 'AD'] = 1
    train_label = train_label.astype(np.int16)
    accs = []
    for i in tqdm.tqdm(range(iteration)):
        classify_ml = ML(train_data,train_label,model)
        accs.append(classify_ml.kfoldML())
    print(accs)
    np.save('Plos_ref/{}/S3_prediction/{}/{}_Classify_AD_AUC_all_vertex_Net-{}_Feature-{}.npy'.format(result,dataset,model_name,netname, feature), accs)

def reg_cog_by_topo(topo,cog,fold_num = 2):
    accs = []
    coefs =[]
    kfold = KFold(fold_num,shuffle=True,random_state=None)
    for train_index,test_index in kfold.split(topo):
        x_train,x_test,=topo[train_index],topo[test_index]
        y_train,y_test=cog[train_index],cog[test_index]
        corr, Y_predict, Y_, coef_   = my_kernel_ridge_regression(x_train, y_train, x_test, y_test)
        accs.append(corr)
        coefs.append(coef_)
    return accs, coefs


def get_group_ind(dataset='mcad'):
    nc_mean = np.load("Plos_ref/results_py/S1_mean_group_fn/{}/mean_nc_fn_fsa4.npy".format(dataset))
    nc_mean = np.argmax(nc_mean,axis=2).astype(np.int16)
    nc_mean = np.concatenate([nc_mean[0],nc_mean[1]])
    return nc_mean


ven_l = nib.load("Plos_ref/HFR_ai/Templates/Parcellation_template/lh_network_1_asym_fs4.mgh").get_fdata().squeeze()
ven_r = nib.load("Plos_ref/HFR_ai/Templates/Parcellation_template/rh_network_1_asym_fs4.mgh").get_fdata().squeeze()
def get_ind_par_py(sub,dataset='mcad', iteration = 9, base_par = "Plos_ref/IndiPar/{}/Py_fs4_from_fs5/"):
    '''
    return 19 * 5124
    '''
    base_dir = base_par.format(dataset)
    network_l = np.load(base_dir + '/' + sub + '/Network_Par_L_18.npz')['data'][iteration]
    network_r = np.load(base_dir + '/' + sub + '/Network_Par_R_18.npz')['data'][iteration]
    network_l = np.concatenate([ven_l.reshape([1,-1]),network_l])
    network_r = np.concatenate([ven_r.reshape([1,-1]),network_r])
    networks = np.concatenate([network_l,network_r],axis=1)
    return networks
def get_ind_par_confi(sub,dataset='mcad', iteration = 9, base_par = "Plos_ref/IndiPar/{}/Py_fs4_from_fs5/"):
    base_dir = base_par.format(dataset)
    confi_l = np.load(base_dir + '/' + sub + "/Network_Confi_L_18.npz")['data'][iteration]
    confi_r = np.load(base_dir + '/' + sub + "/Network_Confi_R_18.npz")['data'][iteration]
    confi_l = np.concatenate([ven_l.reshape([1,-1]),confi_l])
    confi_r = np.concatenate([ven_r.reshape([1,-1]),confi_r])
    confi =  np.concatenate([confi_l,confi_r],axis=1)
    return confi
def get_ind_par_corr(sub,dataset='mcad', base_par = "Plos_ref/IndiPar/{}/Py_fs4_from_fs5/"):
    base_dir = base_par.format(dataset)
    confi_l = np.load(base_dir + '/' + sub + "/Network_Corr_L_18.npz")['data']
    confi_r = np.load(base_dir + '/' + sub + "/Network_Corr_R_18.npz")['data']
    confi_l = np.concatenate([ven_l.reshape([1,-1]),confi_l])
    confi_r = np.concatenate([ven_r.reshape([1,-1]),confi_r])
    confi =  np.concatenate([confi_l,confi_r],axis=1)
    return confi

def get_ind_par_by_subs(subs,dataset,iteration=9 ,base_par = "Plos_ref/IndiPar/{}/Py_fs4_from_fs5/",high_confi = 0) -> tuple:
    '''
    return 3 809 * 19 * 5124
    '''
    data = [[] for i in range(3)]
    for sub in subs:
        indipar = get_ind_par_py(sub,dataset,iteration, base_par)
        indiconfi = get_ind_par_confi(sub,dataset,iteration, base_par)
        indicorr = get_ind_par_corr(sub,dataset, base_par)
        if(high_confi):
            indipar[indiconfi < high_confi] = 0
            indicorr[indiconfi < high_confi] = 0
        data[0].append(indipar)
        data[1].append(indiconfi)
        data[2].append(indicorr)
    data[0] = np.array(data[0]); data[1] = np.array(data[1]); data[2] = np.array(data[2]); 
    return tuple(data)

def get_net_weight(indi_corr):
    indi_weight = indi_corr.copy()
    indi_weight[indi_weight < 0] = 0 
    indi_weight = (indi_weight - indi_weight.min(axis=0)) / (indi_weight.max(axis=0) - indi_weight.min(axis=0))
    return indi_weight

def get_ind_weight_by_corr(sub,dataset='mcad'):
    data = get_ind_par_corr(sub,dataset)
    data = get_net_weight(data)
    return data

def cal_overlay_between_subj_and_average(subj_par,group_par):
    '''
    subj_par : 18 * 5124
    group_par : 18 * 5124
    '''
    dice = 2* subj_par.flatten() @ group_par.flatten() / (np.sum(subj_par) + np.sum(group_par))
    return dice

def changeName(x):
    if(x == "CN"):
        return 1
    elif(x == "MCI"):
        return 2
    elif(x == "Dementia"):
        return 3
def changeName_oasis(x):
    if(x == 'Cognitively normal'):
        return 1
    elif(x == 'AD Dementia'):
        return 3

class dataset_infotable:
    def __init__(self, fd_threshold = None, tSNR_threshold = None ,move_threshold = None,rotate_threshold=None ) -> None:

        self.fd_threshold = 0.37 if fd_threshold is None else fd_threshold
        self.tSNR_threshold= 61.23 if tSNR_threshold is None else tSNR_threshold
        self.move_threshold= 3 if move_threshold is None else move_threshold
        self.rotate_threshold=3 if rotate_threshold is None else rotate_threshold

    def get_screen(self):
        return self.screen

        
class mcad_infotable(dataset_infotable):
    def __init__(self, fd_threshold = None, tSNR_threshold = None ,move_threshold = None, rotate_threshold = None, group_name = 'string') -> None:
        super().__init__(fd_threshold, tSNR_threshold, move_threshold, rotate_threshold)
        infotable = pd.read_csv("infoTable/MCAD_info_809_fmriprep_withTIV_ts_length_new.csv")
        screen =((infotable.tSNR > self.tSNR_threshold) &
                 (infotable['mean-FD'] < self.fd_threshold) &
                 (infotable['max-move'] < self.move_threshold) & 
                 (infotable['max-rotate'] < self.rotate_threshold))
        
        infotable = infotable.loc[screen]

        infotable = infotable.loc[~infotable['sub'].isin(['sub-0151','sub-0185', 'sub-0201', 'sub-0229', 
                                                          'sub-0344', 'sub-0372', 'sub-0504', 'sub-0810', 
                                                          'sub-0857', 'sub-0860'])]
        

        self.screen = screen
        self.infotable = infotable.reset_index(drop=True)
        print("{} 名被试通过了QC，其中NC: {} MCI: {}  AD: {}".format(infotable.shape[0],
                                                            np.sum(infotable.Group == 1),
                                                            np.sum(infotable.Group == 2),
                                                            np.sum(infotable.Group == 3)))
        if(group_name == 'string'):
            self.infotable.loc[:,'Group'] = self.infotable['Group'].apply(change_group_name)

    def get_baseline(self):
        return self.infotable

class adni_infotable(dataset_infotable):
    def __init__(self, fd_threshold = None, tSNR_threshold = None ,move_threshold = None, rotate_threshold = None, group_name = 'string') -> None:
        super().__init__(fd_threshold, tSNR_threshold, move_threshold, rotate_threshold)
        infotable = pd.read_csv("infoTable/ADNI_Info_Screen_by_post.csv")
        infotable.loc[:,'Group'] = infotable['DX'].apply(changeName)
        

        infotable.loc[pd.isna(infotable.AGE),'AGE'] = 0
        infotable.loc[pd.isna(infotable.Years_bl),'Years_bl'] = 0

        infotable.loc[:,'Age'] = infotable['AGE'] + infotable['Years_bl']
        infotable.loc[:,'Gender'] = infotable['PTGENDER'].apply(lambda x : 1 if x == 'Male' else 2)
        infotable.loc[:,'Gender'] = infotable.loc[:,'Gender'].astype('float')
        infotable.loc[:,'center'] = infotable['SITE']
        self.o_infotable = infotable.copy()

        screen =((infotable.tSNR > self.tSNR_threshold) &
                 (infotable['FD'] < self.fd_threshold) &
                 (infotable['max-move'] < self.move_threshold) & 
                 (infotable['max-rotate'] < self.rotate_threshold))

        infotable = infotable.loc[screen]
        infotable = infotable.loc[ ~infotable['sub'].isin(['sub-4835m12','sub-4744m72','sub-0473m138',
                                                        'sub-6142m24',
                                                        'sub-6467sc',
                                                        'sub-6591sc',
                                                        'sub-6754m12'])]

        self.infotable = infotable.reset_index(drop=True)

        print("{} 个记录通过了QC，其中NC: {} MCI: {}  AD: {}".format(infotable.shape[0],
                                                            np.sum(infotable.Group ==1),
                                                            np.sum(infotable.Group ==2),
                                                            np.sum(infotable.Group ==3)))
        print("{} 个被试通过了QC，其中NC: {} MCI: {}  AD: {}".format(np.unique(infotable['PTID']).shape[0],
                                                            np.unique(infotable.loc[infotable.Group ==1]['PTID']).shape[0],
                                                            np.unique(infotable.loc[infotable.Group ==2]['PTID']).shape[0],
                                                            np.unique(infotable.loc[infotable.Group ==3]['PTID']).shape[0]) )
        
        if(group_name == 'string'):
            self.infotable.loc[:,'Group'] = self.infotable['Group'].apply(change_group_name)
            self.group_labels = ['NC','MCI','AD']
        else:
            self.group_labels = [1,2,3]

    def get_baseline(self,group_labels = ['NC','MCI','AD']):
        '''
        To maxium the number of AD, any subject have AD visit will be classfied to AD and delete the previous visit follow-up.
        The first visit of AD DX will be treated as the Baseline.
        '''
        bl_infotable = pd.DataFrame(columns=self.infotable.columns)
        infotable = self.infotable.loc[self.infotable.Group.isin(group_labels)]
        AD_PTID = infotable.loc[infotable.Group == self.group_labels[2],'PTID'].values.tolist()

        infotable = infotable.loc[(infotable.PTID.isin(AD_PTID) & (infotable.Group == self.group_labels[2])) | ~infotable.PTID.isin(AD_PTID)]
        for sub in np.unique(infotable.PTID.values):
            res = infotable.loc[infotable.PTID == sub]
            if(res.shape[0] > 1):
                bl_infotable = bl_infotable.append(res.sort_values('Years_bl').iloc[0])
            elif(res.shape[0] == 1):
                bl_infotable = bl_infotable.append(res)
        bl_infotable.reset_index(inplace=True, drop = True)
        return bl_infotable
    

    def get_two_session(self):
        bl_infotable = self.get_baseline()
        vs1_infotable = pd.DataFrame(columns = bl_infotable.columns)
        vs2_infotable = pd.DataFrame(columns = bl_infotable.columns)

        subjects = np.unique(bl_infotable.PTID.values)
        for sub in subjects:
            rows = self.infotable.loc[self.infotable.PTID == sub]
            if(rows.shape[0] > 1):
                vs1_infotable = vs1_infotable.append(rows.sort_values('Month_bl').iloc[0])
                vs2_infotable = vs2_infotable.append(rows.sort_values('Month_bl').iloc[1])

        vs1_infotable.reset_index(inplace=True, drop =True)
        vs2_infotable.reset_index(inplace=True, drop =True)

        return vs1_infotable, vs2_infotable

class oasis_infotable(dataset_infotable):
    def __init__(self, fd_threshold = None, tSNR_threshold = None ,move_threshold = None, rotate_threshold = None, group_name = 'string') -> None:
        super().__init__(fd_threshold, tSNR_threshold , move_threshold, rotate_threshold)
        infotable = pd.read_csv("infoTable/OASIS_table_all_fmri_screen_by_post.csv")
        infotable.reset_index(inplace=True,drop = True)
        n_info = pd.DataFrame(columns=infotable.columns)
        for index,row in infotable.iterrows():
            p = 0
            for rp in range(row['run_number']):
                if((row['tSNR_run_%d' % (rp + 1)] > self.tSNR_threshold) & 
                   (row['mean-FD_run_%d' % (rp + 1)] < self.fd_threshold) &
                   (row['max-move_run_%d' % (rp + 1)] < self.move_threshold) & 
                   (row['max-rotate_run_%d' % (rp + 1)] < self.rotate_threshold)):
                    
                    row['run_%d' % (p+1)] = row['run_%d' % (rp + 1)]
                    row['tSNR_run_%d' % (p + 1)] = row['tSNR_run_%d' % (rp + 1)]
                    row['mean-FD_run_%d' % (p + 1)] = row['mean-FD_run_%d' % (rp + 1)]
                    row['max-move_run_%d' % (p + 1)] = row['max-move_run_%d' % (rp + 1)]
                    row['max-rotate_run_%d' % (p + 1)] = row['max-rotate_run_%d' % (rp + 1)]
                    p += 1
                row['run_number'] = p
            n_info = n_info.append(row)

        n_info = n_info.loc[n_info.run_number > 0]
        

        n_info.loc[:,'Group'] = n_info['dx1'].apply(changeName_oasis)
        

        n_info.loc[:,'Gender'] = n_info['Gender'].apply(lambda x : 1 if x == 'male' else 2)
        n_info['Gender'] = n_info['Gender'].astype('float')
        n_info.loc[:,'center'] = n_info['Scaner']
        n_info['Age'] = n_info['ageAtEntry'] + n_info['Days'] / 365
        n_info['Age'] = pd.to_numeric(n_info['Age'])
        n_info.mmse = pd.to_numeric(n_info.mmse)
        n_info['sub1'] = n_info['sub'] + "_" + n_info['run_1']
        n_info['sub2'] = n_info['sub'] + "_" + n_info['run_2']
        self.o_infotable = n_info.copy()
        
        n_info = n_info.loc[~n_info['sub'].isin(['sub-OAS30391d1547', 'sub-OAS30443d1480', 'sub-OAS30805d7028','sub-OAS30955d0133'])]
        self.infotable = n_info
        self.infotable.reset_index(inplace=True,drop=True)

        print("{} 个记录通过了QC，其中NC: {} MCI: {}  AD: {}".format(n_info.shape[0],
                                                            np.sum(n_info.Group == 1),
                                                            np.sum(n_info.Group == 2),
                                                            np.sum(n_info.Group == 3)))
        
        print("{} 个被试通过了QC，其中NC: {} MCI: {}  AD: {}".format(np.unique(n_info['Subject']).shape[0],
                                                            np.unique(n_info.loc[n_info.Group == 1]['Subject']).shape[0],
                                                            np.unique(n_info.loc[n_info.Group == 2]['Subject']).shape[0],
                                                            np.unique(n_info.loc[n_info.Group == 3]['Subject']).shape[0]) )
        
        if(group_name == 'string'):
            self.infotable.loc[:,'Group'] = self.infotable['Group'].apply(change_group_name)

    def get_baseline(self):
        '''
        return baseline
        '''
        bl_infotable = pd.DataFrame(columns=self.infotable.columns)
        for subject in pd.unique(self.infotable.Subject):
            screen = self.infotable.Subject == subject
            bl_row = self.infotable.loc[screen].sort_values('Days').iloc[0]
            bl_infotable = bl_infotable.append(bl_row)

        bl_infotable.reset_index(inplace=True, drop = True)
        return bl_infotable

    def get_two_run(self):
        infotable = self.get_baseline()
        two_run = infotable.loc[infotable.run_number > 1]
        two_run.reset_index(inplace=True,drop=True)
        return two_run.copy()

    def get_two_session(self):

        bl_infotable = self.get_baseline()
        vs1_infotable = pd.DataFrame(columns = bl_infotable.columns)
        vs2_infotable = pd.DataFrame(columns = bl_infotable.columns)
        subjects = np.unique(bl_infotable.Subject.values)

        for sub in subjects:
            rows = self.infotable.loc[self.infotable.Subject == sub]
            if(rows.shape[0] > 1):
                vs1_infotable = vs1_infotable.append(rows.sort_values('Days').iloc[0])
                vs2_infotable = vs2_infotable.append(rows.sort_values('Days').iloc[1])
            

        vs1_infotable.reset_index(inplace=True, drop =True)
        vs2_infotable.reset_index(inplace=True, drop =True)

        return vs1_infotable, vs2_infotable
    

# FCN 

def cog_predict_by_svr(yeo_connectome,ind_connectome,infotable,target, iteration=50, fold_num=5) -> pd.DataFrame:
    '''
    predict cognitive score by using functional connectivity based on SVR
    '''
    tinfotable = infotable.loc[pd.notna(infotable[target])]

    idx = np.zeros([18,18])
    idx[np.triu_indices(idx.shape[0],1)] = 1
    yeo_connectome = yeo_connectome[:,idx > 0]
    ind_connectome = ind_connectome[:,idx > 0]

    yeo_connectome = yeo_connectome[pd.notna(infotable[target])]
    ind_connectome = ind_connectome[pd.notna(infotable[target])]
    print("样本数：",str(tinfotable.shape[0]))

    Accs_group = []
    for i in tqdm.tqdm(range(iteration)):
        acc_group = cog_ref(yeo_connectome,tinfotable[target].values, fold_num)
        Accs_group.extend(acc_group)
    Accs_ind = []
    for i in tqdm.tqdm(range(iteration)):
        acc_ind = cog_ref(ind_connectome,tinfotable[target].values, fold_num) 
        Accs_ind.extend(acc_ind)

    df = pd.DataFrame()
    df['Corr_group'] = Accs_group
    df['Corr_ind'] = Accs_ind
    df = pd.melt(df,value_name='Corr',var_name='Atlas')

    fig,ax = plt.subplots(1,1,figsize=(2,2))

    sns.stripplot(data=df,x='Atlas',y='Corr',jitter=0.15,marker='o',ax=ax, linewidth=1,alpha=0.6,zorder=0)
    sns.boxplot(data=df,x='Atlas',y='Corr',width=0.4,ax=ax,zorder=1,linewidth=1)
    for patch in ax.patches:
        r, g, b, a = patch.get_facecolor()
        patch.set_facecolor((r, g, b, .3))
    
    box_pairs = [['Corr_group','Corr_ind']]
    annotator = Annotator(ax, box_pairs, data=df, x='Atlas', y='Corr', order=['Corr_group','Corr_ind'])
    annotator.configure(test='t-test_ind', text_format='simple', loc='inside',show_test_name=False,text_offset=5)
    annotator.apply_and_annotate()
    return df


def cog_predict_by_svm(yeo_connectome,ind_connectome,infotable,target, iteration=50, fold_num=5) -> pd.DataFrame:
    '''
    predict cognitive score by using functional connectivity based on SVM
    '''
    tinfotable = infotable.loc[pd.notna(infotable[target])]

    idx = np.zeros([18,18])
    idx[np.triu_indices(idx.shape[0],1)] = 1
    yeo_connectome = yeo_connectome[:,idx > 0]
    ind_connectome = ind_connectome[:,idx > 0]

    yeo_connectome = yeo_connectome[pd.notna(infotable[target])]
    ind_connectome = ind_connectome[pd.notna(infotable[target])]
    print("样本数：",str(tinfotable.shape[0]))

    Accs_group = []
    for i in tqdm.tqdm(range(iteration)):
        acc_group = cog_ref(yeo_connectome,tinfotable[target].values, fold_num)
        Accs_group.extend(acc_group)
    Accs_ind = []
    for i in tqdm.tqdm(range(iteration)):
        acc_ind = cog_ref(ind_connectome,tinfotable[target].values, fold_num) 
        Accs_ind.extend(acc_ind)

    df = pd.DataFrame()
    df['Corr_group'] = Accs_group
    df['Corr_ind'] = Accs_ind
    df = pd.melt(df,value_name='Corr',var_name='Atlas')

    fig,ax = plt.subplots(1,1,figsize=(2,2))

    sns.stripplot(data=df,x='Atlas',y='Corr',jitter=0.15,marker='o',ax=ax, linewidth=1,alpha=0.6,zorder=0)
    sns.boxplot(data=df,x='Atlas',y='Corr',width=0.4,ax=ax,zorder=1,linewidth=1)
    for patch in ax.patches:
        r, g, b, a = patch.get_facecolor()
        patch.set_facecolor((r, g, b, .3))
    
    box_pairs = [['Corr_group','Corr_ind']]
    annotator = Annotator(ax, box_pairs, data=df, x='Atlas', y='Corr', order=['Corr_group','Corr_ind'])
    annotator.configure(test='t-test_ind', text_format='simple', loc='inside',show_test_name=False,text_offset=5)
    annotator.apply_and_annotate()
    return df