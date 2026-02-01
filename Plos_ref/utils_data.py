
def get_net_weight(indi_corr):
    indi_weight = indi_corr.copy()
    indi_weight[indi_weight < 0] = 0 
    indi_weight = (indi_weight - indi_weight.min(axis=0)) / (indi_weight.max(axis=0) - indi_weight.min(axis=0))
    return indi_weight
def get_net_weight_all_subj(indi_corr):
    indi_weight = indi_corr.copy()
    for i in range(indi_corr.shape[0]):
        indi_weight[i] = get_net_weight(indi_corr[i])
    return indi_weight