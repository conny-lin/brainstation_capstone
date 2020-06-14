import numpy as np
import pandas as pd

def nutcracker_reducesample(X, y, identifier_df, n_sample, randomseed):
    '''
    Take n_sample of each y group (0 or 1), and take the same sample from X, and identifier_df
    Return a dataframe with X, y, and identifier combined in one.
    ----------
    PARAMETERS
    ----------
    X, y and identifier_df has to be the same lengths.
    n_sample = the number of samples to take from each of the y group. ie. 1000000
    '''
    # transform y array for indexing
    y_ind = np.array(y.index)
    ethanol = y.values == 1
    ethanol = ethanol.transpose()
    ethanol = ethanol[0]
    # get index for each group
    y_ind_etoh = y_ind[ethanol]
    y_ind_noetoh = y_ind[~ethanol]
    # define random seed
    np.random.seed(randomseed)

    X_sample = []
    for i, ind_array in enumerate([y_ind_noetoh, y_ind_etoh]):
        # take random 1M for each group from X
        size_array = ind_array.shape[0]
        print(f'taking {n_sample} of {size_array} from ethanol={i}')
        random_ind = np.random.choice(size_array, n_sample, replace=False)
        # index random index to the index array
        ind_choice = ind_array[random_ind]
        ind_choice.shape
        # double check if this random choice will index to right group
        ind_correct = all(y.iloc[ind_choice] == i)
        print(f'all random choices found data from the correct ethanol group: {ind_correct}')
        if not ind_correct:
            assert False, 'random choice incorrect'
        # take sample
        print('taking sample from X')
        sample_x = X.iloc[ind_choice,:]
        # get from identifer
        print('taking sample from identifers')
        sample_identifer = identifier_df.iloc[ind_choice,:]
        # get from y
        print('taking sample from y')
        sample_y = y.iloc[ind_choice,:]
        # correct data type
        sample_y['etoh'].astype(int)
        # combine in one array
        print('combining X, y, identifier arrays')
        sample_df = pd.concat([sample_y, sample_identifer, sample_x], axis=1)
        # add to list
        print('putting array in list')
        X_sample.append(sample_df)

    # combine
    print('combine samples from all groups')
    df = pd.concat(X_sample, ignore_index=True)
    # validate concat correct
    assert (df.shape[0] == n_sample * 2), 'resulting dataframe does not have expected n_sample * 2 length'
    return 