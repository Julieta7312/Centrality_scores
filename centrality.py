import pandas as pd
import numpy as np



def eig_vec_cent(cov, n_components=3, largest=True):

    ''' Return "Eigenvector Centrality" from the covariance matrix
            
    Args:
        
        covar (required): pandas.core.frame.DataFrame
            Estimated covariance matrix on specific date.
        
        n_components (optional): int 
            Number of components used to estimate the centrality score
            
        largest: boolean
           If True, LARGEST n_components number of (eigen values, eigen vectors) will be used to compute centrality score.
           Else, LOWEST n_componets number of (eigen values, eigen vectors) will be used to compute centrality score.
    
    Returns:
        
        Centrality score for every asset.
        
        Centrality scores = ( EigenValue_i * ( abs(EigenVector_i) / sum(abs(EigenVector_i)) ) ) / (Σ EigenValue_i)
        
        i: ith of eigen value/vector
        - Sum of Centrality scores of every asset will sum up to 1
    
    Example:

        cov_dfs = cov_dfs.groupby('Dates').apply(eig_vec_cent)

    * np.linalg.eigh returns the same result as np.linalg.eig when CovMatrix is symmetric.

    '''
    if cov.isnull().values.any(): 
        return None 

    eig_vals, eig_vecs = np.linalg.eigh(cov.values)
    sort_permutation = eig_vals.argsort()
    eig_vals.sort()
    eig_vecs = eig_vecs[:, sort_permutation]
    
    if largest:
        eig_vals = eig_vals[-n_components:]
        eig_vecs = abs(eig_vecs[:, -n_components:])
    else:
        eig_vals = eig_vals[:n_components]
        eig_vecs = abs(eig_vecs[:, :n_components])
    
    return pd.Series(np.sum(eig_vals * eig_vecs / np.sum(eig_vecs, axis=0), axis=1) / np.sum(eig_vals), index=cov.columns)

