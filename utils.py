import numpy as np

def plt_linefit(X,coef):
    idx = [np.argmin(X[:,1]), np.argmax(X[:,1])]
    minmax = [X[idx[0],:].dot(coef) , X[idx[1],:].dot(coef)]
    return [x.item() for x in minmax]
