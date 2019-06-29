from scipy.stats import ortho_group
import numpy as np

def encryption_train(X,y):
    # U1 is an orthogonal matrix
    U1 = ortho_group.rvs(dim=X.shape[0])

    # U2 is an invertible matrix
    if X.shape[1] > 1:
        U2 = ortho_group.rvs(dim=X.shape[1])
    else:
        U2 = np.random.rand(1,1)

    X_enc = U1.dot(X).dot(U2)
    y_enc = U1.dot(y)
    return [X_enc,y_enc,U1,U2]

def decryption_train(X,y,U1,U2):
    X_dec = U1.T.dot(X).dot(np.linalg.inv(U2))
    y_dec = U1.T.dot(y)
    return [X_dec,y_dec]

def encryption_test(X,U2):
    # U3 is an invertible matrix
    if X.shape[0] > 1:
        U3 = ortho_group.rvs(dim=X.shape[0])
    else:
        U3 = np.random.rand(1,1)
    #from IPython import embed; embed()
    X_enc = U3.dot(X).dot(np.linalg.inv(U2))
    return [X_enc,U3]

def decryption_test(y_enc,U3):
    y_dec = np.linalg.inv(U3).dot(y_enc)
    return y_dec

def estimator_OLS(X,y):
    β̂ = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
    return β̂

def predict(β̂,X):
    return X.dot(β̂)
