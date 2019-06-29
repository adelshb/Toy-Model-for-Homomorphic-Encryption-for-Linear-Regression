import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def generate_1d_dataset(m, n, factor=2.0):
    X = np.matrix(np.expand_dims(np.arange(m), 1))
    y = X + np.random.random((m, n)) * factor
    return train_test_split(X, y, test_size=0.5, random_state=42)
