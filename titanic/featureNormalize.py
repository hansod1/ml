import numpy as np
def getMeanAndStdVar(X):
    mean = np.mean(X, axis = 0)
    stdev = np.std(X, axis = 0)
    return mean, stdev

def findBestLevel(price, level_medians):
    best_diff = 100
    best_level = 0
    for k,v in level_medians.iteritems():
        t_diff = abs(price - v)    
        if(t_diff < best_diff):
            best_diff = t_diff
            best_level = k
    return best_level