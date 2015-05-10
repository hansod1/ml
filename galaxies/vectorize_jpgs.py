import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import scipy.ndimage as ndimage
import skimage.morphology as morphology
import copy
import re
import csv
from sklearn import cross_validation
from sklearn.ensemble import RandomForestRegressor
import math
from skimage.transform import resize
import skimage
from skimage.data import imread
import sys
import time
from sklearn.decomposition import RandomizedPCA

#full color
#gray scale

#masked / not masked

#masking parameters

#resized

#images are 424x424

def xformImage(filename, doPlot=False):
    
    #config area
    
    sigma=4
    mean_multiplier=2
    
    #end config area
    
#     img_gray = misc.imread(filename, True)
    img_gray = imread(filename, as_grey=True)
    img_color = imread(filename)
#     print(img_color.shape)
#     img_gray = resize(skimage.img_as_float(img_gray), (400,400))
    
    img_color_masked = copy.deepcopy(img_color)
#     img_color_masked = resize(img_color_masked, (400,400))
#     img_color_resized = resize(img_color, (200,200))
#     img_color = misc.imread(filename, False)
    
    img_gray = ndimage.gaussian_filter(img_gray, sigma=sigma)
    
    m,n = img_gray.shape
    
#     sx = ndimage.sobel(img_gray, axis=0, mode='constant')
#     sy = ndimage.sobel(img_gray, axis=1, mode='constant')
#     sob = np.hypot(sx, sy)
    
    mask = (img_gray > img_gray.mean()*mean_multiplier).astype(np.float)
    
    labels = morphology.label(mask)
    
    center_label = labels[m/2, m/2]
    
    labels[labels != center_label] = 0
    labels[labels == center_label] = 1
    
#     img_test = copy.deepcopy(img_color)
    
#     img_test[ labels == 0, : ] = 0
#     sob [ labels == 0] = 0
    img_color_masked [labels == 0, :] = 0
#     img_test = ndimage.gaussian_filter(img_test, 3)
    if doPlot:
        f, (ax_gray, ax_color, ax_sob) = plt.subplots(ncols=3)
        ax_gray.imshow(img_color_masked, cmap=plt.cm.get_cmap('gray'))
        ax_gray.axis('off')
#         ax_sob.imshow(sob, cmap=plt.cm.get_cmap('gray'))
        ax_sob.axis('off')
        ax_color.imshow(img_color, cmap=plt.cm.get_cmap('gray'))
        ax_color.axis('off')
        plt.show()    
    
    
    return np.reshape(img_color_masked, -1)

def loadPreProcessedVectors(n_samples):
    #each of the vector files is this large
    m = 6157
    
    pattern = 'color_masked_424by424_sigma2_mm1.5_vectors*'
    
    X = []
    y = []
    
    rand_idx = np.random.permutation(m)
    
    vector_files = glob(pattern)
    
    rand_idx = rand_idx[0:int(n_samples/len(vector_files)) ]
#     print(rand_idx)
    
    #take n_samples / len(vector_files) records from each file
    for vfile in vector_files:
        print('Loaded file: '+vfile)
        vfo = np.load(vfile, mmap_mode='r')
        if X != [] and y != []:
#             print(X.shape)
#             print(vfo['X'][rand_idx, 0::].shape)
            X = np.concatenate( (X,vfo['X'][rand_idx, 0::]) )
            y = np.concatenate( (y,vfo['y'][rand_idx]) )
        else:
            X = vfo['X'][rand_idx, 0::]
            y = vfo['y'][rand_idx]
        print("sample_size: " + str(len(y)))
        vfo.close()
    
    return X,y
    

#parse the galaxyId out of the filename of the image
def getGID(filename):
    return int(re.sub('[^0-9]*', '', filename))
    
if __name__ == '__main__':
    
    ##config
    train_solutions_file ='training_solutions_rev1.csv'
    dir_name = 'train_images'
    images = glob(dir_name+"/*.jpg")
    if len(sys.argv) > 1:
        n=int(sys.argv[1])
    else:
        n=1
    pre_process_images = True
    do_train = False
    dimensionality_redux = False
    plot = True
    ##end config
    
    if pre_process_images:
#         rand_idx = np.random.permutation(len(images))
        
        #read in the entire training solutions file
        csv_obj = csv.reader(open(train_solutions_file, 'r'))
        header = csv_obj.next()
        train_solutions = dict()
        
        for row in csv_obj:
            train_solutions[int(row[0])] = row[1:]
        
        sample_size = int(len(images)*.1)
        print('Building sample number '+str(n))
     
        X = []
        y = []
        print('Building sample set of size '+str(sample_size*n - sample_size*(n-1)))
        for i in range(sample_size*(n-1), sample_size*n):
            filename = images[i]
            X.append(xformImage(filename, plot))
            y.append(train_solutions[getGID(filename)])
            
        X = np.asarray(X)
        y = np.asarray(y, dtype=np.float64)
        description = 'color_masked_424by424_sigma2_mm1.5_vectors'+str(n)
        
        np.savez_compressed(description, X=X, y=y)
    else:
        X,y = loadPreProcessedVectors(len(images))
    
    m,n = X.shape
    
    print('X has ' + str(n) + ' features')
    
    if dimensionality_redux:
        pca = RandomizedPCA(n_components=7000, copy=False)
        X = pca.fit_transform(X)
        
        m,n = X.shape
        
        print('Reduced to ' + str(n) + ' features')
        
        variance_kept = np.sum(pca.explained_variance_ratio_);
        print('Kept ' + str(variance_kept) + ' of variance')
    
    start = time.time()
    if do_train:
        X, X_test, y, y_test = cross_validation.train_test_split(X,y, test_size=0.4)
        
        print('Training Model...')
        model = RandomForestRegressor(n_estimators=10, n_jobs=1)
        
        model.fit(X, y)
        
        print('Training done, score is:')
        print(model.score(X_test, y_test))
        
        predict_test = model.predict(X_test)
        
        print('RMSE is:')
        print(math.sqrt(np.mean( (y_test - predict_test)**2 ) ) )
        
        print('Some predictions:')
        print(predict_test[1:3,0:3])
        print('Ground truth:')
        print(y_test[1:3,0:3])
        end = time.time()
        elapsed = end - start
        
        print("Training took " + str(elapsed) + " seconds!")
    
    
