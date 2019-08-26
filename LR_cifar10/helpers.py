import numpy as np
import os
#import cv2

def get_data(dataset, n_channels, n_samples = 0,greyscale = False):
    OUT_DIM = 0
    if dataset == "mnist":
        from keras.datasets import mnist
        (X, y), (X_test, y_test) = mnist.load_data()
        X = X.reshape(-1,28*28).astype(np.float32)
        y = y.astype(np.float32)
        OUT_DIM = 10
    if dataset == "cifar10":
        from keras.datasets import cifar10
        (X, y), (X_test, y_test) = cifar10.load_data()
        if (n_channels == 1):
            if (greyscale):
                X = X.mean(axis=3)
                X = X.reshape(-1,32*32).astype(np.float32)
                X_test = X_test.mean(axis=3)
                X_test = X_test.reshape(-1,32*32).astype(np.float32)
            else:
                X = X.reshape(-1,32*32*3).astype(np.float32)
                X_test = X_test.reshape(-1,32*32*3).astype(np.float32)
        else:
            X = np.moveaxis(X,3,1).astype(np.float32)
            X_test = np.moveaxis(X_test,3,1).astype(np.float32)
        y = y.squeeze().astype(np.float32)
        y_test = y_test.squeeze().astype(np.float32)
        OUT_DIM = 10
    #never mind cifar100
    if dataset == "cifar100":
        from keras.datasets import cifar100
        (X, y), (X_test, y_test) = cifar100.load_data()
        if (n_channels == 1):
            if (greyscale):
                X = X.mean(axis=3)
                X = X.reshape(-1,32*32).astype(np.float32)
                X_test = X_test.mean(axis=3)
                X_test = X_test.reshape(-1,32*32).astype(np.float32)
            else:
                X = X.reshape(-1,32*32*3).astype(np.float32)
                X_test = X_test.reshape(-1,32*32*3).astype(np.float32)
        else:
            X = np.moveaxis(X,3,1).astype(np.float32)
            X_test = np.moveaxis(X_test,3,1).astype(np.float32)
        y = y.squeeze().astype(np.float32)
        y_test = y_test.squeeze().astype(np.float32)
        OUT_DIM = 100
    #never mind cat/dog
    if dataset == "catdog":
        TRAIN_DIR = 'C:/Users/Nico/Documents/Datasets/train/'
        TEST_DIR = 'C:/Users/Nico/Documents/Datasets/test1/'
        ROWS = 50
        COLS = 50
        CHANNELS = 3
        train_images = [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR)]
        train_dogs =   [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR) if 'dog' in i]
        train_cats =   [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR) if 'cat' in i]
        X_tr = prep_data(CHANNELS,ROWS,COLS,train_images)
        labels = []
        for i in train_images:
            if 'dog' in i:
                labels.append(1)
            else:
                labels.append(0)
        if (n_channels == 1):
            X_tr = X_tr.mean(axis=1)
            X_tr = X_tr.reshape(-1,ROWS*COLS).astype(np.float32)
        else:
            X = np.moveaxis(X,3,1).astype(np.float32)
        y = np.asarray(labels).astype(np.float32)
        OUT_DIM = 2
    
    #If want to sample less than the whole data
    if n_samples != 0:
        idxs = np.random.randint(0,X.shape[0],n_samples)
        X = X[idxs]
        y = y[idxs]
    return X,y,X_test,y_test,X.shape[1],OUT_DIM


def read_image(file_path,rows,cols):
    img = cv2.imread(file_path,cv2.IMREAD_COLOR)
    return cv2.resize(img,(rows,cols),interpolation = cv2.INTER_CUBIC)

def prep_data(channels,rows, cols,images):
    count = len(images)
    data = np.ndarray((count, channels, rows, cols), dtype=np.uint8)
    for i, image_file in enumerate(images):
        image = read_image(image_file,rows,cols)
        data[i] = image.T
        if i%2000 == 0: print('Processed {} of {}'.format(i, count))
    
    return data
    
#get probabilities for each class
def get_class_proba(y):
    import pandas as pd
    s = pd.Series(y)
    p = (s.value_counts().sort_index()/len(y)).values
    return p


#if several clusters
def get_data_partitions(OUT_DIM):
    Xs = []
    for i in range(OUT_DIM):
        Xs.append(X[np.argwhere(y==i)])
    return tuple(Xs)


#if several clusters
def get_class_clusters(n_clusters,data_partitions):
    from sklearn.cluster import KMeans
    kmeans = []
    for i in range(len(data_partitions)):
        kmeans.append(KMeans(n_clusters = n_clusters,random_state =0).fit(data_partitions[i].squeeze()))
    return kmeans