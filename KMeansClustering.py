import numpy as np
import matplotlib.pyplot as plt
import scipy.io


# RANDOM INTIALIZATION OF CENTROIDS
def random_intialise(X,K):
    (m,n) = X.shape
    per = np.random.permutation(m)
    cent_idx = per[:K]
    centroids = X[cent_idx,:]
    return centroids


# CENTROID ASSIGNMENT
def findClosestCentroid(X,centroids):
    (m,n) = X.shape
    (K,n) = centroids.shape
    index = np.zeros((m,1),int)
    for i in range(1,m):
        diff = X[i,:]-centroids
        diff = diff * diff
        d = np.sum(diff,axis = 1)
        k = np.argmin(d,axis = 0)
        index[i]=k
    return (index+1)

# MOVE CENTROID
def computeCentroid(X,index,K):
    (m,n) = X.shape
    centroids = np.zeros((K,n),float)

    for i in range(1,K+1):
        l_array = (index == i)
        idx = np.where(l_array==1)
        idx = np.array(idx)
        idx = idx[0,:]
        idx = idx.reshape(-1,1)
        x = X[idx[:,0],:]
        m = np.mean(x,axis = 0)
        centroids[i-1,:]=m

    return centroids

# Plot function 1
def plotDataPoints(X,index,K):
    plt.scatter(X[:,0],X[:,1],s=20,cmap='prism',c=index[:,0])
    cbar = plt.colorbar()
    cbar.set_label('indices')

# Plot function 2
def plotKMeansProgress(X,index,centroids,previous_centroids,K,i):
    plotDataPoints(X,index,K)
    plt.scatter(previous_centroids[:,0],previous_centroids[:,1],c='red',marker='x',s=30)
    plt.scatter(centroids[:,0],centroids[:,1],c='black',marker='x',s=30)
    plt.title(i)


# Main RUN K-MEANS FUNCTION
def runKMeans(X,initial_centroids,max_iters,plot_progress):
    (m,n) = X.shape
    (K,n) = initial_centroids.shape
    centroids = initial_centroids
    previous_centroids=centroids
    for i in range(1,max_iters+1):
        index = findClosestCentroid(X,centroids)
        #plt.figure(i)
        if(plot_progress):

            plotKMeansProgress(X,index,centroids,previous_centroids,K,i)
            previous_centroids = centroids

        centroids = computeCentroid(X,index,K)

    cent_index ={'centroids':centroids,'index':index}
    return cent_index

def runKMeans_Image(image_name,K,max_iters):
    image = plt.imread(image_name)
    (d1,d2,d3) = image.shape
    X = image.reshape(d1*d2,3)
    (m,n) = X.shape
    centroids = random_intialise(X,K)
    cent_index_image = runKMeans(X,centroids,max_iters,False)
    centroids = cent_index_image['centroids']
    index = cent_index_image['index']

    index = findClosestCentroid(X,centroids)
    X_recovered = centroids[index-1,:]
    X_recovered =X_recovered.reshape(d1,d2,3)

    plt.style.use('fivethirtyeight')
    plt.subplot(1,2,1)
    plt.grid(b=None)
    plt.imshow(image)
    plt.title('original')

    plt.subplot(1,2,2)
    plt.grid(b=None)
    plt.imshow(X_recovered)
    plt.title(f'compressed with {K} colors')

image_name = input('enter the name of the image with extension only .png images')
K = int(input('enter the no. of colors to which the image should be compressed'))
runKMeans_Image(image_name,K,10)