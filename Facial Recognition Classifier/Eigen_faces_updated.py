from Libraries import np, time,plt

def PCA(D=np.zeros((1,1)),efficient=True,xdim=0,ydim=0,for_fisher=False,percent_dim=90,labels_unique=np.zeros((1,1)),plot_canonical=False,set=""):
    """
    Computes the eigen faces for the data matrix such that we get an N-c dimensionality (where N is the number of images and C are the number of classes)
    """
    print("Computing Eigenfaces: ")
    t1 = time.time()
    
    print(f"The shape of the data matrix is: {D.shape}")
    
    if efficient == True:
        C_w = D.T@D # computing the co-variance matrix efficiently to get an nxn matrix (400x400)
        val,vect = np.linalg.eig(C_w) #should return the eigen vectors and values for the 400x400 matrix
        vect = D@vect #should give me the eigen vectors for the covariance matrix (4096,400)
    else:
        C = D@D.T
        val,vect = np.linalg.eig(C)
    e = np.real(vect)
    
    if for_fisher == True: # this bit reduces the features of the eigen vectors to N-c
        eigen_vectors_total = D.shape[1]-labels_unique.shape[0] #N-c
        print(f"The number of eigenvectors extracted are N-c i.e.,: {eigen_vectors_total}")
    else:
    # code to determine the number number of eigen-vectors to be computed based on the fall-off threshold provided as percent_dim
        sum_vals = 0
        total_sum = np.sum(val)

        for i in range(val.shape[0]):
            sum_vals += val[i]
            if (sum_vals/total_sum)*100 >= percent_dim:
                print(f"Total number of eigen-vectors to be used based on {percent_dim}% fall off: {i}")
                eigen_vectors_total = i
                break
                
    print(f"the shape of eigen vectors of covariance matrix: {e.shape}")
    
    if plot_canonical == True:
        print("the first 12 canonical basis for Eigenfaces!") 
        plt.figure(figsize=[15,15])
        plt.title("First 12 Canonical Basis (Eigenfaces)")
        for i in range(12):
            plt.subplot(4,3,i+1)
            plt.imshow(np.reshape(e[:,i],(xdim,ydim)),cmap="gray")
        plt.savefig(f"Canonical Basis {set}.png") 
        plt.show()
  
    #print(f"shape of vects: {vect.shape}")
    Projection = vect[:,0:eigen_vectors_total].T@D
    print(vect[:,0:eigen_vectors_total])
    print(Projection.shape)
  
    t2 = time.time()    
    print(f"EIGEN-FACES COMPUTED in {t2-t1} seconds! \n\n\n")

    return Projection