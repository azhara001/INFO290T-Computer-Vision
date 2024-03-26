from Libraries import time,np,eigh,plt

def fisher_faces(D=np.zeros((1,1)),labels=np.zeros((1,1)),xdim=0,ydim=0,print_canonical=False,add_noise=False,set=""):
    """
    This function takes as input a zero-meaned flattened image dataset with the shape [features,number_of_images] and computes the fisher faces 
    """
    print("Computing Fisher Faces")
    t1 = time.time()
    
    print(f"Dataset loaded with the following dimensions: {D.shape} and ready for fisher-face linear\n"
    "discriminant analysis")
    
    N_images = D.shape[1]
    C = np.unique(labels)

    print(f"The number of classes are: {C.shape[0]}")
    
    Sw = np.zeros((D.shape[0],D.shape[0])) #initializing within class covariance matrix
    Sb = np.zeros((D.shape[0],D.shape[0])) #initializing between class covariance matrix
    mu = np.reshape(np.mean(D,axis=1),(D.shape[0],1)) #total mean mu of the dataset
  
    for i in range(len(C)): #for-loop that iterates through the class
        indexes = np.where(labels==i)[0]  
        C_i = np.zeros((D.shape[0],len(indexes)))
        
        for iter,ind in enumerate(indexes): # for loop that captures all the images belonging to one class
            C_i[:,iter] = D[:,ind]
        # now C_i should contain all the images belonging to one class so we can proceed with computing 
        # the within class mean 
        
        mean_c = np.reshape(np.mean(C_i,axis=1),(C_i.shape[0],1)) #within class-2 mean 
        Normalized_c = C_i-mean_c
        C_p1 = Normalized_c@Normalized_c.T
        Sw += C_p1 # adds the within variance of each class to the final Sw matrix 
        Sb += len(indexes)*np.outer(mean_c-mu,mean_c-mu)
        
        C_i = np.zeros((D.shape[0],len(indexes))) #redeclaring C_i
        
    if add_noise == True:
        Sw += Sw + 0.00001*np.identity(Sw.shape[0]) #regularizing Sw: http://www.scholarpedia.org/anrticle/Fisherfaces
    
    val,vec = eigh(Sb, Sw) # compute generalize eigenvector
    e = vec[:,vec.shape[1]-C.shape[0]+1:vec.shape[1]] #pick the last C-1 eigen_vectors

    if print_canonical == True and set == "Olivetti":
        print("the first 12 canonical basis for Fisherfaces!") 
        plt.figure(figsize=[15,15])
        plt.title("Canonical Basis Fisher Faces Olivetti")
        xdim = int(np.sqrt(e.shape[0]))
        for i in range(12):
            plt.subplot(4,3,i+1)
            plt.imshow(np.reshape(e[:,30-i],(xdim,-1)),cmap="gray")
        plt.savefig("Canonical Basis Fisher Faces Olivetti.png")
        plt.show()
    elif print_canonical == True and set == "LFW_sklearn":
        print("the first 12 canonical basis for Fisherfaces!") 
        plt.figure(figsize=[15,15])
        plt.title("Canonical Basis Fisher Faces LFW_sklearn")
        for i in range(12):
            plt.subplot(4,3,i+1)
            plt.imshow(np.reshape(e[:,e.shape[1]-1-i],(50,37)),cmap="gray")
        plt.savefig("Canonical Basis Fisher Faces LFW_sklearn.png")
        plt.show()
 
    Fisher = e.T@D
    print(e.shape)
    
    t2 = time.time()    
    print(f"FISHER-FACES COMPUTED in {t2-t1} seconds! \n\n\n")
    
    return Fisher