from Libraries import np, plt,clear_output

def initialize_image_matrix(dataset=np.zeros((1,1)),zero_mean=True,set="olivetti"):
    """
    This function takes an input dataset such that the shape is [number_of_images,xdim,ydim] and returns a flattened and zero_meaned dataset such that the shape if [xdim*ydim,number_of_images]
    """
    N_images = dataset.shape[0]
    X_dim = dataset.shape[1]
    Y_dim = dataset.shape[2]

    print(f"{N_images} number of images loaded with {X_dim} rows and {Y_dim} columns!")

    Data = np.zeros((X_dim*Y_dim,N_images))
    #print(Data.shape) 

    for image in range(N_images):
        Data[:,image] = dataset[image,:,:].flatten()
    print(f"Data flattened with the shape: {Data.shape}")

    
    if zero_mean == True:
        mu = np.mean(Data,axis=1)
        print(f"Mean across a {N_images} dimension computed for each pixel with {mu.shape} dimensions!")
    
        # displaying the feature wise (pixel wise) mean
        mean_image = np.reshape(mu,(X_dim,Y_dim))
        plt.figure(figsize=[8,8])
        plt.imshow(mean_image,cmap="gray")
        plt.title("Means of all the faces computed!")
        plt.savefig(set+"Means.png") #-> COMMENTED OUT 
        plt.title(f"The mean for the dataset: {set}")
        plt.show()
        clear_output(1)
    
    Data = Data-np.reshape(mu,(mu.shape[0],1))
    print('returning!')
    return Data