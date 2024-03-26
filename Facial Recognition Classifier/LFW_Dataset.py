## Code for importing the LFW dataset 
# Deciding on which images to take and categorizing their class labels 

from Libraries import pd,os,np,plt,cv2

def LFW_Dataset_func_equal_weights(path="",path_labels="",num_classes = 10): #returns same number of images per class
    # extracting the labels for the dataset
    print('i was here')
    labels_LFW = pd.read_csv(path_labels,sep="\t",header=None,names=["Person Name","Person Images"]) #loads the labels
    labels_LFW = labels_LFW.sort_values(by='Person Images',ascending=False) #sorting the classes by number of image per class
    labels_LFW_100 = labels_LFW.head(num_classes) #using only the first num_classes for now
    max_images_per_class = labels_LFW_100['Person Images'].min() # to ensure that our model sees equal number of images per class
    print(f"Total number of images to be used for this dataset are: {max_images_per_class*num_classes} with {num_classes} distinct classes!")

    labels_LFW_100['Person Images'].describe()
    class_labels = list(range(0,num_classes,1))
    labels_LFW_100['Class Code'] = class_labels
    labels_LFW_100.tail(100)

    # creating a datamatrix that will be used for PCA/FLDA
    directory = os.listdir(path)

    dataset = np.zeros((max_images_per_class*num_classes,250,250))
    labels = np.zeros((max_images_per_class*num_classes,1))


    iterr = 0

    for folder in directory: # would loop over the folders where each face has distinct images
        if folder in list(labels_LFW_100['Person Name']):
            images = os.listdir(os.path.join(path,folder))
            print(f"number of images for this class: {len(images)}")
            num_images = 0
            for one in images:
                path_image = os.path.join(path,folder,one)
                temp = plt.imread(path_image)
                temp = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY) # for now, I am not using the three RGB images
                plt.imshow(temp,cmap="gray")
                dataset[iterr,:,:] = temp
                labels[iterr] = labels_LFW_100[labels_LFW_100['Person Name']==folder]['Class Code'].values[0]
                num_images += 1
                iterr += 1
                if num_images == max_images_per_class:
                    print(f"{num_images} image loaded for class: {iterr}")
                    break #break out of this folder   
                       
    #np.save('LFW_Dataset.npy', dataset)
    #np.save('LFW_Dataset_Labels.npy',labels)

    return dataset,labels

def LFW_Dataset_func_unequal_weights(path="",path_labels="",num_classes = 10): #returns total number of images per class
    # extracting the labels for the dataset
    labels_LFW = pd.read_csv(path_labels,sep="\t",header=None,names=["Person Name","Person Images"]) #loads the labels
    labels_LFW = labels_LFW.sort_values(by='Person Images',ascending=False) #sorting the classes by number of image per class
    labels_LFW_100 = labels_LFW.head(num_classes) #using only the first num_classes for now
    
    labels_LFW_100['Person Images'].describe()
    class_labels = list(range(0,num_classes,1))
    labels_LFW_100['Class Code'] = class_labels
    #labels_LFW_100.tail(100)

    # creating a datamatrix that will be used for PCA/FLDA
    directory = os.listdir(path)
    
    dataset = np.zeros((labels_LFW_100['Person Images'].sum(),250,250))
    labels = np.zeros((labels_LFW_100['Person Images'].sum(),1))
    iterr = 0

    for folder in directory: # would loop over the folders where each face has distinct images
        if folder in list(labels_LFW_100['Person Name']):
            images = os.listdir(os.path.join(path,folder))
            print(f"number of images for this class: {len(images)}")
            num_images = 0
            for one in images:
                path_image = os.path.join(path,folder,one)
                temp = plt.imread(path_image)
                temp = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY) # for now, I am not using the three RGB images
                plt.imshow(temp,cmap="gray")
                #plt.show()
                dataset[iterr,:,:] = temp
                #print('i was here?')
                labels[iterr] = labels_LFW_100[labels_LFW_100['Person Name']==folder]['Class Code'].values[0]
                num_images += 1
                iterr += 1

        
    #np.save('LFW_Dataset.npy', dataset)
    #np.save('LFW_Dataset_Labels.npy',labels)

    return dataset,labels