from sklearn.datasets import fetch_lfw_people
from time import sleep 
import matplotlib.pyplot as plt 
from IPython.display import clear_output
import numpy as np

def fetch_LFW_sklearn(display=True,min_faces = 0):
    lfw_people = fetch_lfw_people(min_faces_per_person=min_faces, resize=0.4)
    D = lfw_people['data']
    xdim = lfw_people['images'].shape[1]
    ydim = lfw_people['images'].shape[2]
    print(D.shape)
    print(xdim)
    print(ydim)

    if display == True:
        for name in lfw_people.target_names:
            print(name)

        for i in range(lfw_people['images'].shape[0]):
            clear_output(wait=True)
            plt.imshow(lfw_people['images'][i,:,:])
            plt.show()
            plt.title(lfw_people['target'][i])
            sleep(0.00000000001)
    return lfw_people