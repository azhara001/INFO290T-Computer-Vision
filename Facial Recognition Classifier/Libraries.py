#importing libraries 
import numpy as np
from numpy import random
from numpy import linalg 
import numpy.matlib 
from matplotlib import pyplot as plt
from matplotlib import cm
from sklearn.manifold import TSNE
from scipy.linalg import eigh
import cv2
from IPython.display import clear_output
from time import sleep
import time
import pandas as pd
import os


### SVM for classification
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.utils.fixes import loguniform
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix

print('libraries imported')
