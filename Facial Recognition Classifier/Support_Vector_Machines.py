from Libraries import train_test_split, RandomizedSearchCV, fetch_lfw_people, classification_report, ConfusionMatrixDisplay, StandardScaler, PCA, SVC, loguniform, GridSearchCV, confusion_matrix, np,pd


def SVM_classifier(dataset=np.zeros((1,1)),labels=np.zeros((1,1)),SVM_type="linear",flip_test=False,shuffle=False):
    X_train, X_test, y_train, y_test = train_test_split(dataset.T, labels, test_size=0.30, random_state=42)
    
    if shuffle == True:
        np.random.shuffle(y_test) #randomly shuffles the y labels of the test dataset only
    
    if flip_test == True:
        y_test = np.flipud(y_test) #flipping the test set data
    
    labels = np.ravel(labels)

    print("Fitting the classifier to the training set")
    param_grid = {
            'C': [1e3, 5e3, 1e4, 5e4, 1e5],
            'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1],
            }
    if SVM_type == 'linear':
        clf = GridSearchCV(SVC(kernel='linear', class_weight='balanced'), param_grid) # linear SVM!)
    else:
        clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid) # non-linear SVM (Radial Basis Function Kernel!)
    clf = clf.fit(X_train, np.reshape(y_train,(y_train.shape[0],)))
    print("Best estimator found by grid search:")
    print(clf.best_estimator_)

    print("Predicting the people names on the testing set")
    y_pred = clf.predict(X_test)

    from sklearn.metrics import accuracy_score
    Accuracy_Score = accuracy_score(y_test, y_pred)
    #print(Accuracy_Score)

    Classification_Report = classification_report(y_test, y_pred,output_dict=True)
    Classification_Report = pd.DataFrame(Classification_Report).transpose()
    Confusion_Matrix = confusion_matrix(y_test, y_pred)
    return Accuracy_Score,Classification_Report,Confusion_Matrix
