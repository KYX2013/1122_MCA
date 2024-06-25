from feature_extraction import feature_extraction
import os
import sklearn
import librosa
import pandas as pd
import numpy as np

def list_dir(directory):
    directories = []
    for item in os.listdir(directory):
        if os.path.isdir(os.path.join(directory, item)) and not item.startswith('_'):
            directories.append(item)
    return directories

def list_wav(dir):
    files = os.listdir(dir)
    wav_files = [file for file in files if file.endswith('.wav')]
    return wav_files

# Find files
rootpath = os.getcwd()+'/genre'
dirs = list_dir(rootpath)

files = {}
for gen in dirs:
    files[gen] = list_wav(rootpath+'/'+gen)

# Feature Extraction
df = pd.DataFrame(columns=['sp_ce','sp_ro','sp_fl','zcr','mfccs','energy','sp_ce_skew','tempo','chroma','genre'])
for music in dirs:
    for file in files[music]:
        new_audio = feature_extraction(rootpath+'/'+music+'/'+file,music)
        df.loc[len(df)] = new_audio

# Model Training
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score,confusion_matrix
import xgboost as xgb

df = df.sample(frac=1.0,random_state=118)

X = df.drop(columns=['genre'])
y = df['genre']


kf = KFold(n_splits=5, shuffle=True, random_state=42)
svm_acc = []
knn_acc = []
rf_acc = []
clf_acc = []
## SVM
svm = SVC(kernel='linear')

for i, (train_index, test_index) in enumerate(kf.split(X)):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Train SVM on training data
    svm.fit(X_train, y_train)
    
    # Predict labels on test data
    y_pred = svm.predict(X_test)
    
    # Calculate accuracy
    print(f'{i+1}th validation:')
    accuracy = accuracy_score(y_test, y_pred)
    print('accuracy:',accuracy)
    svm_acc.append(accuracy)

    print('confusion matrix:\n',confusion_matrix(y_test,y_pred))

print('overall acc of svm:',np.mean(svm_acc))
print()
## KNN
knn = KNeighborsClassifier(n_neighbors=8)

for i, (train_index, test_index) in enumerate(kf.split(X)):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Train SVM on training data
    knn.fit(X_train, y_train)
    
    # Predict labels on test data
    y_pred = knn.predict(X_test)
    
    # Calculate accuracy
    print(f'{i+1}th validation:')
    accuracy = accuracy_score(y_test, y_pred)
    print('accuracy:',accuracy)
    knn_acc.append(accuracy)

    print('confusion matrix:\n',confusion_matrix(y_test,y_pred))

print('overall acc of knn:',np.mean(knn_acc))
print()
## Random Forest
rf = RandomForestClassifier()

for i, (train_index, test_index) in enumerate(kf.split(X)):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Train SVM on training data
    rf.fit(X_train, y_train)
    
    # Predict labels on test data
    y_pred = rf.predict(X_test)
    
    # Calculate accuracy
    print(f'{i+1}th validation:')
    accuracy = accuracy_score(y_test, y_pred)
    print('accuracy:',accuracy)
    rf_acc.append(accuracy)

    print('confusion matrix:\n',confusion_matrix(y_test,y_pred))

print('overall acc of random forest:',np.mean(rf_acc))
print()
## XGBoost
clf = xgb.XGBClassifier()

for i, (train_index, test_index) in enumerate(kf.split(X)):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Train SVM on training data
    clf.fit(X_train, y_train)
    
    # Predict labels on test data
    y_pred = clf.predict(X_test)
    
    # Calculate accuracy
    print(f'{i+1}th validation:')
    accuracy = accuracy_score(y_test, y_pred)
    print('accuracy:',accuracy)
    clf_acc.append(accuracy)

    print('confusion matrix:\n',confusion_matrix(y_test,y_pred))

print('overall acc of xgboost:',np.mean(clf_acc))
print()