import os
import sys
import pandas as pd
import numpy as np

datadir = "/home/mi/Zhankun_work/data/FGADR/FGADR-Seg-set_Release/Seg-set/"

labelpath = datadir + "/DR_Seg_Grading_Label.csv"

labeldf = pd.read_csv(labelpath, header=None)

print(labeldf.shape)

X = list(labeldf.iloc[:, 0])
Y = list(labeldf.iloc[:, 1])

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X, Y, test_size=0.3, stratify=Y, random_state=2021)
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.286, stratify=y_train, random_state=2021)

f = open("./train.txt", 'w', encoding="utf8")
for imgname_i, label_i in zip(X_train, y_train):
    print("{}-{}".format(imgname_i, label_i))
    f.write("{} {}\n".format(imgname_i, label_i))
f.close()
f = open("./validation.txt", 'w', encoding="utf8")
for imgname_i, label_i in zip(X_validation, y_validation):
    print("{}-{}".format(imgname_i, label_i))
    f.write("{} {}\n".format(imgname_i, label_i))
f.close()
f = open("./test.txt", 'w', encoding="utf8")
for imgname_i, label_i in zip(X_test, y_test):
    print("{}-{}".format(imgname_i, label_i))
    f.write("{} {}\n".format(imgname_i, label_i))
f.close()


unique_els, counts = np.unique(Y, return_counts=True)
print("all: {}\n{}".format(unique_els, counts))

unique_els, counts = np.unique(y_train, return_counts=True)
print("train: {}\n{}".format(unique_els, counts))
unique_els, counts = np.unique(y_validation, return_counts=True)
print("validation: {}\n{}".format(unique_els, counts))
unique_els, counts = np.unique(y_test, return_counts=True)
print("test: {}\n{}".format(unique_els, counts))


