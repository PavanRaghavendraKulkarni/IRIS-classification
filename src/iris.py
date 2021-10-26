import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pickle

iris=datasets.load_iris()
x_train=iris.data
y_train=iris.target
clf=DecisionTreeClassifier().fit(x_train,y_train)
pickle.dump(clf,open('iri.pkl','wb'))