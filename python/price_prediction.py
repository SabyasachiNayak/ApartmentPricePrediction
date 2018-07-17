import requests
import io
import argparse
from argparse import ArgumentParser
import pandas
import sklearn
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import preprocessing

def runML():
	# Load dataset
	url = 'http://ucanalytics.com/blogs/wp-content/uploads/2016/07/Regression-Analysis-Data.csv'
	response = requests.get(url)
	file_object = io.StringIO(response.content.decode('utf-8'))
	#names = ['Observation', 'Dist_Taxi','Dist_Market','Dist_Hospital','Carpet','Builtup','Parking','City_Category','Rainfall','House_Price']
	#dataset = pandas.read_csv(file_object, names=names)
	dataset = pandas.read_csv(file_object)
	le = preprocessing.LabelEncoder()
	dataset['Parking'] = le.fit_transform(dataset['Parking'].astype(str))
	dataset['City_Category'] = le.fit_transform(dataset['City_Category'].astype(str))
	
	#dataset.iloc[1,0:10] = le.fit_transform(dataset.iloc[1,0:10].astype(str))
	#dataset = dataset.apply(le.fit_transform)

	# Split-out validation dataset (20% validation data)
	array = dataset.values
	X = array[:,0:9]
	Y = array[:,9]
	#X.reshape(1,-1)
	validation_size = 0.20
	seed = 7
	
	X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)
	
	# Spot Check Algorithms
	models = []
	models.append(('LR', LogisticRegression()))
	models.append(('LDA', LinearDiscriminantAnalysis()))
	models.append(('KNN', KNeighborsClassifier()))
	models.append(('CART', DecisionTreeClassifier()))
	models.append(('NB', GaussianNB()))
	models.append(('SVM', SVC()))

	# evaluate each model in turn
	results = []
	names = []
	for name, model in models:
	 kfold = model_selection.KFold(n_splits=10, random_state=seed)
	 cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold,scoring='accuracy')
	 results.append(cv_results)
	 names.append(name)
	 msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	 print(msg)

	# Make predictions on validation dataset
	knn = KNeighborsClassifier()
	knn.fit(X_train, Y_train)
	#predictions = svc.predict(X_validation)
	#print(accuracy_score(Y_validation, predictions))
	#print(confusion_matrix(Y_validation, predictions))
	#print(classification_report(Y_validation, predictions))
	

	predictions = knn.predict(X_validation[0])   #this is for scikit 0.19  version issue fix 
	#predictions = knn.predict([5.9, 3.0, 5.1, 1.8]) 
	#predictions = knn.predict([args.sLength, args.sWidth, args.pLength, args.pWidth]) 
	print(predictions[0])
	return predictions[0]

runML()



