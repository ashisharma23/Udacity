#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data, test_classifier

from time import time
import math
import pandas as pd
#from explore_preprocess_data import remove_outliers

#Imputation transformer for completing missing values.
from sklearn.preprocessing import Imputer
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import svm
from sklearn.feature_selection import SelectKBest,chi2
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.cross_validation import StratifiedShuffleSplit, train_test_split
from sklearn.preprocessing import MinMaxScaler,RobustScaler
from sklearn.neighbors import KNeighborsClassifier, BallTree
from sklearn.metrics import classification_report

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".StratifiedShuffleSplit
####features types:
##### 1.financial features:(all units are in US dollars)
financial_features = ['salary', 'bonus', 'long_term_incentive', 'deferred_income',
'deferral_payments', 'loan_advances', 'other','expenses','director_fees','total_payments', 
'exercised_stock_options','restricted_stock','restricted_stock_deferred', 'total_stock_value']
##### 2. email features: (units are generally number of emails messages; 
#####notable exception is 'email_address', 
##### which is a text string)
email_features = ['email_address', 'to_messages', 'from_poi_to_this_person', 
'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi'] 
###### 3. POI label: (boolean, represented as integer)
POI_label = ['poi'] 


'''
Function Name : manual_adjustment
Input : DataFrame - df
Returns : DataFrame - df
Description	:  Updates specific data points based on validations  performed 
on financial features and enron61702insiderpay.pdf manual lookup.
'''
def manual_adjustment(df):

	 #The above validations errors are corrected for 'BELFER ROBERT' and 
	 #'BHATNAGAR SANJAY' with below manual adjustment per enron61702insiderpay.pdf
	df.at['BELFER ROBERT','expenses'] = df.at['BELFER ROBERT','director_fees']
	df.at['BELFER ROBERT','total_payments'] = df.at['BELFER ROBERT','director_fees']
	df.at['BELFER ROBERT','director_fees'] = \
		math.fabs(df.at['BELFER ROBERT','deferral_payments'])
	
	df.at['BHATNAGAR SANJAY','expenses'] = df.at['BHATNAGAR SANJAY','other']
	df.at['BHATNAGAR SANJAY','total_payments'] = df.at['BHATNAGAR SANJAY','other']
	df.at['BHATNAGAR SANJAY','director_fees'] = 0
	df.at['BHATNAGAR SANJAY','other'] = 0
	
	df.at['BELFER ROBERT','restricted_stock'] = \
	df.at['BELFER ROBERT','restricted_stock_deferred']
	
	df.at['BELFER ROBERT','restricted_stock_deferred'] = \
	df.at['BELFER ROBERT','total_stock_value']
	
	df.at['BELFER ROBERT','total_stock_value'] = 0
	df.at['BELFER ROBERT','exercised_stock_options'] = 0
	
	df.at['BHATNAGAR SANJAY','total_stock_value'] = \
	df.at['BHATNAGAR SANJAY','restricted_stock_deferred']
	
	df.at['BHATNAGAR SANJAY','restricted_stock_deferred'] = \
	df.at['BHATNAGAR SANJAY','restricted_stock']
	
	df.at['BHATNAGAR SANJAY','restricted_stock'] = \
	df.at['BHATNAGAR SANJAY','exercised_stock_options']
	
	df.at['BHATNAGAR SANJAY','exercised_stock_options'] = \
	df.at['BHATNAGAR SANJAY','total_stock_value']
		
	return df
'''
Function Name : plot_features
Input : Series - X
Returns: Series - y
Description	: Plot scatter plot 
'''	
def plot_features(X, y):
	from matplotlib import pyplot
	pyplot.scatter(X, y)
	pyplot.show()
	
'''
Function Name : remove_outliers
Input : DataFrame - df
Returns: DataFrame - df
Description	: This functions explore financial data to identify outliers to remove from
dataset.
'''	
def remove_outliers(df):
	#Step1 : Explore financial data
	print df.describe()
	
	#Removing totals as outliers as inferred from scatter plot of salary and bonus - below.
	df = df.drop('TOTAL')
	# Removing 'THE TRAVEL AGENCY IN THE PARK', which  was a company co-owned 
	# by Enron's former Chairman's sister and is clearly not an individual that should be
	# included in the dataset.
	df = df.drop('THE TRAVEL AGENCY IN THE PARK')
	
	# Check whether all financial features are 0 and data is non relevant. 
	# LOCKHART EUGENE E was dropped here- due to that reason.
	for index, row in df.iterrows():
		all_values_zero = True
		for feature in financial_features:
			if row[feature] != 0.0:
				all_values_zero = False
		if all_values_zero:
			df = df.drop(index)
			
	# Updating below features to their absolute value, as they are marked as negative
	df['deferred_income'] = df['deferred_income'].abs()
	df['deferral_payments'] = df['deferral_payments'].abs()
	df['restricted_stock_deferred'] = df['restricted_stock_deferred'].abs()
	
	#Exploring outliers to identify is there any outlier which is non-person entity
	for feature in financial_features:
		print "#######Financial feature Name: ", feature,'  Start ###########'
		print df.nlargest(3, feature)
		print "#######Financial feature Name: ", feature,'  End ###########'
	
	for feature in email_features[1:]:
		print "#######Email feature Name: ", feature,'  Start ###########'
		print df.nlargest(3, feature)
		print "#######Email feature Name: ", feature,'  End ###########'
	
	
	IQR = df.quantile(q=0.75) - df.quantile(q=0.25)
	first_quartile = df.quantile(q=0.25)
	third_quartile = df.quantile(q=0.75)
	outliers = \
	df[(df>(third_quartile + 1.5*IQR) ) | (df<(first_quartile - 1.5*IQR) )].count(axis=1)
	outliers.sort_values(axis=0, ascending=False, inplace=True)
	print outliers.head(15)
	
	#plot_features(df['salary'],df['bonus'])
	#plot_features(df['to_messages'],df['from_messages'])
	
	
	# As observed in enron61702insiderpay.pdf, 'Director Fees' is essentially 
	# salary paid to Director
	# Running check whether there is an row where salary and Director Fees 
	# both are greater than 0.
	#print df[ (df['salary'] == 0) & (df['director_fees'] != 0)]
	df.loc[(df['salary'] == 0) & (df['director_fees'] != 0), ['salary']] = \
	df.loc[(df['salary'] == 0) & (df['director_fees'] != 0), 'director_fees']
	# Review comment #2 update
	#df = df.drop(columns = ['director_fees'])
	df = df.drop(['director_fees'], axis=1)
			
	#The below boxplot shows there are several outliers which lies beyond the 
	# IQR in the distribution, 	#however all these points correspond to 
	# valid persons for poi envestigation.
	df[financial_features[:8]].boxplot()
	plt.show()
	
	return df
	
'''
Function Name : preprocess_data
Input : numpy array - data, columns - features_list, keys  - person names
Returns	: DataFrame - df after preprocessing.
Description	: Creates dataframe and perform below preprocessing steps:
	1. Update any financial data point having NaN with 0
	2. Updating email features data point having NaN with median for that column.
	3. Perform manual adjustment based on updates
	4. Perform validations on total values based on financial features.
'''	
def preprocess_data(data, columns, keys):
	
	#Converting dataset to pandas dataframe
	# Review comment #1 - Update  Start
	# Error thrown: ValueError: Empty data passed with indices specified.
	# however , I am not getting the error while running locally on my system
	#df = pd.DataFrame(data, columns = columns, index=keys)
	# Changed the above line as below
	df = pd.DataFrame(data, columns = columns, index=keys, copy = True)
	# Review comment #1 - Update  End
	missing_counts = {}
	total_count = len(df.index)
	for  col in columns:
		missing_counts[col] = (df[col].isnull().sum() *100.0)/total_count
	print "Missing count in data set\n"
	print missing_counts
	
		
	# Step1 : Update NaN with appropriate values.
	# For financial (payment and stock) data, values of NaN represent 0 and 
	# not unknown quantities.
	for feature in financial_features:
		df[feature]= df[feature].fillna(0)
		
	# For email data, NaN represent Unknown information
	# Hence updating numeric email data values with mean of the column group by poi
	# using sklearn Imputer class 
	# Since we have several outliers and data is skewed, using median as strategy.
	#Imputation transformer for completing missing values.
	imp = Imputer(missing_values='NaN', strategy = 'median', axis=0)
	# Creating seperate dfs for poi and non-poi
	df_poi = df[df['poi'] == True]
	df_nonpoi = df[df['poi'] == False]
	
	# Lastly, I am removing string column - email_address, 
	# which does not provide much information to the dataset.
	df_poi.ix[:, email_features[1:]] = imp.fit_transform(df_poi.ix[:,email_features[1:]])
	df_nonpoi.ix[:, email_features[1:]] = \
	imp.fit_transform(df_nonpoi.ix[:,email_features[1:]])
	df = df_poi.append(df_nonpoi)
	
	print df[email_features[1:]].describe()
	#	Performing  manual adjustment after validating 
	# sum of all payment_data and stock against totals
	# as in below statements.
	df = manual_adjustment(df)
	
	# splitting payment and stock features
	payment_features = financial_features[:10]  
	stock_features = financial_features[10:]
	
	# Validate if sum of all payment_data equals to total_payments
	'''
	Commenting now, in the final version
	print "Check for validation Error: SUM(payment data) != total_payments"
	print df[df[payment_features[:-1]].sum(axis='columns') != \
	df['total_payments']][payment_features]
	#Validate if sum of all stock_data equals to total_stock_value
	print "Check for validation Error: SUM(stock data) != total_stock_value"
	print df[df[stock_features[:-1]].sum(axis='columns') != \
	df['total_stock_value']][stock_features]
	'''
	return df
		
'''
Function Name : create_new_features
Input :  DataFrame - df
Returns : DataFrame - df
Description	:  Creates new features - to_poi_ratio, from_poi_ratio, shared_poi_ratio,
bonus_to_salary and bonus_to_total using existing features
'''	
def create_new_features(df):

	# Create the new email features and add to the dataframe
	df['to_poi_ratio'] = (df['from_poi_to_this_person'] ) / df['to_messages']
	df['to_poi_ratio'] = df['to_poi_ratio'].fillna(0)
	
	df['from_poi_ratio'] = (df['from_this_person_to_poi']) / df['from_messages']
	df['from_poi_ratio'] = df['from_poi_ratio'].fillna(0)
	
	df['shared_poi_ratio'] = (df['shared_receipt_with_poi'] ) / df['to_messages']
	df['shared_poi_ratio'] = df['shared_poi_ratio'].fillna(0)
	
	#Create the new financial features and add to the dataframe
	df['bonus_to_salary_ratio'] = (df['bonus']) / df['salary']
	df['bonus_to_salary_ratio'] = df['bonus_to_salary_ratio'].fillna(0)
	
	df['bonus_to_total_ratio'] = (df['bonus']) / df['total_payments']
	df['bonus_to_total_ratio'] = df['bonus_to_total_ratio'].fillna(0)
	
	df['non_salary']  = df['bonus']  + df['long_term_incentive']  +  \
		df['loan_advances']  + df['other']  + df['expenses']  + \
		df['exercised_stock_options']  + df['restricted_stock']
	df['non_salary']  = df['non_salary'] /7
	df['non_salary'] = df['non_salary'].fillna(0)
	
	return df

'''
Function Name : build_classifier_pipeline
Input	: classifier_type - name of classifier, kbest - number of features to use in kbest,
f_list - features_list
Returns	:
Description		: Build classifier pipeline to perform exhaustive search over 
specified parameter values for an estimator.
'''	
def build_classifier_pipeline(classifier_type, kbest, f_list):
    # Build pipeline and tune parameters via GridSearchCV

    data = featureFormat(my_dataset, f_list, sort_keys=True)
    labels, features = targetFeatureSplit(data)
     
    # StratifiedShuffleSplit cross-validation object is a merge of StratifiedKFold and 
    # ShuffleSplit which returns stratified randomized folds. 
    # The folds are made by preserving the percentage of samples for each class.  
    # Using stratified shuffle split cross validation because of the small size of the dataset
    sss = StratifiedShuffleSplit(labels, 500, test_size=0.45, random_state=42)
    
    scaler = MinMaxScaler()
    
    
    #Select features according to the k highest scores.
    kbest = SelectKBest( k=kbest)
    
    #Transforms features by scaling each feature to a given range.
    #This estimator scales and translates each feature individually such that
    #it is in the given range on the training set, i.e. between zero and one.
    scaler = MinMaxScaler()
    classifier = set_classifier(classifier_type)
    
    #Pipeline sequentially apply a list of transforms and a final estimator. 
    #Intermediate steps of the pipeline must be 'transforms', that is, 
    #they must implement fit and transform methods. 
    #The final estimator only needs to implement fit. 
    #The transformers in the pipeline can be cached using memory argument.

    #The purpose of the pipeline is to assemble several steps that can be cross-validated
    # together while setting different parameters. For this, it enables setting parameters
    # of the various steps using their names and the parameter name separated by 
    # a '__'.
    # A step's estimator may be replaced entirely by setting the parameter with its name
    # to another estimator, or a transformer removed by setting to None.
    
    # Build pipeline
    pipeline = Pipeline(steps=[('minmax_scaler', scaler), \
    ('feature_selection', kbest), (classifier_type, classifier)])

    # Set parameters for random forest
    parameters = []
    if classifier_type == 'random_forest':
        parameters = dict(random_forest__n_estimators=[10,15,25, 50],
                          random_forest__min_samples_split=[2, 3, 4],
                          random_forest__criterion=['gini', 'entropy'])
     # Set parameters for logical regression
    if classifier_type == 'logistic_reg':
        parameters = dict(logistic_reg__class_weight=['balanced'],
                          logistic_reg__solver=['liblinear'],
                          logistic_reg__C=range(1, 5),
                          logistic_reg__random_state=[42])
     # Set parameters for decision_tree
    if classifier_type == 'decision_tree':
        parameters = dict(decision_tree__min_samples_leaf=range(1, 5),
                          decision_tree__max_depth=range(1, 5),
                          decision_tree__class_weight=['balanced'],
                          decision_tree__criterion=['gini', 'entropy']
                          )
    # Set parameters for svm
    if classifier_type == 'svm':
        parameters = dict(svm__kernel=('linear', 'rbf'),
                          svm__C=[1,5,10],
                          svm__gamma=np.logspace(-9, 3, 13))
    # Set parameters for KNeighbors
    if classifier_type == 'kneighbors':
        parameters = dict(kneighbors__n_neighbors=range(2,5),
                          kneighbors__weights = ['uniform'],
                          kneighbors__algorithm=['auto','ball_tree'],
                          kneighbors__leaf_size=[10,20,30], 
                          kneighbors__p=[1,2], kneighbors__metric=['minkowski'])              
	print "GridSearchCV started"	                         
    # Get optimized parameters for F1-scoring metrics
    #GridSearchCV implements a 'fit' method and a 'predict' method like any classifier 
    #except that the parameters of the classifier used to predict is optimized by 
    # cross-validation.
    cv = GridSearchCV(pipeline, param_grid=parameters, scoring='f1', cv=sss)
    t0 = time()
    cv.fit(features, labels)
    print 'Classifier tuning: %r' % round(time() - t0, 3)
    print "GridSearchCV ended"	
    return cv

'''
Function Name : set_classifier
Input :  x - classifier name
Returns : classifier object
Description	: the object for appropriate classifier object based on x value
'''	
def set_classifier(x):
    # switch statement Python replacement - http://stackoverflow.com/a/103081
    return {
        'random_forest': RandomForestClassifier(),
        'decision_tree': DecisionTreeClassifier(),
        'logistic_reg': LogisticRegression(),
        'gaussian_nb': GaussianNB(),
        'svm':svm.SVC(),
        'kneighbors': KNeighborsClassifier ()
    }.get(x)

#######################################################
# Main starts ...
######################################################
if __name__ == "__main__":
	# Load the dictionary containing the dataset
	print '########## Loading Dataset ##########'
	with open("final_project_dataset.pkl", "r") as data_file:
		data_dict = pickle.load(data_file)
	print '##### data_dict of length %d loaded successfully #####'  % len(data_dict)
	
	# I am removing string column - email_address, 
	# which does not provide much information to the dataset.
	### Select all features list excluding email address, poi being the 1st feature.
	features_list = POI_label + financial_features + email_features[1:]
	data = featureFormat(data_dict, features_list, remove_NaN=False, \
	remove_all_zeroes=False, sort_keys=False)
	
	### Task 1 : Step2 : Preprocess Data
	df = preprocess_data(data, features_list, data_dict.keys())
	
	### Task 2: Remove outliers
	df = remove_outliers(df)
	### Task 3: Create new feature(s)	
	df  = create_new_features(df)
	
	#Scaling the data excluding first column -poi before exporting.
	df.ix[:, df.columns[1:]] = MinMaxScaler().fit_transform(df.ix[:,df.columns[1:]])
	
	### Store to my_dataset for easy export below.
	my_dataset= df.to_dict(orient='index')
	
	### Extract features and labels from dataset for local testing
	features_list = df.columns.tolist()
	print "Total new features = ",features_list
	
	data = featureFormat(my_dataset, features_list, sort_keys = False)
	labels, features = targetFeatureSplit(data)
	
	
	### Task 4: Try a varity of classifiers
	### Please name your classifier clf for easy export below.
	### Note that if you want to do PCA or other multi-stage operations,
	### you'll need to use Pipelines. For more info:
	### http://scikit-learn.org/stable/modules/pipeline.html
	# Extract features and labels from dataset for local testing
	features_train, features_test, labels_train, labels_test = \
		train_test_split(features, labels, train_size=.55, random_state = 42, stratify=labels)
	# Provided to give you a starting point. Try a variety of classifiers.
	print '########## Gaussian NB Starts ##########'
	from sklearn.naive_bayes import GaussianNB
	clf = GaussianNB()

	# Validate model precision, recall and F1-score
	test_classifier(clf, my_dataset, features_list)
	print '########## Gaussian NB Ends ##########'
	
	print '########## Feature Selection Starts ##########'
	# Feature select is performed with SelectKBest where k is selected by GridSearchCV
	# Using Stratify for small and minority POI dataset
	
	
	#skbest = SelectKBest(k='all')  
	#By Running SelectKBest above, it was found 15/24 have scores > 9.8
	#KBest was ran multiple times to identify best features count = 22
	#Those 15 features were selected as final set. 
	kbest_feature_count = 21#22
	skbest = SelectKBest(k=kbest_feature_count)  
	sk_transform = skbest.fit_transform(features_train, labels_train)
	indices = skbest.get_support(True)
	print skbest.scores_
	
	n_list = ['poi']
	for index in indices:
		print '%s score: %f' % (features_list[index + 1], skbest.scores_[index])
		n_list.append(features_list[index + 1])
		
	print "Kbest Features :", n_list
	print '########## Feature Selection Ends ##########'
	# Final features list determined from SelectKBest 
	'''
	Final feature set
	n_list =   ['poi', 'salary', 'bonus', 'long_term_incentive', 
	'deferred_income', 'deferral_payments', 'loan_advances', 'other', 'expenses', 
	'total_payments', 'exercised_stock_options', 'restricted_stock',
	 'restricted_stock_deferred', 'total_stock_value', 'to_messages', 
	 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 
	 'shared_receipt_with_poi', 'to_poi_ratio', 'from_poi_ratio', 'shared_poi_ratio', 
	 'bonus_to_salary_ratio', 'bonus_to_total_ratio', 'non_salary']
	'''
	
	### Task 5: Tune your classifier to achieve better than .3 precision and recall 
	### using our testing script. Check the tester.py script in the final project
	### folder for details on the evaluation method, especially the test_classifier
	### function. Because of the small size of the dataset, the script uses
	### stratified shuffle split cross validation. For more info: 
	### http://scikit-learn.org/stable/modules/generated/sklearn.
	####cross_validation.StratifiedShuffleSplit.html
	
	# Example starting point. Try investigating other evaluation techniques!
	
	# Update features_list with new values
	
	''''
	# The below code was used exploring various algorithms and fine tuning the 
	parameters. The final Question 4 from Task	.
	This part is being commented out, since its take long time for computation
	Please un-comment, if further fine tuning is required.
	
	for  clf in ['random_forest', 'logistic_reg','decision_tree','svm','kneighbors']:
		print '######## ', clf, ' Starts ######'
		cross_val = build_classifier_pipeline(clf, kbest_feature_count, features_list)
		print 'Best parameters: ',clf,'  ' ,cross_val.best_params_
		print 'Best estimators: ', clf,'  ' , cross_val.best_estimator_
		pred = cross_val.predict(features_test)
		print "Classification_report:"
		print classification_report(pred, labels_test)
		print '######## ', clf, ' Ends ########'
	'''
	
	
	#Finally I decided what to use Decision Tree with Ensemble algorithm Adaboost.
	print '########## KBest with AdaBoost Starts ##########'
	features_list = n_list # Use k best features we selected earlier.
	print "Final feature set ",  features_list
	#Adaptive Boosting
	#https://www.analyticsvidhya.com/blog/2015/11/quick-introduction-boosting-algorithms-machine-learning/
	'''
	dt_classifier = DecisionTreeClassifier(criterion='gini', max_depth=1, \
	min_samples_leaf=2, class_weight='balanced')
    
	adaboost_classifier = AdaBoostClassifier(base_estimator = dt_classifier, \
	n_estimators=50, learning_rate=.8)
	
	pipeline = Pipeline(steps=[('minmax_scaler', MinMaxScaler()), \
    ('feature_selection', skbest), (classifier_type, adaboost_classifier)])
    
    param_grid = {"base_estimator__criterion" : ["gini", "entropy"],
              "base_estimator__splitter" :   ["best", "random"],
              "n_estimators": [1, 2]
             }
     
	clf = GridSearchCV(adaboost_classifier, param_grid=param_grid, scoring = 'f1')
	
	clf = svm.SVC(kernel = 'rbf', C = 10, gamma=1) 
	
	clf = KNeighborsClassifier(n_neighbors=3 ,   weights = 'uniform',  \
		algorithm='auto', 	p=1, metric ='minkowski', leaf_size=10)
	'''
	# Since the overall purpose of this model is to 
	#identify more - true persons of interest even if some non-persons of interest 
	#are falsely identified as person of interest by the model i.e. 
	# False Negative > False Positive, which means Recall weighs more, 
	# while choosing between two similar comparable algorithms
	# Hence, the final algorithm I chose is Ensemble algorithm AdaBoostClassifier

	clf = AdaBoostClassifier(DecisionTreeClassifier(criterion='gini', max_depth=1, 
	min_samples_leaf=2, 	class_weight='balanced'), \
	n_estimators=50, learning_rate=.8)
		
	# Validate model precision, recall and F1-score
	test_classifier(clf, my_dataset, features_list)

	# Dump classifier, dataset and features_list
	print '\n'
	print '########## Dump Classifiers, dataset and features_list ##########'
	# Dump your classifier, dataset, and features_list so anyone can
	# check your results. You do not need to change anything below, but make sure
	# that the version of poi_id.py that you submit can be run on its own and
	# generates the necessary .pkl files for validating your results.
	dump_classifier_and_data(clf, my_dataset, features_list)
	print 'Successfully created clf, my_dataset and features_list pkl files'

	# References
	print '########## References ##########'
	print '#https://www.analyticsvidhya.com/blog/2015/11/quick-introduction-boosting-algorithms-machine-learning/ \n ' \
		      'https://civisanalytics.com/blog/data-science/2016/01/06/workflows-python-using-pipeline-gridsearchcv-for-compact-code/ \n' \
		      'http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html \n' \
		      'http://scikit-learn.org/stable/modules/pipeline.html \n' \
	'https://civisanalytics.com/blog/data-science/2015/12/17/workflows-in-python-getting-data-ready-to-build-models/'

#######################################################
# Main ends.
######################################################
