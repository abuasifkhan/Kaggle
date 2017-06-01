# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('train.csv')

# Drop useless tables
del dataset['Name']
del dataset['Cabin']
del dataset['Ticket']


# The columns we'll use to predict the target
predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

# Taking care of missing data
dataset['Age'] = dataset['Age'].fillna(dataset['Age'].mean())
dataset['Embarked'] = dataset['Embarked'].fillna('S')

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
sexEncoder = LabelEncoder()
dataset["Sex"] = sexEncoder.fit_transform(dataset["Sex"])

embarkedEncoder = LabelEncoder()
dataset['Embarked'] = embarkedEncoder.fit_transform(dataset['Embarked'])

# Get Trainset
#X = dataset[predictors].iloc[:,:]
y = dataset['Survived']

# Importing the test dataset
testset = pd.read_csv('test.csv')


# Drop useless tables
del testset['Name']
del testset['Cabin']
del testset['Ticket']

# Taking care of missing data
testset['Fare'] = testset['Fare'].fillna(testset['Fare'].mean())
testset['Age'] = testset['Age'].fillna(testset['Age'].mean())

# Encoding categorical data
# Encoding the Independent Variable
testset['Sex'] = sexEncoder.transform(testset['Sex'])
testset['Embarked'] = embarkedEncoder.transform(testset['Embarked'])

marged = np.append(dataset[predictors], testset[predictors], axis=0)


# Feature Scaling
from sklearn.preprocessing import StandardScaler
ageScale = StandardScaler()
ageScale.fit(marged)
X_train = ageScale.transform(dataset[predictors])
X_test = ageScale.transform(testset[predictors])

# Splitting the dataset into the Training set and Test set
#from sklearn.cross_validation import train_test_split
#X_train, X_test_, y_train, y_test_ = train_test_split(X, y, test_size = 0.00, random_state = 0)


'''
# Fitting Kernel SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf')
'''

'''
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 0)
'''

'''
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 1)
'''
'''
from sklearn import tree
classifier = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth=7)
'''
'''
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
'''

#classifier.fit(dataset[predictors], y)


'''
# Compute the accuracy score for all the cross validation folds.  (much simpler than what we did before!)
from sklearn import cross_validation
scores = cross_validation.cross_val_score(classifier, dataset[predictors], y, cv=10)
# Take the mean of the scores (because we have one for each fold)
print("Accuracy on 10-fold classification: ", scores.mean())
'''


'''## Predicting the Test set results
y_pred_ = classifier.predict(X_test_)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test_, y_pred_)
'''

'''
# Predicting the Test set results
y_pred = classifier.predict(testset[predictors])
'''


# Neural networks try
# Importing the Keras libraries and packages
#import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 8, init = 'uniform', activation = 'relu', input_dim = X_train.shape[1]))

# Adding the second hidden layer
classifier.add(Dense(output_dim = 8, init = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# adam is stocastic gradient descent algorithm
# loss function 'binary_crossentropy' for binary outpur; more than two output 'categorical_crossentropy

# Fitting the ANN to the Training set
classifier.fit(X_train, y, batch_size = 10, nb_epoch = 100)

y_pred = classifier.predict(X_test)
y_pred = (y_pred>0.5)*1


# Compute the accuracy score for all the cross validation folds.  (much simpler than what we did before!)
from sklearn.model_selection import StratifiedKFold
# define 10-fold cross validation test harness
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
cvscores = []
for train, test in kfold.split(X_train, y):
	scores = classifier.evaluate(X_train[test], y[test], verbose=0)
	print("%s: %.2f%%" % (classifier.metrics_names[1], scores[1]*100))
	cvscores.append(scores[1] * 100)
print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))




# Submission csv Generate
print('Making Submission CSV file')
submission = pd.DataFrame({
        'PassengerId':testset['PassengerId'],
        "Survived": y_pred.flatten()
        })
submission.to_csv('titanic.csv',index=False)


## Building the optimal model using Backward Elimination
#import statsmodels.formula.api as sm
#X = np.append(arr = np.ones((dataset[predictors].shape[0], 1)).astype(int), values = dataset[predictors], axis = 1)
#X_opt = X[:, [0,1,2,3,4,5,6]]
#regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
#regressor_OLS.summary()
#X_opt = X[:, [0,1,2,3,4,5]]
#regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
#regressor_OLS.summary()









