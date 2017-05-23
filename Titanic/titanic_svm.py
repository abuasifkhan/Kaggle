# Kernel SVM

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('train.csv')

# Drop useless columns
del dataset['Name']
del dataset['Ticket']
del dataset['Cabin']


# Replace all the missing values
dataset['Age'] = dataset['Age'].fillna(dataset['Age'].median())
dataset['Embarked'] = dataset['Embarked'].fillna('S')

# Importing testset
testset = pd.read_csv('test.csv')

# Drop useless columns
del testset['Name']
del testset['Ticket']
del testset['Cabin']

# Replace all the missing values
testset['Age'] = testset['Age'].fillna(testset['Age'].median())
testset['Embarked'] = testset['Embarked'].fillna('S')

# Get test data
X_test = testset.iloc[:,1:-1]

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder
embarkedLabelEncoder = LabelEncoder()
embarkedLabelEncoder.fit(pd.concat([dataset['Embarked'],testset['Embarked']] ))
dataset['Embarked'] = embarkedLabelEncoder.transform(dataset['Embarked'])
testset['Embarked'] = embarkedLabelEncoder.transform(testset['Embarked'])

sexLabelEncoder = LabelEncoder()
sexLabelEncoder.fit(pd.concat( [dataset['Sex'],testset['Sex']] ) )
dataset['Sex'] = sexLabelEncoder.transform(dataset['Sex'])
testset['Sex'] = sexLabelEncoder.transform(testset['Sex'])

# The columns we'll use to predict the target
predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

# Get data
X = dataset.iloc[:,2:-1]
y = dataset.iloc[:,1]

## Splitting the dataset into the Training set and Test set
#from sklearn.cross_validation import train_test_split
#X_train, X_test_, y_train, y_test_ = train_test_split(X, y, test_size = 0.0, random_state = 0)

X_train = X
y_train = y

# Fitting Kernel SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_train)

## Predicting the Test set results
#y_pred = classifier.predict(X_test_)





# Predicting the Test set results
y_pred = classifier.predict(X_test)
















