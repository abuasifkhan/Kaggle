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

# Get Trainset
X = dataset.iloc[:, range(2,len(dataset.columns))].values
y = dataset.iloc[:, 1].values


# Taking care of missing data
from sklearn.preprocessing import Imputer
ageImputerTrain = Imputer(missing_values=np.nan,strategy = 'mean', axis=0)
X[:,2] = ageImputerTrain.fit_transform(X[:, 2].reshape(-1,1)).flatten()


# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
sexEncoder = LabelEncoder()
X[:,1] = sexEncoder.fit_transform(X[:,1])

embarkedEncoder = LabelEncoder()
X[:,6] = embarkedEncoder.fit_transform(X[:,6])


## Feature Scaling
#from sklearn.preprocessing import StandardScaler
#ageScale = StandardScaler()
#X[:,2] = ageScale.fit_transform(X[:,2].reshape(-1,1)).flatten()

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test_, y_train, y_test_ = train_test_split(X, y, test_size = 0.00, random_state = 0)


# Fitting Kernel SVM to the Training set
#from sklearn.svm import SVC
#classifier = SVC(kernel = 'rbf', random_state = 0)
#classifier.fit(X_train, y_train)


#from sklearn.ensemble import RandomForestClassifier
#classifier = RandomForestClassifier(n_estimators = 1000, criterion = 'entropy', random_state = 0)
#classifier.fit(X_train, y_train)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 1)
classifier.fit(X_train, y_train)

'''## Predicting the Test set results
y_pred_ = classifier.predict(X_test_)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test_, y_pred_)



'''


# Importing the test dataset
testset = pd.read_csv('test.csv')


# Drop useless tables
del testset['Name']
del testset['Cabin']
del testset['Ticket']

# Get testset
X_test = testset.iloc[:,range(1,len(testset.columns))].values

# Taking care of missing data
X_test[:,2] = ageImputerTrain.transform(X_test[:,2].reshape(-1,1)).flatten()

fareImputer = Imputer(missing_values=np.nan,strategy = 'mean', axis=0)
X_test[:,5] = fareImputer.fit_transform(X_test[:, 5].reshape(-1,1)).flatten()

# Encoding categorical data
# Encoding the Independent Variable
X_test[:,1] = sexEncoder.transform(X_test[:,1])
X_test[:,6] = embarkedEncoder.transform(X_test[:,6])

## Feature Scaling
#X_test[:,2] = ageScale.transform(X_test[:,2].reshape(-1,1)).flatten()


# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Submission csv Generate
submission = pd.DataFrame({
        'PassengerId':testset['PassengerId'],
        "Survived": y_pred
        })
submission.to_csv('titanic.csv',index=False)













