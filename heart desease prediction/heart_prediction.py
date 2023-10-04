import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

## data collection and processing
heart_data = pd.read_csv('heart_disease_data.csv')

## checking for missing values
# print(heart_data.isnull().sum())

## checking the distribution of the target variable
### if target == 1 -> has heart disease
### if target == 0 -> no heart disease
# print(heart_data['target'].value_counts())

## splitting the features and target
x = heart_data.drop(columns='target', axis=1)
y = heart_data['target']

## splitting data into testing and training data
### stratify - distribute data into 1 and 0 so that 
### the model does not choose 1's only or 0's only when training and testing
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=2) 
# print(x.shape, x_train.shape, x_test.shape)

## model training
### logisting regression
model = LogisticRegression()

## training the logisticRegression model with the training data
model.fit(x_train, y_train)

## model evaluation
### accuracy score
### accuracy on the training data
x_train_prediction = model.predict(x_train)
training_data_accuracy = accuracy_score(y_train, x_train_prediction)
print('Accuracy on training data: ', training_data_accuracy)

### accuracy on the test data
x_test_prediction = model.predict(x_test)
training_data_accuracy = accuracy_score(y_test, x_test_prediction)
print('Accuracy on test data: ', training_data_accuracy)

## building the predictive system
input_data = (56,0,1,140,294,0,0,153,0,1.3,1,0,2)
input_data_1 = (55,1,0,140,217,0,1,111,1,5.6,0,0,3)

## change input data to a numpy array
input_data_as_numpy_array = np.asarray(input_data_1)

## reshape the numpy array as we are predicting for only one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

prediction = model.predict(input_data_reshaped)

print(prediction)

if prediction[0] == 0:
    print('The person does not have heart disease')
else:
    print('The person have heart disease')

## saving the trained model
import pickle
filename = 'heart_disease_model.sav'
pickle.dump(model, open(filename, 'wb'))

## loading the saved model
loaded_model = pickle.load(open('heart_disease_model.sav', 'rb'))
    
## building the predictive system
input_data = (56,0,1,140,294,0,0,153,0,1.3,1,0,2)
input_data_1 = (55,1,0,140,217,0,1,111,1,5.6,0,0,3)

## change input data to a numpy array
input_data_as_numpy_array = np.asarray(input_data_1)

## reshape the numpy array as we are predicting for only one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

prediction = model.predict(input_data_reshaped)

print(prediction)

if prediction[0] == 0:
    print('The person does not have heart disease')
else:
    print('The person have heart disease')


















