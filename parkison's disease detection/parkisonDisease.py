import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import accuracy_score

## data collection and analysis
### loading the data from csv file to a pandas dataframe
parkinsons_data = pd.read_csv('parkinsons.csv')
# print(parkinsons_data.head(5))

# ### number of rows and columns in the data frame
# parkinsons_data.shape

# ### getting more info about the date frame
# parkinsons_data.info()

# ### checking for missing values in each column
# print(parkinsons_data.isnull().sum())

# ### getting some statistical measures about the data
# parkinsons_data.describe()

# ## distribution of target variable
# ### status == 1 -> has parkinson
# ### status == 0 -> no parkinson
# parkinsons_data['status'].value_counts()
# 
# ## grouping the data based on the target variable
# parkinsons_data.groupby('status').mean()

## data pre-processing
### separating the features and target
#### dropping a column axis == 1
#### dropping a row axis == 0
x = parkinsons_data.drop(columns=['name', 'status'], axis=1)
y = parkinsons_data['status']

## spliting the data into training data and tets data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)
# print(x.shape, x_train.shape, x_test.shape)

## data standardization
scaler = StandardScaler()
scaler.fit(x_train)

### convert data into same scale
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

## model training
### support vector machine learning model
#### support vector machine works by separating values by they classification
#### here it will work by separating positive parkinson and negative parkisons
model = svm.SVC(kernel='linear')  ### support vector classifier

## training the svm with training data
model.fit(x_train, y_train)

## model evaluation
### accuracy score
#### accuracy score on training data
x_train_prediction = model.predict(x_train)
training_data_accuracy = accuracy_score(y_train, x_train_prediction)
print('Accuracy score of training data = ', training_data_accuracy)

#### accuracy score on test data
x_test_prediction = model.predict(x_test)
training_data_accuracy = accuracy_score(y_test, x_test_prediction)
print('Accuracy score of test data = ', training_data_accuracy)

## building a predictive system
input_data = (88.33300,112.24000,84.07200,0.00505,0.00006,0.00254,0.00330,0.00763,0.02143,
              0.19700,0.01079,0.01342,0.01892,0.03237,0.01166,21.11800,
              0.611137,0.776156,-5.249770,0.391002,2.407313,0.249740)
input_data_1 = (197.07600,206.89600,192.05500,0.00289,0.00001,0.00166,0.00168,
                0.00498,0.01098,0.09700,0.00563,0.00680,0.00802,0.01689,0.00339,
                26.77500,0.422229,0.741367,-7.348300,0.177551,1.743867,0.085569)
## changing input data into a numpy array
input_data_as_numpy_array = np.asarray(input_data_1)

## reshape the numpy array
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

## standardize the data
std_data = scaler.transform(input_data_reshaped)

## now predict
prediction = model.predict(std_data)

print(prediction)

if (prediction[0] == 0):
    print('The person does not have Parkisons Disease')
else:
    print('The person has Parkinsons Disease')


## saving the trained model
import pickle
filename = 'parkison.sav'
pickle.dump(model, open(filename, 'wb'))

## loading the saved model
loaded_model = pickle.load(open('parkison.sav', 'rb'))
  

## building a predictive system
input_data = (88.33300,112.24000,84.07200,0.00505,0.00006,0.00254,0.00330,0.00763,0.02143,
              0.19700,0.01079,0.01342,0.01892,0.03237,0.01166,21.11800,
              0.611137,0.776156,-5.249770,0.391002,2.407313,0.249740)
input_data_1 = (197.07600,206.89600,192.05500,0.00289,0.00001,0.00166,0.00168,
                0.00498,0.01098,0.09700,0.00563,0.00680,0.00802,0.01689,0.00339,
                26.77500,0.422229,0.741367,-7.348300,0.177551,1.743867,0.085569)
## changing input data into a numpy array
input_data_as_numpy_array = np.asarray(input_data_1)

## reshape the numpy array
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

## standardize the data
std_data = scaler.transform(input_data_reshaped)

## now predict
prediction = model.predict(std_data)

print(prediction)

if (prediction[0] == 0):
    print('The person does not have Parkisons Disease')
else:
    print('The person has Parkinsons Disease')
























