import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import accuracy_score

## data collection and analysis
### loading the data from csv file to a pandas dataframe
diabetes_data = pd.read_csv('diabetes.csv')
# print(diabetes_data.head(5))

### number of rows and columns in the data frame
# diabetes_data.shape

# # ### getting more info about the date frame
# # diabetes_data.info()

# # ### checking for missing values in each column
# diabetes_data.isnull().sum()

### getting some statistical measures about the data
# diabetes_data.describe()

## distribution of target variable
### Outcome == 1 -> diabetic
### Outcome == 0 -> non-diabetic
# diabetes_data['Outcome'].value_counts()

## grouping the data based on the target variable
# diabetes_data.groupby('Outcome').mean()

## data pre-processing
### separating the features and target
#### dropping a column axis == 1
#### dropping a row axis == 0
x = diabetes_data.drop(columns='Outcome', axis=1)
y = diabetes_data['Outcome']

## data standardization
scaler = StandardScaler()
scaler.fit(x)

### convert data into same scale
standardized_data = scaler.transform(x)
x = standardized_data
y = diabetes_data['Outcome']

## spliting the data into training data and tets data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=2)
# print(x.shape, x_train.shape, x_test.shape)

# ## model training
# ### support vector machine learning model
# #### support vector machine works by separating values by they classification
# #### here it will work by separating positive parkinson and negative parkisons
classifier = svm.SVC(kernel='linear')  ### support vector classifier

## training the svm with training data
classifier.fit(x_train, y_train)

## model evaluation
### accuracy score
#### accuracy score on training data
x_train_prediction = classifier.predict(x_train)
training_data_accuracy = accuracy_score(y_train, x_train_prediction)
print('Accuracy score of training data = ', training_data_accuracy)

#### accuracy score on test data
x_test_prediction = classifier.predict(x_test)
training_data_accuracy = accuracy_score(y_test, x_test_prediction)
print('Accuracy score of test data = ', training_data_accuracy)

## building a predictive system
input_data = (6,148,72,35,0,33.6,0.627,50)
input_data_1 = (1,85,66,29,0,26.6,0.351,31)
## changing input data into a numpy array
input_data_as_numpy_array = np.asarray(input_data)

## reshape the numpy array
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

## standardize the data
std_data = scaler.transform(input_data_reshaped)

## now predict
prediction = classifier.predict(std_data)

print(prediction)

if (prediction[0] == 0):
    print('The person does not have Diabetic Disease')
else:
    print('The person has Diabetic Disease')

## saving the trained model
import pickle
filename = 'diabetes_model.sav'
pickle.dump(classifier, open(filename, 'wb'))

## loading the saved model
loaded_model = pickle.load(open('diabetes_model.sav', 'rb'))

## building a predictive system
input_data = (6,148,72,35,0,33.6,0.627,50)
input_data_1 = (1,85,66,29,0,26.6,0.351,31)
## changing input data into a numpy array
input_data_as_numpy_array = np.asarray(input_data_1)

## reshape the numpy array
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

## standardize the data
std_data = scaler.transform(input_data_reshaped)

## now predict
prediction = classifier.predict(std_data)

print(prediction)

if (prediction[0] == 0):
    print('The person does not have Diabetic Disease')
else:
    print('The person has Diabetic Disease')



























