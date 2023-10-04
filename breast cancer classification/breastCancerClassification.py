import pandas as pd
import numpy as np
import sklearn.datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# breast_cancer_dataset = pd.read_csv("breast cancer.csv")
breast_cancer_dataset = sklearn.datasets.load_breast_cancer()

## loading the data to a data frame
data_frame = pd.DataFrame(breast_cancer_dataset.data, columns= breast_cancer_dataset.feature_names)

## adding the 'target' column to the data frame
data_frame['label'] = breast_cancer_dataset.target

## number of rows and columns in the dataset
data_frame.shape

## getting some information about the data
# data_frame.info()

## checking for missing values
data_frame.isnull().sum()

## statistical measures about the data
data_frame.describe()

## checking the distribution of the target variable
### 1 - > Benign - do not spread and not risky
### 0 - > Malignant - spread in the body and very risky
data_frame['label'].value_counts()

# data_frame.groupby('label').mean()

## separating the features and target
x = data_frame.drop(columns='label', axis=1)
y = data_frame['label']

## spliting the data into training data and testing data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)

## (569, 30) (455, 30) (114, 30)
### (569, 30) -> it is all the data (x.shape)
### (455, 30) -> it is the training data (x_train.shape)
### (144, 30) -> it is the testing data (x_test.shape)
#print(x.shape, x_train.shape, x_test.shape)

## model training
## logistic regression
model = LogisticRegression()

## training the logistic regression model using training data
model.fit(x_train, y_train)

## model evaluation
## using accuracy score
### first evaluate the model using the training data (x_train)
### accuracy on training data
x_train_prediction = model.predict(x_train)
### y_train is the true value and x_train_prediction is the predicted value by the model
training_data_accuracy = accuracy_score(y_train, x_train_prediction)
print('Accuracy on training data = ', training_data_accuracy)

### second evaluate the model using the test data (x_test)
### accuracy on test data
x_test_prediction = model.predict(x_test)
test_data_accuracy = accuracy_score(y_test, x_test_prediction)
print('Accuracy on test data = ', test_data_accuracy)

## bulding a predictive system
input_data = (11.42, 20.38, 77.58, 386.1, 0.1425, 0.2839, 0.2414, 0.1052, 0.2597,
              0.09744, 0.4956, 1.156, 3.445, 27.23, 0.00911, 0.07458, 0.05661, 0.01867,
              0.05963, 0.009208, 14.91,	26.5, 98.87, 567.7,	0.2098,	0.8663,	0.6869,
              0.2575, 0.6638, 0.173)

input_data_1 = (11.52, 18.75, 73.34, 409, 0.09524, 0.05473,	0.03036, 0.02278, 0.192, 0.05907,
                0.3249,	0.9591,	2.183,	23.47,	0.008328,	0.008722,	0.01349, 0.00867,
                0.03218, 0.002386,	12.84,	22.47,	81.81,	506.2,	0.1249,	0.0872,	0.09076,
                0.06316, 0.3306, 0.07036)

### change the input data to a numpy array
input_data_as_numpy_array = np.asarray(input_data_1)

### reshape the numpy array as we are predicting for one datapoint
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

prediction = model.predict(input_data_reshaped)

print(prediction)

if (prediction[0] == 0):
    print('The braest cancer is Malignant')
else:
    print('The breast cancer is Benign')

































