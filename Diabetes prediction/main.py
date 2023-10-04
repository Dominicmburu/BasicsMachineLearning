import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

## data collection and analysis
### loading the data from csv file to a pandas dataframe
df = pd.read_csv('diabetes.csv')

# ### checking for missing values in each column
# print(diabetes_data.isnull().sum())

# print(sns.displot(["BloodPressure"]))

# plitting the dataset
x= df.drop(['Outcome'], axis=1)
y=df['Outcome']


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=2)


rf = RandomForestClassifier()

model = rf.fit(x_train,y_train)

y_train_pred = model.predict(x_train)
accuracy_train = accuracy_score(y_train, y_train_pred)

y_test_pred = model.predict(x_test)
accuracy_test = accuracy_score(y_test, y_test_pred)

print("Accuracy on training data:", accuracy_train)
print("Accuracy on test data:", accuracy_test)

