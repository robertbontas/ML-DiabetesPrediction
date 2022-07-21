import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

# loading the dataset of diabetes to a pandas DataFrame: 
diabetes_db=pd.read_csv('D:\Programming\machine learning\diabetes prediction\diabetes.csv')
print("Database is loading...")
print("------------------------------------------------------------------------------------------------------------------")
# printing the first 5 rows of the diabetes dataset:
print('First 5 rows of the diabetes database:')
print(diabetes_db.head())
print("------------------------------------------------------------------------------------------------------------------")

#shape of the database:
print('The shape of the dataset:') 
print(diabetes_db.shape)
print("------------------------------------------------------------------------------------------------------------------")
print("Statistical measures of the data:")
print(diabetes_db.describe())
print("------------------------------------------------------------------------------------------------------------------")
print('0 -> Non-Diabetic | 1 -> Diabetic')
print(diabetes_db['Outcome'].value_counts())
print("------------------------------------------------------------------------------------------------------------------")
print("Group subjects by outcome and get the mean:")
print(diabetes_db.groupby('Outcome').mean())

print("------------------------------------------------------------------------------------------------------------------")
#Separating the data and labels.
# X will be the rest of dataset
X=diabetes_db.drop(columns='Outcome',axis=1)
# Y will be just the outcome column
Y=diabetes_db["Outcome"]
print("------------------------------------------------------------------------------------------------------------------")
# Let's do Data Standardization with a scaler.
scaler=StandardScaler()
scaler.fit(X) # .fit() computes the mean and std for later scalling
standardized_data=scaler.transform(X) # .transform() perform standardization and centering 
X=standardized_data
# Train test split: 
X_train, X_test, Y_train,Y_test=train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=2)
print(X.shape,X_train.shape, X_test.shape)
#Training the model:
classifier=svm.SVC(kernel='linear')
classifier.fit(X_train,Y_train)
#Model evaluation:


#accuracy score on the training data:
X_train_pred=classifier.predict(X_train)
training_data_accuracy=accuracy_score(X_train_pred,Y_train)
print('Accuracy score of the training data : ', training_data_accuracy)

#accuracy score on the test data:
X_test_pred=classifier.predict(X_test)
test_data_accuracy=accuracy_score(X_test_pred,Y_test)
print('Accuracy score of the test data : ', test_data_accuracy)

#Making a Predictive System:
input_data = (5,166,72,19,175,25.8,0.587,51)
# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

# standardize the input data
std_data = scaler.transform(input_data_reshaped)
print(std_data)

prediction = classifier.predict(std_data)
print(prediction)

if (prediction[0] == 0):
  print('The person is not diabetic')
else:
  print('The person is diabetic')