# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Read the CSV file into a DataFrame
df = pd.read_csv("/kaggle/input/icr-identify-age-related-conditions/train.csv")

# Fill missing values with the mean
df.fillna(df.mean(), inplace=True)

# Split the data into input features (x) and labels (y)
# and drop non-numeric values
x = df.drop(['Id', 'Class', 'EJ'], axis=1)
y = df['Class']

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=0)

# Define the model architecture
model = Sequential() 
model.add(Dense(128, activation='relu', input_dim=len(x.columns)))
model.add(Dense(1, activation='sigmoid')) 
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) 
model.summary()

# Train the model
hist = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=20, batch_size=100)

# Print training accuracy
print(hist.history['accuracy'])

# Print training loss
print(hist.history['loss'])

# Print validation loss
print(hist.history['val_loss'])

# Print validation accuracy
print(hist.history['val_accuracy'])

# Plot training and validation accuracy
sns.set()
acc = hist.history['accuracy']
val = hist.history['val_accuracy']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, '-', label='Training accuracy')
plt.plot(epochs, val, ':', label='Validation accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()

# Predict labels for the test data
y_predicted = model.predict(x_test) > 0.5

# Create a confusion matrix
mat = confusion_matrix(y_test, y_predicted)
labels = ['diagnosable', 'not diagnosable']

# Plot the confusion matrix
sns.heatmap(mat, square=True, annot=True, fmt='d', cbar=False, cmap='Blues',
            xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted label')
plt.ylabel('Actual label')


# Read the test data from a CSV file
test_df = pd.read_csv("/kaggle/input/icr-identify-age-related-conditions/test.csv")

# Fill missing values with the mean
test_df.fillna(df.mean(), inplace=True)

# Extract the features (input data) from the test data
test_x = test_df.drop(['Id', 'EJ'], axis=1)

# Predict the labels for the test data using the trained model
predictions = model.predict(test_x) 

# Print the predicted labels
print(predictions)

# Round the probability values to one decimal place
rounded_predictions = np.round(predictions.flatten(), decimals=1)
rounded_class_0 = np.round(1 - predictions.flatten(), decimals=1)

# Create a DataFrame for the rounded predictions
#pred_df = pd.DataFrame({'Id': test_df['Id'], 'class_0': rounded_class_0, 'class_1': rounded_predictions})
#Implementing suggested solution for how to write submission file
sample = pd.read_csv('/kaggle/input/icr-identify-age-related-conditions/sample_submission.csv')
sample['class_1'] = rounded_predictions
sample['class_0'] = rounded_class_0
sample.to_csv('submission.csv', index = False)
# Write the predictions to a CSV file
#pred_df.to_csv('submission.csv', index=False)
