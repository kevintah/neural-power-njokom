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

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
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

# Scale the data using StandardScaler
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

# Define the model architecture
model = Sequential()
model.add(Dense(128, activation='relu', input_dim=len(x.columns)))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# Perform KFold cross-validation
kf = KFold(n_splits=5, random_state=0, shuffle=True)

best_val_loss = float('inf')
best_epoch = 0

for train_index, val_index in kf.split(x_scaled):
    x_train, x_val = x_scaled[train_index], x_scaled[val_index]
    y_train, y_val = y[train_index], y[val_index]

    # Define the model architecture
    model = Sequential()
    model.add(Dense(128, activation='relu', input_dim=len(x.columns)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    # Define early stopping callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, mode='min', restore_best_weights=True)

    # Train the model
    hist = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=100, batch_size=100, callbacks=[early_stopping])
    
    # Track the best epoch based on validation loss
    if np.min(hist.history['val_loss']) < best_val_loss:
        best_val_loss = np.min(hist.history['val_loss'])
        best_epoch = np.argmin(hist.history['val_loss']) + 1

# Train the final model with the best epoch
model.fit(x_scaled, y, epochs=best_epoch, batch_size=100)

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

# Load the test data
test_df = pd.read_csv("/kaggle/input/icr-identify-age-related-conditions/test.csv")

# Fill missing values with the mean
test_df.fillna(df.mean(), inplace=True)

# Scale the test data using StandardScaler
test_x_scaled = scaler.transform(test_df.drop(['Id', 'EJ'], axis=1))

# Predict the probabilities for the test data using the trained model
probabilities = model.predict(test_x_scaled)

# Create a DataFrame for the predictions
sample = pd.read_csv('/kaggle/input/icr-identify-age-related-conditions/sample_submission.csv')
sample['class_1'] = probabilities
sample['class_0'] = 1 - probabilities
sample.to_csv('submission.csv', index=False)
