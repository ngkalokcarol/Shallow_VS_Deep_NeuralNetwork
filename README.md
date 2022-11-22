# Shallow_VS_Deep_NeuralNetwork


Shallow V.S Deep Neural Network
a. Generate the simulated data first using following equation. Sample 120k data as X from uniform distribution [-2Pi, 2Pi], then feed the sampled X into the equation to get Y. Randomly select 60K as training and 60 K as testing.

b. Train 3 versions of Neural Network, with different numbers of hidden layer (NN with 1 hidden layer, 2 hidden layers and 3 hidden layers), using Mean squared error as objective function and error measurement.

c. For each version, try different number of neurals in your NN and replicate the following left plot (source: https://ojs.aaai.org/index.php/AAAI/article/view/10913). (You don’t need to replicate exactly same results below but need to show the performance difference of 3 versions of Neural Networks)


#### Loading Required Libraries

```python
pip install --upgrade pip

import numpy as np #Data Manipulation and Linear Algebra
import pandas as pd #Dataframe operations
import math #Math Functions
from tensorflow import keras #Deep Learning
from keras.models import Sequential
from statistics import mean
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.metrics import mean_squared_error
import warnings
import matplotlib.pyplot as plt # Plotting Graphs
from sklearn.model_selection import cross_val_score, train_test_split # Cross Validation and Splitting Test/Tra
from sklearn.model_selection import GridSearchCV # Parameter Tuning
import warnings
warnings.filterwarnings('ignore') # Ignore warnings
```

#### Preparing the Data

```python
pi = math.pi
data = np.random.uniform(-2*pi,2*pi,120000)
data
df = pd.DataFrame()
df['X'] = data
df.head()
```
![Screen Shot 2022-11-22 at 9 52 34 AM](https://user-images.githubusercontent.com/50436546/203359893-9d2c6583-1487-4590-b984-de8274f392a1.png)

```python
# Using the function to map data according to given equation

df['y'] = df['X'].map(lambda x: ((((2*(math.cos(x)**2))-1))**2)*2)-1
df.head()
```

![Screen Shot 2022-11-22 at 9 52 54 AM](https://user-images.githubusercontent.com/50436546/203359981-14026876-673e-4596-824d-4206c3e83272.png)

```python
plt.scatter(X, y, color='blue')
plt.show()
```
![Screen Shot 2022-11-22 at 9 53 09 AM](https://user-images.githubusercontent.com/50436546/203360039-09b09c42-e6c5-47c8-90ff-48b9c7d05a28.png)

#### Preparing Test-Train Sets
```python
X = df['X']
y = df['y']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
```
#### Defining the Models
```python
# List for using different number of neurons

neurons = [20,40,60,80]
# Shallow Model

def model_1(unit):
    
    model1 = Sequential()
    model1.add(Dense(units=unit, activation ='relu', input_dim=1))
    model1.add(Dense(units=1))
    model1.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse'])
    return model1
    
# Medium Model

def model_2(unit):
    model2 = Sequential()
    model2.add(Dense(units=unit/2, activation='relu', input_dim=1))
    model2.add(Dense(units=unit/2, activation ='relu', input_dim=1))
    model2.add(Dense(units=1))
    model2.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse'])
    return model2
    
# Deep Model

def model_3(unit):
    model3 = Sequential()
    model3.add(Dense(units=round(unit/3), activation='relu', input_dim=1))
    model3.add(Dense(units=round(unit/3), activation ='relu', input_dim=1))
    model3.add(Dense(units=round(unit/3) + 1, activation ='relu', input_dim=1))
    model3.add(Dense(units=1))
    model3.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse'])
    return model3
    
error_shallow = []
error_medium = []
error_deep = []
units = []
```

#### Running the models with different number of Neurons

```python
for neuron in neurons:
    
    model_shallow = model_1(neuron)
    model_medium = model_2(neuron)
    model_deep = model_3(neuron)
    
    #Running with verbose = 0 for a concise output for submitting
    result1 = model_shallow.fit(X_train,y_train, epochs=50, batch_size=100, verbose=0)
    result2 = model_medium.fit(X_train,y_train, epochs=50, batch_size=100, verbose=0)
    result3 = model_deep.fit(X_train,y_train, epochs=50, batch_size=100, verbose=0)
    
    y_pred_shallow = model_shallow.predict(X_test)
    y_pred_medium = model_medium.predict(X_test)
    y_pred_deep = model_deep.predict(X_test)
    
    mse_shallow = mean_squared_error(y_test, y_pred_shallow)
    mse_medium = mean_squared_error(y_test, y_pred_medium)
    mse_deep = mean_squared_error(y_test, y_pred_deep)
    
    error_shallow.append(mse_shallow)
    error_medium.append(mse_medium)
    error_deep.append(mse_deep)
    
    units.append(neuron)
```


#### Comparing the different models (Shallow V.S. Deep)

Below is the graph replicated for the models performance with MSE

```python
plt.plot(units,error_shallow,label='1 Layer')
plt.plot(units,error_medium,label='2 Layers')
plt.plot(units,error_deep,label='3 Layers')
plt.legend(loc='upper right')
plt.ylabel('MSE')
plt.xlabel('Units')
plt.show()
```

![Screen Shot 2022-11-22 at 9 53 49 AM](https://user-images.githubusercontent.com/50436546/203360203-c8d56041-c333-4c6f-8071-31b40e29e254.png)
