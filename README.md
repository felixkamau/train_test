# train_test

This repository provides a comprehensive example of implementing the train-test split technique for data preprocessing in machine learning projects. Train-test split is a fundamental step in model development, enabling the assessment of model performance on unseen data. This repository includes detailed code examples and explanations to demonstrate how to split a dataset into training and testing sets using popular Python libraries such as scikit-learn.

```
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
```
Import the required libraries.

Load you prefered dataset: 
```
df = pd.read_csv('carprices.csv')
df
```
Splitting data into training and testing sets.
```
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2,random_state =10)
```
