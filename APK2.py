#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[7]:


from fastapi import FastAPI
import uvicorn
from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB
from pydantic import BaseModel


# In[8]:


import joblib


# In[9]:


# Load the saved logistic regression model
logreg_model = joblib.load("logistic_regression_model.joblib")

# Load the saved KNN model
knn_model = joblib.load("knn_model.joblib")

# Load the saved random forest model
rf_model = joblib.load("random_forest_model.joblib")

# Load the saved SVM model
svm_model = joblib.load("svm_model.joblib")


# In[10]:


# uvicorn APK2:app --reload


# In[11]:


# Creating FastAPI instance
app = FastAPI()

# Creating class to define the request body
# and the type hints of each attribute
class request_body(BaseModel):
    sepal_length : float
    sepal_width : float
    petal_length : float
    petal_width : float

# Loading Iris Dataset
iris = load_iris()

# Getting our Features and Targets
X = iris.data
Y = iris.target

# Creating and Fitting our Model
clf = GaussianNB()
clf.fit(X,Y)

# Creating an Endpoint to receive the data
# to make prediction on.

@app.post('/Predict Logarithmic Regression Classifiers')
def predict_logreg(data : request_body):
     # Making the data in a form suitable for prediction
    test_data = [[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]]

    # Predicting the Class
    class_idx = logreg_model.predict(test_data)[0]

    # Return the Result
    return { 'prediction' : iris.target_names[class_idx]}


@app.post('/Predict KNN Classifiers')
def predict_knn(data : request_body):
     # Making the data in a form suitable for prediction
    test_data = [[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]]

    # Predicting the Class
    class_idx = knn_model.predict(test_data)[0]

    # Return the Result
    return { 'prediction' : iris.target_names[class_idx]}


@app.post('/Predict Random Forrest Classifiers')
def predict_rf(data : request_body):
     # Making the data in a form suitable for prediction
    test_data = [[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]]

    # Predicting the Class
    class_idx = rf_model.predict(test_data)[0]

    # Return the Result
    return { 'prediction' : iris.target_names[class_idx]}


@app.post('/Predict SVM Classifiers')
def predict_svm(data : request_body):
     # Making the data in a form suitable for prediction
    test_data = [[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]]

    # Predicting the Class
    class_idx = svm_model.predict(test_data)[0]

    # Return the Result
    return { 'prediction' : iris.target_names[class_idx]}


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




