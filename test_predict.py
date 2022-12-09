#!/usr/bin/env python
# coding: utf-8
import requests

url = 'http://localhost:3000/classify'

wine = {
  "fixed acidity": 7.8000,
  "volatile acidity": 0.8800,
  "citric acid": 0.0000,
  "residual sugar": 2.6000,
  "chlorides": 0.0980,
  "free sulfur dioxide": 25.0000,
  "total sulfur dioxide": 67.0000,
  "density": 0.9968,
  "pH": 3.2000,
  "sulphates": 0.6800,
  "alcohol": 9.8000,
  "type": 1.0000
}

response = requests.post(url, json=wine).json()
print(response)