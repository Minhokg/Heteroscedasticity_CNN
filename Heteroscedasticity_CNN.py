import numpy as np
import matplotlib.pyplot as plt 
import tensorflow as tf 
from tensorflow.keras.layers import Dense, Input, Concatenate
from tensorflow.keras.models import Model, Sequential 

# generate n sample data
np.random.seed(0)
n = 2000
x = np.linspace(0, np.pi/2, n)
mx = np.sin(4*x)
fx = mx*np.sin(5*x) # function part of x 
sigma2 = 0.02 + 0.02*(1-mx)**2
error = np.random.normal(0,np.sqrt(sigma2),n)
y = fx + error 

