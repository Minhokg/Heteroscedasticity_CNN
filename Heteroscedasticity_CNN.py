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

# To do a prediction, we need to divide data into train and test 
ntest = int(n/4)
test_idx = np.random.choice(n,ntest,replace=False)
train_idx = np.setdiff1d(np.arange(n),test_idx)
ntrain = train_idx.shape[0]

xr = x[train_idx] ; xt = x[test_idx]
fxr = fx[train_idx]; fxt = fx[test_idx]
sigma2r = sigma2[train_idx] ; sigma2t = sigma2[test_idx]
yr = y[train_idx].reshape(ntrain,1); yt = y[test_idx].reshape(ntest,1)

# Plotting train data (y and variance) 
plt.figure(1)
plt.subplot(2,1,1);plt.title('training data: given')
plt.plot(xr, fxr,'r.'); plt.plot(xr,yr,'k.');plt.ylabel('yr')
plt.subplot(2,1,2)
plt.plot(xr, sigma2r,'r.'); plt.ylabel('sigma2r');plt.xlabel('xr')

# Also plotting test data (y and variance)

plt.figure(2)
plt.subplot(2,1,1); plt.title('test data: given')
plt.plot(xt, fxt,'r.'); plt.plot(xt,yt,'k.');plt.ylabel('yt')
plt.subplot(2,1,2)
plt.plot(xt, sigma2t,'r.'); plt.ylabel('sigma2t');plt.xlabel('xt')

# First, build a standard CNN 

model =  Sequential()
model.add(Dense(20, activation='tanh',batch_input_shape=(None, 1)))  # tanh
model.add(Dense(10, activation='tanh'))
model.add(Dense(2, activation='linear'))
