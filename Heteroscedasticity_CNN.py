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

# Plotting x vs y and x vs sigma square (the variance of error)
fig = plt.figure(figsize=(10,5))
plt.subplot(2,1,1);plt.title('Heteroscedasticity')
plt.plot(x, fx,'r.'); plt.plot(x,y,'k.');plt.ylabel('y')
plt.subplot(2,1,2)
plt.plot(x, sigma2,'r.'); plt.ylabel('sigma2');plt.xlabel('x')
plt.savefig('Heteroscedasticity.jpg',bbox_inches='tight')

# To do a prediction, we need to divide data into train and test 
ntest = int(n/4)
test_idx = np.random.choice(n,ntest,replace=False)
train_idx = np.setdiff1d(np.arange(n),test_idx)
ntrain = train_idx.shape[0]

xr = x[train_idx] ; xt = x[test_idx]
fxr = fx[train_idx]; fxt = fx[test_idx]
sigma2r = sigma2[train_idx] ; sigma2t = sigma2[test_idx]
yr = y[train_idx].reshape(ntrain,1); yt = y[test_idx].reshape(ntest,1)


# First, build a standard CNN (loss function with mean square error)

model =  Sequential()
# By setting batch_input_shape is (None,1), the model can handle batches of any size flexibily 
model.add(Dense(20, activation='tanh',batch_input_shape=(None, 1)))  
model.add(Dense(10, activation='tanh'))
model.add(Dense(2, activation='linear'))

# Setting a model environment and fitting 
nepochs = 1000 # You can adjust the number of epochs according to the your CPU ability 
model.compile(loss=losses.mean_squared_error, optimizer="adam", metrics=[losses.mean_squared_error])
model.fit(xr, np.stack((yr,sigma2r.reshape(-1,1)),axis=1).reshape(1500,2), epochs=nepochs,verbose=1)

# Next is a Negative Log Likelihood(NLL) CNN 
# Before building a model, we need to create a customized loss function like below

def NN1_loss(y_true,y_pred):
    mu = tf.slice(y_pred,[0,0],[-1,1])
    sigma = tf.math.exp(tf.slice(y_pred,[0,1],[-1,1]))     
    loss = tf.reduce_sum(tf.square(y_true - mu)/sigma + tf.math.log(sigma),axis=0)
    return loss 

# And same as what we did just moment ago, coding a model.
model_nn1 =  Sequential()
model_nn1.add(Dense(20, activation='tanh',batch_input_shape=(None, 1)))  
model_nn1.add(Dense(10, activation='tanh'))
model_nn1.add(Dense(2, activation='linear'))

model_nn1.compile(loss=NN1_loss,optimizer="adam",metrics=[NN1_loss])
model_nn1.fit(xr, yr, epochs=nepochs,verbose=1)  

# Graph for test data in a standard Neural Network
yth_standard = model.predict(xt) 
sig2th_standard = yth_standard[:,1]

plt.figure(11)
plt.subplot(2,1,1); plt.title('test data: One time')
plt.plot(xt,fxt,'r.',label='true') 
plt.plot(xt,yth_standard[:,0],'b.',label='pred') 
plt.ylabel('yt')
plt.legend(('true', 'pred'))

plt.subplot(2,1,2)
plt.plot(xt,sigma2t,'r.',label='true'); 
plt.plot(xt,sig2th_standard,'b.',label='pred')
plt.ylabel('sigma2t'); plt.xlabel('xt')
plt.legend(('true', 'pred'))

# Graph for test data in a NLL Neural Network
yth_nll = model_nn1.predict(xt) 
sig2th_nll = np.exp(yth_nll[:,1])

plt.figure(11)
plt.subplot(2,1,1); plt.title('test data: One time')
plt.plot(xt,fxt,'r.',label='true') 
plt.plot(xt,yth_nll[:,0],'b.',label='pred') 
plt.ylabel('yt')
plt.legend(('true', 'pred'))

plt.subplot(2,1,2)
plt.plot(xt,sigma2t,'r.',label='true'); 
plt.plot(xt,sig2th_nll,'b.',label='pred')
plt.ylabel('sigma2t'); plt.xlabel('xt')
plt.legend(('true', 'pred'))

# Comparing the performance between MSE CNN and NLL CNN by using RMSE (Root MSE) and MAE 
# RMSE
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print('RMSE for standard : ', RMSE(fxt, yth_standard[:,0])) 
print('RMSE for nll : ', RMSE(fxt, yth_nll[:,0])) 

# MAE 
from sklearn.metrics import mean_absolute_error
print('MAE for standard: ',mean_absolute_error(fxt,yth_standard[:,0]))
print('MAE for nll: ', mean_absolute_error(fxt,yth_nll[:,0]))

# Thank you 
