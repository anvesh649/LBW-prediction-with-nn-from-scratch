import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

"""
In this neural network there are 5 layers
1-input 3-hidden 1-output
activation function is sigmoid in all layers
binary_cross entropy cost function is used
learing rate is 0.03 in this network
adam optimizer is used to achieve the results quickly(less epochs)
epochs are 300
weight matrixes dimensions
w0-(512,9),w1=(256,512),w2=(128,256),w3=(1,128)
bias matrixes dimensions
b0-(512,1),b1=(256,1),b2=(128,1),b3=(1,1)
we used adam optimizer because it uses momentum and adaptive learning rates to converge faster
beta1=0.99,beta2=0.999,eps=1e-8
"""

class NN():
    #initializing the number of layers
    def __init__(self):
        self.param={}
        self.L=5
    #initializing weight,bias and adam hyperparameters
    def initialization(self):
        np.random.seed(2)
        for i in range(self.L-1):
            self.param["w"+str(i)]=np.random.randn(self.n[i+1],self.n[i])
            self.param["b"+str(i)]=np.random.randn(self.n[i+1],1)
            self.param["mw"+str(i)]=0
            self.param["mb"+str(i)]=0
            self.param["vw"+str(i)]=0
            self.param["vb"+str(i)]=0
        self.param["t"]=1
    #sigmoid_activation
    def sigmoid(self,z):
        return 1/(1+np.exp(-z))
    #derivative of sigmoid
    def sigmoid_der(self,z):
        return self.sigmoid(z)*self.sigmoid(1-z)
    #forward propagation of neural network
    def feedforward(self,X):
          self.param["z"+str(0)]=np.dot(self.param["w"+str(0)],X.T)+self.param["b"+str(0)]
          self.param["A"+str(0)]=self.sigmoid(self.param["z"+str(0)])
          for i in range (1,self.L-1):
              self.param["z"+str(i)]=np.dot(self.param["w"+str(i)],self.param["A"+str(i-1)])+self.param["b"+str(i)]
              self.param["A"+str(i)]=self.sigmoid(self.param["z"+str(i)])
    #cost function-binary cross entropy
    def cost(self,Y,A):
        m=Y.shape[0]
        logprobs = np.multiply(np.log(A), Y.T) + np.multiply((1 - Y.T), np.log(1 - A))
        cost= -np.sum(logprobs)/m
        return cost
    #backpropagation of neural network
    def backprop(self,X,Y):
        m=Y.shape[0]
        self.param['dz'+str(self.L-2)]=self.param['A'+str(self.L-2)]-Y.T
        self.param['dw'+str(self.L-2)]=np.dot(self.param['dz'+str(self.L-2)],self.param['A'+str(self.L-3)].T)/m
        self.param['db'+str(self.L-2)]=np.sum(self.param['dz'+str(self.L-2)],axis=1,keepdims=True)/m
        for i in reversed(range(1,self.L-2)):
            self.param['dz'+str(i)]=np.dot(self.param['w'+str(i+1)].T,self.param['dz'+str(i+1)])*self.sigmoid_der(self.param['z'+str(i)])
            self.param['dw'+str(i)]=np.dot(self.param['dz'+str(i)],self.param['A'+str(i-1)].T)/m
            self.param['db'+str(i)]=np.sum(self.param['dz'+str(i)],axis=1,keepdims=True)/m
        self.param['dz'+str(0)]=np.dot(self.param['w'+str(1)].T,self.param['dz'+str(1)])*self.sigmoid_der(self.param['z'+str(0)])
        self.param['dw'+str(0)]=np.dot(self.param['dz'+str(0)],X)/m
        self.param['db'+str(0)]=np.sum(self.param['dz'+str(0)],axis=1,keepdims=True)/m  
    #updating weights with adam optimizer
    def update(self,lr,beta1=0.99,beta2=0.999,eps=1e-8):
        for i in range(self.L-1):
            
            self.param["mw"+str(i)]=beta1*self.param["mw"+str(i)]+(1-beta1)*self.param["dw"+str(i)]
            self.param["mb"+str(i)]=beta1*self.param["mb"+str(i)]+(1-beta1)*self.param["db"+str(i)]
            
            self.param["vw"+str(i)]=beta2*self.param["vw"+str(i)]+(1-beta2)*self.param["dw"+str(i)]*self.param["dw"+str(i)]
            self.param["vb"+str(i)]=beta2*self.param["vb"+str(i)]+(1-beta2)*self.param["db"+str(i)]*self.param["db"+str(i)]
            
            self.param["hat_mw"+str(i)]=self.param["mw"+str(i)]/(1-beta1**self.param["t"])
            self.param["hat_mb"+str(i)]=self.param["mb"+str(i)]/(1-beta1**self.param["t"])
    
            self.param["hat_vw"+str(i)]=self.param["vw"+str(i)]/(1-beta2**self.param["t"])
            self.param["hat_vb"+str(i)]=self.param["vb"+str(i)]/(1-beta2**self.param["t"])
            
            self.param["t"]+=1
            
            self.param['w'+str(i)]=self.param['w'+str(i)]-lr*(self.param["hat_mw"+str(i)]/np.sqrt(self.param["hat_vw"+str(i)]+eps))
            self.param['b'+str(i)]=self.param['b'+str(i)]-lr*(self.param["hat_mb"+str(i)]/np.sqrt(self.param["hat_vb"+str(i)]+eps))
    #training the neural network using forward,backprop and update functions with 300 epochs
    def fit(self,X,y,epochs=300,lr=0.03):
        self.n=[X.shape[1],512,256,128,y.shape[1]]#neurons in each layer of neural networks
        self.initialization()
        for i in range(epochs):
            self.feedforward(X)
            self.backprop(X,y)    
            self.update(lr)  
    #forward propagation with updated weights and biases of neural network
    def predict(self,X):
        self.feedforward(X)
        yhat=self.param["A"+str(self.L-2)]
        return yhat[0]
    #prints the accuracy of neural network by comparing predicted y with true y
    def accuracy(self,y,y_obs):
        y_obs=y_obs.reshape(y_obs.shape[0],1)
        for i in range(len(y_obs)):
            if(y_obs[i]>0.6):
                y_obs[i]=1
            else:
                y_obs[i]=0
        print("Accuracy:",np.sum(y==y_obs)/len(y_obs),"\n")
    #prints the confusion matrix,precision,recall,f1 score of predicted y
    def CM(self,y_test,y_test_obs):
        for i in range(len(y_test_obs)):
            if(y_test_obs[i]>0.6):
                y_test_obs[i]=1
            else:
                y_test_obs[i]=0
        cm=[[0,0],[0,0]]
        fp=0
        fn=0
        tp=0
        tn=0
        for i in range(len(y_test)):
            if(y_test[i]==1 and y_test_obs[i]==1):
                tp=tp+1
            if(y_test[i]==0 and y_test_obs[i]==0):
                tn=tn+1
            if(y_test[i]==1 and y_test_obs[i]==0):
                fp=fp+1
            if(y_test[i]==0 and y_test_obs[i]==1):
                fn=fn+1
        cm[0][0]=tn
        cm[0][1]=fp
        cm[1][0]=fn
        cm[1][1]=tp

        p= tp/(tp+fp)
        r=tp/(tp+fn)
        f1=(2*p*r)/(p+r)
		
        print("Confusion Matrix : ")
        print(cm)
        print("\n")
        print(f"Precision : {p}")
        print(f"Recall : {r}")
        print(f"F1 SCORE : {f1}")
			

df=pd.read_csv("data_preprocessing.csv")

#Partitioning X and y from df and converting y into numpy array and reshaping it for computations 
X=df.iloc[:,:-1]
y=df.iloc[:,-1]
y=np.array(y)
y=y.reshape((y.shape[0],1))

#splitting x and y into train_test
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=0)

#training neural network
neuralnetwork=NN()
neuralnetwork.fit(X_train,y_train)
print("\n")

#train accuracy
print("train:")
y_hat_train=neuralnetwork.predict(X_train)
neuralnetwork.CM(y_train,y_hat_train)
neuralnetwork.accuracy(y_train,y_hat_train)

#test accuracy
print("test:")
y_hat_test=neuralnetwork.predict(X_test)
neuralnetwork.CM(y_test,y_hat_test)
neuralnetwork.accuracy(y_test,y_hat_test)