Implementation:
A 5 layer neural network is used to predict if a person has lbw or not.in this 
5 layer network there are input and output layer and three hidden layers,weights
are initialized randomly for the hidden and output layers,and we use forward propagation 
to predict the output for the current set of weights,and we use a cost function (binary cross entropy)  
to find how much close the output with current set of weights to the actual output,and 
we update the weights while backpropagating from the cost function at the output layer 
to input and update the weights accordingly,we used adam optimization alogorithm to update 
the network weights,Adam computes individual learning rates for different parameters,it uses 
momentum and adaptive learning rates to converge faster.

Hyperparameters:
X-input matrix
y-output matrix
L-5(number of layers)
n=[X.shape[1],512,256,128,y.shape[1]](neurons in each layer)
lr-0.03(learning rate)
beta1-0.99
beta2-0.999
eps-1e-8
epochs-300

Key features:
instead of using same learning rate for every layer we used adam which combines momentum and 
adaptive learning rates to get the result quicker.

Steps:
the preprocessed.csv file is in the same folder as main python file
Run the .py file and in the same folder as .csv file
then we get the output in two segments train and test
train contains its confusion_matrix,precision,recall,f1_score,accuracy of the X_train matrix
test contains its confusion_matrix,precision,recall,f1_score,accuracy of the X_test matrix

we create an object using neuralnetwork=NN()
we train using neuralnetwork.fit(X_train,y_train)
we can predict values for x_test using y_hat_test=neuralnetwork.predict(X_test)
we can get the confusion matrix and other scores using neuralnetwork.CM(y_train,y_hat_train)
we can get the accuracy using neuralnetwork.accuracy(y_test,y_hat_test)



 


