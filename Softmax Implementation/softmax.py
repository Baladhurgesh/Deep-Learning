import numpy as np
from matplotlib import pyplot as plt

lrs = [1e-1, 1e-3, 1e-4, 1e-5] #Set of Learning Rates
n_epochs = [50, 100, 150, 200] #Set of Number of Epochs
alphas = [0.001, 0.0001, 0.01, 0.1] #Set of Regularisation Value
minbatch_sizes = [8, 16, 32, 64] #Set of Mini Batch Size values


    
def soft_max(yhat):
    yhat = np.exp(yhat).T/np.sum(np.exp(yhat),axis=1)
    return yhat
    
def train_MNIST(X_tr, y_tr, X_val, yval, X_te, yte):
    prev_val_error = 1000 #Initialising high error value to find Optimal Validation error in various configurations
    for lr in lrs:
        for n_epoch in n_epochs:
            for alpha in alphas:
                for minbatch_size in minbatch_sizes:
                    w = np.random.randn(X_tr.shape[1],10)
                    b = np.random.randn(1,10)

                    for n in range(epochs):
                                        
                        for i in range(int(X_tr.shape[0]/minbatch_size)): #Dividing the Training set to Mini Batch 
                            if i*minbatch_size+minbatch_size < X_tr.shape[0]:
                                X_tr_mini = X_tr[(i*minbatch_size):(i*minbatch_size+minbatch_size),:]
                                ytr_mini = y_tr[(i*minbatch_size):(i*minbatch_size+minbatch_size)]
                            else:
                                X_tr_mini = X_tr[(i*minbatch_size):X_tr.shape[0],:]
                                ytr_mini = y_tr[(i*minbatch_size):X_tr.shape[0]]

                            yhat = soft_max(np.dot(X_tr_mini,w)+b).T 
                            
                            grad_w = (1/X_tr_mini.shape[0])*np.matmul(X_tr_mini.T,(yhat-ytr_mini))+alpha*w #Gradients wrt Weights
                            grad_b = np.mean((yhat-ytr_mini),axis=0) #Gradient wrt Bias
                            
                            w = w - lr*grad_w # Weight Updation
                            b = b - lr*grad_b # Bias Updation

                    y_hat_val = np.dot(X_val, w)+b #Calculating Yhat for Validation data
                    y_hat_val_final = soft_max(y_hat_val).T

                    tval_error = -1* np.mean(np.multiply(yval,np.log(y_hat_val_final))) #Calculating Mean Square Error for Validation data
                    print("Validation Error:", tval_error) #Printing Validation error for every configuration
                    if (prev_val_error>tval_error): #Checking if the configuration has less validation error and stoing all the hyper parameters
                        prev_val_error = tval_error
                        final_w = w
                        final_b = b
                        final_lr = lr
                        final_epoch = n_epoch
                        final_alpha = alpha
                        final_batch_size = minbatch_size
                        print("Final Validation Error:", tval_error)

    y_hat_test = np.dot(X_te, final_w)+final_b
    y_hat_test_final = soft_max(y_hat_test).T
    result_te = np.argmax(y_hat_test_final,axis=1)
    actual_y_te = np.argmax(yte,axis=1)
    test_error = -1* np.mean(np.multiply(yte,np.log(y_hat_test_final))) #Calculating Mean Square Error for Validation data
    print("Testing Error:", test_error) #Printing Validation error for every configuration
    accuracy = sum(result_te==actual_y_te)/len(result_te)*100
    print("Testing Accuracy:",accuracy)
            


def main ():
    # Load data
    X_tr = np.reshape(np.load("mnist_train_images.npy"), (-1, 28*28))    
    ytr = np.load("mnist_train_labels.npy")

    X_val = np.reshape(np.load("mnist_validation_images.npy"), (-1, 28*28))  
    yval = np.load("mnist_validation_labels.npy")
    
    X_te = np.reshape(np.load("mnist_test_images.npy"), (-1, 28*28))    
    yte = np.load("mnist_test_labels.npy") 

    print("Train set:", ytr.shape)
    print("Validation set:", yval.shape)
    print("Test set:", yte.shape)
    
    train_MNIST(X_tr, ytr, X_val, yval, X_te, yte)

    

if __name__ == '__main__':
    main()
