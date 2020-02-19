import numpy as np
from matplotlib import pyplot as plt

lrs = [1e-3, 1e-4, 1e-5, 5e-4] #Set of Learning Rates
n_epochs = [100, 150, 200, 250] #Set of Number of Epochs
alphas = [0.2, 0.5, 0.01, 0.1] #Set of Regularisation Value
minbatch_sizes = [8, 16, 32, 64] #Set of Mini Batch Size values


    

def linear_regression (X_tr, y_tr, X_val, yval, X_te, yte):
    
    prev_val_error = 1000 #Initialising high error value to find Optimal Validation error in various configurations
    for lr in lrs:
        for n_epoch in n_epochs:
            for alpha in alphas:
                for minbatch_size in minbatch_sizes:
                    w = np.random.rand(X_tr.shape[1],1) #Initialising Weights and bias
                    b = 0
                    for n in range(n_epoch):
                        
                        for i in range(int(X_tr.shape[0]/minbatch_size)+1): #Lines 23-29 Dividing the Training set to Mini Batch 
                            if i*minbatch_size+minbatch_size < X_tr.shape[0]:
                                X_tr_mini = X_tr[(i*minbatch_size):(i*minbatch_size+minbatch_size),:]
                                ytr_mini = y_tr[(i*minbatch_size):(i*minbatch_size+minbatch_size)]
                            else:
                                X_tr_mini = X_tr[(i*minbatch_size):X_tr.shape[0],:]
                                ytr_mini = y_tr[(i*minbatch_size):X_tr.shape[0]]
                                
                            yhat = np.dot(X_tr_mini,w)+b #Calculating Y hat value
                            
                            grad_w = (1/X_tr_mini.shape[0])*np.dot(X_tr_mini.T,(yhat-ytr_mini))+alpha*w #Gradients wrt Weights
                            grad_b = np.mean((1/X_tr_mini.shape[0])*(yhat-ytr_mini)) #Gradient wrt Bias
                            
                            w = w - lr*grad_w # Weight Updation
                            b = b - lr*grad_b # Bias Updation

                    y_hat_val = np.dot(X_val, w)+b #Calculating Yhat for Validation data
                    tval_error = 0.5*np.mean((y_hat_val-yval)**2) #Calculating Mean Square Error for Validation data
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
    y_hat_te = np.dot(X_te, final_w)+final_b #Calculating Yhat for Test data from the best configuration of learning rate, epochs, alpha, mini batch size
    te_error = 0.5*np.mean((y_hat_te-yte)**2) #Calculating Unregularised Mean Square Error for Test data

    print("Final Configuration:") #Printing the Best configuration parameters and Testing error
    print("Learning Rate: ", final_lr)
    print("Number of Epochs: ", final_epoch)
    print("Regularisation: ", final_alpha)
    print("Mini Batch Size: ", final_batch_size)
    print("Testing Error: ", te_error)
    print("")

    return final_w, final_b
    

def train_age_regressor ():
    # Load data
    X_tr = np.reshape(np.load("age_regression_Xtr.npy"), (-1, 48*48))    
    ytr = np.load("age_regression_ytr.npy")

    X_val = X_tr[4000:5000,:]
    X_tr = X_tr[0:3999,:]
    yval = ytr[4000:5000]
    ytr = ytr[0:3999]
    
    yval = np.reshape(yval,(-1,1))
    ytr = np.reshape(ytr,(-1,1))
    
    X_te = np.reshape(np.load("age_regression_Xte.npy"), (-1, 48*48))    
    yte = np.load("age_regression_yte.npy") 
    yte = np.reshape(yte,(-1,1))   

    w, b = linear_regression(X_tr, ytr, X_val, yval, X_te, yte)
        

def main():
    train_age_regressor()

if __name__ == '__main__':
    main()