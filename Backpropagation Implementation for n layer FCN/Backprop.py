import numpy as np
from matplotlib import pyplot as plt

lr = [4e-2, 1.7e-2, 1.5e-2, 7e-3, 3.5e-2, 3e-2, 1e-2, 5e-3, 2e-2, 5e-2] #Set of Learning Rates
n_epoch = [50, 70, 60, 60, 50, 40, 50, 40, 70, 30] #Set of Number of Epochs
alpha = [0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001] #Set of Regularisation Value
minbatch_size = [64, 64, 64, 64, 128, 128, 128, 256, 128, 128] #Set of Mini Batch Size values
n_hidden_layers = [1, 2, 1, 2, 3, 1, 2, 5, 3, 4] # Set of number of Hidden Layers
n_hidden_units = [100, 100, 90, 80, 90, 80, 75, 60, 70, 65] # Set of number of hidden units
n_output_units = 10

def soft_max(yhat):
    yhat = (np.array(yhat).T - np.max(yhat, axis = 1)).T
    yhat = np.exp(yhat).T/np.sum(np.exp(yhat),axis=1)
    return yhat.T

def relu(z):
    return np.maximum(0,z)

def relu_diff(z):
    z[z>0] = 1
    z[z<=0] = 0
    return z

def forward(X, w, b, n_hidden_layer):
    z = []
    h = []
    for i in range(n_hidden_layer):
        if i==0:
            z.append((np.dot(X,w[i])+b[i]))
            h.append(relu(z[i]))
        else:
            z.append((np.dot(h[i-1],w[i])+b[i]))
            h.append(relu(z[i]))
    # print(len(h))
    # print(len(w))
    z.append((np.dot(h[-1],w[-1])+b[-1]))
    h.append(soft_max(z[-1]))

    return z,h

def backward(w, z, h, X, y):
    dJ_dw = []
    dJ_db = []
    
    g = h[-1]-y

    for i in range(len(w)-1,0,-1):
        
        if i == len(w)-1:
            dJ_db.append(g.T)
            dJ_dw.append(np.dot(g.T,h[i-1]))
            g = np.dot(g,w[i].T)
            
        else:
            
            g = np.multiply(g,relu_diff(z[i]))
            dJ_db.append(g.T)
            dJ_dw.append(np.dot(g.T,h[i]))
            g = np.dot(g,w[i].T)
    
    g = np.multiply(g,relu_diff(z[0]))
    dJ_db.append(g.T)
    dJ_dw.append(np.dot(g.T,X))

    return dJ_dw, dJ_db





def train_MNIST(X_tr, y_tr, X_val, yval, X_te, yte, n_hidden_layers, n_hidden_units, n_output_units):
    prev_val_accu = 10 #Initialising high error value to find Optimal Validation error in various configurations
    for k in range(len(lr)):  
        print("")
        print("Next configuration")
        w = []
        b = []
        #Weights Initialisation
        for j in range(n_hidden_layers[k]):
            if j == 0:
                w.append(np.random.randn(X_tr.shape[1], n_hidden_units[k]))
                b.append(np.random.randn(1, n_hidden_units[k]))
            else:
                w.append(np.random.randn(n_hidden_units[k], n_hidden_units[k]))
                b.append(np.random.randn(1, n_hidden_units[k]))
        w.append(np.random.randn(n_hidden_units[k], n_output_units))
        b.append(np.random.randn(1, n_output_units))
        w = [w[i]/10 for i in range(len(w))]
        
        for n in range(n_epoch[k]):
            for i in range(int(X_tr.shape[0]/minbatch_size[k])): #Dividing the Training set to Mini Batch 
                
                if i*minbatch_size[k]+minbatch_size[k] < X_tr.shape[0]:
                    X_tr_mini = X_tr[(i*minbatch_size[k]):(i*minbatch_size[k]+minbatch_size[k]),:]
                    ytr_mini = y_tr[(i*minbatch_size[k]):(i*minbatch_size[k]+minbatch_size[k])]
                else:
                    X_tr_mini = X_tr[(i*minbatch_size[k]):X_tr.shape[0],:]
                    ytr_mini = y_tr[(i*minbatch_size[k]):X_tr.shape[0]]
                z, h = forward(X_tr_mini, w, b, n_hidden_layers[k])
             
                dJ_dw, dJ_db = backward(w, z, h, X_tr_mini, ytr_mini)
                
                dJ_dw.reverse()
                dJ_db.reverse()
                #Weight Updation
                for i in range(len(w)):
                    w[i] = w[i] - (lr[k]*dJ_dw[i].T/minbatch_size[k] + alpha[k]*w[i])
                    b[i] = b[i] - lr[k]*np.sum(dJ_db[i],axis=1)
            #Calculating Validation Accuracy
            z_val, h_val = forward(X_val, w, b, n_hidden_layers[k])
            y_hat_val = h_val[-1]
            result_val = np.argmax(y_hat_val,axis=1)
            actual_val = np.argmax(yval,axis=1)
            val_accuracy = sum(result_val==actual_val)/len(result_val)*100
            print("Validation Accuracy:",val_accuracy)

        if val_accuracy>prev_val_accu:
            prev_val_accu = val_accuracy
            final_w = w
            final_b = b
            final_lr = lr[k]
            final_epoch = n_epoch[k]
            final_batch_size = minbatch_size[k]
            final_n_hidden_layers = n_hidden_layers[k]
            final_n_hidden_units = n_hidden_units[k]
            final_alpha = alpha[k]
            print("Final Validation Accuracy:", val_accuracy)

    #Calculating Test Accuracy
    z_te, h_te = forward(X_te, final_w, final_b, final_n_hidden_layers)
    y_hat_test = h_te[-1]
    result_te = np.argmax(y_hat_test,axis=1)
    actual_y_te = np.argmax(yte,axis=1)
    accuracy = sum(result_te==actual_y_te)/len(result_te)*100
    print("")
    print("")
    print("--------------------------------------------------------------------------------------------")
    print("")
    print("Best configuration:")
    print("Learning Rate: ", final_lr)
    print("Number of Epochs: ", final_epoch)
    print("Regularisation: ", final_alpha)
    print("Mini Batch Size: ", final_batch_size)
    print("No of Hidden Layers:", final_n_hidden_layers)
    print("No of Hidden Units in each hidden layer:", final_n_hidden_units)
    print("Testing Accuracy:",accuracy)
    print("")
    print("-------------------------------------------------------------------------------------------")
            


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
    
    train_MNIST(X_tr, ytr, X_val, yval, X_te, yte, n_hidden_layers, n_hidden_units, n_output_units)

    

if __name__ == '__main__':
    main()