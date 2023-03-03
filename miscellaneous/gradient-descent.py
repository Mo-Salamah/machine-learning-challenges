#%%
import numpy as np
from numpy import linalg
import pandas as pd
# %%
X = np.array([[1,2],
              [1,4],
            #   [1,6],
            #   [1,33],
              [1,5],])

m = X.shape[0]
x = X[:, 1]

X_new = np.zeros(m)
 
for i in range(m):
    x_i = x
    x_i = x_i ** i 
    X_new = np.column_stack((X_new, x_i))
    print(X_new)

X_new = X_new[:, 1:]
# y = np.random.randint(0, 100, m)
y = [2,4,6]



# %%
def loss_prime_w0(w0, w1):
    slope = (6*w0) + (11*w1) - 42
    return slope


def loss_prime_w1(w0, w1):
    slope = (11*w0) + (28*w1) - 98
    return slope

def loss(w0, w1):
    loss = 3*np.square(w0) + 14*np.square(w1) 
    loss = loss + 11*w0*w1 - 42*w0 - 98*w1 + 173 
    
    return loss


#%%
w0 = 0
w1 = 0
loss_diff = 2
alpha = 0.001
loss_score = 1000000

# gradient for w0 and w1
gradient = [0 , 0]
gradient[0] = loss_prime_w0(w0, w1)
gradient[1] = loss_prime_w1(w0, w1)

counter = 1

# while (loss_diff > 0.00000000000000000001):
while(counter<10000):
    gradient[0] = loss_prime_w0(w0, w1)
    gradient[1] = loss_prime_w1(w0, w1)
    
    w0 = w0 - (alpha * gradient[0])
    w1 = w1 - (alpha * gradient[1])
    
    old_loss = loss_score
    loss_score = loss(w0, w1)
    loss_diff = loss_score - old_loss
    
    print_table([counter, loss_score, gradient, [w0, w1]], counter==1)
    counter+=1
    

#%%
def print_table(data, header=False, headers=['Iteration',
        'Loss Score', 'Gradiant', 'New Params']):
    
    if header:
        # print the header row
        for item in headers:
            print("{:<15}".format(item), end="")
        print()

    # print a row for each item in the data list
    # for item in data:
    item = data
    # unpack the item list into the relevant variables
    item_name, quantity, price, date = item
    # loop over the two integers in the quantity list
    for i in range(2):
        # print the item name in the first column
        if i == 0:
            print("{:<15}".format(item_name), end="")
            # print the quantity integer in the second column
            print("{:<15.1f}".format(quantity), end="")
        else:
            print("{:<15}".format(""), end="")
            print("{:<15}".format(""), end="")
            
        
        # print the price and date in the third and fourth columns
        print("{:<15.1f}{:<15.1f}".format(price[i], date[i]))
    print('\n\n')













# %%
