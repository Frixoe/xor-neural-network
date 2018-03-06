import numpy as np
import matplotlib.pyplot as plt
import sys

X = np.array([
    [0, 1],
    [1, 0],
    [1, 1],
    [0, 0]
])
y = np.array([
    [1],
    [1],
    [0],
    [0]
])

num_i_units = 2
num_h_units = 2
num_o_units = 1

learning_rate = 0.1 # 0.001, 0.01 <- Magic values
reg_param = 0 # 0.001, 0.01 <- Magic values
max_iter = 5000 # 5000 <- Magic value
m = 4 # Number of training examples

# The model needs to be over fit to make predictions. Which 
np.random.seed(1)
W1 = np.random.normal(0, 1, (num_h_units, num_i_units)) # 2x2
W2 = np.random.normal(0, 1, (num_o_units, num_h_units)) # 1x2

B1 = np.random.random((num_h_units, 1)) # 2x1
B2 = np.random.random((num_o_units, 1)) # 1x1

def sigmoid(z, derv=False):
    if derv: return z * (1 - z)
    return 1 / (1 + np.exp(-z))

def forward(x, predict=False):
    a1 = x.reshape(x.shape[0], 1) # Getting the training example as a column vector.

    z2 = W1.dot(a1) + B1 # 2x2 * 2x1 + 2x1 = 2x1
    a2 = sigmoid(z2) # 2x1

    z3 = W2.dot(a2) + B2 # 1x2 * 2x1 + 1x1 = 1x1
    a3 = sigmoid(z3)

    if predict: return a3
    return (a1, a2, a3)

dW1 = 0
dW2 = 0

dB1 = 0
dB2 = 0

cost = np.zeros((max_iter, 1))
for i in range(max_iter):
    c = 0

    dW1 = 0
    dW2 = 0

    dB1 = 0
    dB2 = 0
    for j in range(m):
        sys.stdout.write("\rIteration: {} and {}".format(i + 1, j + 1))

        # Forward Prop.
        a0 = X[j].reshape(X[j].shape[0], 1) # 2x1

        z1 = W1.dot(a0) + B1 # 2x2 * 2x1 + 2x1 = 2x1
        a1 = sigmoid(z1) # 2x1

        z2 = W2.dot(a1) + B2 # 1x2 * 2x1 + 1x1 = 1x1
        a2 = sigmoid(z2) # 1x1

        # Back prop.
        dz2 = a2 - y[j] # 1x1
        dW2 += dz2 * a1.T # 1x1 .* 1x2 = 1x2

        dz1 = np.multiply((W2.T * dz2), sigmoid(a1, derv=True)) # (2x1 * 1x1) .* 2x1 = 2x1
        dW1 += dz1.dot(a0.T) # 2x1 * 1x2 = 2x2

        dB1 += dz1 # 2x1
        dB2 += dz2 # 1x1

        c = c + (-(y[j] * np.log(a2)) - ((1 - y[j]) * np.log(1 - a2)))
        sys.stdout.flush() # Updating the text.
    W1 = W1 - learning_rate * (dW1 / m) + ( (reg_param / m) * W1)
    W2 = W2 - learning_rate * (dW2 / m) + ( (reg_param / m) * W2)

    B1 = B1 - learning_rate * (dB1 / m)
    B2 = B2 - learning_rate * (dB2 / m)
    cost[i] = (c / m) + ( 
        (reg_param / (2 * m)) * 
        (
            np.sum(np.power(W1, 2)) + 
            np.sum(np.power(W2, 2))
        )
    )


for x in X:
    print("\n")
    print(x)
    print(forward(x, predict=True))

plt.plot(range(max_iter), cost)
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.show()
