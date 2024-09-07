# import numpy as np
import torch
import torch.nn as nn

# X = np.array([1,2,3,4],dtype=np.float32)
# Y = np.array([2,4,6,8],dtype=np.float32)
# w = 0.0
X = torch.tensor([1,2,3,4],dtype=torch.float32)
Y = torch.tensor([2,4,6,8],dtype=torch.float32)
W = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)
# 1) model prediction
def forward(x):
    return W * x

# 2) loss
# def loss(y,y_pred):
#     return ((y_pred - y)**2).mean()
loss = nn.MSELoss()

# gradient 
# MSE = 1/N * (w*x - y)**2
# dJ/dw = 1/N 2x (w*x - y)
# def gradient(x,y,y_pred):
#     return np.dot(2*x,y_pred - y).mean()

print(f'before training: f(5) = {forward(5):.3f}')

# Training
learning_rate = 0.01
n_epochs = 100

optimizer = torch.optim.SGD(params=[W], lr=learning_rate)

for epoch in range(n_epochs):
    # 1) forward
    y_pred = forward(X)
    l = loss(Y,y_pred)

    # 2) compute gradients
    # grad = gradient(X,Y,y_pred)
    l.backward() # dl/dw

    # 3) update weights

    # a) manually update
    # a.1)numpy
    # W -= learning_rate * grad
    # a.2)torch
    # with torch.no_grad():
    #     W -=learning_rate * W.grad

    # b) auto update
    optimizer.step()

    # zero gradients
    # W.grad.zero_()
    optimizer.zero_grad()

    if epoch % 10 == 0:
        print(f'epoch {epoch + 1}:w = {W:.3f},loss = {l:.8f}')

print(f'after training: f(5) = {forward(5):.3f}')






