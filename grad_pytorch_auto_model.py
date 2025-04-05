# import numpy as np
import torch
import torch.nn as nn

# X = np.array([1,2,3,4],dtype=np.float32)
# Y = np.array([2,4,6,8],dtype=np.float32)
# w = 0.0
# X = torch.tensor([1,2,3,4],dtype=torch.float32)
# Y = torch.tensor([2,4,6,8],dtype=torch.float32)
# W = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)
X = torch.tensor([[1], [2], [3], [4]], dtype=torch.float32)
Y = torch.tensor([[2], [4], [6], [8]], dtype=torch.float32)

X_test = torch.tensor([5], dtype=torch.float32)
batch_size , n_features = X.shape
print(batch_size , n_features)
input_size = n_features
output_size = n_features

# 1) model prediction
# def forward(x):
#     return W * x
model = nn.Linear(input_size, output_size)

class Net(nn.Module):
    def __init__(self,input_dim,output_dim):
        super(Net, self).__init__()
        #define layers
        self.l1 = nn.Linear(input_dim, output_dim)
    def forward(self, x):
        return self.l1(x)
model = Net(input_size, output_size)

# 2) loss
# def loss(y,y_pred):
#     return ((y_pred - y)**2).mean()
loss = nn.MSELoss()     # criterion

# gradient
# MSE = 1/N * (w*x - y)**2
# dJ/dw = 1/N 2x (w*x - y)
# def gradient(x,y,y_pred):
#     return np.dot(2*x,y_pred - y).mean()

# print(f'before training: f(5) = {forward(5):.3f}')
# print(f'before training: f(5) = {model(5):.3f}')
print(f'before training: f(5) = {model(X_test).item():.3f}')

# Training
learning_rate = 0.01
n_epochs = 100

# optimizer = torch.optim.SGD([W], lr=learning_rate)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(n_epochs):
    # 1) forward
    # y_pred = forward(X)
    y_pred = model(X)
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
        # print(f'epoch {epoch + 1}:w = {W:.3f},loss = {l:.8f}')
        [W, b] = model.parameters()
        print(f'epoch {epoch + 1}:w = {W[0][0].item():.3f},loss = {l:.8f}')

print(f'after training: f(5) = {model(X_test).item():.3f}')






