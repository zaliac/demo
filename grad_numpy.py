import numpy as np

X = np.array([1,2,3,4],dtype=np.float32)
Y = np.array([2,4,6,8],dtype=np.float32)
W = 0.0

# 1) model prediction
def forward(x):
    return W * x

# 2) loss
def loss(y,y_pred):
    return ((y_pred - y)**2).mean()

# gradient
# MSE = (1/N) * (w*x - y)**2
# dJ/dw = (1/N) 2x (w*x - y)
def gradient(x,y,y_pred):
    return np.dot(2*x,y_pred - y).mean()

# def updateParameters(W,lr):
#     W -= lr * gradient()

print(f'before training: f(5) = {forward(5):.3f}')

# Training
learning_rate = 0.01
n_epochs = 12

for epoch in range(n_epochs):
    # 1) forward
    y_pred = forward(X)                     # y_pred = W * x

    l = loss(Y,y_pred)                      # l = ((y_pred - y)**2).mean()

    # 2) compute gradients: backward() in pytorch
    grad = gradient(X,Y,y_pred)             # grad = np.dot(2*x,y_pred - y).mean()  : dJ/dw = 1/N 2x (w*x - y)

    # 3) update weights: in-place
    W -= learning_rate * grad               # w = w - lr * grad -> optimizer.step()
    # with torch.no_grad():
    #   W -=learning_rate * W.grad
    # W.grad.zero_()

    if epoch % 1 == 0:
        print(f'epoch {epoch + 1}:w = {W:.3f},loss = {l:.8f}')

print(f'after training: f(5) = {forward(5):.3f}')






