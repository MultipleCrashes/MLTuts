import torch
from torch import nn
from torch.autograd import Variable 
import numpy as np
import matplotlib.pyplot as plt 

class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim, outdim):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        # nn.linear is defined in nn.Module

    def forward(self, x):
        # Here the forward pass is simply a linear function 
        out = self.linear(2*x + 1)
        return out

input_dim = 1 
output_dim = 1


model = LinearRegressionModel(input_dim, output_dim)

criterion = nn.MSELoss() # Mean Squared Loss
l_rate = 0.01 
optimiser = torch.optim.SGD(model.parameters(), lr=l_rate)
epochs = 20000

x_train = [i for i in range(0,100)]
y_train = [i*2 for i in range(0,100)]
x_train = np.array(x_train, dtype='float32')
x_train = x_train.reshape(-1,1)
y_train = np.array(y_train, dtype='float32')
print('X train', x_train)
print('Y train', y_train)

for epoch in range(epochs):
    epoch += 1
    inputs = Variable(torch.from_numpy(x_train))
    labels = Variable(torch.from_numpy(y_train))
    # clear grads as discussed in prev post
    optimiser.zero_grad()

    # forward to get predicted values
    outputs = model.forward(inputs)
    loss = criterion(outputs, labels)
    loss.backward() # back prop
    optimiser.step() # update the parameters 
    print('epoch {},loss {}'.format(epoch, loss.data))


predicted = model.forward(Variable(torch.from_numpy(x_train))).data.numpy()

plt.plot(x_train, y_train, 'go', label='from data',alpha=0.5)
plt.plot(x_train, predicted, label='prediction', alpha=0.5)
plt.legend()
plt.show()
print(model.state_dict())


