import torch.nn as nn

class BobNetLinear2(nn.Module):
  def __init__(self):
    super(BobNetLinear2,self).__init__()
    # W x H = 28 x 28
    # Cout = 128
    self.fc1 = nn.Linear(28*28,128)
    # Cout = 128
    self.relu = nn.ReLU()
    # Cout = 10
    self.fc2 = nn.Linear(128,10)
  
  def forward(self, x):
    x = self.relu(self.fc1(x))
    x = self.fc2(x)
    return x