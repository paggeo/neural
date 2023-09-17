import torch.nn as nn 

class BobNetConv1Linear3(nn.Module):
  def __init__(self):
    super(BobNetConv1Linear3,self).__init__()
    # W x H  = 28 x 28
    # Cout x Wout x Hout = Cin x  [(W_in - K_w + 2 * P) / S] + 1 x [(H_in - K_h + 2 * P) / S] + 1 =  32 x 26 x 26
    self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1) # output size = 32 * (28-(3-1)) * (28-(3-1))
    # Cout x Wout x Hout = 32 x 25 x 25
    self.relu1 = nn.ReLU()
    # Cout x Wout x Hout = Cin x Win/kw x Hin/kh = 32 x 13 (26/2) x 13 (26/2)
    self.pool1 = nn.MaxPool2d(kernel_size=2) 

    # Cout = 128 
    self.fc1 = nn.Linear(32*13*13,128)
    # Cout = 128 
    self.relu2 = nn.ReLU()
    # Cout = 64 
    self.fc2 = nn.Linear(128,64)
    # Cout = 64 
    self.relu3 = nn.ReLU()
    # Cout = 10 
    self.fc3 = nn.Linear(64,10)
  
  def forward(self, x):
    x = self.relu1(self.conv1(x))
    x = self.pool1(x)
    x = self.relu2(self.fc1(x.view(x.size(0),-1)))
    x = self.relu3(self.fc2(x))
    x = self.fc3(x)
    return x