#!/usr/bin/env python3
import numpy as np
from tqdm import trange
import matplotlib.pyplot as plt

def fetch(url):
  import os, hashlib
  import requests, gzip
  fp = os.path.join("/tmp",hashlib.md5(url.encode('utf-8')).hexdigest())
  if os.path.isfile(fp):
    print(f"Stored in {fp}")
    with open(fp, "rb") as f:
      dat = f.read()
  else: 
    print(f"Fetching {url}")
    with open(fp, "wb") as f:
      dat = requests.get(url).content
      f.write(dat)
  return np.frombuffer(gzip.decompress(dat), dtype=np.uint8).copy()

import torch
import torch.nn as nn

from BobNetLinear2 import BobNetLinear2
from BobNetLinear3 import BobNetLinear3
from BobNetConv1Linear3 import BobNetConv1Linear3
from BobNetConv2Linear2 import BobNetConv2Linear2

model = BobNetConv2Linear2()

BatchSize = 128
Step = 100
loss_function = torch.nn.CrossEntropyLoss()
optim = torch.optim.Adam(model.parameters())

def train(X_train,Y_train):
  losses , accuracies = [], []
  for i in (t := trange(Step)):
    # Batch size
    samp = np.random.randint(0,X_train.shape[0],size = BatchSize)
    # NOTE: If Linear reshape to line, if Conv reshape to square
    X = torch.tensor(X_train[samp].reshape(-1,1,28,28)).float()
    # X = torch.tensor(X_train[samp].reshape(-1,28*28)).float()
    Y = torch.tensor(Y_train[samp]).long()

    # forward
    optim.zero_grad()
    out = model(X)
    # outside the mode argmax
    cat = torch.argmax(out,dim=1)
    accuracy = (cat == Y).float().mean()
    
    # backward pass
    loss = loss_function(out,Y)
    loss.backward()
    optim.step()

    losses.append(loss.item())
    accuracies.append(accuracy.item())

    t.set_description(f"loss {loss.item():.2f} accuracy {accuracy.item():.2f}")
  return losses, accuracies
  

def evaluation(X_test,Y_test):
  # Y_test_preds = torch.argmax(model(torch.tensor(X_test.reshape((-1,28*28))).float()),dim=1).numpy()
  Y_test_preds = torch.argmax(model(torch.tensor(X_test.reshape((-1,1,28,28))).float()),dim=1).numpy()
  return (Y_test == Y_test_preds).mean()

if __name__ == "__main__":

  X_train = fetch("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz")[0x10:].reshape((-1, 28, 28))
  Y_train = fetch("http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz")[8:]
  X_test = fetch("http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz")[0x10:].reshape((-1, 28, 28))
  Y_test = fetch("http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz")[8:]

  print(model)

  loss, accuracy = train(X_train,Y_train)
  eval = evaluation(X_test,Y_test)
  print(f"Evaluation: {eval}")

  # plt.plot(loss)
  # plt.plot(accuracy)
  # plt.show()


  