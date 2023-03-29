import torch
from GCN import GCN

import networkx as nx
import time
K = 100
model = GCN(K,16)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()
EPOCHS = 100

def train(nodes, edgelst):
      losses = []
      t_start = time.time()
      for epoch in range(EPOCHS):
            model.train()
            optimizer.zero_grad()
            out = model(nodes, edgelst)

            loss = criterion(out[data.train_mask], data.y[data.train_mask])
            #train_acc = accuracy(out[train_set_ind], labels[train_set_ind])
            losses.append(loss)
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
            loss.backward()
            optimizer.step()
      print(f"Training duration: {time.time()-t_start}")
      return losses

def test(nodes, edgelst):
      with torch.no_grad():
            model.eval()
            out = model(nodes, edgelst)
            pred = out.argmax(dim=1)
            test_correct = pred[data.test_mask] == data.y[data.test_mask]
            test_acc = int(test_correct.sum()) / int(data.test_mask.sum())
      return test_acc

for epoch in range(1, 101):
    loss = train()
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')

