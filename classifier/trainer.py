from classifier.bgp_dataset import BGPDataset
#from torch.utils.data import DataLoader
import math,os
import torch.nn as nn, torch, numpy as np
from classifier.GCN import GNN
from feature_extraction.predicate_features import PredicateFeaturesQuery

from torch_geometric.loader.dataloader import DataLoader

torch.manual_seed(12345)
np.random.seed(12345)

#Model hyper parameters
EPOCHS = 2
BATCH_SIZE = 200
LR = 1e-3
WEIGHT_DECAY = 5e-4
PREDICATE_BINS = 30


data_file = '/work/data/train_data.json'
feat_generation_path = '/work/data/pred_feat.pickle'
pickle_file = '/work/data/bgps.pickle'
pickle_data_loader = '/work/data/dataloader.pickle'
pickle_dataset = '/work/data/dataset.pickle'
if not os.path.isfile(pickle_data_loader):
    dataset = BGPDataset(data_file,feat_generation_path, pickle_file,bin=PREDICATE_BINS)
    loader = DataLoader(dataset, batch_size=math.ceil(len(dataset)/BATCH_SIZE))
    torch.save(loader,pickle_data_loader)
    torch.save(dataset,pickle_dataset)
else:
    loader = torch.load(open(pickle_data_loader,'rb'))
    dataset = torch.load(open(pickle_dataset,'rb'))

#Model definition
model = GNN(loader.dataset.node_features[0].shape[1] , 10, hidden_dimension=10)
#optimizer and loss function
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (nodefeatures,edgelists, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(nodefeatures,edgelists)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(nodefeatures)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def train_loop02(dataset, model, loss_fn, optimizer):
    size = len(dataset)
    for batch, sample in enumerate(dataset):
        # Compute prediction and loss
        pred = model(sample['nodes'],sample['edges'],sample['join_index'])
        loss = loss_fn(pred, sample['target'])
        print(f"Gt : {sample['target']}, pred: {pred}")
        #
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #
        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(sample['nodes'])
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        #TODO outcomment this
        if batch % 300 == 0:
            exit()

def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

def run():
    for t in range(EPOCHS):
        print(f"Epoch {t+1}\n-------------------------------")
        #train_loop(loader, model, loss_fn, optimizer)
        train_loop02(dataset, model, loss_fn, optimizer)
        #test_loop(test_dataloader, model, loss_fn)
    print("Done!")
#TODO outcomment this
for i in range(5):
    print(dataset[i])
#data = next(iter(loader))
#print(data)
run()