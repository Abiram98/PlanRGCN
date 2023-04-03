from classifier.bgp_dataset import BGPDataset, return_graph_dataloader
#from torch.utils.data import DataLoader
import math,os
import torch.nn as nn, torch, numpy as np
from classifier.GCN import GNN
from feature_extraction.predicate_features import PredicateFeaturesQuery

from torch_geometric.loader.dataloader import DataLoader
from torch_geometric import seed_everything

torch.manual_seed(12345)
np.random.seed(12345)
seed_everything(12345)

#Model hyper parameters
EPOCHS = 100
BATCH_SIZE = 200
LR = 1e-3
WEIGHT_DECAY = 5e-4
PREDICATE_BINS = 30


data_file = '/work/data/train_data.json'
test_file = '/work/data/test_data.json'
feat_generation_path = '/work/data/confs/newPredExtractionRun/pred_feat_01_04_2023_07_48.pickle'
train_bgp_file_pickle = '/work/data/confs/newPredExtractionRun/train_bgps.pickle'
test_bgp_file_pickle = '/work/data/confs/newPredExtractionRun/test_bgps.pickle'
pickle_data_loader = '/work/data/confs/newPredExtractionRun/train_dataloader.pickle'
pickle_dataset = '/work/data/confs/newPredExtractionRun/train_dataset.pickle'
pickle_testdataset = '/work/data/confs/newPredExtractionRun/test_dataset.pickle'
if not os.path.isfile(pickle_dataset):
    dataset = BGPDataset(data_file,feat_generation_path, train_bgp_file_pickle,bin=PREDICATE_BINS)
    test_dataset = BGPDataset(test_file,feat_generation_path, test_bgp_file_pickle,bin=PREDICATE_BINS)
    #loader = DataLoader(dataset, batch_size=math.ceil(len(dataset)/BATCH_SIZE))
    #torch.save(loader,pickle_data_loader)
    torch.save(dataset,pickle_dataset)
    torch.save(test_dataset,pickle_testdataset)
else:
    #loader = torch.load(open(pickle_data_loader,'rb'))
    dataset = torch.load(pickle_dataset)
    test_dataset= dataset = torch.load(pickle_testdataset)
    print(f"Dataset size: (Train) {len(dataset)}, (Test) {len(test_dataset)}")
#
#temporary code
#def nan_yielder(dataset):
#    for sample in dataset:
#        if torch.sum(torch.isnan( sample['nodes'])) > 0:
#            yield sample
#gen = nan_yielder(dataset)
       
#Model definition
model = GNN(dataset.node_features[0].shape[1] , dataset.node_features[0].shape[1]*2, hidden_dimension=10)
model = model.float()

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

def train_epoch_loop02(dataset=None, model=None, loss_fn=None, optimizer=None, epoch:int=None):
    size = len(dataset)
    train_loss = 0
    print(f'Training dataset size: {size}')
    model.train()
    for sample_no, sample in enumerate(dataset):
        # Compute prediction and loss
        pred = model(sample['nodes'],sample['edges'],sample['join_index'])
        loss = loss_fn(pred, sample['target'])
        train_loss += loss.item()
        #print(f"Gt : {sample['target']}, pred: {pred}")
        #
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #
        if sample_no % 5000 == 0:
            #loss_current, current = loss.item(), (sample_no + 1) * len(sample['nodes'])
            #print(f"Epoch: {epoch:4} {sample_no:5} loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            print(f"Epoch: {epoch+1:4} {sample_no:8} Total loss: {train_loss:>7f}  [{(sample_no/size)*100:.2f}]")

def val_loop02(dataset=dataset, model=model, loss_fn=loss_fn,epoch_no=None, prev_loss = None, path_to_save ='/work/data/confs/newPredExtractionRun/'):
    size = len(dataset)
    
    test_loss, correct = 0, 0
    with torch.no_grad():
        for no,sample in enumerate(dataset):
            pred = model(sample['nodes'],sample['edges'],sample['join_index'])
            test_loss += loss_fn(pred, sample['target']).item()
            #correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    new_loss = prev_loss
    if path_to_save != None and (prev_loss == None or test_loss < new_loss):
        torch.save(model,f"{path_to_save}model_{epoch_no}.pt")
        new_loss = test_loss
    print(f"Val Error {epoch_no+1:4}: Total Loss: {test_loss:>8f}  \n")
    return new_loss
    #test_loss /= num_batches
    #correct /= size
    #print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

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

def run(dataset, test_dataset,model,loss_fn,optimizer):
    #global train_loader
    #train_loader = return_graph_dataloader(dataset, batch_size=BATCH_SIZE)
    
    prev_val_loss = None
    for t in range(EPOCHS):
        print(f"Epoch {t+1}\n--------------------------------------------------------------")
        #train_loop(loader, model, loss_fn, optimizer)
        train_epoch_loop02(dataset=dataset, model=model, loss_fn=loss_fn, optimizer=optimizer, epoch=t)
        prev_val_loss = val_loop02(dataset=test_dataset, model=model, loss_fn=loss_fn,epoch_no=t, prev_loss = prev_val_loss, path_to_save ='/work/data/confs/newPredExtractionRun/')
        print(f"Optimal Val loss : {prev_val_loss}\n")
        #test_loop(test_dataloader, model, loss_fn)
    print("Done!")
#TODO outcomment this
#for i in range(5):
#    print(dataset[i])
#data = next(iter(loader))
#print(data)
if __name__ == "__main__":
    run(dataset, test_dataset,model,loss_fn,optimizer)