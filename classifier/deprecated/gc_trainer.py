from classifier.bgp_dataset import BGPDataset, return_graph_dataloader
from classifier.bgp_dataset_v2 import BGPDataset_v2
#from torch.utils.data import DataLoader
import math,os
import torch.nn as nn, torch, numpy as np
from classifier.deprecated.graph_classification import GNN
from feature_extraction.predicates.predicate_features import PredicateFeaturesQuery

from torch_geometric.loader.dataloader import DataLoader
from torch_geometric import seed_everything
import configparser
from feature_extraction.constants import PATH_TO_CONFIG_GRAPH
torch.manual_seed(12345)
np.random.seed(12345)
seed_everything(12345)



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

def train_epoch_loop02(dataset=None, loss_fn=None, optimizer=None, epoch:int=None):
    global model
    size = len(dataset)
    train_loss = 0
    print(f'Training dataset size: {size}')
    model.train()
    
    for sample_no, sample in enumerate(dataset):
        # Compute prediction and loss
        pred = model(sample['nodes'],sample['edges'],sample['join_index'])
        optimizer.zero_grad()
        loss = loss_fn(pred, torch.reshape(sample['target'],(1,1)))
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        #print(f"Gt : {sample['target']}, pred: {pred}")
        #
        # Backpropagation
        
        #
        if sample_no % 5000 == 0:
            #loss_current, current = loss.item(), (sample_no + 1) * len(sample['nodes'])
            #print(f"Epoch: {epoch:4} {sample_no:5} loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            print(f"Epoch: {epoch+1:4} {(sample_no+1):8} Average loss: {train_loss/(sample_no+1):>7f}  [{((sample_no+1)/size)*100:.2f}]")

def val_loop02(dataset=None, loss_fn=None,epoch_no=None, prev_loss = None, path_to_save ='/work/data/confs/newPredExtractionRun/'):
    global model
    test_loss= 0
    with torch.no_grad():
        for no,sample in enumerate(dataset):
            pred = model(sample['nodes'],sample['edges'],sample['join_index'])
            test_loss += loss_fn(pred, torch.reshape(sample['target'],(1,1))).item()
    test_loss= test_loss/len(dataset)
    
    new_loss = prev_loss
    if path_to_save != None and (prev_loss == None or test_loss < new_loss):
        torch.save(model,f"{path_to_save}model_{epoch_no}.pt")
        new_loss = test_loss
    print(f"Val Error {epoch_no+1:4}: Avg Loss: {test_loss:>8f}  \n")
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

def run(dataset, test_dataset,loss_fn,optimizer, path_to_save ='/work/data/confs/newPredExtractionRun/', epoch=50):
    #global train_loader
    #train_loader = return_graph_dataloader(dataset, batch_size=BATCH_SIZE)
    global model
    prev_val_loss = None
    for t in range(epoch):
        print(f"Epoch {t+1}\n--------------------------------------------------------------")
        #train_loop(loader, model, loss_fn, optimizer)
        train_epoch_loop02(dataset=dataset, loss_fn=loss_fn, optimizer=optimizer, epoch=t)
        prev_val_loss = val_loop02(dataset=test_dataset, loss_fn=loss_fn,epoch_no=t, prev_loss = prev_val_loss, path_to_save =path_to_save)
        print(f"Optimal Val loss : {prev_val_loss}\n")
        #test_loop(test_dataloader, model, loss_fn)
    print("Done!")

def run_undivided(dataset, test_dataset,loss_fn,optimizer, path_to_save ='/work/data/confs/newPredExtractionRun/', epoch=50, early_stop=10):
    #global train_loader
    #train_loader = return_graph_dataloader(dataset, batch_size=BATCH_SIZE)
    global model
    prev_val_loss = None
    val_hist = []
    for t in range(epoch):
        print(f"Epoch {t+1}\n--------------------------------------------------------------")
        #train_loop(loader, model, loss_fn, optimizer)
        #train_epoch_loop02(dataset=dataset, loss_fn=loss_fn, optimizer=optimizer, epoch=t)
        size = len(dataset)
        train_loss = 0
        print(f'Training dataset size: {size}')
        model.train()
        
        for sample_no, sample in enumerate(dataset):
            # Compute prediction and loss
            pred = model(sample['nodes'],sample['edges'],sample['join_index'])
            #optimizer.zero_grad()
            loss = loss_fn(pred, torch.reshape(sample['target'],(1,1)))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            #print(f"Gt : {sample['target']}, pred: {pred}")
            #
            # Backpropagation
            
            #
            if sample_no % 5000 == 0:
                #loss_current, current = loss.item(), (sample_no + 1) * len(sample['nodes'])
                #print(f"Epoch: {epoch:4} {sample_no:5} loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
                print(f"Epoch: {t+1:4} {(sample_no+1):8} Average loss: {train_loss/(sample_no+1):>7f}  [{((sample_no+1)/size)*100:.2f}]")
        
        #prev_val_loss = val_loop02(dataset=test_dataset, loss_fn=loss_fn,epoch_no=t, prev_loss = prev_val_loss, path_to_save =path_to_save)
        test_loss= 0
        with torch.no_grad():
            for no,sample in enumerate(test_dataset):
                pred = model(sample['nodes'],sample['edges'],sample['join_index'])
                test_loss += loss_fn(pred, torch.reshape(sample['target'],(1,1))).item()
        test_loss= test_loss/len(test_dataset)
        p_val_hist = val_hist
        val_hist.append(test_loss)
        if path_to_save != None and (prev_val_loss == None or test_loss < prev_val_loss):
            torch.save(model,f"{path_to_save}model_{t}.pt")
            prev_val_loss = test_loss
        
        print(f"Val Error {t+1:4}: Avg Loss: {test_loss:>8f}  \n")
        print(f"Optimal Val loss : {prev_val_loss}\n")
        
        if np.sum([1 for v_l in p_val_hist[early_stop:] if v_l <= test_loss]) == early_stop:
            print(f"Early Stopping invoked after epoch {t}")
            exit()
        #test_loop(test_dataloader, model, loss_fn)
    print("Done!")
#TODO outcomment this
#for i in range(5):
#    print(dataset[i])
#data = next(iter(loader))
#print(data)
if __name__ == "__main__":
    #Model hyper parameters
    
    parser = configparser.ConfigParser()
    parser.read(PATH_TO_CONFIG_GRAPH)
    
    EPOCHS = int(parser['Training']['EPOCHS'])
    BATCH_SIZE = int(parser['Training']['BATCH_SIZE'])
    LR = float(parser['Training']['LR'])
    WEIGHT_DECAY = float(parser['Training']['WEIGHT_DECAY'])
    PREDICATE_BINS = int(parser['Training']['PREDICATE_BINS'])


    data_file = '/work/data/train_data.json'
    test_file = '/work/data/test_data.json'
    feat_generation_path = '/work/data/confs/newPredExtractionRun/pred_feat_01_04_2023_07_48.pickle'
    train_bgp_file_pickle = '/work/data/confs/newPredExtractionRun/train_bgps.pickle'
    test_bgp_file_pickle = '/work/data/confs/newPredExtractionRun/test_bgps.pickle'
    #pickle_data_loader = '/work/data/confs/newPredExtractionRun/train_dataloader.pickle'
    pickle_train_dataset = '/work/data/confs/newPredExtractionRun/train_dataset.pickle'
    pickle_testdataset = '/work/data/confs/newPredExtractionRun/test_dataset.pickle'
    
    data_file = parser['DebugDataset']['train_data']
    val_file = parser['DebugDataset']['val_data']
    test_file = parser['DebugDataset']['test_data']
    feat_generation_path = parser['PredicateFeaturizerSubObj']['load_path']
    train_bgp_file_pickle = parser['DebugDataset']['train_bgp_file_pickle']
    val_bgp_file_pickle = parser['DebugDataset']['val_bgp_file_pickle']
    test_bgp_file_pickle = parser['DebugDataset']['test_bgp_file_pickle']
    
    pickle_train_dataset = parser['DebugDataset']['pickle_train_dataset']
    pickle_testdataset = parser['DebugDataset']['pickle_testdataset']
    pickle_valdataset = parser['DebugDataset']['pickle_valdataset']
    
    dataset = BGPDataset_v2(parser,data_file)
    test_dataset = BGPDataset_v2(parser,test_file)
    val_dataset = BGPDataset_v2(parser,val_file)
    
    torch.save(dataset,pickle_train_dataset)
    torch.save(test_dataset,pickle_testdataset)
    torch.save(val_dataset,pickle_valdataset)
    """if not os.path.isfile(pickle_train_dataset):
        dataset = BGPDataset_v2(parser,data_file)
        test_dataset = BGPDataset_v2(parser,test_file)
        val_dataset = BGPDataset_v2(parser,val_file)
        
        torch.save(dataset,pickle_train_dataset)
        torch.save(test_dataset,pickle_testdataset)
        torch.save(val_dataset,pickle_valdataset)
    else:
        #loader = torch.load(open(pickle_data_loader,'rb'))
        dataset = torch.load(pickle_train_dataset)
        test_dataset= torch.load(pickle_testdataset)
        val_dataset= torch.load(pickle_valdataset)
        print(f"Dataset size: (Train) {len(dataset)}, (Val) {len(val_dataset)}, (Test) {len(test_dataset)}")"""
        
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
    
    #run(dataset, test_dataset,loss_fn,optimizer, epoch=EPOCHS, path_to_save=parser['Results']['path_to_save_model'])
    run_undivided(dataset, test_dataset,loss_fn,optimizer, epoch=EPOCHS, path_to_save=parser['Results']['path_to_save_model'])