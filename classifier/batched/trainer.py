import math,os
import torch.nn as nn, torch, numpy as np
from torch_geometric import seed_everything
from classifier.batched.gcn import GNN
import configparser
from classifier.batched.hetero_dataset import get_graph_representation
from feature_extraction.constants import PATH_TO_CONFIG_GRAPH
from torch_geometric.loader import DataLoader
torch.manual_seed(12345)
np.random.seed(12345)
seed_everything(12345)

def run_undivided(train_loader:DataLoader, validation_loader:DataLoader, path_to_save ='/work/data/confs/newPredExtractionRun/', epoch=50, early_stop=10, layer_size=100):
    #global train_loader
    #train_loader = return_graph_dataloader(dataset, batch_size=BATCH_SIZE)
    model = GNN(train_loader.dataset[0].x.shape[1] , train_loader.dataset[0].x.shape[1]*2)
    model = model.float()

    #optimizer and loss function
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    
    prev_val_loss = None
    val_hist = []
    best_epoch = 0
    for t in range(epoch):
        print(f"Epoch {t+1}\n--------------------------------------------------------------")

        train_loss = 0
        model.train()
        for sample_no, batch in enumerate(train_loader):
            pred = model(batch)
            #optimizer.zero_grad()
            #loss = loss_fn(pred, torch.reshape(sample['target'],(1,1)))
            loss = loss_fn(pred, batch.y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            c_train_loss = loss.item()
            
            print(f"Epoch: {t+1:4} {(sample_no+1):8} Average loss: {c_train_loss:>7f}")
            train_loss += c_train_loss
        val_loss= 0
        with torch.no_grad():
            for no,sample in enumerate(validation_loader):
                pred = model(sample)
                c_test_loss = loss_fn(pred, sample.y).item()
                #test_loss += loss_fn(pred, torch.reshape(sample['target'],(1,1))).item()
                val_loss += c_test_loss
        val_loss= val_loss/len(val_loader)
        train_loss = train_loss/len(train_loader)
        
        p_val_hist = val_hist
        val_hist.append(val_loss)
        if path_to_save != None and (prev_val_loss == None or val_loss < prev_val_loss):
            torch.save(model,f"{path_to_save}model_{t+1}.pt")
            prev_val_loss = val_loss
            best_epoch = t+1
        
        print(f"Train Error {t+1:4}: Avg Loss: {train_loss:>8f}  \n")
        print(f"Val Error {t+1:4}: Avg Loss: {val_loss:>8f}  \n")
        print(f"Optimal Val loss (Epoch {best_epoch}): {prev_val_loss}\n")
        
        if np.sum([1 for v_l in p_val_hist[early_stop:] if v_l <= val_loss]) == early_stop:
            print(f"Early Stopping invoked after epoch {t+1}")
            exit()
    print("Done!")

if __name__ == "__main__":
    
    parser = configparser.ConfigParser()
    parser.read(PATH_TO_CONFIG_GRAPH)
    
    #Model hyper parameters
    EPOCHS = int(parser['Training']['EPOCHS'])
    BATCH_SIZE = int(parser['Training']['BATCH_SIZE'])
    LR = float(parser['Training']['LR'])
    WEIGHT_DECAY = float(parser['Training']['WEIGHT_DECAY'])
    PREDICATE_BINS = int(parser['Training']['PREDICATE_BINS'])
    
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
    train_loader = DataLoader(get_graph_representation(parser,data_file), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(get_graph_representation(parser,test_file), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(get_graph_representation(parser,val_file), batch_size=BATCH_SIZE, shuffle=True)
    
    torch.save(train_loader,pickle_train_dataset)
    torch.save(test_loader,pickle_testdataset)
    torch.save(val_loader,pickle_valdataset)
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
    
    
    #run(dataset, test_dataset,loss_fn,optimizer, epoch=EPOCHS, path_to_save=parser['Results']['path_to_save_model'])
    print("Model with variable positions in join nodes")
    run_undivided(train_loader, val_loader, epoch=EPOCHS, path_to_save=parser['Results']['path_to_save_model_batched'], layer_size=int(parser['Training']['layer_size']))