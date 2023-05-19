from classifier.deprecated.trainer import run
import configparser
from feature_extraction.constants import PATH_TO_CONFIG
from classifier.bgp_dataset_v2 import BGPDataset_v2
from classifier.GCN import GNN
import torch.nn as nn, torch, numpy as np
from torch_geometric import seed_everything

torch.manual_seed(12345)
np.random.seed(12345)
seed_everything(12345)

if __name__ == "__main__":
    parser = configparser.ConfigParser()
    parser.read(PATH_TO_CONFIG)
    EPOCHS = 50
    BATCH_SIZE = 200
    LR = 1e-3
    WEIGHT_DECAY = 5e-4
    
    data_file = parser['DebugDataset']['train_data']
    train_dataset = BGPDataset_v2(parser,data_file)
    val_dataset = BGPDataset_v2(parser,parser['DebugDataset']['val_data'])
    test_dataset = BGPDataset_v2(parser,parser['DebugDataset']['test_data'])
    model = GNN(train_dataset.node_features[0].shape[1] , train_dataset.node_features[0].shape[1]*2, hidden_dimension=10)
    model = model.float()

    #optimizer and loss function
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    run(train_dataset,val_dataset,model,loss_fn,optimizer, path_to_save=parser['Results']['path_to_save_model'], epoch=EPOCHS)

