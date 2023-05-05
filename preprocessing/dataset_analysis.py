# The goal with this script is to analyse the batching of the dataset (even distributions in the batches)
import os
import configparser
from classifier.batched.trainer import get_debug_data_loader
from feature_extraction.constants import PATH_TO_CONFIG_GRAPH
import torch

def ground_truth_distibution(batch, verbose= False, returnDistribution= False):
    ground_truth_1 = 0
    ground_truth_0 = 0
    for y in batch.y:
        if y == 1:
            ground_truth_1 += 1
        elif y == 0:
            ground_truth_0 += 1
    if verbose:
        print(f"Ground truth distribtution:\n\t1: {ground_truth_1}/{ground_truth_1+ground_truth_0}, {ground_truth_1/(ground_truth_1+ground_truth_0)}")
        print(f"\t0: {ground_truth_0}/{ground_truth_1+ground_truth_0}, {ground_truth_0/(ground_truth_1+ground_truth_0)}")
    if returnDistribution:
        total = ground_truth_0 + ground_truth_1
        return ground_truth_0/total, ground_truth_1/total
    
    return ground_truth_0, ground_truth_1

if __name__ == "__main__":
    parser = configparser.ConfigParser()
    parser.read(PATH_TO_CONFIG_GRAPH)
    train_path = 'trainloader.pth'
    val_path = 'valloader.pth'
    test_path = 'testloader.pth'
    if os.path.exists(train_path):
        train_loader = torch.load(train_path)
        val_loader = torch.load(val_path)
        test_loader = torch.load(test_path)
    else:
        train_loader, val_loader, test_loader = get_debug_data_loader(parser)
        torch.save(train_loader,train_path)
        torch.save(val_loader,val_path)
        torch.save(test_loader,test_path)
    total_gt_0, total_gt_1 = 0,0
    for batch in train_loader:
        gt_0,gt_1 = ground_truth_distibution(batch,verbose=True,returnDistribution=True)
        total_gt_0 += gt_0
        total_gt_1 += gt_1
    print(f"Mean distribution acrosss batches:\n\t1: {total_gt_1/len(train_loader)}\n\t0: {total_gt_0/len(train_loader)}")