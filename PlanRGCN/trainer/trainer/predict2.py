import argparse
import pickle

import torch as th
from trainer.model import Classifier2RGCN
import os
import time
import numpy as np
import pandas as pd

def predict(
    model,
    train_loader,
    val_loader,
    test_loader,
    is_lsq,
    path_to_save="/PlanRGCN/temp_results",
):
    def predict_helper(dataloader, model, is_lsq):
        all_preds = []
        all_preds_unprocessed = []
        all_ids = []
        all_truths = []
        all_time = []
        for graphs, labels, id in dataloader.dataset:
            start = time.time()
            feats = graphs.ndata["node_features"]
            edge_types = graphs.edata["rel_type"]
            pred = model(graphs, feats, edge_types)
            end = time.time()
            all_time.append(end-start)
            or_pred = pred.tolist()
            all_preds_unprocessed.append(or_pred)
            pred = np.argmax(or_pred)
            truths = np.argmax(labels.tolist())
            all_truths.append(truths)
            if not is_lsq:
                id = f"http://lsq.aksw.org/{id}"
            else:
                id = f"{id}"
            all_ids.append(id)
            all_preds.append(pred)
        return all_ids, all_preds, all_truths, all_time,all_preds_unprocessed
    
    os.system(f"mkdir -p {path_to_save}")
    model.eval()
    with th.no_grad():
        train_p = os.path.join(path_to_save,'train_pred.csv')
        val_p = os.path.join(path_to_save,"val_pred.csv")
        test_p = os.path.join(path_to_save,"test_pred.csv")
        
        for loader, path in zip(
            [ val_loader, test_loader],
            [ val_p, test_p],
        ):
            print('predict helper begin')
            ids, preds, truths, durations,all_preds_unprocessed = predict_helper(loader, model, is_lsq)
            df = pd.DataFrame()
            df["id"] = ids
            df["time_cls"] = truths
            df["planrgcn_prediction"] = preds
            df["inference_durations"] = durations
            df["planrgcn_prediction_no_thres"] = all_preds_unprocessed
            df.to_csv(path, index=False)






parser = argparse.ArgumentParser(prog='PredUtil', description ='Prediction utility of existing model')
parser.add_argument('-p', '--prepper', help='Path to prepper')
parser.add_argument('-m', '--model_state', help='Path to model state o f best model')
parser.add_argument('-n', '--n_classes', default=3, type=int, help='time interval numbers')
parser.add_argument('-o', '--save_path', help='path to save the results')
parser.add_argument('--l1',type=int, default=None, help='path to save the results')
parser.add_argument( '--l2', type=int, default=None,help='path to save the results')
parser.add_argument('-d', '--dropout', type=float, default=0.0,help='path to save the results')

args = parser.parse_args()
print(args)
with open(args.prepper, 'rb') as f:
    prepper = pickle.load(f)
print('train loading')
train_loader = prepper.train_loader
print('val loading')
val_loader = prepper.val_loader
print('tes loading')
test_loader = prepper.test_loader
print('model loading')
best_trained_model = Classifier2RGCN(
    prepper.vec_size,
    args.l1 if args.l1 != None else prepper.config["l1"],
    args.l2 if args.l2 != None else prepper.config["l2"],
    args.dropout if args.dropout != None else prepper.config["dropout"],
    args.n_classes,
)
model_state = th.load(args.model_state)
best_trained_model.load_state_dict(model_state["model_state"])

print(prepper.vec_size)
predict(
    best_trained_model,
    train_loader,
    val_loader,
    test_loader,
    True,
    path_to_save=args.save_path,
)
