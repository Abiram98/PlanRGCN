import os
import dgl
from graph_construction.featurizer import FeaturizerPredCo
from graph_construction.query_graph import QueryPlanCommonBi, snap_lat2onehot
from trainer.data_util import DatasetPrep
from trainer.model import Classifier as CLS
import torch as th
import numpy as np
import json
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import torch.nn.functional as F
import pandas as pd


def snap_pred(pred):
    if not isinstance(pred, th.Tensor):
        pred = th.tensor(snap_lat2onehot(pred), dtype=th.float32)
    return th.argmax(pred)


class Trainer:
    def __init__(
        self,
        train_path="/qpp/dataset/DBpedia_2016_12k_sample/train_sampled.tsv",
        val_path="/qpp/dataset/DBpedia_2016_12k_sample/val_sampled.tsv",
        test_path="/qpp/dataset/DBpedia_2016_12k_sample/test_sampled.tsv",
        batch_size=32,
        query_plan_dir="/PlanRGCN/extracted_features/queryplans/",
        pred_stat_path="/PlanRGCN/extracted_features/predicate/pred_stat/batches_response_stats",
        time_col="mean_latency",
        cls_func=snap_lat2onehot,
        # in_dim=12,
        hidden_dim=48,
        n_classes=6,
        featurizer_class=FeaturizerPredCo,
        query_plan=QueryPlanCommonBi,
        model=CLS,
    ) -> None:
        dgl.seed(1223)
        prepper = DatasetPrep(
            train_path=train_path,
            val_path=val_path,
            test_path=test_path,
            batch_size=batch_size,
            query_plan_dir=query_plan_dir,
            pred_stat_path=pred_stat_path,
            time_col=time_col,
            cls_func=cls_func,
            query_plan=query_plan,
            featurizer_class=featurizer_class,
        )
        self.train_loader = prepper.get_trainloader()
        self.val_loader = prepper.get_valloader()
        self.test_loader = prepper.get_testloader()

        self.model = model(prepper.vec_size, hidden_dim, n_classes)

    def train(
        self,
        early_stop=10,
        lr=0.001,
        wd=0.01,
        epochs=100,
        result_path="/PlanRGCN/results/results.json",
        path_to_save="/PlanRGCN/plan_model",
        loss_type="cross-entropy",
        verbosity=1,
    ):
        """Trains a model,  \nHyperparameters: early_stop, lr, wd, epochs"""
    
        opt = th.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=wd)
        # opt = th.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

        prev_val_loss = None
        val_hist = []

        # F1 scores
        train_f1_hist = []
        val_f1_hist = []
        metric_data = {
            "train_loss": [],
            "val_loss": [],
            "test_loss": [],
            "train_f1": [],
            "val_f1": [],
            "test_f1": [],
            "train_prec": [],
            "val_prec": [],
            "test_prec": [],
            "train_recall": [],
            "val_recall": [],
            "test_recall": [],
        }
        best_epoch = 0
        best_f1 = 0
        best_model_path = ""
        for epoch in range(epochs):
            if verbosity >= 2:
                print(
                    f"Epoch {epoch+1}\n--------------------------------------------------------------"
                )

            train_loss = 0
            train_f1 = 0
            train_recall = 0
            train_prec = 0
            self.model.train()
            with th.enable_grad():
                for batch_no, (batched_graph, labels, ids) in enumerate(
                    self.train_loader
                ):
                    feats = batched_graph.ndata["node_features"]
                    edge_types = batched_graph.edata["rel_type"]
                    logits = self.model(batched_graph, feats, edge_types)
                    if loss_type == "cross-entropy":
                        loss = F.cross_entropy(logits, labels)
                    elif loss_type == "mse":
                        loss = F.mse_loss(logits, labels)
                    # loss = F.mse_loss(logits,labels)
                    # cross_entropy(logits, labels)
                    opt.zero_grad()
                    loss.backward()
                    # print(f"{labels}, {logits}")
                    opt.step()

                    c_train_loss = loss.item()

                    # snap_pred(logits,model_thres=pred_thres, add_thres=add_thres)
                    # snap_thres = [pred_thres for x in logits]
                    # snap_add_thres = [add_thres for x in logits]
                    f1_pred = list(map(snap_pred, logits))
                    snapped_labels = list(map(snap_pred, labels))
                    # f1_batch = f1_score(labels, f1_pred)
                    f1_batch = f1_score(snapped_labels, f1_pred, average="micro")
                    prec_batch = precision_score(
                        snapped_labels, f1_pred, average="micro"
                    )
                    recall_batch = recall_score(
                        snapped_labels, f1_pred, average="micro"
                    )
                    if verbosity >= 2:
                        print(
                            f"Epoch: {epoch+1:4} {(batch_no+1):8} Batch loss: {c_train_loss:>7f} Batch F1: {f1_batch}"
                        )
                    train_loss += c_train_loss
                    train_f1 += f1_batch
                    train_recall += recall_batch
                    train_prec += prec_batch

            val_loss = 0
            val_f1 = 0
            val_recall = 0
            val_prec = 0
            self.model.eval()
            with th.no_grad():
                for no, (graphs, labels, ids) in enumerate(self.val_loader):
                    feats = graphs.ndata["node_features"]
                    edge_types = graphs.edata["rel_type"]
                    pred = self.model(graphs, feats, edge_types)
                    """if loss_type == "cross-entropy":
                        c_val_loss = F.cross_entropy(pred, labels).item()
                    elif loss_type == "mse":
                        c_val_loss = F.mse_loss(pred, labels).item()"""
                    if loss_type == "cross-entropy":
                        c_val_loss = F.cross_entropy(pred, labels).item()
                    elif loss_type == "mse":
                        c_val_loss = F.mse_loss(pred, labels).item()
                    val_loss += c_val_loss
                    # snap_thres = [pred_thres for x in pred]
                    # snap_add_thres = [add_thres for x in pred]
                    f1_pred_val = list(map(snap_pred, pred))
                    snapped_lebls = list(map(snap_pred, labels))

                    # f1_batch_val = f1_score(labels, f1_pred_val)
                    f1_batch_val = f1_score(snapped_lebls, f1_pred_val, average="micro")
                    val_f1 += f1_batch_val

                    prec_batch_val = precision_score(
                        snapped_lebls, f1_pred_val, average="micro"
                    )
                    val_prec += prec_batch_val

                    recall_batch_val = recall_score(
                        snapped_lebls, f1_pred_val, average="micro"
                    )
                    val_recall += recall_batch_val
            test_loss = 0
            test_f1 = 0
            test_recall = 0
            test_prec = 0
            self.model.eval()
            with th.no_grad():
                for no, (graphs, labels, ids) in enumerate(self.test_loader):
                    feats = graphs.ndata["node_features"]
                    edge_types = graphs.edata["rel_type"]
                    pred = self.model(graphs, feats, edge_types)
                    # c_test_loss = F.mse_loss(pred, labels).item()
                    if loss_type == "cross-entropy":
                        c_test_loss = F.cross_entropy(pred, labels).item()
                    elif loss_type == "mse":
                        c_test_loss = F.mse_loss(pred, labels).item()
                    # test_loss += loss_fn(pred, torch.reshape(sample['target'],(1,1))).item()
                    test_loss += c_test_loss
                    # snap_thres = [pred_thres for x in pred]
                    # snap_add_thres = [add_thres for x in pred]
                    f1_pred_test = list(map(snap_pred, pred))
                    snapped_test = list(map(snap_pred, labels))

                    # f1_batch_val = f1_score(labels, f1_pred_val)
                    f1_batch_test = f1_score(
                        snapped_test, f1_pred_test, average="micro"
                    )
                    test_f1 += f1_batch_test
                    prec_batch_test = precision_score(
                        snapped_test, f1_pred_test, average="micro"
                    )
                    test_prec += prec_batch_test
                    recall_batch_test = recall_score(
                        snapped_test, f1_pred_test, average="micro"
                    )
                    test_recall += recall_batch_test

            test_loss = test_loss / len(self.test_loader)
            metric_data["test_loss"].append(test_loss)
            test_f1 = test_f1 / len(self.test_loader)
            metric_data["test_f1"].append(test_f1)
            test_prec = test_prec / len(self.test_loader)
            metric_data["test_prec"].append(test_prec)
            test_recall = test_recall / len(self.test_loader)
            metric_data["test_recall"].append(test_recall)

            val_loss = val_loss / len(self.val_loader)
            metric_data["val_loss"].append(val_loss)
            val_f1 = val_f1 / len(self.val_loader)
            metric_data["val_f1"].append(val_f1)
            val_prec = val_prec / len(self.val_loader)
            metric_data["val_prec"].append(val_prec)
            val_recall = val_recall / len(self.val_loader)
            metric_data["val_recall"].append(val_recall)

            train_loss = train_loss / len(self.train_loader)
            metric_data["train_loss"].append(train_loss)
            train_f1 = train_f1 / len(self.train_loader)
            metric_data["train_f1"].append(train_f1)
            train_recall = train_recall / len(self.train_loader)
            metric_data["train_recall"].append(train_recall)
            train_prec = train_prec / len(self.train_loader)
            metric_data["train_prec"].append(train_prec)

            train_f1_hist.append(train_f1)
            val_f1_hist.append(val_f1)

            p_val_hist = val_hist
            val_hist.append(val_loss)
            if path_to_save != None and (
                prev_val_loss == None or val_loss < prev_val_loss
            ):
                th.save(self.model, f"{path_to_save}/model_{epoch+1}.pt")
                prev_val_loss = val_loss
                best_epoch = epoch + 1
            if path_to_save != None and (val_f1 > best_f1):
                th.save(self.model, f"{path_to_save}/best_f1_model_{epoch+1}.pt")
                best_model_path = f"{path_to_save}/best_f1_model_{epoch+1}.pt"
                self.best_model_path = best_model_path
                best_f1 = val_f1
            if verbosity >= 2:
                print(f"Train Avg Loss {epoch+1:4}: {train_loss:>8f}\n")
                print(f"Train Avg F1 {epoch+1:4}: {train_f1}\n")
                print(f"Val Avg Loss {epoch+1:4}: {val_loss:>8f}\n")
                print(f"Val Avg F1 {epoch+1:4}:  {val_f1}\n")
                print(f"Optimal Val loss (Epoch {best_epoch}): {prev_val_loss}\n")

            if (early_stop < len(p_val_hist)) and (
                np.sum([1 for v_l in p_val_hist[-early_stop:] if v_l <= val_loss])
                == early_stop
            ):
                if verbosity >= 1:
                    print(f"Early Stopping invoked after epoch {epoch+1}")
                    json.dump(metric_data, open(result_path, "w"))
                    print("Done!")
                return
        print("Done!")
        json.dump(metric_data, open(result_path, "w"))
        print(best_model_path)
        self.best_model_path = best_model_path
        return best_model_path

    def predict(self, path_to_save="/PlanRGCN/results"):
        os.system(f"mkdir -p {path_to_save}")
        self.model = th.load(self.best_model_path)
        with th.no_grad():
            train_p = f"{path_to_save}/train_pred.csv"
            val_p = f"{path_to_save}/val_pred.csv"
            test_p = f"{path_to_save}/test_pred.csv"
            for loader, path in zip(
                [self.train_loader, self.val_loader, self.test_loader],
                [train_p, val_p, test_p],
            ):
                ids, preds, truths = self.predict_helper(loader)
                df = pd.DataFrame()
                df["id"] = ids
                df["time_cls"] = truths
                df["planrgcn_prediction"] = preds
                df.to_csv(path, index=False)
                # id,time,nn_prediction

    def predict_helper(self, dataloader):
        all_preds = []
        all_ids = []
        all_truths = []
        for graphs, labels, ids in dataloader:
            feats = graphs.ndata["node_features"]
            edge_types = graphs.edata["rel_type"]
            pred = self.model(graphs, feats, edge_types)
            pred = pred.tolist()
            pred = [np.argmax(x) for x in pred]
            truths = [np.argmax(x) for x in labels.tolist()]
            all_truths.extend(truths)
            ids = [f"http://lsq.aksw.org/{x}" for x in ids]
            all_ids.extend(ids)
            all_preds.extend(pred)
        return all_ids, all_preds, all_truths


# t = Trainer()
# t.train()
if __name__ == "__main__":
    from trainer.model import RegressorWSelfTriple as CLS
    from graph_construction.featurizer import FeaturizerPredStats
    from graph_construction.query_graph import QueryPlanCommonBi

    t = Trainer(
        featurizer_class=FeaturizerPredStats,
        query_plan=QueryPlanCommonBi,
        cls_func=lambda x: x,
        model=CLS,
    )
    t.train(
        epochs=100,
        verbosity=2,
        result_path="/PlanRGCN/results/results.json",
        path_to_save="/PlanRGCN/plan_model",
        loss_type="mse",
    )
    t.predict(path_to_save="/PlanRGCN/results_reg")
