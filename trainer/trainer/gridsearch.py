"""
The purpose of this module is hyperparameter search of the models.
"""
import os
from pathlib import Path
import dgl
from graph_construction.featurizer import FeaturizerPredCo, FeaturizerPredCoEnt
from graph_construction.featurizer import FeaturizerSubjPred
from graph_construction.query_graph import (
    QueryPlan,
    snap_lat2onehot,
    snap_lat2onehot_binary,
    snap_lat2onehotv2,
    snap_lat_2onehot_4_cat,
)
from trainer.data_util import DatasetPrep
from trainer.model import (
    Classifier as CLS,
    ClassifierGridSearch,
    RegressorWSelfTriple as CLS,
)
from trainer.model import Classifier2RGCN
import torch as th
import numpy as np
import json
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import torch.nn.functional as F
import pandas as pd
import torch.nn as nn
from trainer.train import Trainer
from time import time

from trainer.utils import tablify
import argparse


def model_gen(in_dim, n_classes):
    rel_layer1 = [256, 128, 512]
    rel_layer2 = [256, 128, 512]
    for l1 in rel_layer1:
        for l2 in rel_layer2:
            layers = [(1, l1), (1, l2)]
            yield ClassifierGridSearch(in_dim, layers, n_classes)


AVG = "macro"


class GridSearch:
    def __init__(
        self,
        splt_data={
            "train_path": None,
            "val_path": None,
            "test_path": None,
            "query_plan": None,
        },
        pred_info={
            "pred_stat_path": "/PlanRGCN/extracted_features/predicate/pred_stat/batches_response_stats",
            "pred_com_path": "/PlanRGCN/data/pred/pred_co/pred2index_louvain.pickle",
        },
        ent_info={
            "ent_path": "/PlanRGCN/extracted_features/entities/ent_stat/batches_response_stats"
        },
        time_col="mean_latency",
        is_lsq=True,
        cls_func=snap_lat2onehot,
        n_classes=6,
        featurizer_class=FeaturizerPredCoEnt,
        scaling="robust",
        # scaling="std",
        query_plan=QueryPlan,
        output_path=None,
        model_gen=model_gen,
    ) -> None:
        dgl.seed(1223)

        self.train_path = splt_data["train_path"]
        self.val_path = splt_data["val_path"]
        self.test_path = splt_data["test_path"]
        if splt_data["query_plan"].endswith("/"):
            self.qp_dir = splt_data["query_plan"]
        else:
            self.qp_dir = splt_data["query_plan"] + "/"

        # Feature Statistics
        self.pred_info = pred_info
        self.ent_info = ent_info

        # Column to use in input files
        self.time_col = time_col
        # Function used to snap prediction to their corresponding categories.
        self.cls_func = cls_func

        # Hyperparameters
        self.batch_sizes = [64]  # [32, 64]
        self.learning_rates = [1e-5]  # [0.001, 1e-5]
        self.weight_decays = [0.01]
        self.scaling = scaling

        self.output_path = output_path

        self.featurizer_class = featurizer_class
        self.query_plan = query_plan
        self.is_lsq = is_lsq
        self.n_classes = n_classes

        self.model_gen = model_gen

        self.epochs = 100

    def search(self):
        results = []
        for batch_size in self.batch_sizes:
            prepper = DatasetPrep(
                train_path=self.train_path,
                val_path=self.val_path,
                test_path=self.test_path,
                batch_size=batch_size,
                query_plan_dir=self.qp_dir,
                pred_stat_path=self.pred_info["pred_stat_path"],
                pred_com_path=self.pred_info["pred_com_path"],
                ent_path=self.ent_info["ent_path"],
                pred_end_path=self.pred_info["pred_ent"],
                time_col=self.time_col,
                cls_func=self.cls_func,
                query_plan=self.query_plan,
                featurizer_class=self.featurizer_class,
                is_lsq=self.is_lsq,
                scaling=self.scaling,
            )
            p = Path(self.train_path)
            loss = "cross-entropy"
            if (p.parent / "loss_weight.json").exists():
                with open(p.parent / "loss_weight.json", "r") as f:
                    W = json.load(f)
                loss = nn.CrossEntropyLoss(th.tensor(W))
            for model in self.model_gen(prepper.vec_size, self.n_classes):
                t = Trainer(
                    is_lsq=self.is_lsq,
                    prepper=prepper,
                    is_model_provided=True,
                    model=model,
                    cls_func=self.cls_func,
                )
                for lr in self.learning_rates:
                    for wd in self.weight_decays:
                        in_dim, rgcn_info, fc_info, _ = model.get_dims()
                        model_folder_name = (
                            f"batch_size_{batch_size}_lr_{lr}_wd_{wd}_{model.model_str}"
                        )
                        os.system(
                            f"mkdir -p {self.output_path}/{model_folder_name}/results"
                        )
                        os.system(
                            f"mkdir -p {self.output_path}/{model_folder_name}/models"
                        )
                        if not os.path.exists(
                            f"{self.output_path}/{model_folder_name}/model_arch.txt"
                        ):
                            with open(
                                f"{self.output_path}/{model_folder_name}/model_arch.txt",
                                "w",
                            ) as f:
                                f.write(str(model))
                                f.flush()
                        start = time()

                        _, val_f1 = t.train(
                            early_stop=10,
                            lr=lr,
                            wd=wd,
                            epochs=self.epochs,
                            result_path=f"{self.output_path}/{model_folder_name}/results/result.json",
                            path_to_save=f"{self.output_path}/{model_folder_name}/models",
                            loss_type=loss,
                            verbosity=2,
                            is_return_f1_val=True,
                        )
                        end = time()
                        t.predict(
                            path_to_save=f"{self.output_path}/{model_folder_name}/results"
                        )
                        res = {
                            "Batch Size": batch_size,
                            "Learning rate": lr,
                            "Weight Decay": wd,
                            "Best Val F1 Score": val_f1,
                            "Input Dimension": in_dim,
                            "Training Time": end - start,
                        }
                        for rgcn_label in rgcn_info.keys():
                            res[rgcn_label] = rgcn_info[rgcn_label]
                        for fc_label in fc_info.keys():
                            res[fc_label] = fc_info[fc_label]
                        results.append(res)

                        df = pd.DataFrame(results)
                        result_sum = f"{self.output_path}/train_summary.txt"
                        result_praw = f"{self.output_path}/train_summary_raw.csv"
                        df.to_csv(result_praw, index=False)
                        print(
                            tablify(
                                df.to_latex(
                                    index=False,
                                    caption="Training Summary over different Confs",
                                    float_format="%.4f",
                                )
                            )
                        )
        with open(result_sum, "w") as f:
            f.write(
                tablify(
                    df.to_latex(
                        index=False,
                        caption="Training Summary over different Confs",
                        float_format="%.4f",
                    )
                )
            )
class RayGridSearch:
    def __init__(
        self,
        splt_data={
            "train_path": None,
            "val_path": None,
            "test_path": None,
            "query_plan": None,
        },
        pred_info={
            "pred_stat_path": "/PlanRGCN/extracted_features/predicate/pred_stat/batches_response_stats",
            "pred_com_path": "/PlanRGCN/data/pred/pred_co/pred2index_louvain.pickle",
        },
        ent_info={
            "ent_path": "/PlanRGCN/extracted_features/entities/ent_stat/batches_response_stats"
        },
        time_col="mean_latency",
        is_lsq=True,
        cls_func=snap_lat2onehot,
        n_classes=6,
        featurizer_class=FeaturizerPredCoEnt,
        scaling="robust",
        # scaling="std",
        query_plan=QueryPlan,
        output_path=None,
        model_gen=model_gen,
    ) -> None:
        dgl.seed(1223)

        self.train_path = splt_data["train_path"]
        self.val_path = splt_data["val_path"]
        self.test_path = splt_data["test_path"]
        if splt_data["query_plan"].endswith("/"):
            self.qp_dir = splt_data["query_plan"]
        else:
            self.qp_dir = splt_data["query_plan"] + "/"

        # Feature Statistics
        self.pred_info = pred_info
        self.ent_info = ent_info

        # Column to use in input files
        self.time_col = time_col
        # Function used to snap prediction to their corresponding categories.
        self.cls_func = cls_func

        # Hyperparameters
        self.batch_sizes = [64]  # [32, 64]
        self.learning_rates = [1e-5]  # [0.001, 1e-5]
        self.weight_decays = [0.01]
        self.scaling = scaling

        self.output_path = output_path

        self.featurizer_class = featurizer_class
        self.query_plan = query_plan
        self.is_lsq = is_lsq
        self.n_classes = n_classes

        self.model_gen = model_gen

        self.epochs = 100

    def search(self):
        results = []
        for batch_size in self.batch_sizes:
            prepper = DatasetPrep(
                train_path=self.train_path,
                val_path=self.val_path,
                test_path=self.test_path,
                batch_size=batch_size,
                query_plan_dir=self.qp_dir,
                pred_stat_path=self.pred_info["pred_stat_path"],
                pred_com_path=self.pred_info["pred_com_path"],
                ent_path=self.ent_info["ent_path"],
                pred_end_path=self.pred_info["pred_ent"],
                time_col=self.time_col,
                cls_func=self.cls_func,
                query_plan=self.query_plan,
                featurizer_class=self.featurizer_class,
                is_lsq=self.is_lsq,
                scaling=self.scaling,
            )
            p = Path(self.train_path)
            loss = "cross-entropy"
            if (p.parent / "loss_weight.json").exists():
                with open(p.parent / "loss_weight.json", "r") as f:
                    W = json.load(f)
                loss = nn.CrossEntropyLoss(th.tensor(W))
            for model in self.model_gen(prepper.vec_size, self.n_classes):
                t = Trainer(
                    is_lsq=self.is_lsq,
                    prepper=prepper,
                    is_model_provided=True,
                    model=model,
                    cls_func=self.cls_func,
                )
                for lr in self.learning_rates:
                    for wd in self.weight_decays:
                        in_dim, rgcn_info, fc_info, _ = model.get_dims()
                        model_folder_name = (
                            f"batch_size_{batch_size}_lr_{lr}_wd_{wd}_{model.model_str}"
                        )
                        os.system(
                            f"mkdir -p {self.output_path}/{model_folder_name}/results"
                        )
                        os.system(
                            f"mkdir -p {self.output_path}/{model_folder_name}/models"
                        )
                        if not os.path.exists(
                            f"{self.output_path}/{model_folder_name}/model_arch.txt"
                        ):
                            with open(
                                f"{self.output_path}/{model_folder_name}/model_arch.txt",
                                "w",
                            ) as f:
                                f.write(str(model))
                                f.flush()
                        start = time()

                        _, val_f1 = t.train(
                            early_stop=10,
                            lr=lr,
                            wd=wd,
                            epochs=self.epochs,
                            result_path=f"{self.output_path}/{model_folder_name}/results/result.json",
                            path_to_save=f"{self.output_path}/{model_folder_name}/models",
                            loss_type=loss,
                            verbosity=2,
                            is_return_f1_val=True,
                        )
                        end = time()
                        t.predict(
                            path_to_save=f"{self.output_path}/{model_folder_name}/results"
                        )
                        res = {
                            "Batch Size": batch_size,
                            "Learning rate": lr,
                            "Weight Decay": wd,
                            "Best Val F1 Score": val_f1,
                            "Input Dimension": in_dim,
                            "Training Time": end - start,
                        }
                        for rgcn_label in rgcn_info.keys():
                            res[rgcn_label] = rgcn_info[rgcn_label]
                        for fc_label in fc_info.keys():
                            res[fc_label] = fc_info[fc_label]
                        results.append(res)

                        df = pd.DataFrame(results)
                        result_sum = f"{self.output_path}/train_summary.txt"
                        result_praw = f"{self.output_path}/train_summary_raw.csv"
                        df.to_csv(result_praw, index=False)
                        print(
                            tablify(
                                df.to_latex(
                                    index=False,
                                    caption="Training Summary over different Confs",
                                    float_format="%.4f",
                                )
                            )
                        )
        with open(result_sum, "w") as f:
            f.write(
                tablify(
                    df.to_latex(
                        index=False,
                        caption="Training Summary over different Confs",
                        float_format="%.4f",
                    )
                )
            )


import argparse


def get_cls_func(name):
    match name:
        case "binary":
            return snap_lat2onehot_binary
        case "0_1_10":
            return snap_lat2onehotv2
        case "0_300ms_1_10":
            return snap_lat_2onehot_4_cat
    raise Exception(f"Classification function {name} undefined")


def main(dbpedia2016, output_path, cls_func, n_classes, base_dir):
    if dbpedia2016:
        pred_info = {
            "pred_stat_path": "/PlanRGCN/extracted_features_dbpedia2016/predicate/pred_stat/batches_response_stats",
            "pred_com_path": "/PlanRGCN/extracted_features_dbpedia2016/predicate/pred_co/pred2index_louvain.pickle",
            "pred_ent": "/PlanRGCN/extracted_features_dbpedia2016/pred_ent/batch_response",
        }
        ent_info = {
            "ent_path": "/PlanRGCN/extracted_features_dbpedia2016/entities/ent_stat/batches_response_stats"
        }
    else:
        pred_info = {
            "pred_stat_path": "/PlanRGCN/extracted_features_dbpedia2016/predicate/pred_stat/batches_response_stats",
            "pred_com_path": "/PlanRGCN/extracted_features_dbpedia2016/predicate/pred_co/pred2index_louvain.pickle",
        }
        ent_info = {
            "ent_path": "/PlanRGCN/extracted_features_dbpedia2016/entities/ent_stat/batches_response_stats"
        }

    splt_data = {
        "train_path": f"{base_dir}/train_sampled.tsv",
        "val_path": f"{base_dir}/val_sampled.tsv",
        "test_path": f"{base_dir}/test_sampled.tsv",
        "query_plan": f"{base_dir}/queryplans",
    }

    g = GridSearch(
        splt_data=splt_data,
        pred_info=pred_info,
        ent_info=ent_info,
        output_path=output_path,
        cls_func=get_cls_func(cls_func),
        n_classes=n_classes,
    )
    g.search()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Grid Searching")
    parser.add_argument(
        "--dbpedia2016", action="store_true", help="Use DBpedia 2016 statistics"
    )
    parser.add_argument("--output_path", type=str, help="Specify the output path")
    parser.add_argument(
        "--cls_func",
        type=str,
        default="binary",
        help="Specify the classification function",
    )
    parser.add_argument(
        "--n_classes", type=int, default=2, help="Specify the number of classes"
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default="/qpp/dataset/DBpedia2016_sample_0_1",
        help="Specify the base directory",
    )

    args = parser.parse_args()

    main(
        args.dbpedia2016, args.output_path, args.cls_func, args.n_classes, args.base_dir
    )


# python3 /PlanRGCN/trainer/trainer/gridsearch.py --dbpedia2016 --output_path "/PlanRGCN/dbpedia2016_temp2" --cls_func binary --n_classes 2 --base_dir "/qpp/dataset/DBpedia2016_sample_0_1"
