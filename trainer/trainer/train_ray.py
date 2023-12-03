
from functools import partial
import os
import tempfile
import dgl
from graph_construction.feats.featurizer import FeaturizerPredCo, FeaturizerPredCoEnt
from graph_construction.query_graph import (
    QueryPlan,
    QueryPlanCommonBi,
    snap_lat2onehot,
    snap_lat2onehotv2,
    snap_reg,
)
import ray
from trainer.data_util import DatasetPrep, GraphDataset
from trainer.model import (
    Classifier as CLS,
    RegressorWSelfTriple as CLS,
    Classifier2RGCN,
)
import torch as th
import numpy as np
import json
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import torch.nn.functional as F
import pandas as pd
import torch.nn as nn
from pathlib import Path
from ray import tune, train

# from ray.air import Checkpoint, session
from ray.train import Checkpoint

# from ray.tune.schedulers import ASHAScheduler
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.air.config import CheckpointConfig
from dgl.dataloading import GraphDataLoader
AVG = "macro"


th.manual_seed(42)
np.random.seed(42)


# Dataloader wrapper in functions:
def get_dataloaders(
    train_path="/qpp/dataset/DBpedia_2016_12k_sample/train_sampled.tsv",
    val_path="/qpp/dataset/DBpedia_2016_12k_sample/val_sampled.tsv",
    test_path="/qpp/dataset/DBpedia_2016_12k_sample/test_sampled.tsv",
    batch_size=32,
    query_plan_dir="/PlanRGCN/extracted_features/queryplans/",
    pred_stat_path="/PlanRGCN/extracted_features/predicate/pred_stat/batches_response_stats",
    pred_com_path="/PlanRGCN/data/pred/pred_co/pred2index_louvain.pickle",
    ent_path="/PlanRGCN/extracted_features/entities/ent_stat/batches_response_stats",
    time_col="mean_latency",
    is_lsq=False,
    cls_func=snap_lat2onehot,
    featurizer_class=FeaturizerPredCoEnt,
    scaling="None",
    query_plan=QueryPlanCommonBi,
    debug=False
):
    train_temp = GraphDataset.load_dataset(train_path, scaling)
    val_temp = GraphDataset.load_dataset(val_path, scaling)
    test_temp = GraphDataset.load_dataset(test_path, scaling)
    if (train_temp is not None) or (val_temp is not None) or (test_temp is not None):
        train_dataloader = GraphDataLoader(
            train_temp, batch_size=batch_size, drop_last=False, shuffle=True
        )
        val_dataloader = GraphDataLoader(
            val_temp, batch_size=batch_size, drop_last=False, shuffle=True
        )
        test_dataloader = GraphDataLoader(
            test_temp, batch_size=batch_size, drop_last=False, shuffle=True
        )
        vec_size = train_temp.vec_size
        return train_dataloader,val_dataloader,test_dataloader, vec_size
    
    prepper = DatasetPrep(
        train_path=train_path,
        val_path=val_path,
        test_path=test_path,
        batch_size=batch_size,
        query_plan_dir=query_plan_dir,
        pred_stat_path=pred_stat_path,
        pred_com_path=pred_com_path,
        ent_path=ent_path,
        time_col=time_col,
        cls_func=cls_func,
        query_plan=query_plan,
        featurizer_class=featurizer_class,
        is_lsq=is_lsq,
        scaling=scaling,
        debug=debug
    )

    train_loader = prepper.get_trainloader()
    train_loader.dataset.save()
    val_loader = prepper.get_valloader()
    val_loader.dataset.save()
    test_loader = prepper.get_testloader()
    test_loader.dataset.save()
    return train_loader, val_loader, test_loader, prepper.vec_size


def train_function(
    config,
    train_path="/qpp/dataset/DBpedia_2016_12k_sample/train_sampled.tsv",
    val_path="/qpp/dataset/DBpedia_2016_12k_sample/val_sampled.tsv",
    test_path="/qpp/dataset/DBpedia_2016_12k_sample/test_sampled.tsv",
    # batch_size=32,
    query_plan_dir="/PlanRGCN/extracted_features/queryplans/",
    pred_stat_path="/PlanRGCN/extracted_features/predicate/pred_stat/batches_response_stats",
    pred_com_path="/PlanRGCN/data/pred/pred_co/pred2index_louvain.pickle",
    ent_path="/PlanRGCN/extracted_features/entities/ent_stat/batches_response_stats",
    time_col="mean_latency",
    is_lsq=False,
    cls_func=snap_reg,
    featurizer_class=FeaturizerPredCoEnt,
    scaling="None",
    n_classes=3,
    query_plan=QueryPlanCommonBi,
    metric_default=0,
    path_to_save=tempfile.TemporaryDirectory(),
):
    if isinstance(path_to_save, str):
        path_to_save = os.path.join(path_to_save, "session_data")
    train_loader, val_loader, _, input_d = get_dataloaders(
        train_path=train_path,
        val_path=val_path,
        test_path=test_path,
        batch_size=config["batch_size"],
        query_plan_dir=query_plan_dir,
        pred_stat_path=pred_stat_path,
        pred_com_path=os.path.join(pred_com_path, config["pred_com_path"]),
        ent_path=ent_path,
        time_col=time_col,
        is_lsq=is_lsq,
        cls_func=cls_func,
        featurizer_class=featurizer_class,
        scaling=scaling,
        query_plan=query_plan,
    )
    net = Classifier2RGCN(
        input_d, config["l1"], config["l2"], config["dropout"], n_classes
    )
    if not isinstance(config["loss_type"], str):
        criterion = config["loss_type"]
    elif config["loss_type"] == "cross-entropy":
        p = Path(train_path)
        if (p.parent / "loss_weight.json").exists():
            with open(p.parent / "loss_weight.json", "r") as f:
                W = json.load(f)
            criterion = nn.CrossEntropyLoss(th.tensor(W))
        else:
            criterion = nn.CrossEntropyLoss()
    elif config["loss_type"] == "mse":
        criterion = nn.MSELoss()
    opt = th.optim.AdamW(net.parameters(), lr=config["lr"], weight_decay=config["wd"])

    checkpoint = train.get_checkpoint()

    if checkpoint:
        with checkpoint.as_directory() as checkpoint_dir:
            checkpoint_dict = th.load(os.path.join(checkpoint_dir, "checkpoint.pt"))
            start_epoch = checkpoint_dict["epoch"] + 1
            net.load_state_dict(checkpoint_dict["model_state"])
            # checkpoint_state = checkpoint.to_dict()
            # start_epoch = checkpoint_state["epoch"]
            # net.load_state_dict(checkpoint_state["net_state_dict"])
            opt.load_state_dict(checkpoint_dict["optimizer_state_dict"])
    else:
        start_epoch = 0
    for epoch in range(start_epoch, config["epochs"]):
        print(
            f"Epoch {epoch+1}\n--------------------------------------------------------------"
        )
        train_loss, train_f1, train_recall, train_prec = train_epoch(
            model=net,
            train_loader=train_loader,
            criterion=criterion,
            opt=opt,
            epoch=epoch,
            verbosity=2,
            snap_pred=snap_pred,
            metric_default=metric_default,
        )

        val_loss, val_f1, val_prec, val_recall = evaluate(
            model=net,
            data_loader=val_loader,
            loss_type=config["loss_type"],
            metric_default=metric_default,
        )
        print(f"Train Avg Loss {epoch+1:4}: {train_loss:>8f}\n")
        print(f"Train Avg F1 {epoch+1:4}: {train_f1}\n")
        print(f"Val Avg Loss {epoch+1:4}: {val_loss:>8f}\n")
        print(f"Val Avg F1 {epoch+1:4}:  {val_f1}\n")
        """checkpoint_data = {
            "epoch": epoch,
            "net_state_dict": net.state_dict(),
            "optimizer_state_dict": opt.state_dict(),
        }"""
        # checkpoint = Checkpoint.from_dict(checkpoint_data)
        with tempfile.TemporaryDirectory() as tempdir:
            th.save(
                {
                    "epoch": epoch,
                    "model_state": net.state_dict(),
                    "optimizer_state_dict": opt.state_dict(),
                },
                os.path.join(tempdir, "checkpoint.pt"),
            )
            train.report(
                metrics={
                    "val loss": val_loss,
                    "val f1": val_f1,
                    "train loss": train_loss,
                    "train f1": train_f1,
                    "input d": input_d,
                    "num_class": n_classes,
                    "batch_size": config["batch_size"],
                },
                checkpoint=Checkpoint.from_directory(tempdir),
            )
    print("Finished Training")


def train_epoch(
    model,
    train_loader,
    criterion,
    opt,
    epoch,
    verbosity,
    snap_pred,
    metric_default,
):
    train_loss = 0
    train_f1 = 0
    train_recall = 0
    train_prec = 0
    model.train()
    with th.enable_grad():
        for batch_no, (batched_graph, labels, ids) in enumerate(train_loader):
            feats = batched_graph.ndata["node_features"]
            edge_types = batched_graph.edata["rel_type"]
            logits = model(batched_graph, feats, edge_types)
            loss = criterion(logits, labels)
            opt.zero_grad()
            loss.backward()
            opt.step()

            c_train_loss = loss.item()

            # snap_pred(logits,model_thres=pred_thres, add_thres=add_thres)
            # snap_thres = [pred_thres for x in logits]
            # snap_add_thres = [add_thres for x in logits]
            f1_pred = list(map(snap_pred, logits))
            snapped_labels = list(map(snap_pred, labels))
            # f1_batch = f1_score(labels, f1_pred)
            f1_batch = f1_score(
                snapped_labels,
                f1_pred,
                average=AVG,
                zero_division=metric_default,
            )
            prec_batch = precision_score(
                snapped_labels,
                f1_pred,
                average=AVG,
                zero_division=metric_default,
            )
            recall_batch = recall_score(
                snapped_labels,
                f1_pred,
                average=AVG,
                zero_division=metric_default,
            )
            if verbosity >= 2:
                print(
                    f"Epoch: {epoch+1:4} {(batch_no+1):8} Batch loss: {c_train_loss:>7f} Batch F1: {f1_batch}"
                )
            train_loss += c_train_loss
            train_f1 += f1_batch
            train_recall += recall_batch
            train_prec += prec_batch
    return (
        train_loss / len(train_loader),
        train_f1 / len(train_loader),
        train_recall / len(train_loader),
        train_prec / len(train_loader),
    )


# evaluate on validation data loader and also test
def evaluate(model, data_loader, loss_type, metric_default=0):
    loss = 0
    f1_val = 0
    recall_val = 0
    precision_val = 0
    model.eval()
    with th.no_grad():
        for _, (graphs, labels, _) in enumerate(data_loader):
            feats = graphs.ndata["node_features"]
            edge_types = graphs.edata["rel_type"]
            pred = model(graphs, feats, edge_types)

            if loss_type == "cross-entropy":
                c_val_loss = F.cross_entropy(pred, labels).item()
            elif loss_type == "mse":
                c_val_loss = F.mse_loss(pred, labels).item()
            else:
                c_val_loss = loss_type(pred, labels).item()

            loss += c_val_loss

            f1_pred_val = list(map(snap_pred, pred))
            snapped_lebls = list(map(snap_pred, labels))

            f1_batch_val = f1_score(
                snapped_lebls,
                f1_pred_val,
                average=AVG,
                zero_division=metric_default,
            )
            f1_val += f1_batch_val

            prec_batch_val = precision_score(
                snapped_lebls,
                f1_pred_val,
                average=AVG,
                zero_division=metric_default,
            )
            precision_val += prec_batch_val

            recall_batch_val = recall_score(
                snapped_lebls,
                f1_pred_val,
                average=AVG,
                zero_division=metric_default,
            )
            recall_val += recall_batch_val

    loss = loss / len(data_loader)
    f1_val = f1_val / len(data_loader)
    precision_val = precision_val / len(data_loader)
    recall_val = recall_val / len(data_loader)
    return loss, f1_val, precision_val, recall_val


def snap_pred(pred, cls_func=None):
    if not isinstance(pred, th.Tensor):
        pred = th.tensor(cls_func(pred), dtype=th.float32)
    return th.argmax(pred)


def predict(
    model,
    train_loader,
    val_loader,
    test_loader,
    is_lsq,
    path_to_save="/PlanRGCN/temp_results",
):
    os.system(f"mkdir -p {path_to_save}")
    with th.no_grad():
        train_p = f"{path_to_save}/train_pred.csv"
        val_p = f"{path_to_save}/val_pred.csv"
        test_p = f"{path_to_save}/test_pred.csv"
        for loader, path in zip(
            [train_loader, val_loader, test_loader],
            [train_p, val_p, test_p],
        ):
            ids, preds, truths = predict_helper(loader, model, is_lsq)
            df = pd.DataFrame()
            df["id"] = ids
            df["time_cls"] = truths
            df["planrgcn_prediction"] = preds
            df.to_csv(path, index=False)


def predict_helper(dataloader, model, is_lsq):
    all_preds = []
    all_ids = []
    all_truths = []
    for graphs, labels, ids in dataloader:
        feats = graphs.ndata["node_features"]
        edge_types = graphs.edata["rel_type"]
        pred = model(graphs, feats, edge_types)
        pred = pred.tolist()
        pred = [np.argmax(x) for x in pred]
        truths = [np.argmax(x) for x in labels.tolist()]
        all_truths.extend(truths)
        if not is_lsq:
            ids = [f"http://lsq.aksw.org/{x}" for x in ids]
        else:
            ids = [f"{x}" for x in ids]
        all_ids.extend(ids)
        all_preds.extend(pred)
    return all_ids, all_preds, all_truths


def stop_bad_run(trial_id: str, result: dict) -> bool:
    """This function should return true when the trial should be stopped and false for continued training.

    Args:
        trial_id (str): _description_
        result (dict): _description_

    Returns:
        bool: _description_
    """
    if result["val f1"] < 0.7 and result["training_iteration"] >= 50:
        return True
    if result["val f1"] < 0.5 and result["training_iteration"] >= 20:
        return True

    return False


def main(
    num_samples=2,
    max_num_epochs=10,
    train_path="/qpp/dataset/DBpedia_2016_12k_sample/train_sampled.tsv",
    val_path="/qpp/dataset/DBpedia_2016_12k_sample/val_sampled.tsv",
    test_path="/qpp/dataset/DBpedia_2016_12k_sample/test_sampled.tsv",
    # batch_size=32,
    query_plan_dir="/PlanRGCN/extracted_features/queryplans/",
    pred_stat_path="/PlanRGCN/extracted_features/predicate/pred_stat/batches_response_stats",
    pred_com_path="/PlanRGCN/data/pred/pred_co/pred2index_louvain.pickle",
    ent_path="/PlanRGCN/extracted_features/entities/ent_stat/batches_response_stats",
    time_col="mean_latency",
    is_lsq=False,
    cls_func=snap_lat2onehot,
    featurizer_class=FeaturizerPredCoEnt,
    scaling="None",
    n_classes=3,
    query_plan=QueryPlanCommonBi,
    path_to_save="/PlanRGCN/temp_results",
    config={
        "l1": tune.choice([128, 256, 512, 1024]),
        "l2": tune.choice([128, 256, 512]),
        "dropout": tune.grid_search([0.0, 0.5]),
        "wd": 0.01,
        "lr": tune.grid_search([1e-5]),
        "epochs": 10,
        "batch_size": tune.choice([64, 256]),
        "loss_type": "cross-entropy",
    },
    checkpoint_config=CheckpointConfig(
        1, "val f1", "max"
    ),  # CheckpointConfig(12, "val f1", "max"),
    resume=False
):
    config["epochs"] = max_num_epochs
    ray_temp_path = os.path.join(path_to_save, "temp_session")
    ray_save = os.path.join(path_to_save, "ray_save")
    os.makedirs(ray_save, exist_ok=True)
    os.makedirs(ray_temp_path, exist_ok=True)
    context = ray.init(
        num_cpus=10,  # only 2 cpus on strato2 sercer
        _system_config={
            "local_fs_capacity_threshold": 0.99,
            "object_spilling_config": json.dumps(
                {"type": "filesystem", "params": {"directory_path": "/PlanRGCN/temp"}},
            ),
        },
    )
    print(context.dashboard_url)

    scheduler = AsyncHyperBandScheduler(
        metric="val f1",
        mode="max",
        max_t=max_num_epochs,
        grace_period=10,
        reduction_factor=2,
    )
    """trainable_with_resources = tune.with_resources(
        partial(
            train_function,
            train_path=train_path,
            val_path=val_path,
            test_path=test_path,
            # batch_size=batch_size,
            query_plan_dir=query_plan_dir,
            pred_stat_path=pred_stat_path,
            pred_com_path=pred_com_path,
            ent_path=ent_path,
            time_col=time_col,
            is_lsq=is_lsq,
            cls_func=cls_func,
            featurizer_class=featurizer_class,
            scaling=scaling,
            n_classes=n_classes,
            query_plan=query_plan,
            path_to_save=path_to_save,
        ),
        {"cpu": 0.5},
    )
    tuner = tune.Tuner(
        trainable_with_resources,
        param_space=config,
        tune_config=tune.TuneConfig(
            num_samples=num_samples,
            scheduler=scheduler,
        ),
    )
    result = tuner.fit()
    result.get_dataframe().to_csv(
        Path(path_to_save).joinpath("hyper_search_results.csv")
    )"""
    trainable = partial(
        train_function,
        train_path=train_path,
        val_path=val_path,
        test_path=test_path,
        query_plan_dir=query_plan_dir,
        pred_stat_path=pred_stat_path,
        pred_com_path=pred_com_path,
        ent_path=ent_path,
        time_col=time_col,
        is_lsq=is_lsq,
        cls_func=cls_func,
        featurizer_class=featurizer_class,
        scaling=scaling,
        n_classes=n_classes,
        query_plan=query_plan,
    )

    result = tune.run(
        trainable,
        checkpoint_config=checkpoint_config,
        resources_per_trial={"cpu": 1},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        local_dir=ray_save,
        stop=stop_bad_run,
        resume=resume
    )
    # best_trial = result.get_best_result("val f1", "max", "last")
    best_trial = result.get_best_trial("val f1", "max", "last")
    print(f"Best trial config: {best_trial.config}")
    print(f"Best trial final validation f1: {best_trial.last_result['val f1']}")
    retrain_config = best_trial.config
    (
        train_loader,
        val_loader,
        test_loader,
        input_d,
    ) = get_dataloaders(
        train_path=train_path,
        val_path=val_path,
        test_path=test_path,
        batch_size=best_trial.last_result["batch_size"],
        query_plan_dir=query_plan_dir,
        pred_stat_path=pred_stat_path,
        pred_com_path=os.path.join(pred_com_path, best_trial.config["pred_com_path"]),
        ent_path=ent_path,
        time_col=time_col,
        is_lsq=is_lsq,
        cls_func=cls_func,
        featurizer_class=featurizer_class,
        scaling=scaling,
        query_plan=query_plan,
    )

    best_trained_model = Classifier2RGCN(
        best_trial.last_result["input d"],
        best_trial.config["l1"],
        best_trial.config["l2"],
        best_trial.config["dropout"],
        n_classes,
    )
    best_checkpoint = os.path.join(
        best_trial.checkpoint.to_directory(), "checkpoint.pt"
    )  # .to_air_checkpoint()
    print(f"path to best checkpoint {best_checkpoint}")
    retrain_config['best_checkpoint'] =best_checkpoint 
    os.system(f"cp {best_checkpoint} {os.path.join(path_to_save,'best_model.pt')}")
    retrain_config["input d"] = best_trial.last_result["input d"] 
    model_state = th.load(best_checkpoint)
    best_trained_model.load_state_dict(model_state["model_state"])
    
    predict(
        best_trained_model,
        train_loader,
        val_loader,
        test_loader,
        is_lsq,
        path_to_save=path_to_save,
    )
    json.dump(retrain_config, open(os.path.join(path_to_save, "model_config.json"),'w'))


if __name__ == "__main__":
    exit()
    train_path = "/qpp/dataset/DBpedia2016_sample_0_1_10/train_sampled.tsv"
    val_path = "/qpp/dataset/DBpedia2016_sample_0_1_10/val_sampled.tsv"
    test_path = "/qpp/dataset/DBpedia2016_sample_0_1_10/test_sampled.tsv"
    qp_path = "/qpp/dataset/DBpedia2016_sample_0_1_10/queryplans/"
    main(
        num_samples=2,
        max_num_epochs=100,
        train_path=train_path,
        val_path=val_path,
        test_path=test_path,
        batch_size=64,
        query_plan_dir=qp_path,
        pred_stat_path="/PlanRGCN/extracted_features_dbpedia2016/predicate/pred_stat/batches_response_stats",
        pred_com_path="/PlanRGCN/extracted_features_dbpedia2016/predicate/pred_co/pred2index_louvain.pickle",
        ent_path="/PlanRGCN/extracted_features_dbpedia2016/entities/ent_stat/batches_response_stats",
        time_col="mean_latency",
        is_lsq=True,
        cls_func=snap_lat2onehotv2,
        featurizer_class=FeaturizerPredCoEnt,
        scaling="std",
        n_classes=3,
        query_plan=QueryPlan,
    )