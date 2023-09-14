from graph_construction.featurizer import FeaturizerPredStats
from graph_construction.query_graph import (
    QueryPlanCommonBi,
    query_graph_w_class_vec,
    snap_lat2onehot,
)

from dgl.dataloading import GraphDataLoader


class GraphDataset:
    def __init__(self, graphs, labels, ids) -> None:
        self.graph = graphs
        self.labels = labels
        self.ids = ids

    def __getitem__(self, i):
        return self.graph[i], self.labels[i], self.ids[i]

    def __len__(self):
        return len(self.labels)


class DatasetPrep:
    def __init__(
        self,
        train_path="/qpp/dataset/DBpedia_2016_12k_sample/train_sampled.tsv",
        val_path="/qpp/dataset/DBpedia_2016_12k_sample/val_sampled.tsv",
        test_path="/qpp/dataset/DBpedia_2016_12k_sample/test_sampled.tsv",
        batch_size=64,
        query_plan_dir="/PlanRGCN/extracted_features/queryplans/",
        pred_stat_path="/PlanRGCN/extracted_features/predicate/pred_stat/batches_response_stats",
        time_col="mean_latency",
        cls_func=snap_lat2onehot,
        featurizer_class=FeaturizerPredStats,
        query_plan=QueryPlanCommonBi,
    ) -> None:
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.cls_func = cls_func

        self.time_col = time_col
        self.feat = featurizer_class(pred_stat_path)
        self.vec_size = self.feat.filter_size + self.feat.tp_size
        self.query_plan_dir = query_plan_dir
        self.batch_size = batch_size
        self.query_plan = query_plan

    def get_dataloader(self, path):
        graphs, clas_list, ids = query_graph_w_class_vec(
            self.query_plan_dir,
            query_path=path,
            feat=self.feat,
            time_col=self.time_col,
            cls_funct=self.cls_func,
            query_plan=self.query_plan,
        )
        train_dataset = GraphDataset(graphs, clas_list, ids)
        train_dataloader = GraphDataLoader(
            train_dataset, batch_size=self.batch_size, drop_last=False, shuffle=True
        )
        return train_dataloader

    def get_trainloader(self):
        return self.get_dataloader(self.train_path)

    def get_valloader(self):
        return self.get_dataloader(self.val_path)

    def get_testloader(self):
        return self.get_dataloader(self.test_path)


if __name__ == "__main__":
    pred_stat_path = (
        "/PlanRGCN/extracted_features/predicate/pred_stat/batches_response_stats"
    )
    query_plan_dir = "/PlanRGCN/extracted_features/queryplans/"

    feat = FeaturizerPredStats(pred_stat_path)
    batch_size = 2
    train_path = "/qpp/dataset/DBpedia_2016_12k_sample/train_sampled.tsv"
    func = lambda x: x
    graphs, clas_list, ids = query_graph_w_class_vec(
        query_plan_dir,
        query_path=train_path,
        feat=feat,
        time_col="mean_latency",
        cls_funct=func,
    )
    train_dataset = GraphDataset(graphs, clas_list, ids)
    train_dataloader = GraphDataLoader(
        train_dataset, batch_size=batch_size, drop_last=False, shuffle=True
    )
    for g, l, i in train_dataloader:
        print(i)
        break
