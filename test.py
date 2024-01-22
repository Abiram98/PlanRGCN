
from graph_construction.feats.featurizer_path import FeaturizerPath
from graph_construction.qp.query_plan_path import QueryPlanPath
import time, json

from trainer.model import Classifier2RGCN

QueryPlanPath
path = '/data/wikidata_0_1_10_v3_path_weight_loss/queryplans/lsqQuery-7dgRzgYGbTZmAp4MKascjK6VEZdwGmPoJksQ1F6isUk'
data = json.load(open(path,'r'))
starttime = time.time()
qp=QueryPlanPath(data)
endtime = time.time()
print(endtime-starttime)
inputd = 415
l1 = 2048
l2 = 4096
dropout = 0.0
wd = 0.01
lr = 1e-05
n = 3

model = Classifier2RGCN(inputd, l1, l2, dropout, n)
pred_stat_path = (
    "/data/planrgcn_features/extracted_features_wd/predicate/pred_stat/batches_response_stats"
)
pred_com_path = "/data/planrgcn_features/extracted_features_wd/predicate/pred_co/pred2index_louvain.pickle"
#ent_path = (
#    "/PlanRGCN/extracted_features_dbpedia2016/entities/ent_stat/batches_response_stats"
#)

ent_path = (
    "/data/planrgcn_features/extracted_features_wd/entities/ent_stat/batches_response_stats"
)
lit_path =(
        "/data/planrgcn_features/extracted_features_wd/literals_stat/batches_response_stats"
)
featurizer = FeaturizerPath(pred_stat_path=pred_stat_path,
        pred_com_path=pred_com_path,
        ent_path=ent_path,
        lit_path = lit_path,
        scaling="binner")
starttime = time.time()
qp.feature(featurizer)
dgl_graph = qp.to_dgl()
feats = dgl_graph.ndata["node_features"]
edge_types = dgl_graph.edata["rel_type"]
pred = model(dgl_graph, feats, edge_types)
endtime= time.time()
print(endtime-starttime)