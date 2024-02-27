import json
import os
import time
import dgl
from graph_construction.feats.featurizer_path import FeaturizerPath
from graph_construction.nodes.PathComplexException import PathComplexException
from graph_construction.qp.QueryPlanCommonBi import QueryPlanCommonBi
from graph_construction.qp.qp_utils import QueryPlanUtils
from graph_construction.qp.query_plan_path import QueryPlanPath
import json5
import networkx as nx
import numpy as np
from graph_construction.stack import Stack
import pandas as pd
from graph_construction.feats.featurizer import FeaturizerBase, FeaturizerPredStats
from graph_construction.node import Node, FilterNode, TriplePattern
from graph_construction.nodes.path_node import PathNode
import torch as th

from graph_construction.qp.query_plan import QueryPlan

def create_query_plan(path, query_plan=QueryPlan):
    try:
        data = json.load(open(path, "r"))
    except Exception:
        data = json5.load(open(path, "r"))
    q = query_plan(data)
    q.path = path
    return q


def create_query_plans_dir(
    source_dir, ids: set[str] = None, add_id=False, query_plan=QueryPlanCommonBi
):
    
    if ids is None:
        # TODO: check whether lsqQuery is necessary
        files = [x for x in os.listdir(source_dir) if x.startswith("lsqQuery")]
    else:
        ids = [str(x) for x in ids]
        files = [x for x in os.listdir(source_dir) if x in ids]
    durationQP = []
    if add_id:
        temp = []
        for x in files:
            try:
                start = time.time()
                temp.append((create_query_plan(f"{source_dir}{x}", query_plan=query_plan), x))
                end = time.time()
                durationQP.append(end-start)
            except PathComplexException:
                pass
        return temp
    
    query_plans = []
    for x in files:
        try:
            start = time.time()
            query_plans.append(create_query_plan(f"{source_dir}{x}", query_plan=query_plan))
            end = time.time()
            durationQP.append(end-start)
        except PathComplexException:
            pass
    return query_plans, durationQP


def create_dgl_graphs(
    qps: list[QueryPlan], featurizer: FeaturizerBase, without_id=True
) -> list:
    if without_id:
        return create_dgl_graph_helper(qps, featurizer)

    dgl_graphs = list()
    for x, id in qps:
        x: QueryPlan
        if len(x.nodes) == 0:
            continue
        x.feature(featurizer)
        dgl_graph = x.to_dgl()
        dgl_graphs.append((dgl_graph, id))
    return dgl_graphs


def create_dgl_graph_helper(qps: list[QueryPlan], featurizer: FeaturizerBase) -> list:
    dgl_graphs = list()
    for x in qps:
        # for now skip single triple patterns
        if len(x.edges) <= 0:
            continue
        x.feature(featurizer)
        dgl_graph = x.to_dgl()
        dgl_graphs.append(dgl_graph)
    return dgl_graphs


def create_query_graphs_data_split(
    source_dir,
    query_path="/qpp/dataset/DBpedia_2016_12k_sample/train_sampled.tsv",
    query_plan=QueryPlan,
    is_lsq=False,
    feat: FeaturizerBase = None,
):
    df = pd.read_csv(query_path, sep="\t")
    if is_lsq:
        ids = set([x[20:] for x in df["queryID"]])
    else:
        ids = set([x for x in df["queryID"]])

    qps, durationQPS = create_query_plans_dir(source_dir, ids, query_plan=query_plan, add_id=True)
    for qp, id in qps:
        try:
            assert len(qp.G.nodes) > 0
        except AssertionError:
            print(qp.data)
            print(qp.path)
            exit()
    return create_dgl_graphs(qps, feat, without_id=False),durationQPS


def query_graphs_with_lats(
    source_dir,
    query_path="/qpp/dataset/DBpedia_2016_12k_sample/train_sampled.tsv",
    feat: FeaturizerBase = None,
    query_plan=QueryPlan,
    time_col="mean_latency",
    is_lsq=False,
    debug = False
):
    df = pd.read_csv(query_path, sep="\t")
    if is_lsq:
        df["queryID"] = df["queryID"].apply(lambda x: x[20:])
    df.set_index("queryID", inplace=True)
    # print(df.loc["lsqQuery-UBZNr7M1ITVUf21mrBIQ9W4f6cdpJr6DQbr0HkWKOnw"][time_col])

    graphs_ids,durationQPS = create_query_graphs_data_split(
        source_dir,
        query_path=query_path,
        feat=feat,
        query_plan=query_plan,
        is_lsq=is_lsq,
    )
    samples = list()
    for g, id in graphs_ids:
        if not is_lsq:
            id = int(id)
        lat = df.loc[id][time_col]
        if isinstance(lat, pd.Series):
            lat = lat.iloc[0]
        samples.append((g, id, lat))
        if len(samples) == 10 and debug is True:
            break
    return samples, durationQPS


def query_graph_w_class_vec(
    source_dir,
    query_path="/qpp/dataset/DBpedia_2016_12k_sample/train_sampled.tsv",
    feat: FeaturizerBase = None,
    time_col="mean_latency",
    cls_funct=lambda x: x,
    query_plan=QueryPlanCommonBi,
    is_lsq=False,
    debug = False
):
    samples,durationQPS = query_graphs_with_lats(
        source_dir=source_dir,
        query_path=query_path,
        feat=feat,
        time_col=time_col,
        query_plan=query_plan,
        is_lsq=is_lsq,
        debug = debug
    )
    return query_graph_w_class_vec_helper(samples, cls_funct),durationQPS


def snap_lat2onehot(lat):
    vec = np.zeros(6)
    if lat < 0.01:
        vec[0] = 1
    elif (0.01 < lat) and (lat < 0.1):
        vec[1] = 1
    elif (0.1 < lat) and (lat < 1):
        vec[2] = 1
    elif (1 < lat) and (lat < 10):
        vec[3] = 1
    elif 10 < lat and lat < 100:
        vec[4] = 1
    elif lat > 100:
        vec[5] = 1

    return vec


def snap_lat2onehotv2(lat):
    vec = np.zeros(3)
    if lat < 1:
        vec[0] = 1
    elif (1 < lat) and (lat < 10):
        vec[1] = 1
    elif 10 < lat:
        vec[2] = 1
    return vec

def snap_reg(lat):
    return np.array([lat])


def snap_lat_2onehot_4_cat(lat):
    vec = np.zeros(4)
    if lat < 0.3:
        vec[0] = 1
    elif (0.3 < lat) and (lat < 1):
        vec[1] = 1
    elif (1 < lat) and (lat < 10):
        vec[1] = 1
    elif 10 < lat:
        vec[2] = 1
    return vec


def snap_lat2onehot_binary(lat):
    vec = np.zeros(2)
    if lat < 1:
        vec[0] = 1
    else:
        vec[1] = 1
    return vec


def query_graph_w_class_vec_helper(samples: list[tuple], cls_funct):
    graphs = []
    clas_list = []
    ids = []
    for g, id, lat in samples:
        graphs.append(g)
        ids.append(id)
        try:
            clas_list.append(cls_funct(lat))
        except Exception:
            print(lat)
            print("Something went wrong")
            exit()
    clas_list = th.tensor(np.array(clas_list), dtype=th.float32)
    return graphs, clas_list, ids


if __name__ == "__main__":
    # feat = FeaturizerBase(5)
    # KG statistics feature paths
    pred_stat_path = "/PlanRGCN/extracted_features_dbpedia2016/predicate/pred_stat/batches_response_stats"
    pred_com_path = "/PlanRGCN/extracted_features_dbpedia2016/predicate/pred_co"
    ent_path = "/PlanRGCN/extracted_features_dbpedia2016/entities/ent_stat/batches_response_stats"
    feat = FeaturizerPath(
        pred_stat_path=pred_stat_path,
        pred_com_path=f"{pred_com_path}/pred2index_louvain.pickle",
        ent_path=ent_path,
        bins=50,
    )
    # feat = FeaturizerPredStats(pred_stat_path)
    query_plan_files = [
        "lsqQuery-4D3PJSkE25IRd0N8ZKCpvslAgij5-THVkQy59w0QpK4",  # good path query example
        "lsqQuery-22Wctl3TpbNuqnugVntifWIf3TtAbnKFMirp0o-gIXI",
        "lsqQuery-zvjkqAQ05pz9dowoV7m_7qFbdI_x032BlCOkMbH3yeM",
        "lsqQuery-zsHnSHpsjWi4goJ6xiDTjkcqzPa1OuvcrA44agmWJJU",
        "lsqQuery--d_KKBdrHgoIkwpxtVcazeWBkdMEe-CarA6kPaDtJWQ",
    ]
    query_plan_files = [f"/query_plans_dbpedia/{x}" for x in query_plan_files]
    qps = list()
    for q_f in query_plan_files:
        qps.append(create_query_plan(q_f, query_plan=QueryPlanPath))
    create_dgl_graphs(qps, feat)
    exit()
    # path2
    q = ""
    # path 1
    q = "/query_plans_dbpedia/lsqQuery-OX07lIynqG7DyAFi_6DElGJNl8gSOnbUsyFp8"
    # print(q.edges)
    q.feature(feat)
    G = q.G
    dgl_g: dgl.DGLGraph = q.to_dgl()
    print("dgl" + str(dgl_g.all_edges()))

    def find_p_f(f):
        t = open(f, "r").readlines()
        if "path" in t:
            return True
        else:
            return False

    exit()
    """query_plan = create_query_plan(
        "/PlanRGCN/extracted_features/queryplans/lsqQuery-Yi0-ewp6Py0hTrLTppeXWTUNGW3z2KhaNz1wHKDCdMw"
    )
    query_plan.feature(feat)
    dgl_grpah = query_plan.to_dgl()"""
    """query_graphs_with_lats(
        "/PlanRGCN/extracted_features/queryplans/",
        query_path="/qpp/dataset/DBpedia_2016_12k_sample/train_sampled.tsv",
        feat=feat,
        time_col="mean_latency",
    )"""
    # qps = create_query_plans_dir("/PlanRGCN/extracted_features/queryplans/")
    # print(len(create_dgl_graphs(qps, feat)))
# q.extract_triples()
