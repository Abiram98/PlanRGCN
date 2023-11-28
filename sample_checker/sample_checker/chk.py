from graph_construction.feats.featurizer import FeaturizerPredCoEnt
from graph_construction.query_graph import (
    QueryPlan,
    create_dgl_graphs,
    create_query_plans_dir,
)
import pandas as pd


# taken from create_query_graphs_data_split (loosely")
def input_check(
    query_plan_dir,
    query_path,
    pred_stat_path,
    pred_com_path,
    ent_path,
    scaling,
    is_lsq=False,
):
    feat = FeaturizerPredCoEnt(
        pred_stat_path=pred_stat_path,
        pred_com_path=pred_com_path,
        ent_path=ent_path,
        scaling=scaling,
    )
    df = pd.read_csv(query_path, sep="\t")
    if is_lsq:
        ids = set([x[20:] for x in df["queryID"]])
    else:
        ids = set([x for x in df["queryID"]])
    # (qp, ids)
    qps = create_query_plans_dir(query_plan_dir, ids, query_plan=QueryPlan, add_id=True)
    # (query graph, id)
    dlg_graphs = create_dgl_graphs(qps, feat, without_id=False)
    for (qp, id), (qg, id2) in zip(qps, dlg_graphs):
        yield (qp, qg, id, id2)
