


import os

from pathlib import Path

if 'QG_JAR' not in os.environ.keys():
    os.environ['QG_JAR']='/PlanRGCN/PlanRGCN/qpe/target/qpe-1.0-SNAPSHOT.jar'


if 'QPP_JAR' not in os.environ.keys():
    os.environ['QPP_JAR']='/PlanRGCN/qpp/qpp_features/sparql-query2vec/target/sparql-query2vec-0.0.1.jar'


import sys

print(sys.path)

import torch
import json
from trainer.model import Classifier2RGCN

import pandas as pd

from graph_construction.jar_utils import get_query_graph
from graph_construction.query_graph import create_query_graph

from graph_construction.qp.query_plan_path import QueryGraph
import pickle as p
import time

import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Argument parser for model inference")

    parser.add_argument('--prep_path', type=str, default='/data/DBpedia_3_class_full/plan_l14096_l21025_no_pred_co/prepper.pcl',
                        help='Path to the preparation file')
    parser.add_argument('--model_path', type=str, default='/data/DBpedia_3_class_full/plan_l14096_l21025_no_pred_co/best_model.pt',
                        help='Path to the model file')
    parser.add_argument('--config_path', type=str, default='/data/DBpedia_3_class_full/plan_l14096_l21025_no_pred_co/model_config.json',
                        help='Path to the model configuration file')
    parser.add_argument('--output_path', type=str, default='/data/DBpedia_3_class_full/test_inf/plan_inference.csv',
                        help='Path to save the inference output')
    parser.add_argument('--query_path', type=str, default='/data/DBpedia_3_class_full/test_sampled.tsv',
                        help='Path to the query input file')
    parser.add_argument('--gpu', type=str, default='yes',
                        help='yes if use gpu otherwise will not use gpu. Default uses gpu')

    return parser.parse_args()

if __name__ == "__main__":
    prep_path = '/data/DBpedia_3_class_full/plan_l14096_l21025_no_pred_co/prepper.pcl'
    model_path = '/data/DBpedia_3_class_full/plan_l14096_l21025_no_pred_co/best_model.pt'
    config_path = '/data/DBpedia_3_class_full/plan_l14096_l21025_no_pred_co/model_config.json'
    output_path = '/data/DBpedia_3_class_full/test_inf/plan_inference.csv'
    query_path = '/data/DBpedia_3_class_full/test_sampled.tsv'

    #Temporary code - for interactive run
    prep_path = "/data/wikidata_3_class_full/planRGCN_no_pred_co/prepper.pcl"
    model_path = "/data/wikidata_3_class_full/planRGCN_no_pred_co/best_model.pt"
    config_path = "/data/wikidata_3_class_full/planRGCN_no_pred_co/model_config.json"
    output_path = "/data/wikidata_3_class_full/test_inf/plan_inference.csv"
    query_path = "/data/wikidata_3_class_full/test_sampled.tsv"

    args = get_args()

    #Terminal version
    """prep_path = '"$prep_path"'
    model_path = '"$model_path"'
    config_path = '"$config_path"'
    output_path = '"$output_path"'
    query_path = '"$query_path"'"""

    # Save the parsed arguments into the same variable names
    prep_path = args.prep_path
    model_path = args.model_path
    config_path = args.config_path
    output_path = args.output_path
    query_path = args.query_path



    config = json.load(open(config_path, 'r'))

    inputd = config['input_d']
    l1 = config['l1']
    l2 = config['l2']
    n_classes = 3

    prepper = p.load(open(prep_path, 'rb'))
    featurizer = prepper.feat
    query_plan = QueryGraph

    query_df = pd.read_csv(query_path, sep='\t')

    get_query_graph(query_df['queryString'].iloc[0])


    def single_qg_con(query_text, featurizer, query_plan):
        try:
            query = query_text
            start = time.time()
            qg = create_query_graph(query, query_plan=query_plan)
            qg.feature(featurizer=featurizer)
            dgl_graph = qg.to_dgl()
            end = time.time()
            qg_time = end - start
            return dgl_graph, qg_time
        except Exception as e:
            return None


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = Classifier2RGCN(inputd, l1, l2, 0.0, n_classes)
    model = model.to(device)
    model.eval()

    entries = []
    with torch.no_grad():
        for idx, row in query_df.iterrows():
            query_text = row['queryString']
            start = time.time()
            data = single_qg_con(query_text, featurizer, query_plan)
            qg_time = time.time() - start
            if data is None:
                continue
            qg, dur = data

            feats = qg.ndata['node_features']
            edge_types = qg.edata['rel_type']
            feats = feats.to(device)
            edge_types = edge_types.to(device)
            qg = qg.to(device)
            start = time.time()
            pred = model(qg, feats, edge_types)
            inference_time = time.time() - start
            pred = pred.detach().to('cpu')
            entries.append((row['queryID'], qg_time, inference_time, pred))
    df = pd.DataFrame(entries, columns=['queryID', 'qg_time', 'inference_time', 'pred'])
    df.to_csv(output_path, index=False)
    info_txt = f'''Inference Statistics
    Features Query Graph: {df['qg_time'].mean()}
    Model Inference: {df['inference_time'].mean()}
    '''
    print(info_txt)
    with open(os.path.join(Path(output_path).parent, 'plan_inf_summary.txt'), 'w') as w:
        w.write(info_txt)

