from graph_construction.feats.featurizer_path import FeaturizerPath

import pandas as pd
import numpy as np
from graph_construction.qp.query_plan_path import QueryPlanPath, QueryGraph
from graph_construction.query_graph import create_query_graphs_data_split
pred_stat_path='/data/planrgcn_features/extracted_features_dbpedia2016/predicate/pred_stat/batches_response_stats'
ent_path='/data/planrgcn_features/extracted_features_dbpedia2016/entities/ent_stat/batches_response_stats'
lit_path='/data/planrgcn_features/extracted_features_dbpedia2016/literals/literals_stat/batches_response_stats'
featurizer = FeaturizerPath(pred_stat_path=pred_stat_path,pred_com_path=None,ent_path =ent_path, lit_path = lit_path, scaling='binner')
create_query_graphs_data_split(query_path='/data/DBpedia_3_class_full/test_sampled.tsv', is_lsq=False, feat=featurizer, query_plan=QueryGraph)


exit()

from graph_construction.jar_utils import get_query_graph
query = '''PREFIX  dc: <http://purl.org/dc/elements/1.1/>
SELECT  ?title
WHERE   { <http://example.org/book/book1> dc:title* ?title }'''




df = pd.read_csv('/PlanRGCN/data/DBpedia_3_class_full/all.tsv', sep='\t')
import time

times = []
not_parse_qs = []
for i, query in enumerate(list(df['queryString'])):
    start = time.time()
    try:
        get_query_graph(query)
    except Exception:
        not_parse_qs.append(query)
    times.append(time.time()-start)
    if (i % 100) == 0:
        print(times[-100:])
        print(i, np.mean(times))

print(len(not_parse_qs), "  queries not parsed")
print(not_parse_qs)


