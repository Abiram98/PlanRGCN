
from graph_construction.qp.query_plan_path import QueryPlanPath
import time, json

QueryPlanPath
path = '/data/wikidata_0_1_10_v2_path_weight_loss/queryplans/lsqQuery-7dgRzgYGbTZmAp4MKascjK6VEZdwGmPoJksQ1F6isUk'
data = json.load(open(path,'r'))
starttime = time.time()
QueryPlanPath(data)
endtime = time.time()
print(endtime-starttime)