
import os
from graph_construction.qp.query_plan_path import QueryPlanPath
from graph_construction.query_graph import create_query_plan
p = '/data/wikidata_3_class_full/queryplans/'
files = [os.path.join(p,x) for x in os.listdir(p)]
data_stat = {'single triple pattern': 0, 'opt': 0, 'all': len(files), 'star':0 , 'path':0, 'hybrid':0}
print(create_query_plan(files[1],query_plan=QueryPlanPath).edges)

exit()
for x in files:
    G = create_query_plan(x,query_plan=QueryPlanPath)
    if len(G.edges) == 1:
        data_stat['single triple pattern'] += 1
    else:
        star = False
        path = False
        opt = False
        filt = False
        for s in G.edges:
            if s[2] in [0, 8]:
                star = True
            if s[2] in [2, 6]:
                path = True
            
        if star and path:
            data_stat['hybrid'] +=1
        elif star:
            data_stat['star'] +=1
        elif path:
            data_stat['path'] +=1
print(data_stat)