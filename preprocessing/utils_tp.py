

#from graph_construction.tps_graph import tps_graph

import json

def get_bgp_predicates_from_path(path):
    data = None
    with open(path,'rb') as f:
        data = json.load(f)
                
    if data == None:
        print('Data could not be loaded!')
        return
    
    BGP_strings = list(data.keys())
    
    preds = []
    for bgp_string in BGP_strings:
        #triple_strings = bgp_string[1:-1].split(',')
        triple_strings = bgp_string[1:-1].split(',')
        bgp_pred = []
        for triple_string in triple_strings:
            splits = triple_string.split(' ')
            splits = [s for s in splits if s != '']
            if triple_string.strip() == '':
                continue
            assert len(splits) == 3
            if not splits[1].startswith('?'):
               bgp_pred.append(splits[1])
        preds.append(bgp_pred)
        del bgp_pred
    return preds