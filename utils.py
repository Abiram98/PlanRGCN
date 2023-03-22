import json

class BGP:
    def __init__(self, BGP_string:str, ground_truth):
        triple_strings = BGP_string[1:-1].split(',')
        self.triples = []
        for t in triple_strings:
            self.triples.append(TriplePattern(t))
        self.ground_truth = 1 if ground_truth else 0
    
    def __str__(self):
        temp_str = 'BGP( '
        for t in self.triples:
            temp_str = temp_str +' '+ str(t)
        temp_str = temp_str +' )'
        return temp_str
        

class TriplePattern:
    def __init__(self, triple_string:str):
        splits = triple_string.split(' ')
        splits = [s for s in splits if s != '']
        assert len(splits) == 3
     
        self.subject = Node(splits[0])
        self.predicate = Node(splits[1])
        self.object = Node(splits[2])
    def __str__(self):
        return f'Triple ({str(self.subject)} {str(self.predicate)} {str(self.object)} )'
    def __eq__(self, other):
        return self.subject == other.subject and self.predicate == other.predicate and self.object == other.object

class Node:
    def __init__(self, node_label:str) -> None:
        self.node_label = node_label
        if node_label.startswith('?'):
            self.type = 'VAR'
        elif node_label.startswith('http'):
            self.type = 'URI'
        else:
            self.type = None
    def __str__(self):
        if self.type == None:
            return self.node_label
        else:
            return f'{self.type} {self.node_label}'
    def __eq__(self, other):
        return self.node_label == other.node_label
    def __hash__(self) -> int:
        return hash(self.node_label)

def load_BGPS_from_json(path):
    data = None
    with open(path,'rb') as f:
        data = json.load(f)
                
    if data == None:
        print('Data could not be loaded!')
        return
    BGP_strings = list(data.keys())
    BGPs = []
    for bgp_string in BGP_strings:
        BGPs.append(BGP(bgp_string, data[bgp_string]))
    return BGPs

def get_predicates(bgps: list):
    predicates = set()
    for bgp in bgps:
        for triple in bgp.triples:
            if triple.predicate.type == 'URI':
                predicates.add(triple.predicate)
    return list(predicates)

def get_entities(bgps: list):
    entities = set()
    for bgp in bgps:
        for triple in bgp.triples:
            for e in [triple.subject, triple.object]:
                if e.type == 'URI':
                    entities.add(e)
    return list(entities)

class BGPGraph:
    def __init__(self, bgp : BGP):
        self.bgp = bgp

bgps = load_BGPS_from_json('/work/data/train_data.json')
print(f'BGPS loaded : {len(bgps)}')
ents = get_entities(bgps)
preds = get_predicates(bgps)

print(f'Entities extracted: {len(ents)}')
print(f'Preds extracted: {len(preds)}')