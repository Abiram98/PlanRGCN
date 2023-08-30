import json
import matplotlib.pyplot as plt
import networkx as nx
import pydot
from networkx.drawing.nx_pydot import graphviz_layout
from networkx.drawing.nx_agraph import write_dot, graphviz_layout
def graph_fy(path):
    data = json.load(open(path,'r'))
    G = nx.Graph()
    G = add_edges(data, G)
    print(G)
    #plt.show()
    
def tree_fy(path):
    plt.clf()
    data = json.load(open(path,'r'))
    G = nx.DiGraph()
    G = add_edges(data, G)
    print(G)
    nx.nx_agraph.write_dot(G,'/PlanRGCN/test.dot')
    plt.title('draw_networkx')
    pos =graphviz_layout(G, prog='dot')
    nx.draw(G, pos, with_labels=False, arrows=True)
    plt.savefig('/PlanRGCN/nx_test.png')
    #nx.draw_networkx(G)
    #plt.savefig('/PlanRGCN/test.png')
    #plt.show()

def tree_fy2(path):
    plt.clf()
    data = json.load(open(path,'r'))
    G = nx.DiGraph()
    G = add_edges2(data)
    nx.nx_agraph.write_dot(G,'/PlanRGCN/test.dot')
    plt.title('draw_networkx')
    pos =graphviz_layout(G, prog='dot')
    nx.draw(G, pos, with_labels=True, arrows=True)
    plt.savefig('/PlanRGCN/nx_test.png')

def add_edges2(data, debug = True):
    graph = nx.DiGraph()
    name_set = set()
    name,name_set = get_unique_name(data['opName'], name_set)
    #name_set.add(name)
    if 'subOp' in data.keys():
        for x in data['subOp']:
            child_name, name_set = get_unique_name(x['opName'], name_set)
            graph.add_edge(name, child_name)
            if debug:
                print(name, child_name)
            graph, name_set = add_egdes2_helper(graph, x, child_name, name_set, debug = debug)
    """elif data['opName'] == "Triple":
        subject_name, name_set = get_unique_name_tp(data['Subject'], name_set, 'Subject')
        pred_name, name_set = get_unique_name_tp(data['Predicate Path'], name_set, 'Predicate Path')
        obj_name, name_set = get_unique_name_tp(data['Object'], name_set, 'Object')
        
        graph.add_edge(name, subject_name)
        graph.add_edge(name, pred_name)
        graph.add_edge(name, obj_name)
        if debug:
            print(name, subject_name)
            print(name, pred_name)
            print(name, obj_name)
    elif data['opName'] == "path":
        subject_name, name_set = get_unique_name_tp(data['Subject'], name_set, 'Subject')
        pred_name, name_set = get_unique_name_tp(data['Predicate Path'], name_set, 'Predicate Path')
        obj_name, name_set = get_unique_name_tp(data['Object'], name_set, 'Object')
        
        graph.add_edge(name, subject_name)
        graph.add_edge(name, pred_name)
        graph.add_edge(name, obj_name)
        
        if debug:
            print(name, subject_name)
            print(name, pred_name)
            print(name, obj_name)"""
    return graph

def add_egdes2_helper(G, node, node_name, name_set, debug = True):
    if 'subOp' in node.keys():
        for x in node['subOp']:
            child_name, name_set = get_unique_name(x['opName'], name_set)
            G.add_edge(node_name, child_name)
            if debug:
                print(node_name, child_name)
            G, name_set = add_egdes2_helper(G, x, child_name, name_set, debug=debug)
    """elif node['opName'] == "Triple":
        subject_name, name_set = get_unique_name_tp(node['Subject'], name_set, 'Subject')
        pred_name, name_set = get_unique_name_tp(node['Predicate'], name_set, 'Predicate')
        obj_name, name_set = get_unique_name_tp(node['Object'], name_set, 'Object')
        
        G.add_edge(node_name, subject_name)
        G.add_edge(node_name, pred_name)
        G.add_edge(node_name, obj_name)
        if debug:
            print(node_name, subject_name)
            print(node_name, pred_name)
            print(node_name, obj_name)
    elif node['opName'] == "path":
        subject_name, name_set = get_unique_name_tp(node['Subject'], name_set, 'Subject')
        pred_name, name_set = get_unique_name_tp(node['Predicate Path'], name_set, 'Predicate Path')
        obj_name, name_set = get_unique_name_tp(node['Object'], name_set, 'Object')
        
        G.add_edge(node_name, subject_name)
        G.add_edge(node_name, pred_name)
        G.add_edge(node_name, obj_name)
        if debug:
            print(node_name, subject_name)
            print(node_name, pred_name)
            print(node_name, obj_name)"""
    return G,name_set

def get_unique_name(name, name_set):
    while (name in name_set):
        name = name + "I"
    name_set.add(name)
    return name, name_set

def get_unique_name_tp(name, name_set, prefix):
    name = prefix + name
    while (name in name_set):
        name = name+''
    name_set.add(name)
    return name, name_set
    

def add_edges(node:dict, G:nx.Graph, super_node=None) -> nx.Graph:
    print(node)
    name = node['opName']
    while (name in G.nodes):
        name = name + "I"
    G.add_node(name)
    if 'subOp' in node.keys():
        for target in node["subOp"]:
            
            G = add_edges(target, G, super_node=name)
        
        return G
    elif node['opName'] == 'path':
        if super_node is None:
            raise Exception("For path operation: Supernode need to be defined")
        G.add_edge(super_node, name)
        print(super_node, name)
        subject_name = f"Subject {node['Subject']}"
        while (subject_name in G.nodes):
            subject_name = " "+ subject_name
        G.add_edge(name, subject_name)
        print(name, subject_name)
        obejct_name = f"Object {node['Object']}"
        while (obejct_name in G.nodes):
            obejct_name = " "+ obejct_name
        G.add_edge(name, obejct_name)
        return G
    elif node['opName'] == 'Triple':
        if super_node is None:
            raise Exception("For path operation: Supernode need to be defined")
        G.add_edge(super_node, name)
        print(super_node, name)
        subject_name = f"Subject {node['Subject']}"
        while (subject_name in G.nodes):
            subject_name = " "+ subject_name
        G.add_edge(name, subject_name)
        print(name, subject_name)
        obejct_name = f"Object {node['Object']}"
        while (obejct_name in G.nodes):
            obejct_name = " "+ obejct_name
        G.add_edge(name, obejct_name)
        print(name, obejct_name)
        pred_name = f"Predicate {node['Predicate']}"
        while (pred_name in G.nodes):
            pred_name = " "+ pred_name
        G.add_edge(name, pred_name)
        print(name, pred_name)
        return G
    else:
        if super_node is not None: 
            G.add_edge(super_node, name)
            print(super_node,name)
            return G
        else:
            print(name)
    
    