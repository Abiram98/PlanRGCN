import json
import os
import networkx as nx
from pyvis.network import Network
import itertools
import time


class PredicateCommunityCreator:
    def __init__(
        self,
        verbose=True,
        ks=[10, 20, 30, 40, 50, 60, 100, 150, 200],
        save_dir="/PlanRGCN/data/pred/pred_co",
    ) -> None:
        self.verbose = verbose
        self.ks = ks
        self.save_dir = save_dir

    def get_predicate_list(self, filepath: str):
        edge_lst = list()
        data = json.load(open(filepath, "r"))
        data = data["results"]["bindings"]
        for bind in data:
            edge_lst.append((bind["p1"]["value"], bind["p2"]["value"]))
        return edge_lst

    def get_predicate_from_dir(self, dir: str):
        files = sorted([x for x in os.listdir(dir) if x.endswith(".json")])
        edge_lst = list()
        for f in files:
            temp_edges = self.get_predicate_list(f"{dir}{f}")
            edge_lst.extend(temp_edges)
        return edge_lst

    def create_pred_graph(self, edge_lst):
        pred_graph = nx.Graph()
        pred_graph.add_nodes_from(edge_lst)
        return pred_graph

    def community_k(self, pred_graph, k=10):
        """_summary_

        Args:
            pred_graph (nx.Graph): Predicate graph
            k (int, optional): the amount of communities to consider. Defaults to 10.
        """
        communities = nx.community.girvan_newman(pred_graph)
        # print(next(communities))
        com_sets = []
        # com_sets = None
        for com in itertools.islice(communities, k):
            extracted_coms = tuple(sorted(c) for c in com)
            for no_com, extr_c in enumerate(extracted_coms):
                if self.verbose:
                    print(f"Working on Commnity {no_com}", end="\r")

                if not extr_c in com_sets:
                    com_sets.append(extr_c)
        if self.verbose:
            print("-" * 40 + "\n")
            print("Beginning Cluster Postprocessing" + "\n")
            print("-" * 40)
        pred_2_index = {}
        for idx, com in enumerate(com_sets):
            for pred in com:
                if pred in pred_2_index.keys():
                    pred_2_index[pred].append(idx)
                else:
                    pred_2_index[pred] = [idx]

        pred2index = pred_2_index
        max_clusters = len(com_sets)
        return pred2index, max_clusters

    def create_clusters(self, predicate_dir: str):
        start = time.time()
        if self.verbose:
            print("-" * 40 + "\n")
            print("Beginning Predicate Graph Construction" + "\n")
            print("-" * 40)
        edge_lst = self.get_predicate_from_dir(predicate_dir)
        pred_graph = self.create_pred_graph(edge_lst)
        print(f"Predicate Graph constructed after: {time.time()-start:,.2f}")

        # print(f"Amount of component in input graph: {nx.number_connected_components(pred_graph)}")
        # exit()
        """if save_pred_graph_png != None:
            net = Network()
            net.from_nx(pred_graph)
            net.save_graph(save_pred_graph_png)"""
        if self.verbose:
            print("-" * 40 + "\n")
            print("Beginning Clustering" + "\n")
            print("-" * 40)
        for k in self.ks:
            pred2index, max_clusters = self.community_k(pred_graph, k=k)
            path = f"{self.save_dir}/{str(k)}/community_{str(k)}.json"
            os.system("mkdir -p " + path)
            with open(path, "w") as f:
                json.dump((pred2index, max_clusters), f)


d = PredicateCommunityCreator()
d.create_clusters(
    "/PlanRGCN/extracted_features/predicate/predicate_cooccurence/batch_response/"
)
exit()


class Pred_clust_feat(Predicate_Featurizer_Sub_Obj):
    def __init__(self, endpoint_url=None, timeout=30):
        super().__init__(endpoint_url, timeout)

    # this function will be invoked during the loading of the function.
    def prepare_pred_feat(self, bins=30, k=20):
        return

    # This needs to implemented for predicates.
    def get_pred_feat(self, pred_label):
        try:
            return self.pred2index[pred_label]
        except KeyError:
            return []

    def print_communities(self, communities, pred_graph):
        for i, com in enumerate(communities):
            print("-" * 25)
            print(f"Communitity {i} ")
            for node in com:
                print(node, end="  ")
            print("\n")
            print("-" * 25)

    def create_cluster_from_pred_file(self, pred_file, save_pred_graph_png=None):
        preds = get_bgp_predicates_from_path(pred_file)
        self.create_clusters(preds, save_pred_graph_png=save_pred_graph_png)

    def create_clusters(
        self,
        predicates: list[list[str]],
        save_pred_graph_png=None,
        community_no: int = 10,
        verbose=False,
    ):
        pred_graph = nx.Graph()
        if verbose:
            print("-" * 40 + "\n")
            print("Beginning Predicate Graph Construction" + "\n")
            print("-" * 40)

        for bgp in predicates:
            pred_graph.add_nodes_from(bgp)
            for u in bgp:
                for v in bgp:
                    if u != v:
                        pred_graph.add_edge(u, v)

        # print(f"Amount of component in input graph: {nx.number_connected_components(pred_graph)}")
        # exit()
        if save_pred_graph_png != None:
            net = Network()
            net.from_nx(pred_graph)
            net.save_graph(save_pred_graph_png)
        if verbose:
            print("-" * 40 + "\n")
            print("Beginning Clustering" + "\n")
            print("-" * 40)
        communities = nx.community.girvan_newman(pred_graph)
        # print(next(communities))
        com_sets = []
        # com_sets = None
        for com in itertools.islice(communities, community_no):
            extracted_coms = tuple(sorted(c) for c in com)
            for no_com, extr_c in enumerate(extracted_coms):
                if verbose:
                    print(f"Working on Commnity {no_com}", end="\r")

                if not extr_c in com_sets:
                    com_sets.append(extr_c)
        if verbose:
            print("-" * 40 + "\n")
            print("Beginning Cluster Postprocessing" + "\n")
            print("-" * 40)
        pred_2_index = {}
        for idx, com in enumerate(com_sets):
            for pred in com:
                if pred in pred_2_index.keys():
                    pred_2_index[pred].append(idx)
                else:
                    pred_2_index[pred] = [idx]

        self.pred2index = pred_2_index
        self.max_clusters = len(com_sets)
        """print("-"*25+ '\n'*2)
        print('Beginning Stats'+'\n'*2)
        print("-"*25)
        zeros,ones, more = 0, 0, 0
        for k,v in pred_2_index.items():
            
            if len(v) == 1:
                ones += 1
            elif len(v) == 0:
                zeros +=1
            else:
                more +=1
        print(zeros, ones, more)
        print("nodes in graph and predicates in table:",len(pred_graph.nodes), len(pred_2_index.keys()))"""
