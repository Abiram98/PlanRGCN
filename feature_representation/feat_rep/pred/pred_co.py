import json
import os
import pickle
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
        nodes = set()
        for f in files:
            temp_edges = self.get_predicate_list(f"{dir}{f}")
            for n1, n2 in temp_edges:
                nodes.add(n1)
                nodes.add(n2)
            edge_lst.extend(temp_edges)
        return edge_lst

    def create_pred_graph(self, edge_lst):
        pred_graph = nx.DiGraph(edge_lst)
        return pred_graph

    # works with directed grpahs
    def get_louvain_communities(
        self,
        dir: str = "/PlanRGCN/extracted_features/predicate/predicate_cooccurence/batch_response/",
    ):
        edges = self.get_predicate_from_dir(dir)
        pred_graph = self.create_pred_graph(edges)
        commun = nx.algorithms.community.louvain_communities(pred_graph)
        commun_2 = []
        for x in commun:
            commun_2.append(list(x))
        path = self.save_dir + "/communities_louvain.pickle"
        with open(path, "wb") as f:
            pickle.dump(commun_2, f)

    def components_preds(self, pred_graph):
        components = nx.strongly_connected_components(pred_graph)
        com_sets = set()
        for i, c in enumerate(components):
            print(f"comp enum: {i:,.0f}")
            com_sets.add(tuple(c))

        pred_2_index = {}
        for idx, com in enumerate(com_sets):
            print(f"{idx:,.0f}")
            for pred in list(com):
                if pred in pred_2_index.keys():
                    pred_2_index[pred].append(idx)
                else:
                    pred_2_index[pred] = [idx]

        pred2index = pred_2_index
        max_clusters = len(com_sets)
        path = f"{self.save_dir}/components/comps_{str(max_clusters)}.json"
        os.system("mkdir -p " + f"{self.save_dir}/components/")
        with open(path, "w") as f:
            json.dump((pred2index, max_clusters), f)
        return pred2index, max_clusters

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

    def create_clusters(self, predicate_dir: str, is_component=True):
        start = time.time()
        if self.verbose:
            print("-" * 40 + "\n")
            print("Beginning Predicate Graph Construction" + "\n")
            print("-" * 40)
        # pred_graph_path = f"{self.save_dir}/pred_graph.pickle"
        pred_graph_path = f"{self.save_dir}/pred_graph2.pickle"
        if os.path.exists(pred_graph_path):
            with open(pred_graph_path, "rb") as f:
                pred_graph = pickle.load(f)
        else:
            edge_lst = self.get_predicate_from_dir(predicate_dir)
            pred_graph = self.create_pred_graph(edge_lst)
            with open(pred_graph_path, "wb") as f:
                pred_graph = pickle.dump(pred_graph, f)

        print(f"Predicate Graph constructed after: {time.time()-start:,.2f}")
        if is_component:
            self.components_preds(pred_graph)
            return
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


def create_louvain_to_p_index(
    path="/PlanRGCN/data/pred/pred_co/communities_louvain.pickle",
    output_path="/PlanRGCN/data/pred/pred_co/pred2index_louvain.pickle",
):
    if not os.path.exists(path):
        raise Exception("File does not exist " + path)
    cs = pickle.load(open(path, "rb"))
    pred2index = {}
    m = len(cs) + 1
    for i, c in enumerate(cs):
        for p in list(c):
            pred2index[p] = i
    print(pred2index)
    print(m)
    pickle.dump((pred2index, m), open(output_path, "wb"))


if __name__ == "__main__":
    d = PredicateCommunityCreator(save_dir="/PlanRGCN/data/pred/pred_co")
    d.get_louvain_communities(
        dir="/PlanRGCN/extracted_features/predicate/predicate_cooccurence/batch_response/"
    )
    create_louvain_to_p_index(
        path="/PlanRGCN/data/pred/pred_co/communities_louvain.pickle",
        output_path="/PlanRGCN/data/pred/pred_co/pred2index_louvain.pickle",
    )
    exit()
    d.create_clusters(
        "/PlanRGCN/extracted_features/predicate/predicate_cooccurence/batch_response/"
    )
    exit()
