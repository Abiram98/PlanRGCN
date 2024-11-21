import sys
sys.path.append('/PlanRGCN/')
import os
os.environ['QG_JAR']='/PlanRGCN/PlanRGCN/qpe/target/qpe-1.0-SNAPSHOT.jar'
os.environ['QPP_JAR']='/PlanRGCN/qpp/qpp_features/sparql-query2vec/target/sparql-query2vec-0.0.1.jar'


import time

from q_gen.pp_gen import PPGenerator
from q_gen.util import Utility


import json

import pickle

import pandas as pd

from feature_extraction.sparql import Endpoint
from graph_construction.jar_utils import get_ent_rel



class instanceData:
    """Loads the extracted instance data of form ?s PPcand ?o . ?o {pred} ?o2
    Then creates a generator for the instance data
    """
    def __init__(self, base_folder, dir_name):
        self.fold_name = os.path.join(base_folder, dir_name)
        pp_text_f = os.path.join(self.fold_name, 'PPcand_text.txt')
        pp_tp_data = os.path.join(self.fold_name, 'PPcand_text.json')
        assert os.path.exists(pp_text_f)
        assert os.path.exists(pp_tp_data)
        with open(pp_text_f) as f:
            self.PP_predicate = f.read()[:-1] # last character is newline
        with open(pp_tp_data, 'r') as f:
            dat = json.load(f)
        self.relations = [ins['pred2']['value'] for ins in dat]

    def gen(self):
        for r in self.relations:
            yield self.PP_predicate, r


class PPCandGenTrain(PPGenerator):
    """
    This class generates property path queries with only relations and entities that are used in the training set.
    """
    def __init__(self, train_file, val_file, test_file, pred_stat_path,url, outputfolder, subj_stat_path, new_qs_folder='/data/DBpedia_3_class_full/newPPs'):
        self.train_file = train_file
        self.val_file = val_file
        self.test_file = test_file
        self.endpoint = Endpoint(url)
        self.train_df = pd.read_csv(self.train_file, sep='\t')
        self.val_df = pd.read_csv(self.val_file, sep='\t')
        self.test_df = pd.read_csv(self.test_file, sep='\t')
        assert os.path.exists(outputfolder)

        self.out_t_rel_folder = os.path.join(outputfolder, 'newPPqueries')
        os.makedirs(self.out_t_rel_folder, exist_ok='True')
        with open(os.path.join(self.out_t_rel_folder, 'README.txt'), 'w') as f:
            f.write("Property path queries generated for augmenting the slow/med queries\n")

        print('Starting entity and relation extraction from training set')
        train_rels, train_ents = Utility.get_ent_rels_from_train(self.train_df)
        print('Extracting frequency of every relation in KG')
        self.pred_freq = self.get_pred_freq(pred_stat_path)

        #extract PP candidates from train set
        print('Extracting PP candidates for relations in train set')
        if not os.path.exists(os.path.join(self.out_t_rel_folder, f"PPcand.pickle")):
            self.extractPPcandFromTrain(train_rels)
        with open(os.path.join(self.out_t_rel_folder, f"PPcand.pickle"), 'rb') as f:
            PPcands = pickle.load(f)

        print('Extracting relations that we can add to values clause')
        valuable_rels = self.get_rels_usable_in_values(PPcands, train_rels)
        rel_str = self.   get_rel_value_clause(valuable_rels)

        folder_cand__pairs = os.path.join(self.out_t_rel_folder, 'PP_instance_data')
        if not os.path.exists(folder_cand__pairs):
            folder_cand__pairs = self.extract_inst_data_temp3(PPcands, rel_str)

        # already run
        #self.generate_pp_qs_template3(folder_cand__pairs, url)

        #Folders generated by 'generate_pp_qs_template3'
        query_folder = os.path.join(self.out_t_rel_folder, 'PP_generated_qs')
        #Temporary folder that should not be used beyond the next step
        above1sec = os.path.join(query_folder, 'above1sec')
        #For our purposes these queries serve no purpose
        under1sec = os.path.join(query_folder, 'under1sec')

        #Folder to copy the slow running queries to
        above10sec = os.path.join(query_folder, 'above10sec')
        bet_1_10_sec = os.path.join(query_folder, 'bet1_10sec')
        #self.distinguish_med_from_slow(above10sec, above1sec, bet_1_10_sec, url)

        #self.collect_slow_qs_lat(above10sec, url)

        med_df = self.collect_med_slow_qs(bet_1_10_sec)
        med_df = self.filter_med_qs(med_df)
        slow_df = self.collect_med_slow_qs(above10sec)
        slow_df = self.filter_slow_qs(slow_df)
        comb_new_pp = pd.concat([med_df, slow_df])
        self.output_new_pp_qs(comb_new_pp, new_qs_folder, test_file)

    def collect_slow_qs_lat(self, above10sec, url):
        self.endpoint2 = Endpoint(url)
        self.endpoint2.sparql.setTimeout(900)
        slow = 0
        timeout = 0
        print(f"Collecting slow queries\n\n")
        for pred_id in os.listdir(above10sec):
            q_fold_w_pred_id = os.path.join(above10sec, pred_id)
            for query_id in os.listdir(q_fold_w_pred_id):
                q_folder = os.path.join(q_fold_w_pred_id, query_id, query_id)
                with open(os.path.join(q_folder, 'query_text.txt'), 'r') as f:
                    query_text = f.read()
                try:
                    start = time.time()
                    card = len(self.endpoint2.run_query_and_results(query_text))
                    duration = time.time() - start
                    dest_fold = os.path.join(above10sec, pred_id, query_id)
                    assert os.path.exists(dest_fold)
                    with open(os.path.join(dest_fold, 'duration_card.txt'), 'w') as f:
                        f.write(f"{duration},{card}")
                    if duration > 10:
                        slow += 1
                except TimeoutError:
                    dest_fold = os.path.join(above10sec, pred_id, query_id)
                    assert os.path.exists(dest_fold)
                    with open(os.path.join(dest_fold, 'duration_card.txt'), 'w') as f:
                        f.write(f"{900},{0}")
                    timeout += 1
                print(f"\r Currently processed data: slow {slow} & timeout {timeout}")

    def filter_med_qs(self, med_df):
        # only queries that returns results
        med_df = med_df[med_df['resultset_1'] > 0]
        # to filter away the queries that are borderlining
        med_df = med_df[(med_df['mean_latency'] > 2) & (med_df['mean_latency'] < 8)]
        med_df = self.reorder_df_to_pred_format(med_df)
        return med_df

    def filter_slow_qs(self, slow_df):
        # only queries that returns results
        slow_df = slow_df[slow_df['resultset_1'] > 0]
        # to filter away the queries that are borderlining
        slow_df = slow_df[(slow_df['mean_latency'] > 15) & (slow_df['mean_latency'] < 900)]
        slow_df = self.reorder_df_to_pred_format(slow_df)
        return slow_df

    def output_new_pp_qs(self, med_df, new_qs_folder, test_file):
        base_test_df = pd.read_csv(test_file, sep='\t')
        assert len(set(base_test_df.columns).intersection(list(med_df.columns)))
        new_df = pd.concat([base_test_df, med_df])
        assert not os.path.exists(new_qs_folder)
        os.makedirs(new_qs_folder)
        new_df.to_csv(os.path.join(new_qs_folder, 'queries.tsv'), sep='\t', index=False)

    def collect_med_slow_qs(self, time_int_fold_name) -> pd.DataFrame:
        med_qs = []
        for pred_id in os.listdir(time_int_fold_name):
            q_fold_w_pred_id = os.path.join(time_int_fold_name, pred_id)
            for query_id in os.listdir(q_fold_w_pred_id):
                q_folder = os.path.join(q_fold_w_pred_id, query_id)
                try:
                    with open(os.path.join(q_folder, query_id, 'query_text.txt'), 'r') as f:
                        query_text = f.read()
                    with open(os.path.join(q_folder, 'duration_card.txt'), 'r') as f:
                        dur_car_text = f.read().split(',')
                        dur = float(dur_car_text[0])
                        card = int(dur_car_text[1])
                    if dur > 1:
                        med_qs.append({
                            'queryID': f"http://lsq.aksw.org/PPtemplate3_{pred_id}_{query_id}",
                            'id': f"http://lsq.aksw.org/PPtemplate3_{pred_id}_{query_id}",
                            'queryString': query_text,
                            'mean_latency': dur,
                            'resultset_1': card
                        })
                except Exception:
                    """this means that the query has not been run yet"""
                    continue
        med_df = pd.DataFrame.from_dict(med_qs)
        return med_df

    def reorder_df_to_pred_format(self, df:pd.DataFrame):
        """
        Reorders the df to be on the same format as test_sampled.tsv such that inference functionality can be used
        @param df:
        @return: df
        """
        cols = ['id', 'queryString', 'query_string_0', 'latency_0', 'resultset_0',
       'query_string_1', 'latency_1', 'resultset_1', 'query_string_2',
       'latency_2', 'resultset_2', 'mean_latency', 'min_latency',
       'max_latency', 'time_outs', 'path', 'triple_count', 'subject_predicate',
       'predicate_object', 'subject_object', 'fully_concrete', 'join_count',
       'filter_count', 'left_join_count', 'union_count', 'order_count',
       'group_count', 'slice_count', 'zeroOrOne', 'ZeroOrMore', 'OneOrMore',
       'NotOneOf', 'Alternative', 'ComplexPath', 'MoreThanOnePredicate',
       'queryID', 'Queries with 1 TP', 'Queries with 2 TP',
       'Queries with more TP', 'S-P Concrete', 'P-O Concrete', 'S-O Concrete']
        for x in cols:
            if x not in df.columns:
                df[x] = None
        return df[cols]




    def distinguish_med_from_slow(self, above10sec, above1sec, bet_1_10_sec, url):
        os.makedirs(bet_1_10_sec)
        os.makedirs(above10sec)
        self.endpoint2 = Endpoint(url)
        self.endpoint2.sparql.setTimeout(10)
        slow = 0
        med = 0
        print(f"Starting distinguishing between med and slow queries\n\n")
        for pred_id in os.listdir(above1sec):
            q_fold_w_pred_id = os.path.join(above1sec, pred_id)
            for query_id in os.listdir(q_fold_w_pred_id):
                q_folder = os.path.join(q_fold_w_pred_id, query_id)
                with open(os.path.join(q_folder, 'query_text.txt'), 'r') as f:
                    query_text = f.read()
                try:
                    start = time.time()
                    card = len(self.endpoint2.run_query_and_results(query_text))
                    duration = time.time() - start
                    dest_fold = os.path.join(bet_1_10_sec, pred_id, query_id)
                    os.makedirs(dest_fold)
                    os.system(f"cp -r {q_folder} {dest_fold}")
                    with open(os.path.join(dest_fold, 'duration_card.txt'), 'w') as f:
                        f.write(f"{duration},{card}")
                    med += 1
                except TimeoutError:
                    dest_fold = os.path.join(above10sec, pred_id, query_id)
                    os.makedirs(dest_fold)
                    os.system(f"cp -r {q_folder} {dest_fold}")
                    slow += 1
                print(f"\r Currently processed data: slow {slow} & med {med}")

    def generate_pp_qs_template3(self, folder_cand__pairs, url):
        self.endpoint2 = Endpoint(url)
        self.endpoint2.sparql.setTimeout(1)
        query_folder = os.path.join(self.out_t_rel_folder, 'PP_generated_qs')
        above1sec = os.path.join(query_folder, 'above1sec')
        under1sec = os.path.join(query_folder, 'under1sec')
        above1sec_no = 0
        under1sec_no = 0
        now = time.time()
        print("Beginning PP query generation phase\n\n")
        for dir_name in os.listdir(folder_cand__pairs):
            gen = instanceData(folder_cand__pairs, dir_name).gen()
            for pp_pred, tp_pred in gen:
                query = self.PPTemplateFromTrain3(pp_pred, tp_pred, op='+')
                try:
                    self.endpoint2.run_query_and_results(query)
                except TimeoutError:
                    above1sec_no += 1
                    q_fold = os.path.join(above1sec, dir_name, str(above1sec_no))
                    os.makedirs(q_fold)
                    with open(os.path.join(q_fold, "query_text.txt"), 'w') as f:
                        f.write(query)
                    with open(os.path.join(q_fold, "PP_predicate.txt"), 'w') as f:
                        f.write(pp_pred)
                    with open(os.path.join(q_fold, "TP_predicate.txt"), 'w') as f:
                        f.write(tp_pred)
                else:
                    under1sec_no += 1
                    q_fold = os.path.join(under1sec, dir_name, str(under1sec_no))
                    os.makedirs(q_fold)
                    with open(os.path.join(q_fold, "query_text.txt"), 'w') as f:
                        f.write(query)
                    with open(os.path.join(q_fold, "PP_predicate.txt"), 'w') as f:
                        f.write(pp_pred)
                    with open(os.path.join(q_fold, "TP_predicate.txt"), 'w') as f:
                        f.write(tp_pred)
                print(f"run time: {time.time() - now:.2f} - Currently evaluated {above1sec_no} med/slow qs and {under1sec_no} fast qs")

    def extract_inst_data_temp3(self, PPcands, rel_str):
        folder_cand__pairs = os.path.join(self.out_t_rel_folder, 'PP_instance_data')
        os.makedirs(folder_cand__pairs)
        print('Stating instance data creation')
        now = time.time()
        for i, cand in enumerate(PPcands):
            query = self.coRelExtractQuery(cand, rel_str, n_co_pred=20)
            res = self.endpoint.run_query_and_results(query)
            self.save_co_pred_inst_data(cand, folder_cand__pairs, i, res)
            print(f'\r Current status: [{i}/{len(PPcands)}], running time: {time.time() - now:.2f}        ')
        return folder_cand__pairs

    def PPTemplateFromTrain3(self, cand, pred, op='+'):
        return f"""
        SELECT ?o ?o2 WHERE {{
            ?s <{cand}>{op} ?o .
            ?s <{pred}> ?o2
        }}
        """

    def save_co_pred_inst_data(self, cand, folder_cand__pairs, i, res):
        q_folder = os.path.join(folder_cand__pairs, str(i))
        os.makedirs(q_folder)
        pp_str_path = os.path.join(q_folder, 'PPcand_text.txt')
        with open(pp_str_path, 'w') as f:
            f.write(cand + '\n')
        pp_temp_inst_path = os.path.join(q_folder, 'PPcand_text.json')
        with open(pp_temp_inst_path, 'w') as f:
            json.dump(res, f)

    def coRelExtractQuery(self, cand, rel_str, n_co_pred=10):
        query = f"""
            SELECT DISTINCT ?pred2 WHERE {{         
                ?s <{cand}> ?o .
                ?o ?pred2 ?o2 .  
                
                VALUES ?pred2 {rel_str}   
            }}
            LIMIT {n_co_pred}
            """
        return query

    def get_rel_value_clause(self, valuable_rels):
        rel_str = '{ '
        for x in valuable_rels:
            rel_str += f'<{x}>'
        rel_str += ' }'
        return rel_str

    def get_rels_usable_in_values(self, PPcands, train_rels):
        if not os.path.exists(os.path.join(self.out_t_rel_folder, 'train_rels_in_values_clause.pickle')):
            def get_train_pred_str(start, end):
                t_ent_str = "{"
                for e in train_rels[start:end]:
                    t_ent_str += f" <{e}> "
                t_ent_str += "}"
                return t_ent_str

            def get_train_pred_str_single_rel(rel):
                t_ent_str = "{"
                t_ent_str += f" <{rel}> "
                t_ent_str += "}"
                return t_ent_str

            # check which rels are hard to process in values clause
            valuable_rels = set()
            cand = list(PPcands)[0]
            for i, r in enumerate(train_rels):
                query = f"""
                SELECT ?pred2 WHERE {{         
                    ?s <{cand}> ?o .
                    ?s ?pred2 ?o2 .  
        
                    VALUES ?pred2 {get_train_pred_str_single_rel(r)}   
                }}
                LIMIT 1
                """
                try:
                    self.endpoint.run_query(query)
                    valuable_rels.add(r)
                except Exception as e:
                    print(e)
            with open(os.path.join(self.out_t_rel_folder, 'train_rels_in_values_clause.pickle'), 'wb') as f:
                pickle.dump(valuable_rels, f)
        else:
            with open(os.path.join(self.out_t_rel_folder, 'train_rels_in_values_clause.pickle'), 'rb') as f:
                valuable_rels = pickle.load(f)
        return valuable_rels

    def extractPPcandFromTrain(self, train_rels):
        access_freq = lambda x: self.pred_freq[x] if x in self.pred_freq.keys() else 0
        sorted_train_rels = list(set(sorted(train_rels, key=access_freq, reverse=True)))
        pp_cands = set()
        print('Starting PP candicate identification', end='')
        for i, rel in enumerate(sorted_train_rels):
            if (i % 10 == 0) and (i > 0):
                with open(os.path.join(self.out_t_rel_folder, f"PPcand_{i}.pickle"), 'wb') as f:
                    pickle.dump(pp_cands, f)
            if self.is_rel_PP_legilable(rel):
                pp_cands.add(rel)
            print(f"\r[{i}/{len(sorted_train_rels)}] PP candidates in train: {len(pp_cands)}", end='')
        with open(os.path.join(self.out_t_rel_folder, f"PPcand.pickle"), 'wb') as f:
            pickle.dump(pp_cands, f)


    @DeprecationWarning
    def pp_template_rel(self, rel, pp_type = "+"):
        """
        This method will not create valid queries for Virtuoso, because Virtuoso does not support proeprty path queries wihth unbound subjects.
        Instantiates property path SPARQL query where
        @param rel:
        @return: Instantiated sparql query with the property path rel
        """
        return f"""
        SELECT ?o WHERE {{
            ?s <{rel}>{pp_type} ?o
        }}
        """


if __name__ == "__main__":
    base_path = '/data/DBpedia_3_class_full'
    train_file = f'{base_path}/train_sampled.tsv'
    val_file = f'{base_path}/val_sampled.tsv'
    test_file = f'{base_path}/test_sampled.tsv'
    url= 'http://172.21.233.14:8891/sparql'
    pred_stat_path = '/data/metaKGStat/dbpedia/predicate/pred_stat/batches_response_stats/freq'
    subj_stat_path = '/data/metaKGStat/dbpedia/entity/ent_stat/batches_response_stats/subj'
    outputfolder = '/data/generatedPP'
    os.makedirs(outputfolder, exist_ok=True)
    gen = PPCandGenTrain(train_file, val_file, test_file,pred_stat_path, url, outputfolder, subj_stat_path)