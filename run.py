import os
from pathlib import Path



os.environ['QG_JAR']='/PlanRGCN/PlanRGCN/qpe/target/qpe-1.0-SNAPSHOT.jar'
os.environ['QPP_JAR']='/PlanRGCN/qpp/qpp_features/sparql-query2vec/target/sparql-query2vec-0.0.1.jar'

from graph_construction.jar_utils import get_query_graph
from load_balance.workload.workload import Workload, WorkloadAnalyzer
from load_balance.result_analysis.adcanal import ADMCTRLAnalyser,ADMCTRLAnalyserV2
#from query_log_analysis import QueryLogAnalyzer
from trainer.new_post_predict import QPPResultProcessor
import pandas as pd
from q_gen.util import Utility

def analyze_Med_interval_qs(sample_path, pred_path, train_log, val_log):
    tql = pd.read_csv(train_log, sep='\t')
    vql = pd.read_csv(val_log, sep='\t')
    train_val_rels, train_val_ents = Utility.get_ent_rels_from_train_val(tql, vql)
    ql = pd.read_csv(sample_path, sep='\t')
    ql['id'] = ql['id'].apply(lambda x: x[20:])
    preds = pd.read_csv(pred_path)
    df = ql.merge(preds, on='id')
    df = df[(df.time_cls==1)&(df.planrgcn_prediction==1)].copy()

    df = df[~df['queryString'].str.contains('rdf:type')]
    from collections import Counter
    qs = []
    for idx, row in df.iterrows():
        try:
            qg = get_query_graph(row['queryString'])
        except Exception:
            continue
        rel_ts = dict(Counter([x[2] for x in qg['edges']]))
        q_data = {}
        for k in rel_ts.keys():
            q_data[k] = rel_ts[k]
        q_data['id'] = row['queryID']
        qs.append(q_data)
    res_df = pd.DataFrame(qs)
    pred_stat_path = '/data/metaKGStat/dbpedia/predicate/pred_stat/batches_response_stats/freq'
    pred_freq = Utility.get_pred_freq(pred_stat_path)

    ### Analysis of chosen template
    pred_card = 2566984
    lower = pred_card*0.5
    upper = pred_card*1.

    keys = [ x for x in pred_freq.keys() if lower <pred_freq[x] < upper]
    keys = [ x for x in keys if x not in train_val_rels]

    f"""
    SELEC
    """








train_log = '/data/DBpedia_3_class_full/train_sampled.tsv'
val_log = '/data/DBpedia_3_class_full/val_sampled.tsv'
analyze_Med_interval_qs('/data/DBpedia_3_class_full/test_sampled.tsv', '/data/DBpedia_3_class_full/plan_l14096_l21025_no_pred_co/test_pred.csv', train_log, val_log)

def query_log_runtimes(train_path, val_path, test_path):
    train = pd.read_csv(train_path, sep='\t')
    val = pd.read_csv(val_path, sep='\t')
    test = pd.read_csv(test_path, sep='\t')
    lat_cols = ['latency_0', 'latency_1', 'latency_2']
    total_runtime = 0
    for df in [train, val, test]:
        for idx, row in df.iterrows():
            for col in lat_cols:
                total_runtime += row[col]
                assert row[col] <= 900

    def convert_seconds_to_time(seconds):
        days = seconds // 86400
        hours = (seconds % 86400) // 3600
        minutes = (seconds % 3600) // 60
        remaining_seconds = seconds % 60
        return f"{days}d {hours}h {minutes}m {remaining_seconds}s"

    return convert_seconds_to_time(total_runtime)



def total_workload_runtime():
    base = f'/data/DBpedia_3_class_full'
    train= os.path.join(base, 'train_sampled.tsv')
    val= os.path.join(base, 'val_sampled.tsv')
    test= os.path.join(base, 'test_sampled.tsv')
    print(query_log_runtimes(train, val, test))

    base = f'/data/wikidata_3_class_full'
    train= os.path.join(base, 'train_sampled.tsv')
    val= os.path.join(base, 'val_sampled.tsv')
    test= os.path.join(base, 'test_sampled.tsv')
    print(query_log_runtimes(train, val, test))


def ql_table():
    from query_class_analyzer import QueryClassAnalyzer, SemiFineGrainedQueryClassAnalyzer
    base = '/data/wikidata_3_class_full/'
    SemiFineGrainedQueryClassAnalyzer(f"{base}/train_sampled.tsv",
                       f"{base}/val_sampled.tsv",
                       f"{base}/test_sampled.tsv",
                       objective_file='/data/wikidata_3_class_full/plan01_n_gq/objective.py'
                       )
    base = '/data/DBpedia_3_class_full/'
    SemiFineGrainedQueryClassAnalyzer(f"{base}/train_sampled.tsv",
                       f"{base}/val_sampled.tsv",
                       f"{base}/test_sampled.tsv",
                       objective_file='/data/wikidata_3_class_full/plan01_n_gq/objective.py'
                       )
def runtime_plot():
    df = pd.read_csv('/data/wikidata_3_class_full/test_sampled.tsv', sep='\t')
    import matplotlib.pyplot as plt
    runtimes = df['mean_latency']
    import matplotlib.pyplot as plt

    def get_hist_data(runtimes:pd.Series, bin_width=0.01):
        start = 0
        bin_edges = []
        while start <= 900:
            bin_edges.append(start)
            start += bin_width

        counts = []
        for i in range(1, len(bin_edges)):
            if i == 0:
                continue
            counts.append( len(runtimes[(bin_edges[i-1] < runtimes) & (runtimes <= bin_edges[i])]) )

        bin_width = [bin_edges[i + 1] - bin_edges[i] for i in range(len(bin_edges) - 1)]




#runtime_plot()

exit()

def qppDBpediaNewqs(exp_type="PP"):
    dbpedia_plan_path = '/data/DBpedia_3_class_full/newPPs/plan_inference.csv'
    base_path = '/data/DBpedia_3_class_full'
    val_sampled_file = os.path.join(base_path,'val_sampled.tsv')
    train_sampled_file = os.path.join(base_path,'train_sampled.tsv')
    p = QPPResultProcessor(obj_fil='/data/DBpedia_3_class_full/objective.py',
                           dataset="DBpedia",
                           exp_type=exp_type,
                           test_sampled_file="/data/DBpedia_3_class_full/newPPs/queries.tsv",
                           val_sampled_file=val_sampled_file,
                           train_sampled_file=train_sampled_file)


    dbpedia_nn_path = '/data/DBpedia_3_class_full/newPPs/nn_prediction.csv'
    dbpedia_svm_path = '/data/DBpedia_3_class_full/newPPs/svm_pred.csv'

    p.evaluate_dataset(path_to_pred=dbpedia_plan_path,
                             sep = ',',
                             ground_truth_col='time_cls',
                             pred_col='pred',
                             id_col='id',
                             approach_name="P",
                                reg_to_cls=False)

    p.evaluate_dataset(path_to_pred=dbpedia_nn_path,
                             sep = ',',
                             ground_truth_col='time_cls',
                             pred_col='nn_prediction',
                             id_col='id',
                             approach_name="NN",reg_to_cls=True )
    p.evaluate_dataset(path_to_pred=dbpedia_svm_path,
                             sep = ',',
                             ground_truth_col='time_cls',
                             pred_col='svm_prediction',
                             id_col='id',
                             approach_name="SVM",reg_to_cls=True)
    print(p.process_results(add_symbol=''))
    print(p.true_counts)

def qppDBpediaNewqsUnseen(exp_type="completly_unseen"):

    base_path = '/data/DBpedia_3_class_full'
    val_sampled_file = os.path.join(base_path,'val_sampled.tsv')
    train_sampled_file = os.path.join(base_path,'train_sampled.tsv')
    p = QPPResultProcessor(obj_fil='/data/DBpedia_3_class_full/objective.py',
                           dataset="DBpedia",
                           exp_type=exp_type,
                           test_sampled_file="/data/DBpedia_3_class_full/newUnseenQs2/queries.tsv",
                           val_sampled_file=val_sampled_file,
                           train_sampled_file=train_sampled_file)

    dbpedia_plan_path = '/data/DBpedia_3_class_full/newUnseenQs3/plan_inference.csv'
    dbpedia_nn_path = '/data/DBpedia_3_class_full/newUnseenQs3/nn_prediction.csv'
    dbpedia_svm_path = '/data/DBpedia_3_class_full/newUnseenQs3/svm_pred.csv'

    p.evaluate_dataset(path_to_pred=dbpedia_plan_path,
                             sep = ',',
                             ground_truth_col='time_cls',
                             pred_col='pred',
                             id_col='id',
                             approach_name="P",
                                reg_to_cls=False)
    p.evaluate_dataset(path_to_pred=dbpedia_nn_path,
                             sep = ',',
                             ground_truth_col='time_cls',
                             pred_col='nn_prediction',
                             id_col='id',
                             approach_name="NN",reg_to_cls=True )
    p.evaluate_dataset(path_to_pred=dbpedia_svm_path,
                             sep = ',',
                             ground_truth_col='time_cls',
                             pred_col='svm_prediction',
                             id_col='id',
                             approach_name="SVM",reg_to_cls=True)

    print(p.process_results(add_symbol=''))
    print(p.true_counts)

#qppDBpediaNewqsUnseen()
#ql_table()


def sample_more_test_queries_DBpedia(path='/data/generatedPP/PP_w_Optionals', original_test_sampled='/data/DBpedia_3_class_full/test_sampled.tsv' ,base_exp_path='/data/DBpedia_3_class_full',output_path = '/data/DBpedia_3_class_full/additionalPPtestQs/queries.tsv'):
    os.listdir(path)
    slow_q_dir = os.path.join(path, 'above10sec','query_executions.tsv')
    med_q_dir = os.path.join(path, 'between1_10sec', 'query_executions.tsv')
    sl = pd.read_csv(slow_q_dir, sep='\t')
    sl_res = sl[sl['resultsetSize']>0]
    sl_res = sl_res[sl_res['latency']>sl_res['latency'].quantile(0.5)]

    m = pd.read_csv(med_q_dir, sep='\t')
    m = m[m['latency']> 1.5]
    m = m[m['latency'] < 9.5]
    import numpy as np
    np.random.seed(21)



    s_sample = sl_res.sample(60)
    s_sample = pd.concat([s_sample,m])
    s_sample['id'] = s_sample['queryID']
    s_sample['mean_latency'] = s_sample['latency']

    cols = ['id', 'queryString', 'query_string_0', 'latency_0', 'resultset_0', 'query_string_1', 'latency_1', 'resultset_1', 'query_string_2', 'latency_2', 'resultset_2', 'mean_latency', 'min_latency', 'max_latency', 'time_outs', 'path', 'triple_count', 'subject_predicate', 'predicate_object', 'subject_object', 'fully_concrete', 'join_count', 'filter_count', 'left_join_count', 'union_count', 'order_count', 'group_count', 'slice_count', 'zeroOrOne', 'ZeroOrMore', 'OneOrMore', 'NotOneOf', 'Alternative', 'ComplexPath', 'MoreThanOnePredicate', 'queryID', 'Queries with 1 TP', 'Queries with 2 TP', 'Queries with more TP', 'S-P Concrete', 'P-O Concrete', 'S-O Concrete']
    for c in cols:
        if c not in s_sample.columns:
            s_sample[c] = 0
    test = pd.read_csv(original_test_sampled,sep='\t' )
    all = pd.concat([test, s_sample])
    all = all[cols]
    all.to_csv(output_path,sep='\t',index=False)

    ql = pd.concat([pd.read_csv(f'{base_exp_path}/train_sampled.tsv', sep ='\t'),pd.read_csv(f'{base_exp_path}/val_sampled.tsv', sep ='\t'), all])
    ql.to_csv(os.path.join(Path(output_path).parent, 'all.tsv'), sep='\t', index=False)

#sample_more_test_queries_DBpedia(path='/data/generatedPP/PP_w_Optionals')


def qppWikidata(exp_type="all", version='VLDB'):
    base_path = '/data/wikidata_3_class_full'
    val_sampled_file = os.path.join(base_path,'val_sampled.tsv')
    train_sampled_file = os.path.join(base_path,'train_sampled.tsv')
    test_sampled_file = os.path.join(base_path,'test_sampled.tsv')

    p = QPPResultProcessor(obj_fil='/data/DBpedia_3_class_full/objective.py', dataset="Wikidata",
                           exp_type=exp_type,
                           val_sampled_file=val_sampled_file,
                           train_sampled_file=train_sampled_file,
                           test_sampled_file=test_sampled_file)
    p.evaluate_dataset(path_to_pred="/data/wikidata_3_class_full/planRGCN_no_pred_co/test_pred.csv",
                       sep=',',
                       ground_truth_col='time_cls',
                       pred_col='planrgcn_prediction',
                       id_col='id',
                       approach_name="P")

    p.evaluate_dataset(path_to_pred="/data/wikidata_3_class_full/nn/k25/nn_test_pred.csv",
                       sep=',',
                       ground_truth_col='time',
                       pred_col='nn_prediction',
                       id_col='id',
                       approach_name="NN")

    p.evaluate_dataset(path_to_pred="/data/wikidata_3_class_full/baseline/svm/test_pred.csv",
                       sep=',',
                       ground_truth_col='time',
                       pred_col='svm_prediction',
                       id_col='id',
                       approach_name="SVM")

    print(p.process_results(add_symbol='', version=version))
    print(p.true_counts)

def qppDBpedia(exp_type="all", version='VLDB'):
    base_path = '/data/DBpedia_3_class_full'
    val_sampled_file = os.path.join(base_path,'val_sampled.tsv')
    train_sampled_file = os.path.join(base_path,'train_sampled.tsv')
    test_sampled_file = os.path.join(base_path,'test_sampled.tsv')
    p = QPPResultProcessor(obj_fil='/data/DBpedia_3_class_full/objective.py',
                           dataset="DBpedia",
                           exp_type=exp_type,
                           test_sampled_file=test_sampled_file,
                           val_sampled_file=val_sampled_file,
                           train_sampled_file=train_sampled_file)
    dbpedia_plan_path = '/data/DBpedia_3_class_full/plan_l14096_l21025_no_pred_co/test_pred.csv'
    #The following seems to be much better model.
    dbpedia_plan_path = '/data/DBpedia_3_class_full/plan16_10_2024/test_pred.csv'
    #The large retrained model
    dbpedia_plan_path = '/data/DBpedia_3_class_full/plan16_10_2024_4096_2048/test_pred.csv'
    dbpedia_nn_path = '/data/DBpedia_3_class_full/nn/test_pred.csv'
    dbpedia_svm_path = '/data/DBpedia_3_class_full/svm/test_pred_cls.csv'

    p.evaluate_dataset(path_to_pred=dbpedia_plan_path,
                             sep = ',',
                             ground_truth_col='time_cls',
                             pred_col='planrgcn_prediction',
                             id_col='id',
                             approach_name="P")
    p.evaluate_dataset(path_to_pred=dbpedia_nn_path,
                             sep = ',',
                             ground_truth_col='time_cls',
                             pred_col='nn_prediction',
                             id_col='id',
                             approach_name="NN",reg_to_cls=False )
    p.evaluate_dataset(path_to_pred=dbpedia_svm_path,
                             sep = ',',
                             ground_truth_col='time_cls',
                             pred_col='svm_prediction',
                             id_col='id',
                             approach_name="SVM")
    res = p.process_results(add_symbol='', version=version)
    print(res)
    print(p.true_counts)

def qppWikidataQuantile(exp_type="all", version='VLDB'):
    base_path = '/data/wikidataV2'
    val_sampled_file = os.path.join(base_path,'val_sampled.tsv')
    train_sampled_file = os.path.join(base_path,'train_sampled.tsv')
    test_sampled_file = os.path.join(base_path,'test_sampled.tsv')

    p = QPPResultProcessor(obj_fil='/data/wikidataV2/plan01/objective.py', dataset="Wikidata",
                           exp_type=exp_type,
                           val_sampled_file=val_sampled_file,
                           train_sampled_file=train_sampled_file,
                           test_sampled_file=test_sampled_file)
    p.evaluate_dataset(path_to_pred="/data/wikidataV2/plan01/test_pred.csv",
                       sep=',',
                       ground_truth_col='time_cls',
                       pred_col='planrgcn_prediction',
                       id_col='id',
                       approach_name="P")

    """p.evaluate_dataset(path_to_pred="/data/wikidata_3_class_full/nn/k25/nn_test_pred.csv",
                       sep=',',
                       ground_truth_col='time',
                       pred_col='nn_prediction',
                       id_col='id',
                       approach_name="NN")

    p.evaluate_dataset(path_to_pred="/data/wikidata_3_class_full/baseline/svm/test_pred.csv",
                       sep=',',
                       ground_truth_col='time',
                       pred_col='svm_prediction',
                       id_col='id',
                       approach_name="SVM")"""

    print(p.process_results(add_symbol='', version=version))
    print(p.true_counts)

qppWikidataQuantile()
#Full query log
#qppWikidata(version='VLDB')
#qppDBpedia(version='VLDB')

#seen property path only
#qppWikidata(exp_type="seen_PP")
#qppDBpedia(exp_type="seen_PP")

#DBpedia new property path queries
#qppDBpediaNewqs(exp_type="seen_PP")



def evaluate_DBpedia_adm_ctrl_scriptV2(print_results=True):
    objective_file='/data/DBpedia_3_class_full/admission_control/planrgcn_44/../../objective.py'

    test_sampled = '/data/DBpedia_3_class_full/test_sampled.tsv'
    a = ADMCTRLAnalyserV2(test_sampled, objective_file=objective_file)

    base_path = '/data/DBpedia_3_class_full/admission_controlV2/workload1'
    plan_res_adm = f'{base_path}/planrgcn'
    nn_res_adm = f'{base_path}/nn'
    svm_res_adm = f'{base_path}/svm'
    a.evaluate_dataset(plan_res_adm, 'PlanRGCN', dataset = 'DBpedia')
    a.evaluate_dataset(nn_res_adm, 'NN', dataset = 'DBpedia')
    a.evaluate_dataset(svm_res_adm, 'SVM', dataset = 'DBpedia')

    if print_results:
        a.process_results()
    return a
def evaluate_wikidata_adm_ctrl_scriptV2(print_results=True, evaluation_class=ADMCTRLAnalyserV2):
    objective_file='/data/DBpedia_3_class_full/admission_control/planrgcn_44/../../objective.py'

    test_sampled = '/data/wikidata_3_class_full/test_sampled.tsv'
    a = evaluation_class(test_sampled, objective_file=objective_file)

    base_path = '/data/wikidata_3_class_full/admission_controlV2/workload1'
    plan_res_adm = f'{base_path}/planrgcn'
    nn_res_adm = f'{base_path}/nn'
    svm_res_adm = f'{base_path}/svm'
    a.evaluate_dataset(plan_res_adm, 'PlanRGCN', dataset = 'wikidata')
    a.evaluate_dataset(nn_res_adm, 'NN', dataset = 'wikidata')
    a.evaluate_dataset(svm_res_adm, 'SVM', dataset = 'wikidata')

    if print_results:
        a.process_results()
    return a

#evaluate_DBpedia_adm_ctrl_scriptV2()
#evaluate_wikidata_adm_ctrl_scriptV2()


def query_log_plot(path, dataset):
    import pandas as pd
    train_df = pd.read_csv( os.path.join(path, 'train_sampled.tsv'), sep='\t')
    val_df = pd.read_csv(os.path.join(path, 'val_sampled.tsv'), sep='\t')
    test_df = pd.read_csv(os.path.join(path, 'test_sampled.tsv'), sep='\t')
    df = pd.concat([train_df, val_df, test_df])

    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    # Assuming df is your input DataFrame with 'mean_latency' column in seconds
    # Convert mean_latency from seconds to milliseconds
    df['mean_latency_ms'] = df['mean_latency'] * 1000
    # Set up the plot style for academic research
    sns.set_style('white')  # Clean academic-style grid

    plt.figure(figsize=(10, 6))
    # Determine the max value to set an appropriate range for the histogram
    max_latency = df['mean_latency'].max()
    bin_edges = []
    thres = 0.01 #ms first threshold
    while thres < max_latency:
        bin_edges.append((thres))
        thres += 0.01
    bin_no = 900/0.01
    #print(bin_edges)
    #plt.yscale('log')
    #plt.xscale('log')
    plt.hist(df['mean_latency'], bins=bin_edges, color='skyblue', edgecolor='black')

    # Add threshold lines at 1 second (1000 ms) and 10 seconds (10000 ms)
    plt.axvline(x=1, color='red', linestyle='--', linewidth=2, label='1 sec')
    plt.axvline(x=10, color='green', linestyle='--', linewidth=2, label='10 sec')

    # Add labels and title
    plt.title(f'Query Execution Time Distribution {dataset}', fontsize=16)
    plt.xlabel('Execution Time (s)', fontsize=14)
    plt.ylabel('Number of Queries', fontsize=14)

    # Add legend
    plt.legend()

    # Display the plot
    plt.tight_layout()
    plt.show()

    from query_log_analysis import QueryLogAnalyzer
    analyse = QueryLogAnalyzer(df)

#query_log_plot('/data/wikidata_3_class_full', "Wikidata")

def workload_check_queries(w_f):
    import pickle
    queries, arrival_times = pickle.load(open(w_f, 'rb'))
    for q in queries:
        print(q)





#qppDBpedia()
#qppDBpedia(exp_type='PP')
#qppDBpedia(exp_type='completly_unseen')

#evaluate_wikidata_adm_ctrl_scriptV2()
#evaluate_DBpedia_adm_ctrl_scriptV2()

