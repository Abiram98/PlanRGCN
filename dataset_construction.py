import json
import os
import pandas as pd
if __name__ == "__main__":
    os.system('apt-get update && apt-get install python3 -y && apt-get install pip -y && pip install pandas')
    queries_folder = '/leapfrog-rdf-benchmark/benchmark/queries'
    res_fold = '/leapfrog-rdf-benchmark/results'
    
    query_paths = [os.path.join(dp, f) for dp, dn, filenames in os.walk(queries_folder) for f in filenames if os.path.splitext(f)[1] == '.txt' and 'warmup' not in f]
    #print(query_paths)
    queries  = []
    jena_rt = []
    bloom_rt = []
    lf_rt = []
    paths = []
    for p in query_paths:
        res_dir = res_fold+'/'+ p[len(queries_folder)+1:-4]
        assert os.path.exists(res_dir)
        lf_path = res_dir +'/leapfrog.csv'#recheck this path
        bloom_path = res_dir + '/jenaclone.csv'
        jena = res_dir +'/jena.csv'
        
        with open(p, 'r') as f:
            for line in f:
                queries.append(line[:-1])
        
        bloom_df = pd.read_csv(bloom_path)
        temp = list(bloom_df['average time (3)'])
        bloom_rt.extend(temp)
        paths.extend([res_dir for x in range(len(temp))])
        j_df = pd.read_csv(jena)
        jena_rt.extend(list(j_df['average time (3)']))
        leap_df = pd.read_csv(lf_path)
        lf_rt.extend(list(leap_df['average time (3)']))
        

        assert len(queries) == len(jena_rt) and len(bloom_rt) == len(jena_rt) and len(lf_rt) == len(jena_rt) and len(lf_rt) == len(paths)
    data = []
    for q, j, bf, l, p in zip(queries, jena_rt, bloom_rt, lf_rt, paths):
        q_data = {'query': q, 'bloom_runtime':bf, 'jena_runtime':j, 'leapfrog':l, 'path':p}
        data.append(q_data)
    json.dump(data,open('all_data.json','w'))
    print('done!')
    
