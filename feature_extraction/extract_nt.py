
#extract predicate statistics from N triple file instead of endpoint

import json
import argparse

def extract_predicate_stats(preds, literals_mapper, path_to_nt, output_path):
    preds_dct = {}
    for p in preds:
        preds_dct[p] = set()
    
    with open(path_to_nt, 'r') as f:
        for line in f:
            line_splits = line.split(' ')
            if len(line_splits) != 4:
                continue
            nt_pred = line_splits[1][1:-1]
            if nt_pred in preds_dct.keys() and line_splits[2].startswith('"'):
                #preds_dct[nt_pred] = preds_dct[nt_pred] + 1
                preds_dct[nt_pred].add(line_splits[2])
    for k in preds_dct.keys():
        literals_mapper[k] = len(list(preds_dct[k]))
    json.dump(literals_mapper, open(output_path,'w'))

def get_absnt_pred(pred_file='/work/data/extracted_statistics/predicates_only.json',
                        freq_map_path= "/work/data/extracted_statistics/updated_pred_freq.json",
                        lit_map_path="/work/data/extracted_statistics/updated_pred_unique_lits.json",
                        ent_map_path="/work/data/extracted_statistics/updated_pred_unique_subj_obj.json", feat_type='lits'):
    all_preds = json.load(open(pred_file))
    
    if feat_type == 'lits':
        lits = json.load(open(lit_map_path))
        absnt_preds = extract_absnt_preds(lits, all_preds)
        print(f"# of absent predicate lits {len(absnt_preds)} of {len(all_preds)}")
        return absnt_preds, lits
    elif feat_type == 'freq': 
        pred_freq = json.load(open(freq_map_path))
        absnt_preds = extract_absnt_preds(pred_freq, all_preds)
        print(f"# of absent predicate freq {len(absnt_preds)} of {len(all_preds)}")
        return absnt_preds
    elif feat_type == 'ents':
        ents = json.load(open(ent_map_path))
        absnt_preds = extract_absnt_preds(ents, all_preds)
        print(f"# of absent predicate ents {len(absnt_preds)} of {len(all_preds)}")
        return absnt_preds
    
    #print(f"# of absent predicate frequencies {len(all_preds)-len(list(pred_freq.keys()))} of {len(all_preds)}")
    #print(f"# of absent predicate ents {len(all_preds)-len(list(ents.keys()))} of {len(all_preds)}")

def extract_absnt_preds(mapper:dict, preds :list[str]):
    absnt_preds = []
    for i in preds:
        if i not in mapper.keys():
            absnt_preds.append(i)
    return absnt_preds

def run():
    parser = argparse.ArgumentParser(
                    prog='extract_nt',
                    epilog='Used to extract statistics from n triple file')
    parser.add_argument('--nt', '--nt')
    parser.add_argument('--task', '--task')
    parser.add_argument('--all_pred_file', '--all_pred_file')
    parser.add_argument('--lit_map_p', '--lit_map_p')
    parser.add_argument('--output', '--output')
    #parser.add_argument('--preds', '--preds')
    args = parser.parse_args()
    
    path_to_nt = args.nt
    task = args.task
    pred_file = args.all_pred_file
    lit_map_path = args.lit_map_p
    output = args.output
    if task == "pred_lits":
        absnt_preds,pred_literals = get_absnt_pred(feat_type='lits', pred_file=pred_file,lit_map_path=lit_map_path)
        
        extract_predicate_stats(absnt_preds, pred_literals, path_to_nt, output_path=output)
        print("Finished literal extraction for missing features")

if __name__ == "__main__":
    run()