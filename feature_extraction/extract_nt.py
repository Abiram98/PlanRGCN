
#extract predicate statistics from N triple file instead of endpoint

import json
import argparse
import os
from pathlib import Path
import subprocess

import urllib
import urllib.parse

def extract_predicate_stats(preds, literals_mapper, path_to_nt, output_path, temp_path=None):
    t_file = open(f"{temp_path}/lit_temp.txt",'w')
    c_lines= 0
    with open(path_to_nt, 'r') as f:
        for line in f:
            c_lines += 1
            line_splits = line.split(' ')
            if len(line_splits) != 4:
                continue
            nt_pred = line_splits[1][1:-1]
            if nt_pred in preds and line_splits[2].startswith('"'):
                #preds_dct[nt_pred] = preds_dct[nt_pred] + 1
                t_file.write(f"{nt_pred} {line_splits[2]}\n")
        #literals_mapper[p] = len(list(pred_lits))
    print(f"Total # of lines in N Triple file is {c_lines}")
    t_file.close()
    res_lit_file = open(f"{temp_path}/lit_pred_count.txt",'w')
    for p_no, p in enumerate(preds):
        print(f'Processing temp files for {p}   [{p_no+1}/{len(preds)}]')
        result = subprocess.run(['grep',f'{p}', f"{temp_path}/lit_temp.txt"], capture_output=True, text=True)
        result = subprocess.run(['wc','-l'], capture_output=True, text=True, input=result.stdout)
        res_lit_file.write(f"{p} {result.stdout}")
    res_lit_file.close()
        
    #json.dump(literals_mapper, open(output_path,'w'))
def extract_unfilt_pred_lits(preds,temp_path=None, literals_mapper=None, output_path = None, print_it = 8000):
    if temp_path == None or literals_mapper == None or output_path == None:
        raise Exception()
    dct = {}
    for p in preds:
        dct[p] = 0
    f = open(f"{temp_path}/lit_temp.txt",'r')
    for eline, line in enumerate(f):
        if eline % print_it == 0:
            print(f"Processing line {eline+1}", end='\r')
        if p in line:
            dct[p] += 1
    for p in preds:
        literals_mapper[p] = dct[p]
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
        return absnt_preds, ents
    
    #print(f"# of absent predicate frequencies {len(all_preds)-len(list(pred_freq.keys()))} of {len(all_preds)}")
    #print(f"# of absent predicate ents {len(all_preds)-len(list(ents.keys()))} of {len(all_preds)}")

def extract_absnt_preds(mapper:dict, preds :list[str]):
    absnt_preds = []
    for i in preds:
        if i not in mapper.keys():
            absnt_preds.append(i)
    return absnt_preds

def extract_absnt_pred_ents(preds= None,ent_mapper=None, path_to_nt=None,temp_path=None, process_lines= 8000):
    temp_path += "/pred_ents"
    subprocess.run(['mkdir','-p',temp_path])
    pred_str_dict_subj = create_empty_pred_dict(preds)
    pred_str_dict_obj = create_empty_pred_dict(preds)
    
    with open(path_to_nt, 'r') as f:
        for n_line, line in enumerate(f):
            if n_line % process_lines == 0:
                write_to_ent_files(preds,pred_str_dict_subj,temp_path, is_sub=True)
                write_to_ent_files(preds,pred_str_dict_obj,temp_path, is_sub=False)
                pred_str_dict_subj = create_empty_pred_dict(preds)
                pred_str_dict_obj = create_empty_pred_dict(preds)
                print(f"Processing line # {n_line}", end='\r')
            
            line_splits = line.split(' ')
            if len(line_splits) != 4:
                continue
            nt_pred = line_splits[1][1:-1]
            if nt_pred in preds:
                if line_splits[2].startswith('<'):
                    pred_str_dict_subj[nt_pred].append(line_splits[2]+'\n')
                if line_splits[0].startswith('<'):
                    pred_str_dict_subj[nt_pred].append(line_splits[0]+'\n')
                
    write_to_ent_files(preds,pred_str_dict_subj,temp_path, is_sub=True)
    write_to_ent_files(preds,pred_str_dict_obj,temp_path, is_sub=False)

def create_empty_pred_dict(preds: list[str]):
    dct = {}
    for p in preds:
        dct[p] = []
    return dct

def write_to_ent_files(preds, dct,temp_path, is_sub=True):
    if is_sub:
        prefix = 'sub'
    else:
        prefix = 'obj'
    for p in preds:
        path = Path(f"{temp_path}/{prefix}_{urllib.parse.quote(p, safe='')}.txt")
        if os.path.exists(path):
            with open(path, 'a') as f:
                f.writelines(dct[p])
        else:
            with open(path, 'w') as f:
                f.writelines(dct[p])
def post_process_pred_ents(preds= None,ent_mapper=None, path_to_nt=None,temp_path=None, output_path = None):
    temp_path += "/pred_ents"
    if not os.path.exists(temp_path):
        print(f'The path {temp_path} does not exist with temporary results!')
        exit()
    files = os.listdir(temp_path)
    obj_file_map, sub_file_map = {},{}
    pred_names = set()
    for f in files:
        pred_name = urllib.parse.unquote( f[4:-4])
        pred_names.add(pred_name)
        if f.startswith('obj'):
            obj_file_map[pred_name] = f
        if f.startswith('sub'):
            sub_file_map[pred_name] = f
    for num, p in enumerate(preds):
        print(f"Processing {p} [{num+1}/{len(preds)}]")
        #res = subprocess.run(['sort', '-us','-o',f"{temp_path}/sorted_{sub_file_map[p]}", f"{temp_path}/{sub_file_map[p]}"], capture_output=True, text=True)
        subprocess.run(['uniq','-u', f"{temp_path}/{sub_file_map[p]}",f"{temp_path}/sorted_{sub_file_map[p]}"], capture_output=False, text=False)
        res = subprocess.run(['wc','-l',f"{temp_path}/sorted_{sub_file_map[p]}"], capture_output=True, text=True)
        sub_val = int(res.stdout.split(' ')[0])
        
        subprocess.run(['uniq','-u', f"{temp_path}/{obj_file_map[p]}",f"{temp_path}/sorted_{obj_file_map[p]}"], capture_output=False, text=False)
        res = subprocess.run(['wc','-l',f"{temp_path}/sorted_{obj_file_map[p]}"], capture_output=True, text=True)
        obj_val = int(res.stdout.split(' ')[0])
        print(f"entity mapper: {p},{sub_val},{obj_val}")
        ent_mapper[p] = (sub_val, obj_val)
    json.dump(ent_mapper, open(output_path,'w'))

def extract_literal_types(path_to_nt, output_path, temp_path=None):
    t_file = open(f"{temp_path}/lits_w_type.txt",'w')
    lan_file = open(f"{temp_path}/lits_w_lan_type.txt",'w')
    with open(path_to_nt, 'r') as f:
        for line in f:
            line_splits = line.split(' ')
            if len(line_splits) != 4:
                continue
            obj = line_splits[2]
            if "^^" in obj:
                t_file.write(obj+'\n')
            if "@" in obj:
                lan_file.write(obj+'\n')
    t_file.close()
    lan_file.close()
    res_lit_file = open(f"{temp_path}/lit_pred_count.txt",'w')
    res_lit_file.close()

def run():
    parser = argparse.ArgumentParser(
                    prog='extract_nt',
                    epilog='Used to extract statistics from n triple file')
    parser.add_argument('--nt', '--nt')
    parser.add_argument('--task', '--task')
    parser.add_argument('--all_pred_file', '--all_pred_file')
    parser.add_argument('--lit_map_p', '--lit_map_p')
    parser.add_argument('--ent_map_p', '--ent_map_p')
    parser.add_argument('--output', '--output')
    parser.add_argument('--temp', '--temp')
    parser.add_argument('--process_lines', '--process_lines')
    #parser.add_argument('--preds', '--preds')
    args = parser.parse_args()
    
    path_to_nt = args.nt
    task = args.task
    pred_file = args.all_pred_file
    lit_map_path = args.lit_map_p
    ent_map_path = args.ent_map_p
    output = args.output
    temp_path = args.temp
    try:
        process_lines= int(args.process_lines)
    except TypeError:
        process_lines= args.process_lines
        
    if task == "pred_lits":
        absnt_preds,pred_literals = get_absnt_pred(feat_type='lits', pred_file=pred_file,lit_map_path=lit_map_path)
        
        extract_predicate_stats(absnt_preds, pred_literals, path_to_nt, output_path=output, temp_path=temp_path)
        print("Finished literal extraction for missing features")
    elif task == "post_process_lits":
        absnt_preds,pred_literals = get_absnt_pred(feat_type='lits', pred_file=pred_file,lit_map_path=lit_map_path)
        extract_unfilt_pred_lits(absnt_preds, literals_mapper=pred_literals,output_path=output, temp_path=temp_path)
    elif task == 'pred_ents':
        absnt_preds,pred_ents = get_absnt_pred(feat_type='ents', pred_file=pred_file,ent_map_path=ent_map_path)
        extract_absnt_pred_ents(preds= absnt_preds,ent_mapper=pred_ents, path_to_nt=path_to_nt, temp_path=temp_path ,process_lines=process_lines)
    elif task == 'post_process_pred_ents':
        absnt_preds,pred_ents = get_absnt_pred(feat_type='ents', pred_file=pred_file,ent_map_path=ent_map_path)
        post_process_pred_ents(preds= absnt_preds,ent_mapper=pred_ents, path_to_nt=path_to_nt,temp_path=temp_path, output_path=output)
    elif task == 'literals_feats':
        #not done
        extract_literal_types(path_to_nt, output_path=output, temp_path=temp_path)
        

if __name__ == "__main__":
    run()