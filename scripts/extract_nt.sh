
temp_path=/work/data/temp
#NT_FILE=/work/data/sample.nt
NT_FILE=/wikidata-prefiltered.nt
all_pred_file=/work/data/extracted_statistics/predicates_only.json
lit_map_path=/work/data/extracted_statistics/updated_pred_unique_lits.json
output=/work/data/extracted_statistics/updated_nt_pred_unique_lits.json
ent_map_path=/work/data/extracted_statistics/updated_pred_unique_subj_obj.json
extracted_dir=/work/data/extracted_statistics
mkdir -p $temp_path

#for collecting predicate literal counts for missing predicates
: '
(cd ../ && python3 -m feature_extraction.extract_nt --nt $NT_FILE --task pred_lits --all_pred_file $all_pred_file --lit_map_p $lit_map_path \
    --output $output \
    --temp $temp_path)

(cd ../ && python3 -m feature_extraction.extract_nt --nt $NT_FILE --task post_process_lits --all_pred_file $all_pred_file --lit_map_p $lit_map_path \
    --output $output \
    --temp $temp_path)
'

#for collecting predicate entities counts for missing predicates
process_lines=8000
output=/work/data/extracted_statistics/updated_nt_pred_unique_subj_obj.json
: '
rm -rf $temp_path
mkdir -p $temp_path
process_lines=8000


(cd ../ && python3 -m feature_extraction.extract_nt \
    --nt $NT_FILE \
    --task pred_ents \
    --all_pred_file $all_pred_file \
    --lit_map_p $lit_map_path \
    --output $output \
    --temp $temp_path \
    --ent_map_p $ent_map_path \
    --process_lines $process_lines)
'
: '
(cd ../ && python3 -m feature_extraction.extract_nt \
    --nt $NT_FILE \
    --task post_process_pred_ents \
    --all_pred_file $all_pred_file \
    --lit_map_p $lit_map_path \
    --output $output \
    --temp $temp_path \
    --ent_map_p $ent_map_path \
    --process_lines $process_lines)
'
#For collecting entity features
: '
output=/work/data/extracted_statistics
mkdir -p /work/data/temp/ents
(cd ../ && python3 -m feature_extraction.extract_nt \
    --nt $NT_FILE \
    --task ent_feats \
    --all_pred_file $all_pred_file \
    --lit_map_p $lit_map_path \
    --output $output \
    --temp $temp_path \
    --ent_map_p $ent_map_path \
    --process_lines $process_lines \
    --query_log /work/data/data_files/processed_gt.json)
'
output=/work/data/extracted_statistics
mkdir -p /work/data/temp/ents
(cd ../ && python3 -m feature_extraction.extract_nt \
    --nt $NT_FILE \
    --task ent_feat_post \
    --all_pred_file $all_pred_file \
    --lit_map_p $lit_map_path \
    --output $output \
    --temp $temp_path \
    --ent_map_p $ent_map_path \
    --process_lines $process_lines \
    --query_log /work/data/data_files/processed_gt.json)

#python3 -m feature_extraction.extract_nt --task ent_in_split --split_dir /work/data/splits/splits_0.05 --temp_dir /work/data/temp/ents --output /work/data/extracted_statistics/ent_stat_5re.json