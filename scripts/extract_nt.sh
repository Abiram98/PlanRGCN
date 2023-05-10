NT_FILE=/work/data/sample.nt
NT_FILE=/work/data/wikidata-prefiltered.nt

all_pred_file=/work/data/extracted_statistics/predicates_only.json
lit_map_path=/work/data/extracted_statistics/updated_pred_unique_lits.json
output=/work/data/extracted_statistics/updated_nt_pred_unique_lits.json

(cd ../ && python3 -m feature_extraction.extract_nt --nt $NT_FILE --task pred_lits --all_pred_file $all_pred_file --lit_map_p $lit_map_path \
    --output $output)