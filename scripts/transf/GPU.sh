

transfer_wikidataV2_qpp2(){
BASE=/data/abiram/data_qpp

params=(--exclude 'baseline' --exclude "data_splitter.pickle" --exclude "geddbtran.sh" --exclude "plan01" --exclude "queryStat.json" --exclude "resampled" --exclude "knn25")
echo "${params[@]}"
rsync -aWP "${params[@]}" qpp2:"${BASE}/wikidataV2" /data/
}