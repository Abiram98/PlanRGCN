import trainer.data_util as u

train_path='/qpp/dataset/DBpedia_2016_12k_simple_opt_filt/train_sampled.tsv'
val_path='/qpp/dataset/DBpedia_2016_12k_simple_opt_filt/val_sampled.tsv'
test_path='/qpp/dataset/DBpedia_2016_12k_simple_opt_filt/test_sampled.tsv'

d = u.DatasetPrep(train_path=train_path,test_path=test_path,val_path=val_path)
print(d.get_testloader())