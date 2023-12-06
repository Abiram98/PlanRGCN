# PlanRGCN

## Prerequisites
Docker should be installed

## Setup
First start a docker container in interactive mode


## To run
In interactive mode:
```
docker build -t bgp_cls:1 .
export path_to_rdf/wikidata-prefiltered.nt
docker run -m 20g -it --name bgp5 -v "$(pwd)"/:/work -v $path_to_rdf:/wikidata-prefiltered.nt -p 85:80 bgp_cls:1 bin/bash

docker run -m 20g -it --name bgp5 -v "$(pwd)"/:/work -v /srv/data/abiram/blf_data/import/wikidata-prefiltered.nt:/wikidata-prefiltered.nt -p 85:80 bgp_cls:1 bin/bash
```
### Configuration File - Not Necessary anymore
First step is to create a configuration file with the appropriate features.
Rember to set the proper path in /work/feature_extraction/constants.py

### Predicate Features - Not Necessary anymore

to collect predicate features:
```
python3 -m feature_check.predicate_check
```

### Python code to run - Not Necessary anymore

``` 
python3 -m classifier.trainer
```

Running the current code
Removed 7263 of 464556 training data
Removed 149 of 57588 test data uased as validation

# Feature Extraction
## Prerequsite
1. have the rdf store running
## Predicate Features
```
python3 -m 
```
## Extracting query plans
Example for dbpedia 2016 queries with limit removed:
```
bash '/PlanRGCN/scripts/qp/qp_extract_lsq.sh' /qpp/dataset/DBpedia2016limitless
```

# Training model:
f1 on validation: 0.19/ 0.10
```
python3 -c """
from trainer.train import Trainer
from graph_construction.featurizer import FeaturizerPredCo
t = Trainer(featurizer_class=FeaturizerPredCo)
t.train(epochs=100,verbosity=2,
result_path='/PlanRGCN/results/results.json',
path_to_save='/PlanRGCN/plan_model')
t.predict(path_to_save='/PlanRGCN/results')
"""
```
f1 on validation:  0.31
```
python3 -c """
from trainer.train import Trainer
from graph_construction.featurizer import FeaturizerPredStats
t = Trainer(featurizer_class=FeaturizerPredStats)
t.train(epochs=100,verbosity=2,
result_path='/PlanRGCN/results/results.json',
path_to_save='/PlanRGCN/plan_model')
t.predict(path_to_save='/PlanRGCN/results')
"""
```

## Experiment with self loop relation
val f1=0.18
```
python3 -c """
from trainer.model import ClassifierWSelfTriple as CLS
from trainer.train import Trainer
from graph_construction.featurizer import FeaturizerPredStats
from graph_construction.query_graph import QueryPlanCommonBi
t = Trainer(featurizer_class=FeaturizerPredStats,query_plan=QueryPlanCommonBi)
t.train(epochs=100,verbosity=2,
result_path='/PlanRGCN/results/results.json',
path_to_save='/PlanRGCN/plan_model')
t.predict(path_to_save='/PlanRGCN/results')
"""
```

# Regression with PlanRGCN
val f1= 0.09566
```
python3 -c """
from trainer.model import RegressorWSelfTriple as CLS
from trainer.train import Trainer
from graph_construction.featurizer import FeaturizerPredStats
from graph_construction.query_graph import QueryPlanCommonBi

t = Trainer(
    featurizer_class=FeaturizerPredStats,
    query_plan=QueryPlanCommonBi,
    cls_func=lambda x: x,
    model=CLS,
)
t.train(
    epochs=100,
    verbosity=2,
    result_path='/PlanRGCN/results/results.json',
    path_to_save='/PlanRGCN/plan_model',
    loss_type='mse',
)
t.predict(path_to_save='/PlanRGCN/results_reg')
"""
```

## Experiment with self loop relation and entity features
val f1=
```
python3 -c """
from trainer.model import ClassifierWSelfTriple as CLS
from trainer.train import Trainer
from graph_construction.featurizer import FeaturizerPredCoEnt
from graph_construction.query_graph import QueryPlanCommonBi
t = Trainer(featurizer_class=FeaturizerPredCoEnt,query_plan=QueryPlanCommonBi)
t.train(epochs=100,verbosity=2,
result_path='/PlanRGCN/results3/results.json',
path_to_save='/PlanRGCN/plan_model')
t.predict(path_to_save='/PlanRGCN/results3')
"""
```

# Training model with simple queries:
f1 on validation: 0.19/ 0.10
```
python3 -c """
from trainer.train import Trainer
from graph_construction.featurizer import FeaturizerPredCo
from graph_construction.featurizer import FeaturizerPredCo
from graph_construction.query_graph import QueryPlanCommonBi, snap_lat2onehot,snap_lat2onehotv2

t = Trainer(featurizer_class=FeaturizerPredCo,
train_path="/qpp/dataset/DBpedia_2016_12k_simple_opt_filt/train_sampled.tsv",
val_path="/qpp/dataset/DBpedia_2016_12k_simple_opt_filt/val_sampled.tsv",
test_path="/qpp/dataset/DBpedia_2016_12k_simple_opt_filt/test_sampled.tsv",
batch_size=32,
query_plan_dir="/PlanRGCN/extracted_features/queryplans/",
pred_stat_path="/PlanRGCN/extracted_features/predicate/pred_stat/batches_response_stats",
time_col="mean_latency",
hidden_dim=48,
n_classes=3,
featurizer_class=FeaturizerPredCo,
query_plan=QueryPlanCommonBi,
model=CLS,
cls_func=snap_lat2onehotv2
)
t.train(epochs=100,verbosity=2,
result_path='/PlanRGCN/results/results.json',
path_to_save='/PlanRGCN/plan_model')
t.predict(path_to_save='/PlanRGCN/results')
"""
```



f1 on validation:  0.31
```
python3 -c """
from trainer.train import Trainer
from graph_construction.featurizer import FeaturizerPredStats
t = Trainer(featurizer_class=FeaturizerPredStats)
t.train(epochs=100,verbosity=2,
result_path='/PlanRGCN/results/results.json',
path_to_save='/PlanRGCN/plan_model')
t.predict(path_to_save='/PlanRGCN/results')
"""
```
