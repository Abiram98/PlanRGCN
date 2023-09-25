# BGPClassifier

## Prerequisites
Docker should be installed
Git lfs
In the working directory, clone this repo and git@github.com:dkw-aau/leapfrog-rdf-benchmark.git
## Data

### Leapfrog dataset
First run the dataset_construction.py file on the leapfrog repo (git clone git@github.com:dkw-aau/leapfrog-rdf-benchmark.git) and mv the file to data/data_files folder.
```
docker run -it --rm -v "$(pwd)"/leapfrog-rdf-benchmark:/leapfrog-rdf-benchmark -v "$(pwd)"/BGPClassifier:/work -m 15g ubuntu:22.10
cd work && python3 dataset_construction.py 
exit

```
Then the following commands should be executed:
```
cd scripts
bash dataset_split.sh convert-leaf
bash dataset_split.sh data-split
```
This will create the training, validation and test splits in the folder, /work/data/splits.

### Training of TP Graph
```
python3 -m dgl_classifier.trainer --train_file /work/data/splits/train.json --val_file /work/data/splits/val.json --test_file /work/data/splits/test.json
```
Alternatively the Jupyter Notebooks also show these.

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

# Executing mavne project interactively
```
mvn exec:java -f "/PlanRGCN/qpe/pom.xml"
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