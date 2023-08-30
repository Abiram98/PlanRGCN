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