# BGPClassifier

## Prerequisites
Docker should be installed
Git lfs

## Data

### Leapfrog dataset
First run the dataset_construction.py file on the leapfrog repo (git clone git@github.com:dkw-aau/leapfrog-rdf-benchmark.git) and mv the file to data/data_files folder.

Then the following commands should be executed:
```
cd scripts
bash dataset_split.sh convert-leaf
bash dataset_split.sh gt-assign
bash dataset_split.sh strat-split
```
This will create the training, validation and test splits in the folder, /work/data/splits.

### Training of TP Graph
```
python3 -m dgl_classifier.trainer --train_file /work/data/splits/train.json --val_file /work/data/splits/val.json --test_file /work/data/splits/test.json
```

## To run
In interactive mode:
```
docker build -t qpp_image:1 .
docker run --rm -it --name BGPClassifier -v "$(pwd)"/:/work qpp_image:1 bin/bash
```
### Configuration File
First step is to create a configuration file with the appropriate features.
Rember to set the proper path in /work/feature_extraction/constants.py

### Predicate Features

to collect predicate features:
```
python3 -m feature_check.predicate_check
```

### Python code to run

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