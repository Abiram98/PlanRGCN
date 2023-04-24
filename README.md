# BGPClassifier

## Prerequisites
Docker should be installed
Git lfs

## Data
Data can be gotten from:
```
git clone ...
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