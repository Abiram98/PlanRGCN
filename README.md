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