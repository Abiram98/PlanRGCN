#!bin/bash

(cd ../ && python3 -m preprocessing.utils_script stratified_split)
(cd ../ && python3 -m preprocessing.dataset_analysis)