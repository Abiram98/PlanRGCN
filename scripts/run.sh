# !bin/bash
( cd ../ && python3 -m utils_script ql_pred_buckets)
( cd ../ && python3 -m classifier.batched.trainer > train_log_pred_feat.txt)