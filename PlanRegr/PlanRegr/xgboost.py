import xgboost as xgb
import numpy as np
from trainer.embed import load_embedding_data
from sklearn.metrics import mean_absolute_error, mean_squared_error
from PlanRegr.metrics import relative_error_mean, relative_error
sample_name = "wikidata_0_1_10_v2_path_hybrid"
config = {
    "reg_train_path" : f"/data/{sample_name}/reg_train_sampled.pickle",
"reg_val_path" : f"/data/{sample_name}/reg_val_sampled.pickle",
"reg_test_path" : f"/data/{sample_name}/reg_test_sampled.pickle",
}


    
def train(config):
    ids, embeds, lats = load_embedding_data(config['reg_train_path'])
    tree_size = [1000, 2000, 5000]
    models = [xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree = 1, learning_rate = 0.01, max_depth = 20, alpha = 10, n_estimators =x) for x in tree_size]
    X = np.vstack(embeds)
    y = np.vstack(lats)
    y = y.reshape((-1,))
    re_s = []
    for m in models:
        m.fit(X,y)
        train_preds = m.predict(X)
        r = relative_error_mean(train_preds,y)
        print(f"Relative error is {r}, mse: {mean_squared_error(train_preds, y)} mae: {mean_absolute_error(train_preds, y)}")
    
    
    #xg_reg = xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree = 1, learning_rate = 0.01, max_depth = 20, alpha = 10, n_estimators = 50)
    #xg_reg.fit(X,y)
    #train_pred = xg_reg.predict(X)
    pass
if __name__ == "__main__":
    train(config)