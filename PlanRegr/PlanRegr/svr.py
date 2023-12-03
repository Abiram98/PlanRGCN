from sklearn.svm import SVR
from trainer.embed import load_embedding_data
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from PlanRegr.metrics import relative_error_mean, relative_error

sample_name = "wikidata_0_1_10_v2_path_hybrid"
config = {
    "reg_train_path" : f"/data/{sample_name}/reg_train_sampled.pickle",
"reg_val_path" : f"/data/{sample_name}/reg_val_sampled.pickle",
"reg_test_path" : f"/data/{sample_name}/reg_test_sampled.pickle",
}

def main():
    ids, embeds, lats = load_embedding_data(config['reg_train_path'])
    X = np.vstack(embeds)
    y = np.vstack(lats)
    y = y.reshape((-1,))
    svr = SVR(kernel='rbf').fit(X,y)
    train_pred = svr.predict(X)
    train_pred2 = train_pred.copy()
    train_pred2[train_pred2<0] = 0
    
    print(train_pred)
    r = relative_error_mean(train_pred,y)
    print(f"Relative error is {r}, mse: {mean_squared_error(train_pred, y)} mae: {mean_absolute_error(train_pred, y)}")
    r = relative_error_mean(train_pred2,y)
    print(f"Non-negative: Relative error is {r}, mse: {mean_squared_error(train_pred2, y)} mae: {mean_absolute_error(train_pred2, y)}")
    pass
if __name__ == "__main__":
    main()