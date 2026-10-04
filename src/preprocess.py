import os
import yaml
import numpy as np
from sklearn.model_selection import train_test_split

with open("params.yaml","r") as f:
    params=yaml.safe_load(f)["preprocess"]

d=np.load("data/raw/fashion_raw.npz")
    
x_train = d["x_train"].astype("float32") / 127.5 - 1.0
x_test = d["x_test"].astype("float32") / 127.5 - 1.0
#normalize by dividing by 255, so range is in 0-255

x_train,x_val,y_train,y_val=train_test_split(
    x_train, d["y_train"],
    test_size=params["test_size"],
    random_state=params["seed"],
    stratify=d["y_train"]
)
os.makedirs("data/processed",exist_ok=True)

np.savez_compressed("data/processed/fashion_processed.npz",
                    x_train=x_train, y_train=y_train,
                    x_val=x_val, y_val=y_val,
                    x_test=x_test, y_test=d["y_test"])

print("Processed:", x_train.shape, x_val.shape, x_test.shape)