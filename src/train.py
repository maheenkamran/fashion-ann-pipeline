import os
import yaml
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras.optimizers import Adam

with open("params.yaml","r") as f:
    params=yaml.safe_load(f)["train"]

d = np.load("data/processed/fashion_processed.npz")

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(params["dense_units"], activation="relu"),
    keras.layers.Dropout(params["dropout_rate"]),
    keras.layers.Dense(10, activation="softmax"),
])
model.compile(
    optimizer=Adam(learning_rate=params["learning_rate"]),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

history = model.fit(
    d["x_train"], d["y_train"],
    validation_data=(d["x_val"], d["y_val"]),
    epochs=params["epochs"], batch_size=params["batch_size"],
) 

os.makedirs("models", exist_ok=True)
#create folder if not exists

model.save("models/model.h5")
pd.DataFrame(history.history).to_csv("models/history.csv", index=False)

#history.history gives loss accuracy val loss val accuracy for all epochs