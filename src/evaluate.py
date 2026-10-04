import json
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

d = np.load("data/processed/fashion_processed.npz")
model = keras.models.load_model("models/model.h5")

loss, acc = model.evaluate(d["x_test"], d["y_test"])

probabilities=model.predict(d["x_test"])
# make prediction on test data
predictions=np.argmax(probabilities,axis=1)
#axis=1 so each row max, not entire array max
#np.argmax to get index not max value

cm = confusion_matrix(d["y_test"], predictions)
#(actual value,predicted value)
ConfusionMatrixDisplay(cm).plot()
plt.savefig("models/confusion_matrix.png")
plt.close()

metrics={
    "test_loss":float(loss),
    "test_accuracy":float(acc)
} #dict
with open("metrics.json","w") as f:
    json.dump(metrics,f,indent=2)
    
    #convert python obj to json, and write to file
    
print(f"Test accuracy: {acc:.4f}")