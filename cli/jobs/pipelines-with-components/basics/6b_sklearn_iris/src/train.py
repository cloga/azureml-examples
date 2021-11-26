import argparse
from pathlib import Path
from uuid import uuid4
from datetime import datetime
import os
import pandas as pd
# from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pickle
from sklearn import preprocessing

parser = argparse.ArgumentParser("train")
parser.add_argument("--data", type=str, help="Path to data")
parser.add_argument("--n_estimators", type=int, default=3, help="The number of trees in the forest.")
parser.add_argument("--max_depth", type=int, default=3, help="The maximum depth of the tree.")
args = parser.parse_args()

print("hello training world...")


print("mounted_path files: ")

arr = os.listdir(args.data)
print(arr)
train_data = pd.read_csv((Path(args.data) / 'iris.csv'))
print(train_data.columns)

# Split the data into input(X) and output(y)
le = preprocessing.LabelEncoder()
y_label = le.fit_transform(train_data["Species"])
y = y_label


# X = train_data.drop(['cost'], axis=1)
X = train_data[
    [
        "SepalLengthCm",
        "SepalWidthCm",
        "PetalLengthCm",
        "PetalWidthCm"
    ]
]

# Split the data into train and test sets
trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.3, random_state=42)
print(trainX.shape)
print(trainX.columns)

regr = RandomForestRegressor(n_estimators=args.n_estimators, max_depth=args.max_depth, random_state=0)
# Train a random forest model with the train set

model = regr.fit(trainX, trainy)
print(model.score(trainX, trainy))

test_pred = model.predict(testX) 
mean_squared_error(testy, test_pred)
# run.log(mean_squared_error)

# Output the model
pickle.dump(model, open((Path(args.data) / "model.sav"), "wb"))

