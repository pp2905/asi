import pickle
from datetime import datetime

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder, LabelEncoder


def object_to_int(dataframe_series):
    if dataframe_series.dtype == "object":
        dataframe_series = LabelEncoder().fit_transform(dataframe_series)
    return dataframe_series


# Wczytanie danych
data = pd.read_csv("data/dataset.csv")

print(data)

data["TotalCharges"] = pd.to_numeric(data.TotalCharges, errors="coerce")
data.dropna(inplace=True)

data = data.drop(["customerID"], axis=1)
data.drop(labels=data[data["tenure"] == 0].index, axis=0, inplace=True)

data["SeniorCitizen"] = data["SeniorCitizen"].map({0: "No", 1: "Yes"})
data = data.apply(lambda x: object_to_int(x))

print(data)

X = data.drop(columns=["Churn"])
y = data["Churn"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=40, stratify=y
)


knn_model = KNeighborsClassifier(n_neighbors=11)
knn_model.fit(X_train, y_train)
predicted_y = knn_model.predict(X_test)
accuracy_knn = knn_model.score(X_test, y_test)
print("KNN accuracy:", accuracy_knn)

now = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
#
# eval_df = pd.DataFrame()
# eval_df = eval_df.append(
#     {
#         "time_stamp": now,
#         "version": "1.0",
#         "metric": "KNeighborsClassifier",
#         "score": accuracy_knn,
#     },
#     ignore_index=True,
# )
# eval_df.to_csv("evaluation/model_eval.csv", index=False)

pickle.dump(knn_model, open("model/model.sv", "wb"))
