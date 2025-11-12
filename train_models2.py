import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib

data = pd.read_csv("synthetic_fashion_data.csv")


X = data.drop(["fabric_meters", "chemical_liters"], axis=1)
y_fabric = data["fabric_meters"]
y_chemicals = data["chemical_liters"]

X_train, X_test, y_f_train, y_f_test = train_test_split(X, y_fabric, test_size=0.2, random_state=42)
_, _, y_c_train, y_c_test = train_test_split(X, y_chemicals, test_size=0.2, random_state=42)

fabric_model = RandomForestRegressor(n_estimators=20, random_state=42)
chemical_model = RandomForestRegressor(n_estimators=20, random_state=42)

fabric_model.fit(X_train, y_f_train)
chemical_model.fit(X_train, y_c_train)

joblib.dump((fabric_model, chemical_model, X.columns.tolist()), "sustainable_models.pkl")
print("Models trained and saved as sustainable_models.pkl")
