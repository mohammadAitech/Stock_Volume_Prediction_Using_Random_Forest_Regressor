
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

data = pd.read_csv("TSLA.csv")

if 'Date' in data.columns:
    data['Date'] = pd.to_datetime(data['Date'])
    data['Date'] = data['Date'].map(pd.Timestamp.toordinal)

data = data.select_dtypes(include=[float, int])


x = data.drop("Volume", axis=1)
y = data['Volume']


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


model = RandomForestRegressor(max_depth=5, min_samples_leaf=1)
model.fit(x_train, y_train)


y_pred = model.predict(x_test)
mse = mean_squared_error(y_test, y_pred)
print(f"mse {mse}")