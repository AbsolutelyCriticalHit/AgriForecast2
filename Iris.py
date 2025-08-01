import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
farmdata = pd.read_csv("yield_df.csv")
print(farmdata)
X = farmdata[["average_rain_fall_mm_per_year","avg_temp"]]
y = farmdata["hg/ha_yield"]
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
model = LinearRegression()
model.fit(X_train,y_train)
print(model.predict([[2702,27.5]]))
print(y_test)
print(model.score(X_test,y_test))