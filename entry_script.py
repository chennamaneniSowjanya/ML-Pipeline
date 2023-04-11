import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from pipeline import Pipeline


# load data
data = pd.read_csv("data.csv")

# split data
X_train, X_test, y_train, y_test = train_test_split(data.drop(columns=["cnt"]), data["cnt"], test_size=0.25, random_state=42)

# fit pipeline
pipe = Pipeline()
pipe.fit(X_train, y_train)

# evaluate pipeline
y_pred = pipe.predict(X_test)

#state the model performance
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean absolute error of the model is:",mae) 

