# Import libraries
from LinearRegression import GDLinearRegressionModel
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Import data
train = pd.read_csv(r"train.csv")
y = train["Happiness Score"]
train.drop(["Happiness Score", "Country"], axis = 1, inplace = True)

# preprocessing
from preprocessing import prepare
pre = prepare()
X = pre.transform(train)

# fit model
model = GDLinearRegressionModel(learning_rate = 0.001, iterations = 3000)
model.fit(X.T, y)

print("Slope is", model.slope)
print("Intercept is", model.intercept)

# plot error
from sklearn.metrics import mean_absolute_error
print("mae with predicted wights", mean_absolute_error(y, model.predict(X)))
plt.plot(model.history)
plt.show()

# Final predictions
test = pd.read_csv(r"test.csv")
test_countries = test["Country"]
test.drop("Country", axis = 1, inplace = True)
pre_test = pre.transform(test)
preds = model.predict(pre_test)
submission = pd.DataFrame({"Country": test_countries, "Happiness Score": preds})
submission.to_csv(r"submission.csv", index = False)
