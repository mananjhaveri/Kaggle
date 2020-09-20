# NOW MOZILLA CLUB MPSTME - Kaggle Competition: Winning Solution

Hey, guys!!!

Here's my code and approach for [this](https://www.kaggle.com/c/now-mozilla-club-mpstme) ML Competition.

## Documentation of the model used:

I have created my own model using the Gradient Descent algorithm for Linear Regression.

How the model works:
* Initialize slope and intercept as 0.
* for a certain number of iterations, keep on modifying the slope and the intercept based on the cost and by a factor equal to learning rate.

### Parameters

| Name | Default Value | Description |
| --- | --- | --- |
| learning_rate | 0.01 | the factor by which weights are backtracked |
| iterations | 2000 | the number of times the algorithm should backtrack to find weights |

### Attributes
| Name | Description |
| --- | --- |
| slope | returns a list of all weights |
| intercept | returns the numeric values of the intercept |
| history | a list of costs calculated at each iteration |

### Usage
```Python
# create an instance
model = GDLinearRegressionModel(learning_rate=lr, iterations=iter)

# fitting the model
model.fit(X, y)

# predicting values
preds = model.predict(test)

# print slope & intercept
print(model.slope)
print(model.intercept)

# plotting error
import matplotlib.pyplot as plt
plt.plot(model.history)
```


### NOTE:
If your data looks like this:

| A | B | C |
| --- | --- | --- |
| 1 | 4 | 7 |
| 2 | 5 | 8 |
| 3 | 6 | 9 |

then, the value of X that is used as model.fit(X, y) should look like this:

**[[1, 2, 3], [4, 5, 6], [7, 8, 9]]**

Basically, it should be the transpose of the above table.
