class preprocess:

    def __init__(self):
        return 

    def fit_transform(self, df, drop_features=False):
        self.df = df.copy() 

        # splitting into X and y
        X, y = self.df.drop(["diagnosis"], axis=1), self.df["diagnosis"]

        # drop features 
        # dropping  "mean_radius", "mean_perimeter" bcause highly correlated with "mean_area"
        if drop_features == True:
            X.drop(["mean_radius", "mean_perimeter"], axis=1, inplace=True)

        # Scaling 
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        # splitting the data into train and test
        from sklearn.model_selection import StratifiedShuffleSplit 
        split = StratifiedShuffleSplit(n_splits=1, test_size=0.2)

        for train_index, test_index in split.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y.loc[train_index], y.loc[test_index] 
        
        return X_train, y_train, X_test, y_test 