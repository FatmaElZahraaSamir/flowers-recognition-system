import pandas as pd
from sklearn.model_selection import train_test_split

# Load dataset
data = pd.read_csv('Iris.csv')

# Change string value to numeric
data["Species"] = data["Species"].map({
    "Iris-setosa": 0,
    "Iris-versicolor": 1,
    "Iris-virginica": 2
}).astype(int)

# Change dataframe to array
data_array = data.values

# Split x and y (feature and target)
X_train, X_test, y_train, y_test = train_test_split(data_array[:,:4],
                                                    data_array[:,4],
                                                    test_size=0.2)


from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(hidden_layer_sizes=(10),
                    solver='sgd',
                    learning_rate_init=0.01,
                    max_iter=500,
                    random_state=113)

# Train the model
mlp.fit(X_train, y_train)

# Test the model
print("Accuracy ",mlp.score(X_test,y_test))

sl = 5.7
sw = 2.8
pl = 4.5
pw = 1.3
data = [[sl,sw,pl,pw]]
print( mlp.predict(data))


import pickle

model = open("model.pickle","wb")
pickle.dump(mlp, model)
model.close()



















