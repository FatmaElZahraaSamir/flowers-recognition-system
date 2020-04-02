import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Importing the dataset

#data = pd.read_csv(
#    filepath_or_buffer='https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', 
#    header=None, sep=',')

data = pd.read_csv('Iris.csv')
#print(data.head())

## shape
#print(data.shape)
##Information on the data
#print(data.info())
## descriptions
#print(data.describe())
## class distribution
#print(data.groupby('Species').size())

# Seperating the Species column
X = data.iloc[:,0:4]
y = data.iloc[:,4]

# Feature Scaling (standardize the data)
X = (X - X.mean()) / X.std(ddof=0) #std()

# compute the covariance matrix
X = np.matrix(X)                # 150* 4
cov = (X.T * X) / X.shape[0]    # 4*150 * 150*4 = 4*4   / 150 (no. of rows)

U, S, V = np.linalg.svd(cov) # singular value decomposition S: values, U:Vectors

#print(np.sum(S))

# plotting the variance explained by each PC 
explained_variance=(S / np.sum(S))*100
plt.figure(figsize=(8,4))
plt.bar(range(4), explained_variance, alpha=0.6)
plt.ylabel('Percentage of explained variance')
plt.xlabel('Dimensions')


# calculating our new axis
pc1 = X.dot(U[:,0])
pc2 = X.dot(U[:,1])


species = data["Species"].tolist()
# plotting in 2D
def plot_scatter(pc1, pc2):
    fig, ax = plt.subplots(figsize=(8, 5))
    
    species_unique = list(set(species))
    species_colors = ["r","b","g"]
    
    for i, spec in enumerate(species):
        plt.scatter([pc1[i]], [pc2[i]], label = spec, s = 20, c=species_colors[species_unique.index(spec)])
    
    from collections import OrderedDict
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), prop={'size': 10}, loc=4)
    
    ax.set_xlabel('Principal Component 1', fontsize = 10)
    ax.set_ylabel('Principal Component 2', fontsize = 10)
    ax.axhline(y=0, color="grey", linestyle="--")
    ax.axvline(x=0, color="grey", linestyle="--")
    
    plt.axis([-4, 4, -3, 3])
    plt.show()
    
plot_scatter(pc1, pc2)



