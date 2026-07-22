import numpy as np
import pandas as pd
from scipy.spatial import distance

"""
Nearest neighbor prediction using Euclidean distance.
"""

def nearest_neighbor_predict(train_features, train_target, new_features):
    # Store distances in a list
    distances = []
    # Iterate through all dataframe rows
    for row in range(len(train_features)):
        actual_vector = train_features.loc[row].values
        # Compare distances between the new vector and the indexed one
        vector_distance = distance.euclidean(
            actual_vector, new_features
        )
        distances.append(vector_distance)

    # Get the index with the closest distance
    best_index = np.array(distances).argsort()[0]
    # Return the target of the closest index
    return train_target.iloc[best_index]