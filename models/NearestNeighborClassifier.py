import numpy as np
import pandas as pd
from scipy.spatial import distance
from BaseMLModel import BaseMLModel

"""
Nearest Neighbor Prediction using Euclidean distance.
"""

class NearestNeighborClassifier(BaseMLModel):

    def _nearest_neighbor_predict(self, train_features, train_target, new_features):
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

    def predict(self, new_features):
        # Store predictions in a list
        values = []
        # Iterate through all new features
        for i in range(new_features.shape[0]):
            features_vector = new_features.loc[i].values
            # Call the nearest neighbor prediction function
            prediciton = self._nearest_neighbor_predict(
                self.train_features,
                self.train_target,
                features_vector
               )
            values.append(prediciton)
        # transform the list of predictions into a pandas series and return it
        predictions = pd.Series(values)
        return predictions
    
columns = [
    'bedrooms',
    'total area',
    'kitchen',
    'living area',
    'floor',
    'total floors',
]

df_train = pd.DataFrame(
    [
        [1.0, 38.5, 6.9, 18.9, 3.0, 5.0],
        [1.0, 38.0, 8.5, 19.2, 9.0, 17.0],
        [1.0, 34.7, 10.3, 19.8, 1.0, 9.0],
        [1.0, 45.9, 11.1, 17.5, 11.0, 23.0],
        [1.0, 42.4, 10.0, 19.9, 6.0, 14.0],
        [1.0, 46.0, 10.2, 20.5, 3.0, 12.0],
        [2.0, 77.7, 13.2, 39.3, 3.0, 17.0],
        [2.0, 69.8, 11.1, 31.4, 12.0, 23.0],
        [2.0, 78.2, 19.4, 33.2, 4.0, 9.0],
        [2.0, 55.5, 7.8, 29.6, 1.0, 25.0],
        [2.0, 74.3, 16.0, 34.2, 14.0, 17.0],
        [2.0, 78.3, 12.3, 42.6, 23.0, 23.0],
        [2.0, 74.0, 18.1, 49.0, 8.0, 9.0],
        [2.0, 91.4, 20.1, 60.4, 2.0, 10.0],
        [3.0, 85.0, 17.8, 56.1, 14.0, 14.0],
        [3.0, 79.8, 9.8, 44.8, 9.0, 10.0],
        [3.0, 72.0, 10.2, 37.3, 7.0, 9.0],
        [3.0, 95.3, 11.0, 51.5, 15.0, 23.0],
        [3.0, 69.3, 8.5, 39.3, 4.0, 9.0],
        [3.0, 89.8, 11.2, 58.2, 24.0, 25.0],
    ],
    columns=columns,
)


train_features = df_train.drop('bedrooms', axis=1)
train_target = df_train['bedrooms']

df_test = pd.DataFrame(
    [
        [1, 36.5, 5.9, 17.9, 2, 7],
        [2, 71.7, 12.2, 34.3, 5, 21],
        [3, 88.0, 18.1, 58.2, 17, 17],
    ],
    columns=columns,
)

test_features = df_test.drop('bedrooms', axis=1)

model = NearestNeighborClassifier()
model.fit(train_features, train_target)
new_predictions = model.predict(test_features)
print(new_predictions)