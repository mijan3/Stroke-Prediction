import numpy as np
import pandas as pd
from joblib import dump

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

def manhattan_distance(x1, x2):
    return np.sum(np.abs(x1 - x2))

class KNNFromScratch:
    def __init__(self, k=5, distance='euclidean'):
        self.k = k
        self.distance = distance

    def fit(self, X, y):
        self.X_train = np.array(X)
        self.y_train = np.array(y)

    def _distance(self, x):
        if self.distance == 'euclidean':
            return np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
        elif self.distance == 'manhattan':
            return np.sum(np.abs(self.X_train - x), axis=1)
        else:
            raise ValueError("Unsupported distance")

    def predict(self, X):
        X = np.array(X)
        predictions = []

        for x in X:
            distances = self._distance(x)
            k_indices = np.argsort(distances)[:self.k]
            k_labels = self.y_train[k_indices]

            # Majority vote
            prediction = np.bincount(k_labels).argmax()
            predictions.append(prediction)

        return np.array(predictions)

    def predict_proba(self, X):
        X = np.array(X)
        probabilities = []

        for x in X:
            distances = self._distance(x)
            k_indices = np.argsort(distances)[:self.k]
            k_labels = self.y_train[k_indices]

            prob = np.mean(k_labels)
            probabilities.append([1 - prob, prob])

        return np.array(probabilities)

def mean_impute(X):
    X = np.array(X, dtype=float)
    col_means = np.nanmean(X, axis=0)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(col_means, inds[1])
    return X

def standardize(X):
    mean = X.mean(axis=0)
    std = X.std(axis=0) + 1e-8
    return (X - mean) / std

def train_test_split(X, y, test_size=0.2, seed=42):
    np.random.seed(seed)
    idx = np.random.permutation(len(X))
    test_len = int(len(X) * test_size)

    test_idx = idx[:test_len]
    train_idx = idx[test_len:]

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

def roc_auc_score(y_true, y_prob):
    sorted_idx = np.argsort(y_prob)
    y_true = y_true[sorted_idx]
    y_prob = y_prob[sorted_idx]

    P = np.sum(y_true)
    N = len(y_true) - P

    tpr = []
    fpr = []

    tp = fp = 0

    for i in range(len(y_true)):
        if y_true[i] == 1:
            tp += 1
        else:
            fp += 1
        tpr.append(tp / P)
        fpr.append(fp / N)

    return np.trapezoid(tpr, fpr)


# Load data
df = pd.read_csv('./stroke-data.csv')
df = df.drop(columns=['id'])

y = df['stroke'].values
X = df.drop(columns=['stroke']).select_dtypes(include=[np.number]).values

# Preprocessing
X = mean_impute(X)
X = standardize(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Train KNN
knn = KNNFromScratch(k=7, distance='euclidean')
knn.fit(X_train, y_train)

# Predict
y_pred = knn.predict(X_test)
y_prob = knn.predict_proba(X_test)[:, 1]

# Evaluate
auc = roc_auc_score(y_test, y_prob)
print("KNN ROC-AUC:", auc)




# After training
dump(knn, 'knn_from_scratch_model.joblib')

print("✅ Scratch KNN model dumped successfully")
