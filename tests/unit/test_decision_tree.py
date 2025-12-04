import numpy as np
from rice_ml.supervised_learning.decision_tree import DecisionTreeClassifier

def test_simple_tree():
    X = np.array([[0,0],[0,1],[1,0],[1,1]])
    y = np.array([0,0,1,1])
    tree = DecisionTreeClassifier(max_depth=3, random_state=0)
    tree.fit(X, y)
    assert np.array_equal(tree.predict(X), y)

def test_proba_shape():
    X = np.array([[0],[1],[2]])
    y = np.array([0,1,1])
    tree = DecisionTreeClassifier()
    tree.fit(X, y)
    proba = tree.predict_proba(X)
    assert proba.shape == (3, 2)