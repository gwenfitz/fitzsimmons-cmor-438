"""
Unit tests for community_detection.py.

These tests validate correctness on small graphs where community
structure is known in advance.
"""

import numpy as np
from rice_ml.unsupervised_learning.community_detection import LabelPropagation


def test_two_communities():
    """
    Two disconnected components should form two communities.
    """
    # adjacency matrix for two disconnected triangles
    A = np.array([
        [0, 1, 1, 0, 0, 0],
        [1, 0, 1, 0, 0, 0],
        [1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1],
        [0, 0, 0, 1, 0, 1],
        [0, 0, 0, 1, 1, 0],
    ])

    model = LabelPropagation(max_iter=50)
    labels = model.fit_predict(A)

    # first three nodes should share a label
    assert len(set(labels[:3])) == 1

    # last three nodes should share a label
    assert len(set(labels[3:])) == 1

    # the two groups should be different
    assert labels[0] != labels[3]


def test_single_component():
    """
    Fully connected graph should yield a single community.
    """
    n = 5
    A = np.ones((n, n)) - np.eye(n)

    model = LabelPropagation(max_iter=50)
    labels = model.fit_predict(A)

    assert len(set(labels)) == 1


def test_isolated_nodes():
    """
    Isolated nodes should retain unique labels.
    """
    A = np.zeros((3, 3))

    model = LabelPropagation(max_iter=10)
    labels = model.fit_predict(A)

    assert len(set(labels)) == 3
