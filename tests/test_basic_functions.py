
from rice_ml.basic_functions import add

def test_add_positive_integers():
    assert add(2,3) == 5

def test_add_negative_integers():
    assert add(-2,-3) == -5