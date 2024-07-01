import numpy as np

def arithmetics_test(a):
    assert np.isclose(a, 32, atol=0.05 / 1000), f"Wrong a. Expected: {32} got: {a}"
    print("\033[92mAll tests passed!")