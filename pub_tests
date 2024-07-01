import numpy as np

def arithmetics_test(Ekin):
    try:
      assert np.isclose(Ekin, 50000, atol=0.05 / 1000), f"\033[91mFehler in Ekin. Erwartet {50000.0} erhalten {Ekin}"
    except AssertionError as msg: 
      print(msg)
    else:
      print("\033[92mAlle Tests erfolgreich!")
