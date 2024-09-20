import numpy as np
import pandas as pd

# ========================================================================== 
# Exercise 1
# ==========================================================================
def arithmetics_test(Ekin):
    try:
      assert np.isclose(Ekin, 50000, atol=0.05 / 1000), f"\033[91mFehler in Ekin. Erwartet {50000.0} erhalten {Ekin}"
    except AssertionError as msg: 
      print(msg)
    else:
      print("\033[92mAlle Tests erfolgreich!")

def dicts_test(cylinder_head,components,with_exists):
    crankshaft = components[0]
    crankshaft_weight = crankshaft["weight"]
    cylinder_head_val = {"name":"cylinder head",
                        "material":"aluminum alloy",
                        "weight":2.5,
                        "dimensions (mm)":{"lenth":150,"width":100,"height":80}}
    if "width" in crankshaft["dimensions (mm)"]:
      with_exists_val = True
    else: with_exists_val = False

    try:
      # check if cylinder head was added
      assert cylinder_head in components, f"\033[91m 'cylinder_head' nicht in Liste gefunden"
      # check if cylinder head correct
      assert cylinder_head == cylinder_head_val , f"\033[91m Error in 'cylinder_head'. \n Expected:\n {cylinder_head_val}\n Got:\n {cylinder_head}  "
      # check modification
      assert crankshaft_weight == 15 , f"\033[91m Error in the weight stored for the crankshaft. \n Expected: 15 Got: {crankshaft_weight}"
      # check with_exists
      assert with_exists_val  == with_exists, f"\033[91m Error in 'with_exists'. \n Expected: {with_exists_val} Got: {with_exists}"
    except AssertionError as msg:
      print(msg)
    else:
      print("\033[92mAlle Tests erfolgreich!")

def arithmetic_arrays_test(A,B,C,D):
    try:
      # check A
      assert np.all(A == np.array([[2,3],[9,2],[4,5]])), f"\033[91mFehler in A. Erwartet: \n{np.array([[2,3],[9,2],[4,5]])}\n\n Erhalten: \n{A}"

      # check B
      assert np.all(B[0] == np.array([[1,1]])), f"\033[91mFehler in B. Erste Zeile enthält nicht nur Einsen."
      assert np.all(B[1] != 1), f"\033[91mFehler in B. Zweite Zeile enthält Einsen."
      assert np.all(B[2] != 1), f"\033[91mFehler in B. Dritte Zeile enthält Einsen."

      # check C
      assert np.all(C == A+B), f"\033[91mFehler in C. Erwartet: \n{A+B}\n\n Erhalten: \n{C}"

      # check D
      assert np.all(D == 2*C), f"\033[91mFehler in C. Erwartet: \n{2*C}\n\n Erhalten: \n{D}"
    except AssertionError as msg:
      print(msg)
    else:
      print("\033[92mAlle Tests erfolgreich!")

def conditions_test(Q,QTQ,eig_vals,singular_vals,cond_diy):
  QTQ_val = np.dot(Q.T,Q)
  eig_vals_val = np.linalg.eig(QTQ_val)[0]
  singular_vals_val = np.sqrt(eig_vals_val)
  cond_diy_val = max(singular_vals_val)/min(singular_vals_val)

  try:
    assert np.all(QTQ == QTQ_val), f"\033[91m Fehler in QTQ Erwartet: \n {QTQ_val}, \n Erhalten: \n {QTQ}"
    assert np.all(eig_vals == eig_vals_val), f"\033[91m Fehler in eig_vals Erwartet: \n {eig_vals_val}, \n Erhalten: \n {eig_vals}"
    assert np.all(singular_vals == singular_vals_val), f"\033[91m Fehler in singular_vals Erwartet: \n {singular_vals_val}, \n Erhalten: \n {singular_vals}"
    assert cond_diy == cond_diy_val, f"\033[91m Fehler in cond_diy Erwartet: {cond_diy_val}, Erhalten: {cond_diy}"
  except AssertionError as msg:
      print(msg)
  else:
      print("\033[92mAlle Tests erfolgreich!")

def orthonormality_test(is_orthonormal):
  class test_base:
    def __init__(self,vectors,dimension,orthonormal):
      self.vectors = vectors
      self.dimension = dimension
      self.orthonormal = orthonormal

  test_bases = []

  vectors = [np.array([[1/np.sqrt(2),1/np.sqrt(2),0]]),
                  np.array([[-1/np.sqrt(2),1/np.sqrt(2),0]]),
                  np.array([[0,0,1]])]
  test_bases.append(test_base(vectors, 3, True))

  vectors = [np.array([[1/np.sqrt(2),1/np.sqrt(2),0]]),
                  np.array([[-1/np.sqrt(3),1/np.sqrt(2),0]]),
                  np.array([[0,0,1]])]
  test_bases.append(test_base(vectors, 3, False))

  vectors = [np.array([[1,0,0]]),
                  np.array([[0,1,0]]),
                  np.array([[0,0,1]])]
  test_bases.append(test_base(vectors, 3, True))

  vectors = [np.array([[1,0,0,0]]),
                  np.array([[0,1,0,0]]),
                  np.array([[0,0,1,0]]),
                  np.array([[0,0,0,1]])]
  test_bases.append(test_base(vectors, 4, True))

  vectors = [np.array([[1,0,0,0]]),
                  np.array([[0,1,0,0]]),
                  np.array([[0,0,1,0]])]
  test_bases.append(test_base(vectors, 4, False))

  vectors = [np.array([[1,0]]),
                  np.array([[0,1]]),
                  np.array([[0,0]])]
  test_bases.append(test_base(vectors, 3, False))

  try:
      for b in test_bases:
        function_result = is_orthonormal(b.vectors,b.dimension)
        assert function_result == b.orthonormal, f"\033[91m Fehler für \n {b.vectors} \n Erwartet: {b.orthonormal}, Erhalten: {function_result}"
  except AssertionError as msg:
      print(msg)
  else:
      print("\033[92mAlle Tests erfolgreich!")

def lgs_circuit_board_test(quantity_of_A, quantity_of_B):
  A = np.array([[3,2],[2,3]])
  b = np.array([[550],[525]])
  quantity_of_A_val, quantity_of_B_val = np.linalg.solve(A, b)
  try:
      assert quantity_of_A == quantity_of_A_val, f"\033[91m Fehler in quantity_of_A. Erwartet: {quantity_of_A_val}, Erhalten: {quantity_of_A}"
      assert quantity_of_B == quantity_of_B_val, f"\033[91m Fehler in quantity_of_A. Erwartet: {quantity_of_B_val}, Erhalten: {quantity_of_B}"
  except AssertionError as msg:
      print(msg)
  else:
      print("\033[92mAlle Tests erfolgreich!")

def plt_derivative_test(time, space, velocity):
  space_val = 0.1 * np.sin(2 * np.pi * time) + 0.05
  velocity_val = 0.1 * 2 * np.pi * np.cos(2 * np.pi * time)
  try:
      assert np.all(space == space_val), f"\033[91m Fehler in 'space'"
      assert np.all(velocity == velocity_val), f"\033[91m Fehler in 'velocity'"
  except AssertionError as msg:
      print(msg)
  else:
      print("\033[92mAlle Tests erfolgreich!")

# ========================================================================== 
# Exercise 2
# ==========================================================================
def combinatorics_df_test(component_dict,components_df):
    val_dict = {"Component": ["Gear", "Gear", "Gear", "Bolt", "Bolt", "Sensor", "Sensor", "Sensor"],
                "Type": ["A", "B", "C", "1", "2", "X", "Y", "Z"],
                "Quantity": [3,3,3,2,2,3,3,3]}
    val_df = pd.DataFrame(val_dict)
    try:
      assert component_dict == val_dict, f"\033[91mFehler in component_dict. Erwartet: \n {val_dict} \n Erhalten: \n {component_dict}"
      pd.testing.assert_frame_equal(val_df, components_df)
    except AssertionError as msg: 
      print(msg)
    else:
      print("\033[92mAlle Tests erfolgreich!")

def combinatorics_num_comp_test(num_gears,num_bolts,num_sensors):
    try:
      assert num_gears == 3, f"\033[91mFehler in num_gears. Erwartet: 3 Erhalten: {num_gears}"
      assert num_bolts == 2, f"\033[91mFehler in num_gears. Erwartet: 2 Erhalten: {num_bolts}"
      assert num_sensors == 3, f"\033[91mFehler in num_gears. Erwartet: 3 Erhalten: {num_sensors}"
    except AssertionError as msg: 
      print(msg)
    else:
      print("\033[92mAlle Tests erfolgreich!")

def combinatorics_perm_test(num_perm,num_perm_rep):
    try:
      assert num_perm == 72, f"\033[91mFehler in num_gears. Erwartet: 72 Erhalten: {num_perm}"
      assert num_perm_rep == 324, f"\033[91mFehler in num_gears. Erwartet: 324 Erhalten: {num_perm_rep}"
    except AssertionError as msg: 
      print(msg)
    else:
      print("\033[92mAlle Tests erfolgreich!")

def combinatorics_comb_test(num_comb,num_comb_rep):
    try:
      assert num_comb == 9, f"\033[91mFehler in num_gears. Erwartet: 9 Erhalten: {num_comb}"
      assert num_comb_rep == 108, f"\033[91mFehler in num_gears. Erwartet: 108 Erhalten: {num_comb_rep}"
    except AssertionError as msg: 
      print(msg)
    else:
      print("\033[92mAlle Tests erfolgreich!")

def probability_space_test1(omega, len_A):
    try:
      assert omega == ["aa", "ab", "ba", "bb"], f"\033[91mFehler in omega. Erhalten: {omega}"
      assert len_A == 16, f"\033[91mFehler in num_gears. Erwartet: 16 Erhalten: {len_A}"
    except AssertionError as msg: 
      print(msg)
    else:
      print("\033[92mAlle Tests erfolgreich!")

def probability_space_test2(A1, P_A1, A2, P_A2, expectation):
    try:
      P_A2_val = (0.6)**2 + 0.6 * 0.4 + 0.4 * 0.6
      P_A1_val = 0.6 * 0.6 + 0.4 * 0.4
      assert A1 == ["aa","bb"], f"\033[91mFehler in A1. Erhalten: {A1}"
      assert P_A1 == P_A1_val, f"\033[91mFehler in P_A1. Erwartet: {P_A1_val} Erhalten: {P_A1}"
      assert A2 == ["aa","ab","ba"], f"\033[91mFehler in A2. Erhalten: {A2}"
      assert P_A2 == P_A2_val, f"\033[91mFehler in P_A2. Erwartet: {P_A2_val} Erhalten: {P_A2}"
      assert expectation == P_A2_val, f"\033[91mFehler in expectation. Erwartet: {P_A2_val} Erhalten: {expectation}"
    except AssertionError as msg: 
      print(msg)
    else:
      print("\033[92mAlle Tests erfolgreich!")

def qlt_location_test(roughness_42):
    try:
      assert np.isclose(roughness_42, 1.30544, atol=0.0001), f"\033[91mFehler in roughness_42. Erwartet {1.30544} erhalten {roughness_42}"
    except AssertionError as msg:
      print(msg)
    else:
      print("\033[92mAlle Tests erfolgreich!")

def compare_dataframes(df1, df2, df_name):
    try:
        pd.testing.assert_frame_equal(df1, df2)
    except AssertionError as e:
        raise AssertionError(f"Fehler in {df_name} (right DataFrame)\n\nDetails:\n{e}")

def qlt_prob_spaces_test(qlt_df, omega_qlt, A3):
    omega_val = qlt_df
    mask = (omega_qlt['surface_roughness'] > 5) & (omega_qlt['hardness'] < 44)
    A3_val = omega_val[mask]

    try:
      compare_dataframes(omega_val, omega_qlt, "omega_qlt")
      compare_dataframes(A3_val, A3, "A3")
    except AssertionError as msg: 
      print(msg)
    else:
      print("\033[92mAlle Tests erfolgreich!")

def qlt_prob_test(A3_prob, A3_defective_prob):
    try:
      assert np.isclose(A3_prob, 0.07, atol=0.01), f"\033[91mFehler in A3_prob. Erwartet 0.07 erhalten {A3_prob}"
      assert np.isclose(A3_defective_prob, 0.58, atol=0.01), f"\033[91mFehler in A3_defective_prob. Erwartet 0.58 erhalten {A3_defective_prob}"
    except AssertionError as msg:
      print(msg)
    else:
      print("\033[92mAlle Tests erfolgreich!")

def motor_errors_test(P_conditional, P_causal):
    try:
      assert P_conditional == 0.8, f"\033[91mFehler in P_conditional."
      assert P_causal == 0.15, f"\033[91mFehler in P_causal."
    except AssertionError as msg: 
      print(msg)
    else:
      print("\033[92mAlle Tests erfolgreich!")

def production_df_test(production_df):
    production_df_val = pd.DataFrame({'Month': ['2024-01', '2024-01', '2024-02', '2024-02','2024-03', '2024-03'],
                                      'Machine_ID': ['M001', 'M002', 'M001', 'M003','M002', 'M003'],
                                      'Production_Quantity': [50, 30, 60, 40, 45, 55]
  })
    
    try:
      pd.testing.assert_frame_equal(production_df_val, production_df)
    except AssertionError as msg: 
      print(msg)
    else:
      print("\033[92mAlle Tests erfolgreich!")

def distribution_binom_test(n, p, expected_value_binom):
    try:
      assert n == 30, f"\033[91mFehler in n."
      assert p == 0.05, f"\033[91mFehler in p."
      assert expected_value_binom == n * p, f"\033[91mFehler in expected_value_binom. Erwartet {n * p} erhalten {expected_value_binom}"
    except AssertionError as msg: 
      print(msg)
    else:
      print("\033[92mAlle Tests erfolgreich!")

def distribution_norm_test(mean, std, expected_value_norm):
    try:
      assert mean == 0, f"\033[91mFehler in mean."
      assert std == 0.02, f"\033[91mFehler in std."
      assert expected_value_norm == mean, f"\033[91mFehler in expected_value_norm. Erwartet {mean} erhalten {expected_value_norm}"
    except AssertionError as msg: 
      print(msg)
    else:
      print("\033[92mAlle Tests erfolgreich!")

def distribution_poisson_test(lambd, expected_value_poisson):
    try:
      assert lambd == 2, f"\033[91mFehler in lambd."
      assert expected_value_poisson == lambd, f"\033[91mFehler in expected_value_poisson. Erwartet {lambd} erhalten {expected_value_poisson}"
    except AssertionError as msg: 
      print(msg)
    else:
      print("\033[92mAlle Tests erfolgreich!")

def distribution_uniform_test(a, b, expected_value_uniform):
    expected_val = (a + b) / 2
    try:
      assert a == 5, f"\033[91mFehler in a."
      assert b == 10, f"\033[91mFehler in b."
      assert expected_value_uniform == expected_val, f"\033[91mFehler in expected_value_uniform. Erwartet {expected_val} erhalten {expected_value_uniform}"
    except AssertionError as msg: 
      print(msg)
    else:
      print("\033[92mAlle Tests erfolgreich!")

def motor_df_test(motor_df, motor_df_final):
    motor_df_final_val = motor_df.reset_index().rename(columns = {"index":"ID"}).drop(["Laufzeit (h)"], axis = 1)
    try:
      pd.testing.assert_frame_equal(motor_df_final_val, motor_df_final)
    except AssertionError as msg: 
      print(msg)
    else:
      print("\033[92mAlle Tests erfolgreich!")
