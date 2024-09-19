import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ========================================================================== 
# Exercise 1
# ==========================================================================
def get_components_dict():
  components = []
  crankshaft = {"name":"crankshaft",
                   "material":"steel",
                   "weight":10,
                   "dimensions (mm)":{"lenth":300,"diameter":50}
                   }
  ball_bearing = {"name":"ball bearing",
                   "material":"chromium steel",
                   "weight":0.5,
                   "dimensions (mm)":{"inner diameter":20,"outer diameter":30,"width":10}
                   }
  components.append(crankshaft)
  components.append(ball_bearing)
  return components

def radius_of_component(x):
  def comp_func(x):
    if x < 0.2:
      return np.sqrt(x)+0.05
    elif x < 0.3:
      return 0.2
    else:
      return 0.05 * (1 + 0.5 * np.cos(50 * np.pi * x)) + 0.1
  return [comp_func(x) for x in x]

def get_profile(length):
  x = np.linspace(0, length, 1000+1)
  y = radius_of_component(x)
  return x,y

def visualize_disc(x_start,x_end,r):
    x = np.linspace(x_start, x_end, 10)
    y = np.full(10, r)
    return x,y

# ========================================================================== 
# Exercise 2
# ==========================================================================
def combinatorics_plot_num(perm_w_rep, perm_wo_rep, comb_w_rep, comb_wo_rep):
  # Save the results in a dictionary
  results = {
      'Permutation w replacement': perm_w_rep,
      'Permutation wo replacement': perm_wo_rep,
      'Combination w replacement': comb_w_rep,
      'Combination wo replacement': comb_wo_rep
  }
  results_df = pd.DataFrame(list(results.items()), columns=['Scenario', 'Number of possibilities'])

  # Plotting
  plt.figure(figsize=(10, 6))
  plt.bar(results_df['Scenario'], results_df['Number of possibilities'], color='skyblue')
  plt.ylabel('Number of possibilities')
  plt.xticks(rotation=45)
  plt.show()
