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
def combinatorics_plot_num(num_perm_rep, num_perm, num_comb_rep, num_comb):
  # Save the results in a dictionary
  results = {
      'Permutation w replacement': num_perm_rep,
      'Permutation wo replacement': num_perm,
      'Combination w replacement': num_comb_rep,
      'Combination wo replacement': num_comb
  }
  results_df = pd.DataFrame(list(results.items()), columns=['Scenario', 'Number of possibilities'])

  # Plotting
  plt.figure(figsize=(10, 6))
  plt.bar(results_df['Scenario'], results_df['Number of possibilities'], color='skyblue')
  plt.ylabel('Number of possibilities')
  plt.xticks(rotation=45)
  plt.show()

def get_production_df(): 
  # Production data per month
  production_df = pd.DataFrame({
      'Month': ['2024-01', '2024-01', '2024-02', '2024-02'],
      'Machine_ID': ['M001', 'M002', 'M001', 'M003'],
      'Production_Quantity': [50, 30, 60, 40]
  })
  return production_df

class regressor():
  def __init__(self):
    pass
    
  def add_data(self, X, Y):
    self.X = self.scale_input(X,Y)
    self.Y = Y

  def set_optimizer(self, optimizer):
    self.optimizer = optimizer

  def scale_input(self, X, Y):
    # Min-Max Scaling
    X_min = np.min(X)
    X_max = np.max(X)
    X_scaled = (X - X_min) / (X_max - X_min)
    return X_scaled

  def train(self, w, b, learning_rate, iterations):
    self.history_iterations = iterations
    self.w, self.b, self.loss_history = self.optimizer(self.X, self.Y, w, b, learning_rate, iterations)

  def plot_loss_history(self):
    # Plot Loss over Iterations
    plt.plot(range(self.history_iterations), self.loss_history)
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.title("Loss over Iterations")
    plt.show()

  def plot_data_prediction(self):
    # Plot Prediction Line
    plt.scatter(self.X, self.Y, color='blue', label='Data Points')
    plt.plot(self.X, self.w * self.X + self.b, color='red', label='Regression Line')
    plt.xlabel('Last (%) (Scaled)')
    plt.ylabel('Energieverbrauch (kWh)')
    plt.legend()
    plt.show()
