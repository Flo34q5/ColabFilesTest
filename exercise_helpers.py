import numpy as np

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
