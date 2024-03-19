import numpy as np
import matplotlib.pyplot as plt

t = np.linspace(0, 1, 101)

def minimal_jerk(t):
    return 10 * t**3 - 15 * t**4 + 6 * t**5

def velocity(t):
    return 30 * t**2 - 60 * t**3 + 30 * t**4

def acceleration(t):
    return 60 * t - 180 * t**2 + 120 * t**3

def jerk(t):
    return - 360 * t + 360 * t**2
    
# Refactored Plotting
fig, axs = plt.subplots(2, 2, figsize=(10, 10))

# Define plot function
def plot_data(ax, data, label, color):
    ax.plot(t, data, label=label, color=color)
    ax.set_xlabel("Time")
    ax.set_ylabel(label)
    ax.legend()

# Define data and labels
data = [minimal_jerk(t), velocity(t), acceleration(t), jerk(t)]
labels = ["Position", "Velocity", "Acceleration", "Jerk"]
colors = ["Blue", "green", "red", "orange"]

# Plot data
for i in range(2):
    for j in range(2):
        plot_data(axs[i, j], data[i*2+j], labels[i*2+j], colors[i*2+j])
        axs[i, j].set_title(labels[i*2+j])

plt.tight_layout()
plt.show()