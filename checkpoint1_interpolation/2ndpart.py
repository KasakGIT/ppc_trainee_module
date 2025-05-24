import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev

# Reading the CSV file
df = pd.read_csv('loop_track_waypoints.csv')

# Extract X and Y coordinates
x = df['X'].values  # Use 'X' column
y = df['Y'].values  # Use 'Y' column

# Interpolate using splprep and splev
tck, u = splprep([x, y], s=0, per=True)  # 'per=True' means the track will be closed which we want
u_new = np.linspace(0, 1, 1000)  # 1000 points for smoothness
x_new, y_new = splev(u_new, tck)

plt.plot(x, y, 'ro', label='Original Waypoints')
plt.plot(x_new, y_new, 'b-', label='Interpolated Path')

plt.title("Path Interpolation")
plt.xlabel("X (meters)")
plt.ylabel("Y (meters)")
plt.legend()

plt.grid(True)
plt.axis('equal')  # Equal aspect ratio
plt.show()