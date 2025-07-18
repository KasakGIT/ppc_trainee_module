import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev

# 1. Read the CSV file
df = pd.read_csv('loop_track_waypoints.csv')

# Extract x and y coordinates
x = df['X'].values
y = df['Y'].values

# 2. Prepare for closed-loop interpolation
# For closed loop, we need to append the first point to the end
x_closed = np.append(x, x[0])
y_closed = np.append(y, y[0])

# 3. Parameterize the curve (t represents the "time" along the curve)
t = np.linspace(0, 1, len(x_closed))

# 4. Perform cubic spline interpolation
# s=0 means no smoothing - exact interpolation
tck, u = splprep([x_closed, y_closed], u=t, s=0, per=True)  # per=True for periodic (closed loop)

# 5. Create finer interpolation for smooth curve
u_new = np.linspace(0, 1, 1000)  # 1000 points for smooth curve
x_new, y_new = splev(u_new, tck)

# 6. Visualization
plt.figure(figsize=(10, 8))

# Plot original waypoints
plt.plot(x, y, 'ro', label='Original Waypoints', markersize=6)

# Plot interpolated curve
plt.plot(x_new, y_new, 'b-', label='Interpolated Path', linewidth=2)

# Add labels and title
plt.xlabel('x (meters)')
plt.ylabel('y (meters)')
plt.title('Loop Track Interpolation')
plt.legend()
plt.grid(True)
plt.axis('equal')  # Ensure equal scaling for x and y axes
plt.show()