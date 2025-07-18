import matplotlib.pyplot as plt
import numpy as np

def cubic_spline_interpolation(x, y):
    """
    Natural cubic spline interpolation for 4 points.
    Returns coefficients (a, b, c, d) for each of the 3 intervals.
    """
    if len(x) != 4 or len(y) != 4:
        raise ValueError("This implementation requires exactly 4 points")
    
    n = 3  # number of intervals (for 4 points)
    h = [x[i+1] - x[i] for i in range(n)]
    
    # Set up the 2x2 system for c1 and c2 (since c0 = c3 = 0 for natural spline)
    # Equation 1: 2(h0+h1)c1 + h1 c2 = 3[(y2-y1)/h1 - (y1-y0)/h0] , got these from the constraints on eqns, after solving them, these can be used for solving for c1 and c2 
    # Equation 2: h1 c1 + 2(h1+h2) c2 = 3[(y3-y2)/h2 - (y2-y1)/h1]
    # after finding c0,c1,c2 we can simply use these values to find bi and di as we have equations for them in terms of c which we got by solving for equations using appropriate conditions
    # Create the matrix and right-hand side

    A = np.array([
        [2*(h[0]+h[1]), h[1]],
        [h[1], 2*(h[1]+h[2])]
    ])
    rhs = np.array([
        3*((y[2]-y[1])/h[1] - (y[1]-y[0])/h[0]),
        3*((y[3]-y[2])/h[2] - (y[2]-y[1])/h[1])
    ])
    
    # Manually solve the 2x2 system Ax = rhs
    det = A[0][0] * A[1][1] - A[0][1] * A[1][0]
    c1 = (A[1][1] * rhs[0] - A[0][1] * rhs[1]) / det
    c2 = (-A[1][0] * rhs[0] + A[0][0] * rhs[1]) / det
    c = [0.0, c1, c2, 0.0]  # Natural spline: c0 = c3 = 0

    
    # Calculate b and d coefficients
    a = y[:-1]
    b = []
    d = []
    for i in range(n):
        b_i = (y[i+1]-y[i])/h[i] - h[i]*(2*c[i]+c[i+1])/3
        d_i = (c[i+1]-c[i])/(3*h[i])
        b.append(b_i)
        d.append(d_i)
    
    # Return coefficients (excluding c3)
    return list(zip(a, b, c[:-1], d))

def evaluate_spline(x_data, coeffs, x):
    """Evaluate spline at point x"""
    if x < x_data[0] or x > x_data[-1]:
        raise ValueError("x is outside interpolation range")
    
    # Find the right interval
    if x <= x_data[1]:
        i = 0
    elif x <= x_data[2]:
        i = 1
    else:
        i = 2
    
    a, b, c, d = coeffs[i]
    dx = x - x_data[i]
    return a + b*dx + c*dx**2 + d*dx**3

def plot_spline(x_data, y_data, coeffs, num_points=100):
    """Plot the spline using matplotlib"""
    plt.figure(figsize=(8, 6))
    
    # Plot original data points
    plt.plot(x_data, y_data, 'o', label='Data points', markersize=8)
    
    # Generate points for the spline curve
    x_vals = np.linspace(min(x_data), max(x_data), num_points)
    y_vals = [evaluate_spline(x_data, coeffs, x) for x in x_vals]
    
    # Plot spline curve
    plt.plot(x_vals, y_vals, '-', label='Cubic spline interpolation')
    
    # Plot each polynomial segment in different colors
    colors = ['red', 'green', 'blue']
    for i in range(3):
        x_segment = np.linspace(x_data[i], x_data[i+1], num_points//3)
        y_segment = [evaluate_spline(x_data, coeffs, x) for x in x_segment]
        plt.plot(x_segment, y_segment, '--', color=colors[i], 
                alpha=0.5, label=f'Segment {i+1}')
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Cubic Spline Interpolation of 4 Points')
    plt.legend()
    plt.grid(True)
    plt.show()

# Example usage with 4 points
if __name__ == "__main__":
    # Input your 4 points here
    x_data = [0, 2, 3, 5]  # Replace with your x-values
    y_data = [8, 3, 5, 2]  # Replace with your y-values
    
    # Compute spline coefficients
    coeffs = cubic_spline_interpolation(x_data, y_data)
    
    # Print coefficients
    print("Cubic Spline Coefficients:")
    for i, (a, b, c, d) in enumerate(coeffs):
        print(f"Interval {i} ({x_data[i]} to {x_data[i+1]}):")
        print(f"  a = {a:.4f}, b = {b:.4f}, c = {c:.4f}, d = {d:.4f}")
        print(f"  Polynomial: {a:.4f} + {b:.4f}(x-{x_data[i]}) + {c:.4f}(x-{x_data[i]})² + {d:.4f}(x-{x_data[i]})³")
    
    # Plot the results
    plot_spline(x_data, y_data, coeffs)