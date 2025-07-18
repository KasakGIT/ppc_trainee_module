import numpy as np
import matplotlib.pyplot as plt
from itertools import product

def interpolation(x_array, y, d2_start, d2_end):
    """matrix-based cubic spline implementation"""
    # Build the system of equations matrix
    matrix_array = np.array([
        [x_array[0]**3, x_array[0]**2, x_array[0], 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [x_array[1]**3, x_array[1]**2, x_array[1], 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, x_array[1]**3, x_array[1]**2, x_array[1], 1, 0, 0, 0, 0],
        [0, 0, 0, 0, x_array[2]**3, x_array[2]**2, x_array[2], 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, x_array[2]**3, x_array[2]**2, x_array[2], 1],
        [0, 0, 0, 0, 0, 0, 0, 0, x_array[3]**3, x_array[3]**2, x_array[3], 1],
        [3*x_array[1]**2, 2*x_array[1], 1, 0, -3*x_array[1]**2, -2*x_array[1], -1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 3*x_array[2]**2, 2*x_array[2], 1, 0, -3*x_array[2]**2, -2*x_array[2], -1, 0],
        [6*x_array[1], 2, 0, 0, -6*x_array[1], -2, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 6*x_array[2], 2, 0, 0, -6*x_array[2], -2, 0, 0],
        [6*x_array[0], 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # d2y/dx2 at start
        [0, 0, 0, 0, 0, 0, 0, 0, 6*x_array[3], 2, 0, 0]   # d2y/dx2 at end
    ])
    
    # Build the right-hand side constants vector
    constants = np.array([
        y[0], y[1], y[1], y[2], y[2], y[3],
        0, 0, 0, 0, d2_start, d2_end
    ])
    
    # Solve system of equations
    coefficients = np.linalg.solve(matrix_array, constants)
    
    return coefficients

def evaluate_spline(x_array, coefficients, x):
    """Evaluate the spline at point x"""
    # Determine which segment x falls into
    if x <= x_array[1]:
        a, b, c, d = coefficients[0:4]
        xi = x
    elif x <= x_array[2]:
        a, b, c, d = coefficients[4:8]
        xi = x
    else:
        a, b, c, d = coefficients[8:12]
        xi = x
    
    return a*xi**3 + b*xi**2 + c*xi + d

def calculate_curvature(x_array, coefficients, x):
    """Calculate curvature at point x using the formula:
    κ(x) = |y''(x)| / (1 + (y'(x))^2)^(3/2)"""
    # Determine which segment x falls into
    if x <= x_array[1]:
        a, b, c, d = coefficients[0:4]
        xi = x
    elif x <= x_array[2]:
        a, b, c, d = coefficients[4:8]
        xi = x
    else:
        a, b, c, d = coefficients[8:12]
        xi = x
    
    # First and second derivatives
    y_prime = 3*a*xi**2 + 2*b*xi + c
    y_double_prime = 6*a*xi + 2*b
    
    return abs(y_double_prime) / (1 + y_prime**2)**1.5

def total_curvature(x_array, coefficients, n_points=100):
    """Calculate total curvature along the spline"""
    total = 0
    # Sample points in each segment
    for i in range(len(x_array)-1):
        x_segment = np.linspace(x_array[i], x_array[i+1], n_points)
        segment_length = x_array[i+1] - x_array[i]
        
        for x in x_segment:
            k = calculate_curvature(x_array, coefficients, x)
            total += k * (segment_length/n_points)  # Riemann sum approximation
    
    return total

def optimize_boundary_conditions(x_array, y_array, d2_range=(-10, 15), step=1):
    """Find boundary conditions that minimize total curvature"""
    min_curvature = float('inf')
    best_bc = (0, 0)
    
    # Generate all combinations of boundary conditions
    d2_options = np.arange(d2_range[0], d2_range[1]+step, step)
    
    for d2_start, d2_end in product(d2_options, repeat=2):
        try:
            coeffs = interpolation(x_array, y_array, d2_start, d2_end)
            current_curv = total_curvature(x_array, coeffs)
            
            if current_curv < min_curvature:
                min_curvature = current_curv
                best_bc = (d2_start, d2_end)
        except np.linalg.LinAlgError:
            continue  # Skip invalid combinations
            
    return best_bc, min_curvature

def plot_results(x_array, y_array, coeffs_opt, coeffs_nat):
    """Plot both natural and optimized splines"""
    x_plot = np.linspace(min(x_array), max(x_array), 100)
    y_opt = [evaluate_spline(x_array, coeffs_opt, x) for x in x_plot]
    y_nat = [evaluate_spline(x_array, coeffs_nat, x) for x in x_plot]
    
    plt.figure(figsize=(10, 6))
    plt.plot(x_array, y_array, 'ko', label='Waypoints', markersize=8)
    plt.plot(x_plot, y_opt, 'r-', label='Optimized spline')
    plt.plot(x_plot, y_nat, 'b--', label='Natural spline')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Cubic Spline Interpolation with Minimum Curvature')
    plt.legend()
    plt.grid(True)
    plt.show()

# Example usage
if __name__ == "__main__":
    # Define your 4 waypoints
    x_array = np.array([0, 2, 3, 5])
    y_array = np.array([8, 3, 5, 2])
    
    # Find optimal boundary conditions
    (opt_d2y0, opt_d2yn), min_curv = optimize_boundary_conditions(x_array, y_array)
    print(f"Optimal boundary derivatives: d2y0={opt_d2y0:.2f}, d2yn={opt_d2yn:.2f}")
    print(f"Minimum total curvature: {min_curv:.4f}")
    
    # Compare with natural spline (d2y0 = d2yn = 0)
    nat_curv = total_curvature(x_array, interpolation(x_array, y_array, 0, 0))
    print(f"Natural spline curvature: {nat_curv:.4f}")
    print(f"Curvature reduction: {100*(nat_curv-min_curv)/nat_curv:.1f}%")
    
    # Plot results
    coeffs_opt = interpolation(x_array, y_array, opt_d2y0, opt_d2yn)
    coeffs_nat = interpolation(x_array, y_array, 0, 0)
    plot_results(x_array, y_array, coeffs_opt, coeffs_nat)