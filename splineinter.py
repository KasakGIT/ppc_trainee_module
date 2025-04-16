import numpy as np
import matplotlib.pyplot as plt

def compute_spline(x, y):
    n = len(x) - 1  

    h = [x[i+1] - x[i] for i in range(n)]  

    alpha = [0] * (n)
    for i in range(1, n):
        alpha[i] = (3/h[i]) * (y[i+1] - y[i]) - (3/h[i-1]) * (y[i] - y[i-1])

    l = [1] + [0] * n
    mu = [0] * n
    z = [0] * (n+1)

    for i in range(1, n):
        l[i] = 2 * (x[i+1] - x[i-1]) - h[i-1] * mu[i-1]
        mu[i] = h[i] / l[i]
        z[i] = (alpha[i] - h[i-1] * z[i-1]) / l[i]

    l[n] = 1
    z[n] = 0

    a = y[:-1]       
    b = [0] * n
    c = [0] * (n+1)
    d = [0] * n

    for j in range(n-1, -1, -1):
        c[j] = z[j] - mu[j] * c[j+1]
        b[j] = (y[j+1] - y[j])/h[j] - h[j] * (c[j+1] + 2*c[j]) / 3
        d[j] = (c[j+1] - c[j]) / (3 * h[j])

    return a, b, c, d

# Step 6: Function to evaluate spline at a given point
def evaluate(x, a, b, c, d, xp):
    n = len(x) - 1
    for i in range(n):
        if x[i] <= xp <= x[i+1]:
            dx = xp - x[i]
            return a[i] + b[i]*dx + c[i]*dx**2 + d[i]*dx**3
    return None

def main():

    x = [1, 2, 3, 4]
    y = [2, 3, 5, 4]

    a, b, c, d = compute_spline(x, y)

    print("Spline Coefficients:")
    for i in range(len(a)):
        print(f"From x = {x[i]} to x = {x[i+1]}:")
        print(f" a = {a[i]:.3f}, b = {b[i]:.3f}, c = {c[i]:.3f}, d = {d[i]:.3f}")

    xs = np.linspace(x[0], x[-1], 100)
    ys = [evaluate(x, a, b, c, d, xi) for xi in xs]

    plt.plot(x, y, 'ro', label='Given Points')     
    plt.plot(xs, ys, 'b-', label='Cubic Spline')   
    plt.title("Cubic Spline Interpolation")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.legend()
    plt.show()

main()
