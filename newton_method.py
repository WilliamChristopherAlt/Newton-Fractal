import numpy as np
import matplotlib.pyplot as plt

colormap = "twilight"
interactive_resolution = 200
high_resolution = 1024


# Initialize draggable roots
roots = np.array([1 + 0j, -0.5 + 0.866j, -0.5 - 0.866j, 1 + 1j, 0 + 0j, -1 -1j])

# Function to compute polynomial and its derivative
def f(z, roots):
    p = np.ones_like(z, dtype=complex)
    dp = np.zeros_like(z, dtype=complex)
    
    for r in roots:
        dp = dp * (z - r) + p
        p *= (z - r)
    
    return p, dp

# Newton's method with travel distance tracking
def newton_method(z, roots, max_iter=50, tol=1e-6):
    z_prev = np.copy(z)
    total_travel = np.zeros_like(z, dtype=float)

    for _ in range(max_iter):
        p, dp = f(z, roots)
        dz = p / dp
        z -= dz

        # Track movement
        total_travel += np.abs(z - z_prev)
        z_prev = np.copy(z)

        if np.all(np.abs(dz) < tol):
            break

    return z, total_travel

# Generate grid
x = np.linspace(-2, 2, interactive_resolution)
y = np.linspace(-2, 2, interactive_resolution)
X, Y = np.meshgrid(x, y)
Z = X + 1j * Y

x_high_res = np.linspace(-2, 2, high_resolution * 2)
y_high_res  = np.linspace(-2, 2, high_resolution * 2)
X_high_res, Y_high_res  = np.meshgrid(x_high_res , y_high_res )
Z_high_res  = X_high_res  + 1j * Y_high_res   # Complex number grid

# Compute Newton fractal and chaos intensity
def compute_fractal(high_res=False):
    grid = Z_high_res if high_res else Z
    new_roots, travel_matrix = newton_method(grid.copy(), roots)

    # Classify each point by closest root (without reordering)
    color_matrix = np.zeros_like(new_roots, dtype=int)

    for i, r in enumerate(roots):
        mask = np.isclose(new_roots, r, atol=1e-3)
        color_matrix[mask] = i  # Assign color based on initial root order

    # Apply a custom contrast adjustment (you can tweak the exponent factor)
    contrast_factor = 0.5 # Adjust this factor as needed
    travel_matrix = np.power(travel_matrix, contrast_factor)

    # Blend colors with chaos intensity
    fractal_img = np.zeros((*color_matrix.shape, 3))  # RGB array
    for i in range(len(roots)):
        mask = color_matrix == i
        base_color = np.array(plt.cm.get_cmap(colormap)(i / len(roots))[:3])  # Get RGB from colormap
        fractal_img[mask] = base_color * (travel_matrix[mask])[:, np.newaxis]
        fractal_img = np.clip(fractal_img, 0, 1)

    if not high_res:
        return fractal_img
    else:
        h, w, c = fractal_img.shape  # Extract height, width, and color channels
        assert h % 2 == 0 and w % 2 == 0, "Image dimensions must be even for downsampling"

        # Downsampling using mean pooling on the first two dimensions
        return fractal_img.reshape(h // 2, 2, w // 2, 2, c).mean(axis=(1, 3))

# Initialize plot
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_title("Newton Fractal with Chaos Highlighting")
ax.set_xlabel("Re(z)")
ax.set_ylabel("Im(z)")

# Compute initial fractal
fractal_img = compute_fractal()
image_display = ax.imshow(fractal_img, extent=(-2, 2, -2, 2), origin="lower")

# Plot draggable roots
root_points, = ax.plot(roots.real, roots.imag, 'wo', markersize=10, markeredgecolor='black')

# Handle dragging
dragging = None

def on_press(event):
    global dragging
    if event.inaxes != ax:
        return
    for i, r in enumerate(roots):
        if np.hypot(event.xdata - r.real, event.ydata - r.imag) < 0.1:
            dragging = i
            break

def on_release(event):
    global dragging
    dragging = None

def on_motion(event):
    global dragging
    if dragging is None or event.inaxes != ax:
        return
    roots[dragging] = event.xdata + 1j * event.ydata
    root_points.set_data(roots.real, roots.imag)
    fractal_img = compute_fractal()  # Recompute fractal
    image_display.set_data(fractal_img)
    plt.draw()

# High-resolution rendering
def on_key(event):
    if event.key == "enter":
        print("Rendering high-resolution fractal...")
        high_res = compute_fractal(True)
        high_res = np.flipud(high_res)
        plt.imsave("newton_fractal.png", high_res)
        print("Saved as newton_fractal.png")

# Connect interactive events
fig.canvas.mpl_connect("button_press_event", on_press)
fig.canvas.mpl_connect("button_release_event", on_release)
fig.canvas.mpl_connect("motion_notify_event", on_motion)
fig.canvas.mpl_connect("key_press_event", on_key)

plt.show()
