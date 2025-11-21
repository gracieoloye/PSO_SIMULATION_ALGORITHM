import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import streamlit.components.v1 as components

# 1. Configuration & Helper Classes
class Particle:
    def __init__(self, bounds):
        self.position = np.random.uniform(bounds[0], bounds[1], 2)
        self.velocity = np.random.uniform(-1, 1, 2)
        self.pbest_position = self.position.copy()
        self.pbest_value = float('inf')
        self.bounds = bounds

    def update(self, global_best_position, w, c1, c2):
        r1 = np.random.rand(2)
        r2 = np.random.rand(2)
        cognitive = c1 * r1 * (self.pbest_position - self.position)
        social = c2 * r2 * (global_best_position - self.position)
        self.velocity = (w * self.velocity) + cognitive + social
        self.position += self.velocity
        self.position = np.clip(self.position, self.bounds[0], self.bounds[1])

# 2. The Objective Functions
def banana_function(x, y):
    return (1 - x)**2 + 100 * (y - x**2)**2

def schwefel_function(x, y):
    return 418.9829 * 2 - (x * np.sin(np.sqrt(np.abs(x))) + y * np.sin(np.sqrt(np.abs(y))))

# 3. Streamlit UI Layout
st.title("Particle Swarm Optimization (PSO) Simulation")
st.markdown("""
This application simulates **Swarm Intelligence** to find the global minimum of complex mathematical functions.
The particles communicate to find the lowest point (dark blue regions).
""")

# Sidebar Controls
st.sidebar.header("Simulation Parameters")
func_choice = st.sidebar.selectbox("Choose Function", ["Banana Function", "Schwefel Function"])
num_particles = st.sidebar.slider("Number of Particles", 10, 100, 30)
iterations = st.sidebar.slider("Iterations", 10, 100, 50)
w = st.sidebar.slider("Inertia (w)", 0.0, 1.0, 0.7)

# Setup based on choice
if func_choice == "Banana Function":
    func = banana_function
    bounds = [-2, 2]
    zoom_level = "Log" # Helper for plotting
else:
    func = schwefel_function
    bounds = [-500, 500]
    zoom_level = "Linear"

# 4. Run Simulation
if st.button("Run Simulation"):
    # Initialize
    particles = [Particle(bounds) for _ in range(num_particles)]
    global_best_pos = np.array([0.0, 0.0])
    global_best_val = float('inf')
    history = []

    # Optimization Loop
    progress_bar = st.progress(0)
    for i in range(iterations):
        current_positions = []
        for p in particles:
            val = func(p.position[0], p.position[1])
            if val < p.pbest_value:
                p.pbest_value = val
                p.pbest_position = p.position.copy()
            if val < global_best_val:
                global_best_val = val
                global_best_pos = p.position.copy()
            current_positions.append(p.position.copy())
        
        for p in particles:
            p.update(global_best_pos, w, 1.4, 1.4)
        
        history.append(current_positions)
        progress_bar.progress((i + 1) / iterations)

    st.success(f"Global Minimum Found: {global_best_val:.5f} at {global_best_pos}")

    # 5. Generate Animation
    with st.spinner("Generating Animation..."):
        fig, ax = plt.subplots(figsize=(6, 6))
        x = np.linspace(bounds[0], bounds[1], 100)
        y = np.linspace(bounds[0], bounds[1], 100)
        X, Y = np.meshgrid(x, y)
        Z = func(X, Y)
        
        if zoom_level == "Log":
            from matplotlib.colors import LogNorm
            ax.contourf(X, Y, Z, levels=50, cmap='viridis', norm=LogNorm())
        else:
            ax.contourf(X, Y, Z, levels=50, cmap='viridis')
            
        scatter = ax.scatter([], [], c='red', s=30)
        ax.set_xlim(bounds[0], bounds[1])
        ax.set_ylim(bounds[0], bounds[1])

        def animate(i):
            scatter.set_offsets(history[i])
            ax.set_title(f"Iteration {i}/{iterations}")
            return scatter,

        anim = FuncAnimation(fig, animate, frames=iterations, interval=100, blit=True)
        
        # Render as HTML5 Video
        components.html(anim.to_jshtml(), height=700)
