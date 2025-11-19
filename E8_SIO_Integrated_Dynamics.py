import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from itertools import product
import os

# --- CONFIGURATION ---
# UPDATE THIS PATH if needed
FFMPEG_PATH = r"C:\ffmpeg\bin\ffmpeg.exe" 

VIDEO_FILENAME = 'E8_SIO_Integrated_Dynamics.mp4'
LIMIT_X = 20000
FPS = 30
DURATION = 20
FRAMES = FPS * DURATION

if os.path.exists(FFMPEG_PATH):
    plt.rcParams['animation.ffmpeg_path'] = FFMPEG_PATH

print(f"--- INITIALIZING E8-SIO INTEGRATION ---")
print(f"Mapping Dynamical Phase Space onto Geometric Lattice Space...")

# ==========================================
# 1. E8 GEOMETRY ENGINE
# ==========================================

def generate_e8_projection():
    print("Generating E8 Roots...", end="")
    roots = []
    types = [] # 0=Vector, 1=Spinor
    
    # Vector Roots (Mod 0,2,4) - Structural
    for i in range(8):
        for j in range(i+1, 8):
            for s1 in [-1, 1]:
                for s2 in [-1, 1]:
                    r = np.zeros(8); r[i]=s1; r[j]=s2
                    roots.append(r); types.append(0)
    
    # Spinor Roots (Mod 1,5) - Prime
    for signs in product([0.5, -0.5], repeat=8):
        if np.sum(np.array(signs) < 0) % 2 == 0:
            roots.append(np.array(signs)); types.append(1)
            
    roots = np.array(roots)
    types = np.array(types)
    
    # Coxeter Projection
    u = np.array([1, np.cos(np.pi/15), np.cos(2*np.pi/15), np.cos(3*np.pi/15),
                  np.cos(4*np.pi/15), np.cos(5*np.pi/15), np.cos(6*np.pi/15), np.cos(7*np.pi/15)])
    v = np.array([0, np.sin(np.pi/15), np.sin(2*np.pi/15), np.sin(3*np.pi/15),
                  np.sin(4*np.pi/15), np.sin(5*np.pi/15), np.sin(6*np.pi/15), np.sin(7*np.pi/15)])
    
    # Orthonormalize
    v = v - np.dot(v, u) / np.dot(u, u) * u
    u = u / np.linalg.norm(u)
    v = v / np.linalg.norm(v)
    
    x = roots @ u
    y = roots @ v
    
    print(" Done.")
    return x, y, types

x_roots, y_roots, root_types = generate_e8_projection()
root_radii = np.sqrt(x_roots**2 + y_roots**2)
root_angles = np.arctan2(y_roots, x_roots)

# ==========================================
# 2. SIO DYNAMICS ENGINE
# ==========================================

def generate_sio_state(limit, frames):
    print("Computing SIO Dynamics...", end="")
    
    # Prime Walk
    is_prime = np.ones(limit+1, dtype=bool); is_prime[0:2]=False
    for i in range(2, int(np.sqrt(limit))+1):
        if is_prime[i]: is_prime[i*i::i]=False
    
    primes = np.where(is_prime)[0]; primes=primes[primes>3]
    sieve = np.zeros(limit)
    val = 0
    for p in primes:
        val += (1 if p%6==1 else -1)
        sieve[p] = val
    for i in range(1, limit):
        if sieve[i]==0 and sieve[i-1]!=0: sieve[i]=sieve[i-1]
        
    # Time steps
    t_idx = np.linspace(100, limit-1, frames).astype(int)
    x_val = t_idx
    
    # State q (Position) - Normalized
    # We use a larger normalization to keep it inside the lattice ring
    q = sieve[t_idx] / (np.sqrt(x_val)/np.log(x_val)) * 0.5 
    
    # State p (Momentum) - Zeta Driven (Approximated by rotation for speed)
    # In the full model, this comes from the zeros. Here we simulate the phase
    # angle evolution which corresponds to the 'Arg(Zeta)'
    # The prime number theorem implies a rotation in the complex plane.
    phase = np.log(x_val) * 14.1347 # Driven by first zero
    p = np.sin(phase) 
    
    print(" Done.")
    return q, p, x_val

q_vals, p_vals, x_vals = generate_sio_state(LIMIT_X, FRAMES)

# ==========================================
# 3. ANIMATION SETUP
# ==========================================

fig, ax = plt.subplots(figsize=(10, 10), facecolor='black')
ax.set_axis_off()
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)

# Static Lattice (Dimmed)
# Vector roots (Structure) - Grey
mask_vec = (root_types == 0)
ax.scatter(x_roots[mask_vec], y_roots[mask_vec], c='#333', s=30, marker='s', alpha=0.5)

# Spinor roots (Primes) - Dimmed Base
mask_spin = (root_types == 1)
base_scat = ax.scatter(x_roots[mask_spin], y_roots[mask_spin], c='#220022', s=50, alpha=0.3)

# Dynamic Activation Scatter
# We will update sizes and colors based on resonance
active_scat = ax.scatter([], [], s=[], c=[], alpha=0.9, edgecolors='white', linewidth=0.5)

# The SIO Pilot Wave (The Orbit Tracer)
pilot_line, = ax.plot([], [], c='white', linewidth=1, alpha=0.5)
pilot_head, = ax.plot([], [], marker='o', c='white', markersize=6)

# UI Text
info_text = ax.text(0.05, 0.95, "", transform=ax.transAxes, color='#00ff00', fontfamily='monospace', fontsize=10)
ax.text(0.5, 0.02, "E8 LATTICE // SIO RESONANCE", transform=ax.transAxes, color='#444', ha='center', fontsize=14)

def update(frame):
    # Current SIO State
    q = q_vals[frame] # Radial bias (Inert vs Split)
    p = p_vals[frame] # Phase driver
    
    # Map State to Polar Coordinates
    # Angle is driven by the Zeta momentum (p) + Time evolution
    # Radius is driven by the Prime Position (q)
    
    # SIO Orbit visualization coordinates
    orbit_x = q
    orbit_y = p * 0.5 # Scale momentum to fit
    
    # 1. Update Pilot Trace
    tail = 50
    start = max(0, frame-tail)
    pilot_line.set_data(q_vals[start:frame], p_vals[start:frame]*0.5)
    pilot_head.set_data([orbit_x], [orbit_y])
    
    # 2. Calculate Lattice Resonance
    # We activate roots that align with the SIO phase angle
    sio_angle = np.arctan2(orbit_y, orbit_x)
    sio_mag = np.sqrt(orbit_x**2 + orbit_y**2)
    
    # Angular alignment check (Dot product in angle space)
    # We look for roots within a sector
    angle_diff = np.abs(np.arctan2(np.sin(root_angles - sio_angle), np.cos(root_angles - sio_angle)))
    
    # Activation Logic:
    # Roots light up if the SIO 'beam' passes near them (angle < 30 degrees)
    # AND if the Parity matches the SIO state (q)
    
    # Color Logic:
    # q < 0 (Inert) -> Magenta
    # q > 0 (Split) -> Cyan
    active_color = '#ff00ff' if q < 0 else '#00ffff'
    
    # Filter for active roots
    # Only affect Spinor roots (Type 1)
    mask_active = (root_types == 1) & (angle_diff < 0.5)
    
    if np.sum(mask_active) > 0:
        # Update the active scatter plot
        active_x = x_roots[mask_active]
        active_y = y_roots[mask_active]
        
        # Size pulsates with magnitude
        sizes = np.ones(len(active_x)) * 80 * (sio_mag + 0.5)
        
        active_scat.set_offsets(np.c_[active_x, active_y])
        active_scat.set_sizes(sizes)
        active_scat.set_color(active_color)
    else:
        active_scat.set_offsets(np.zeros((0, 2)))

    # Update Text
    state_str = "INERT (MASS)" if q < 0 else "SPLIT (ENERGY)"
    info_text.set_text(
        f"X: {x_vals[frame]}\n"
        f"SIO State:  {state_str}\n"
        f"Lattice Resonance: LOCKED"
    )
    
    return active_scat, pilot_line, pilot_head, info_text

# --- RENDER ---
print("Rendering E8-SIO Integration...")
try:
    writer = animation.FFMpegWriter(fps=FPS, bitrate=6000)
    anim = animation.FuncAnimation(fig, update, frames=FRAMES, blit=True)
    anim.save(VIDEO_FILENAME, writer=writer)
    print(f"DONE. Saved to {VIDEO_FILENAME}")
except Exception as e:
    print(f"Error: {e}")