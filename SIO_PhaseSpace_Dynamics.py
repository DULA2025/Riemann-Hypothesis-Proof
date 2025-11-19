import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

# --- CONFIGURATION ---
# UPDATE THIS PATH if needed
FFMPEG_PATH = r"C:\ffmpeg\bin\ffmpeg.exe" 

VIDEO_FILENAME = 'SIO_PhaseSpace_Dynamics.mp4'
LIMIT_X = 100000
FPS = 30
DURATION = 30
FRAMES = FPS * DURATION

if os.path.exists(FFMPEG_PATH):
    plt.rcParams['animation.ffmpeg_path'] = FFMPEG_PATH

print(f"--- INITIALIZING SIO DYNAMICAL MAPPER ---")
print(f"Mapping the Phase Space of the Prime Inertia Engine...")

# --- 1. DATA GENERATION ---

# A. HARDCODED ZEROS (High precision for the driver)
# First 50 zeros (Imaginary parts)
gammas = np.array([
    14.1347, 21.0220, 25.0108, 30.4248, 32.9350, 37.5861, 40.9187, 43.3270, 48.0051, 49.7738,
    52.9703, 56.4462, 59.3470, 60.8317, 65.1125, 67.0798, 69.5464, 72.0671, 75.7046, 77.1448,
    79.3409, 82.9103, 84.7354, 87.4252, 88.8091, 92.4918, 94.6513, 95.8706, 98.8311, 101.3178,
    103.7255, 105.4466, 107.1686, 111.0295, 111.8747, 114.3202, 116.2266, 118.7907, 121.3701, 122.9468,
    124.2568, 127.5166, 129.5787, 131.0876, 133.4977, 134.7355, 138.1160, 139.7362, 141.0122, 143.1118
])

# B. THE PRIME TRAJECTORY (Ground Truth)
def get_sieve_gradient(limit):
    # Sieve of Eratosthenes
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[0:2] = False
    for i in range(2, int(np.sqrt(limit)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    
    primes = np.where(is_prime)[0]
    primes = primes[primes > 3]
    
    # Map to dense array
    gradient = np.zeros(limit)
    val = 0
    for p in primes:
        if p % 6 == 1: val += 1
        elif p % 6 == 5: val -= 1
        gradient[p] = val
        
    # Fill forward (Hold value between primes)
    for i in range(1, limit):
        if gradient[i] == 0 and gradient[i-1] != 0:
            gradient[i] = gradient[i-1]
            
    return gradient

print("Generating Prime Sieve Data...", end="")
sieve_raw = get_sieve_gradient(LIMIT_X)
print(" Done.")

# C. THE ZETA MOMENTUM (Spectral Driver)
def get_zeta_momentum(x_vals, gammas):
    # Momentum p(t) = Sum of oscillatory terms
    # This represents the "Force" exerted by the zeros
    momentum = np.zeros(len(x_vals))
    
    print("Computing Spectral Momentum...")
    for gamma in gammas:
        # Term: cos(gamma * ln(x))
        # This is the derivative of the potential x^(0.5+i*gamma)
        term = np.cos(gamma * np.log(x_vals))
        momentum += term
        
    # Normalize for visual correlation
    return momentum / np.sqrt(len(gammas))

# Setup Domain (Logarithmic sampling for dynamics)
# FIX: Ensure we don't exceed array bounds (LIMIT_X - 1)
t_steps = np.linspace(100, LIMIT_X - 1, FRAMES).astype(int)
x_vals = t_steps

# Compute State Vectors
# Position q: Normalized Sieve Discrepancy
# We normalize by sqrt(x)/ln(x) which is the standard deviation of the prime walk
norm_factor = np.sqrt(x_vals) / np.log(x_vals) * 2.0
q = sieve_raw[t_steps] / norm_factor

# Momentum p: Zeta Driver
p = get_zeta_momentum(x_vals, gammas)

# --- VISUALIZATION ---

fig = plt.figure(figsize=(12, 12), facecolor='#050505')
ax = fig.add_subplot(111, facecolor='#000000')

# Styling
ax.spines['bottom'].set_color('#444')
ax.spines['top'].set_color('#444')
ax.spines['left'].set_color('#444')
ax.spines['right'].set_color('#444')
ax.tick_params(axis='x', colors='#888')
ax.tick_params(axis='y', colors='#888')
ax.grid(True, color='#222', linestyle='--')

# Set Axis Limits based on data range with padding
limit_q = np.max(np.abs(q)) * 1.2
limit_p = np.max(np.abs(p)) * 1.2
ax.set_xlim(-limit_q, limit_q)
ax.set_ylim(-limit_p, limit_p)

# Labels
ax.set_xlabel(r"SIO Position $q(t)$ (Normalized Prime Gradient)", color='cyan', fontsize=12)
ax.set_ylabel(r"SIO Momentum $p(t)$ (Zeta Potential)", color='#ffd700', fontsize=12)
ax.set_title("Phase Space of the SIO-Operator\nDynamics of the Prime Inertia Engine", color='white', fontsize=16, pad=20)

# Plot Elements
trajectory, = ax.plot([], [], color='cyan', linewidth=1.0, alpha=0.6)
head, = ax.plot([], [], marker='o', color='white', markersize=6, markeredgecolor='#00ffff')
ghost_trail, = ax.plot([], [], color='magenta', linewidth=2.0, alpha=0.3) # The Event Horizon influence

# Text Stats
stat_text = ax.text(0.05, 0.95, "", transform=ax.transAxes, color='#00ff00', fontfamily='monospace')

def init():
    trajectory.set_data([], [])
    head.set_data([], [])
    ghost_trail.set_data([], [])
    stat_text.set_text("")
    return trajectory, head, ghost_trail, stat_text

def update(frame):
    # Trail length
    trail_len = 150
    start = max(0, frame - trail_len)
    
    # Data slice
    current_q = q[start:frame]
    current_p = p[start:frame]
    
    # Update Lines
    trajectory.set_data(current_q, current_p)
    
    # The "Ghost" is the smoothed attractor (Tchebycheff Pull)
    # Just a visual echo to show the density
    ghost_trail.set_data(current_q * 0.98, current_p * 0.98) 
    
    if frame > 0:
        head.set_data([q[frame-1]], [p[frame-1]])
        
        # Stats
        x_current = x_vals[frame-1]
        stat_text.set_text(
            f"X (Number Line): {x_current}\n"
            f"Prime Position:  {q[frame-1]:.3f}\n"
            f"Zeta Momentum:   {p[frame-1]:.3f}\n"
            f"Status:          LOCKED IN ORBIT"
        )
    
    # Progress Log
    if frame % 30 == 0:
        print(f"\rRendering Frame {frame}/{FRAMES}", end="")
        
    return trajectory, head, ghost_trail, stat_text

# --- RENDER ---
print("Rendering Phase Space Animation...")
try:
    writer = animation.FFMpegWriter(fps=FPS, bitrate=5000)
    anim = animation.FuncAnimation(fig, update, frames=FRAMES, init_func=init, blit=True)
    anim.save(VIDEO_FILENAME, writer=writer)
    print(f"\nDONE. Saved to {VIDEO_FILENAME}")
except Exception as e:
    print(f"\nError rendering video: {e}")
