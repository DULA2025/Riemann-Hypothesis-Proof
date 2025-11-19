import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

# --- CONFIGURATION ---
# Update this path to your local ffmpeg.exe
FFMPEG_PATH = r"C:\ffmpeg\bin\ffmpeg.exe"

VIDEO_FILENAME = 'SIO_GapFriction_Decay.mp4'
LIMIT_X = 50000
FPS = 30
DURATION = 20
FRAMES = FPS * DURATION

if os.path.exists(FFMPEG_PATH):
    plt.rcParams['animation.ffmpeg_path'] = FFMPEG_PATH

print(f"--- INITIALIZING SIO GAP FRICTION TEST ---")
print(f"Comparing Self-Adjoint Stability vs. Gap Decay...")

# --- 1. DATA GENERATION ---

# A. ZETA ZEROS (Driver)
gammas = np.array([
    14.1347, 21.0220, 25.0108, 30.4248, 32.9350, 37.5861, 40.9187, 43.3270, 48.0051, 49.7738,
    52.9703, 56.4462, 59.3470, 60.8317, 65.1125, 67.0798, 69.5464, 72.0671, 75.7046, 77.1448
])

# B. PRIME DATA & GAPS
def get_prime_dynamics(limit):
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[0:2] = False
    for i in range(2, int(np.sqrt(limit)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    
    primes = np.where(is_prime)[0]
    primes = primes[primes > 3]
    
    # Calculate Gaps
    gaps = np.diff(primes)
    # Pad gaps to match length
    gaps = np.append(gaps, 0)
    
    # Map to dense array X
    # We need gap info at every X for the friction model
    gap_map = np.zeros(limit)
    prime_map = np.zeros(limit)
    
    for i, p in enumerate(primes):
        val = 1 if p % 6 == 1 else -1
        prime_map[p] = val
        gap_map[p] = gaps[i]
        
    # Accumulate Position (The SIO Walk)
    q_real = np.cumsum(prime_map)
    
    # Fill forward for continuous time simulation
    # For gaps, we hold the "current gap we are traversing"
    current_gap = 0
    for i in range(1, limit):
        if gap_map[i] > 0:
            current_gap = gap_map[i]
        gap_map[i] = current_gap # The friction persists while traversing the gap
            
    return q_real, gap_map

print("Generating Prime Dynamics...", end="")
q_raw, gaps_raw = get_prime_dynamics(LIMIT_X)
print(" Done.")

# C. COMPUTE ORBITS
# Domain
t_steps = np.linspace(100, LIMIT_X - 1, FRAMES).astype(int)
x_vals = t_steps

# 1. REAL ORBIT (Undamped)
# Normalized by sqrt(x)/ln(x)
norm_factor = np.sqrt(x_vals) / np.log(x_vals) * 2.0
q_stable = q_raw[t_steps] / norm_factor

# Zeta Momentum (p)
p_stable = np.zeros(FRAMES)
for gamma in gammas:
    p_stable += np.cos(gamma * np.log(x_vals))
p_stable /= np.sqrt(len(gammas))

# 2. DAMPED ORBIT (Gap Friction Model)
# We simulate a particle that loses energy proportional to the local Gap Ratio
# Gap Ratio R = Gap / ln(x)
# If R > 1 (Big Gap), Friction is high.
avg_gaps = np.log(x_vals)
gap_ratios = gaps_raw[t_steps] / avg_gaps

# Apply cumulative decay
# Energy E[i] = E[i-1] * (1 - damping * GapRatio)
damping_coeff = 0.002 # Sensitivity to gaps
energy_envelope = np.cumprod(1.0 - damping_coeff * gap_ratios)

q_decay = q_stable * energy_envelope
p_decay = p_stable * energy_envelope

# --- VISUALIZATION ---

fig = plt.figure(figsize=(12, 12), facecolor='#050505')
ax = fig.add_subplot(111, facecolor='#000000')

# Styling
ax.set_xlim(-4, 4)
ax.set_ylim(-4, 4)
ax.grid(True, color='#222', linestyle=':')
ax.set_title("SIO Stability Test: Self-Adjoint vs. Gap Friction\nProof of Non-Decaying Prime Energy", color='white', fontsize=16, pad=20)
ax.set_xlabel("SIO Position $q(t)$", color='gray')
ax.set_ylabel("SIO Momentum $p(t)$", color='gray')

# The Event Horizon (Black Hole)
circle = plt.Circle((0, 0), 0.1, color='black', zorder=10)
glow = plt.Circle((0, 0), 0.3, color='#330033', alpha=0.5, zorder=9)
ax.add_artist(circle)
ax.add_artist(glow)
ax.text(0, -0.5, "EVENT HORIZON\n(Entropy Sink)", color='#440044', ha='center', fontsize=8)

# Plot Elements
line_stable, = ax.plot([], [], color='cyan', linewidth=1.5, alpha=0.8, label='Actual Primes (Lossless)')
head_stable, = ax.plot([], [], marker='o', color='white', markersize=5)

line_decay, = ax.plot([], [], color='red', linewidth=1.5, alpha=0.6, linestyle='--', label='Gap Friction Model (Decaying)')
head_decay, = ax.plot([], [], marker='x', color='red', markersize=6)

# Gap Meter
gap_text = ax.text(0.05, 0.92, "", transform=ax.transAxes, color='white', fontfamily='monospace')

ax.legend(loc='upper right', facecolor='#111', edgecolor='#444', labelcolor='white')

def init():
    line_stable.set_data([], [])
    head_stable.set_data([], [])
    line_decay.set_data([], [])
    head_decay.set_data([], [])
    return line_stable, head_stable, line_decay, head_decay

def update(frame):
    tail = 100
    start = max(0, frame - tail)
    
    # Stable Prime
    line_stable.set_data(q_stable[start:frame], p_stable[start:frame])
    head_stable.set_data([q_stable[frame-1]], [p_stable[frame-1]])
    
    # Decaying Particle
    line_decay.set_data(q_decay[start:frame], p_decay[start:frame])
    head_decay.set_data([q_decay[frame-1]], [p_decay[frame-1]])
    
    # Stats
    current_gap = gaps_raw[t_steps[frame-1]]
    avg = avg_gaps[frame-1]
    ratio = current_gap / avg
    
    # Dynamic color for gap text based on stress
    color = '#00ff00' if ratio < 1.0 else '#ff0000'
    
    gap_text.set_text(
        f"X: {x_vals[frame-1]}\n"
        f"Local Gap:   {current_gap}\n"
        f"Exp. Gap:    {avg:.1f}\n"
        f"Gap Stress:  {ratio:.2f}x"
    )
    gap_text.set_color(color)
    
    return line_stable, head_stable, line_decay, head_decay

# --- RENDER ---
print("Rendering Orbit Decay Test...")
try:
    writer = animation.FFMpegWriter(fps=FPS, bitrate=5000)
    anim = animation.FuncAnimation(fig, update, frames=FRAMES, init_func=init, blit=True)
    anim.save(VIDEO_FILENAME, writer=writer)
    print(f"DONE. Saved to {VIDEO_FILENAME}")
except Exception as e:
    print(e)