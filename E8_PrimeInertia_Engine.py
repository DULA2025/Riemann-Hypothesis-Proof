import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from itertools import product
import time
import os

# --- CONFIGURATION ---
# IMPORTANT: Paste the path to ffmpeg.exe here if it's not in your system PATH
# Example: r"C:\ffmpeg\bin\ffmpeg.exe"
FFMPEG_PATH = r"C:\ffmpeg\bin\ffmpeg.exe" 

VIDEO_FILENAME = 'E8_PrimeInertia_Engine.mp4'
DURATION_SEC = 30
FPS = 30
TOTAL_FRAMES = DURATION_SEC * FPS

# HD Resolution settings (1920x1080)
DPI = 100
FIG_WIDTH = 1920 / DPI
FIG_HEIGHT = 1080 / DPI
COMPUTE_RES = 400 

# --- SETUP FFMPEG ---
if os.path.isfile(FFMPEG_PATH):
    plt.rcParams['animation.ffmpeg_path'] = FFMPEG_PATH
    print(f"Found FFmpeg at: {FFMPEG_PATH}")
else:
    print(f"WARNING: FFmpeg not found at {FFMPEG_PATH}")
    print("Please verify the path or add ffmpeg to your system PATH variables.")
    # Proceeding hoping it's in PATH...

print(f"--- INITIALIZING PRIME INERTIA ENGINE ---")
print(f"Target: {VIDEO_FILENAME}")
print(f"Resolution: 1920x1080 (Full HD)")
print(f"Duration: {DURATION_SEC}s ({TOTAL_FRAMES} frames)")
print("-" * 40)

# --- MATH FUNCTIONS ---

def generate_spinor_roots():
    roots = []
    for signs in product([0.5, -0.5], repeat=8):
        if np.sum(np.array(signs) < 0) % 2 == 0:
            roots.append(np.array(signs))
    return np.array(roots)

def get_coxeter_plane_projection(roots):
    u = np.array([1, np.cos(np.pi/15), np.cos(2*np.pi/15), np.cos(3*np.pi/15),
                  np.cos(4*np.pi/15), np.cos(5*np.pi/15), np.cos(6*np.pi/15), np.cos(7*np.pi/15)])
    v = np.array([0, np.sin(np.pi/15), np.sin(2*np.pi/15), np.sin(3*np.pi/15),
                  np.sin(4*np.pi/15), np.sin(5*np.pi/15), np.sin(6*np.pi/15), np.sin(7*np.pi/15)])
    v = v - np.dot(v, u) / np.dot(u, u) * u
    u = u / np.linalg.norm(u)
    v = v / np.linalg.norm(v)
    return roots @ u, roots @ v

def assign_parity_charges(roots):
    scaled = (roots * 2).astype(int)
    return np.array([1 if np.sum(r)%4 == 0 else -1 for r in scaled])

# --- PRE-CALCULATION ---

print("Generating E8 Lattice Geometry...", end="")
roots = generate_spinor_roots()
charges = assign_parity_charges(roots)
base_x, base_y = get_coxeter_plane_projection(roots)
print(" Done.")

# --- ANIMATION SETUP ---

fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT), facecolor='black', dpi=DPI)
ax = plt.axes([0, 0, 1, 1], frameon=False)
ax.set_axis_off()

limit = 12
kx = np.linspace(-limit, limit, COMPUTE_RES)
ky = np.linspace(-limit, limit, COMPUTE_RES)
KX, KY = np.meshgrid(kx, ky)

img = ax.imshow(np.zeros((COMPUTE_RES, COMPUTE_RES)), 
                cmap='inferno', 
                extent=[-limit, limit, -limit, limit],
                origin='lower',
                interpolation='bicubic',
                vmin=0, vmax=5)

# Removed letter_spacing to fix AttributeError
title_text = ax.text(0.5, 0.95, "PRIME INERTIA ENGINE // E8 SPECTRAL INTERFERENCE", 
        transform=ax.transAxes, color='white', ha='center', fontsize=24, fontweight='normal')

subtitle_text = ax.text(0.5, 0.91, "Reciprocal Space Scan", 
        transform=ax.transAxes, color='#00ffff', ha='center', fontsize=14)

twin_prime_label = ax.text(0, 0, "DESTRUCTIVE INTERFERENCE\n(TWIN PRIME VOID)", 
        color='cyan', ha='center', va='center', fontsize=12, fontweight='bold', alpha=0.8)

start_time = time.time()

def update(frame):
    t = frame / FPS
    
    # PHYSICS
    angle = t * 0.15
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    
    rot_x = base_x * cos_a - base_y * sin_a
    rot_y = base_x * sin_a + base_y * cos_a
    zoom_pulse = 1.0 + 0.1 * np.sin(t * 0.5)
    
    # DIFFRACTION
    Amplitude = np.zeros_like(KX, dtype=complex)
    scaled_x = rot_x * zoom_pulse
    scaled_y = rot_y * zoom_pulse
    
    # Core Loop (Optimized)
    for x, y, q in zip(scaled_x, scaled_y, charges):
        phase_term = 1j * (KX * x + KY * y)
        Amplitude += q * np.exp(phase_term)
        
    Intensity = np.abs(Amplitude)**2
    LogIntensity = np.log1p(Intensity)
    
    img.set_data(LogIntensity)
    subtitle_text.set_text(f"Mod 6 Spinor Diffraction | Phase Angle: {np.degrees(angle):.1f}°")
    
    # Progress Log
    elapsed = time.time() - start_time
    if frame > 0:
        eta = (elapsed / frame) * (TOTAL_FRAMES - frame)
        print(f"\rRendering Frame {frame}/{TOTAL_FRAMES} - ETA: {eta:.0f}s   ", end="")
    
    return img, title_text, subtitle_text, twin_prime_label

# --- RENDER ---

print("\nStarting FFmpeg render...")

try:
    writer = animation.FFMpegWriter(fps=FPS, metadata=dict(artist='DULA_Grok'), bitrate=8000)
    anim = animation.FuncAnimation(fig, update, frames=TOTAL_FRAMES, blit=False)
    anim.save(VIDEO_FILENAME, writer=writer)
    print(f"\n\nSUCCESS. Video saved as: {VIDEO_FILENAME}")
except FileNotFoundError:
    print("\n\nERROR: FFmpeg still not found.")
    print("Please set the FFMPEG_PATH variable at the top of the script to the location of ffmpeg.exe")
except Exception as e:
    print(f"\n\nAn error occurred: {e}")
