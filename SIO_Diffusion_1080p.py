import numpy as np
import imageio.v2 as iio
from matplotlib import cm

# --- Helper functions for video generation ---
def rescale01(a):
    a = np.asarray(a)
    # Using percentiles is crucial for high-res to avoid a few hot pixels
    # from making the entire image look pale.
    p_low, p_high = np.percentile(a, [1, 99])
    a = np.clip(a, p_low, p_high)
    return ((a - a.min()) / (a.max() - a.min() + 1e-12)).astype(np.float32)

def to_rainbow8(a01):
    a01 = np.clip(a01, 0, 1)
    colored_float = cm.jet(a01)
    return (colored_float[..., :3] * 255).astype(np.uint8)

def writer(path, fps):
    return iio.get_writer(path, fps=fps, codec="libx264", quality=9, macro_block_size=1, format="FFMPEG", output_params=["-pix_fmt", "yuv420p"])

# --- Reverted to the stable 2D local interaction operator ---
def lap(A):
    """
    Calculates the 2D local interaction field.
    This is necessary to handle the high resolution without crashing.
    """
    return (-4 * A + np.roll(A, 1, axis=0) + np.roll(A, -1, axis=0) +
            np.roll(A, 1, axis=1) + np.roll(A, -1, axis=1))

def main(
    out="prime_physics_sio_1080p.mov",
    width=1920, # Full HD Width
    height=1080, # Full HD Height
    fps=30,
    frames=3000, # Increased for a longer video (15 seconds)
    steps=4,    # Reduced to balance performance with the larger grid
    Du=0.16,
    Dv=0.08
):
    # Parameters for a stable, evolving pattern
    F = 0.055
    k = 0.062

    # === Initialize 2D Full HD Prime Potential Fields ===
    # The grid is now (height, width)
    P_5 = np.ones((height, width), dtype=np.float32)
    P_1 = np.zeros((height, width), dtype=np.float32)

    # Introduce an initial perturbation (a central rectangle)
    s = height // 10 # Scale perturbation to the smaller dimension
    center_h_slice = slice(height // 2 - s, height // 2 + s)
    center_w_slice = slice(width // 2 - s, width // 2 + s)
    
    P_5[center_h_slice, center_w_slice] = 0.50
    P_1[center_h_slice, center_w_slice] = 0.25
    
    # Add random noise over the full grid
    P_5 += 0.02 * (np.random.rand(height, width).astype(np.float32) - 0.5)
    P_1 += 0.02 * (np.random.rand(height, width).astype(np.float32) - 0.5)

    with writer(out, fps) as w:
        print(f"Simulating {width}x{height} 2D Prime Physics (this may take a few minutes)...")
        for frame in range(frames):
            for _ in range(steps):
                L5 = lap(P_5)
                L1 = lap(P_1)
                p5_p1_p1 = P_5 * P_1 * P_1
                P_5 += (Du * L5 - p5_p1_p1 + F * (1 - P_5)).astype(np.float32)
                P_1 += (Dv * L1 + p5_p1_p1 - (F + k) * P_1).astype(np.float32)
            
            w.append_data(to_rainbow8(rescale01(P_1)))
            # Simple progress indicator
            if (frame + 1) % 10 == 0:
                print(f"  ... Frame {frame + 1} / {frames} rendered.")
                
        print(f"\nSimulation complete! Video saved to '{out}'.")

if __name__ == "__main__":
    main()
