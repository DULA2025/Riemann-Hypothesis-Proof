import numpy as np
import imageio.v2 as iio
from matplotlib import cm # Import matplotlib's colormap functionality

# --- Helper functions for video generation ---
def rescale01(a):
    a = np.asarray(a)
    # Clip to handle outliers and prevent pale images
    p_low, p_high = np.percentile(a, [1, 99])
    a = np.clip(a, p_low, p_high)
    return ((a - a.min()) / (a.max() - a.min() + 1e-12)).astype(np.float32)

def to_rainbow8(a01):
    """
    NEW: Converts a normalized grayscale array to an 8-bit RGB rainbow image.
    """
    a01 = np.clip(a01, 0, 1)
    # Use the 'jet' colormap for a classic rainbow effect
    # The colormap returns an (N, N, 4) RGBA array with float values
    colored_float = cm.jet(a01)
    # Convert to 8-bit RGB by taking the first 3 channels and scaling to 255
    return (colored_float[..., :3] * 255).astype(np.uint8)

def writer(path, fps):
    return iio.get_writer(path, fps=fps, codec="libx264", quality=8, macro_block_size=1, format="FFMPEG", output_params=["-pix_fmt", "yuv420p"])

# --- The local interaction operator (Laplacian) ---
def lap(A):
    """
    Calculates the local interaction field, analogous to the SIO's global operator.
    This represents the "diffusion" of prime potential.
    """
    return (-4 * A + np.roll(A, 1, 0) + np.roll(A, -1, 0) + np.roll(A, 1, 1) + np.roll(A, -1, 1))

def main(
    out="prime_physics_sio_rainbow.mov",
    N=256,
    fps=30,
    frames=3000,
    steps=8,
    Du=0.16,  # Diffusion rate for P_5
    Dv=0.08   # Diffusion rate for P_1
):
    # === PARAMETERS for a "Coral Growth" Pattern ===
    # These values are chosen to create a dynamic and visually interesting system
    # that we interpret as the evolution of prime potential fields.
    F = 0.055
    k = 0.062

    # === Initialize the Prime Potential Fields ===
    P_5 = np.ones((N, N), dtype=np.float32)
    P_1 = np.zeros((N, N), dtype=np.float32)

    # Introduce an initial perturbation to start the dynamics
    s = N // 10
    P_5[N // 2 - s:N // 2 + s, N // 2 - s:N // 2 + s] = 0.50
    P_1[N // 2 - s:N // 2 + s, N // 2 - s:N // 2 + s] = 0.25
    P_5 += 0.02 * (np.random.rand(N, N).astype(np.float32) - 0.5)
    P_1 += 0.02 * (np.random.rand(N, N).astype(np.float32) - 0.5)

    with writer(out, fps) as w:
        print(f"Simulating Prime Physics dynamics with F={F:.4f} and k={k:.4f}...")
        for frame in range(frames):
            for _ in range(steps):
                L5 = lap(P_5)
                L1 = lap(P_1)
                p5_p1_p1 = P_5 * P_1 * P_1
                P_5 += (Du * L5 - p5_p1_p1 + F * (1 - P_5)).astype(np.float32)
                P_1 += (Dv * L1 + p5_p1_p1 - (F + k) * P_1).astype(np.float32)
            
            # Use the new rainbow color function to generate the video frame
            w.append_data(to_rainbow8(rescale01(P_1)))
        print(f"Simulation complete. Video saved to '{out}'.")

if __name__ == "__main__":
    main()
