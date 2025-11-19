## 🧬 The E8 Prime Inertia Engine (V2)

**Status:** `Live` | **Type:** `Quantum-Arithmetic Simulation`

This script (`E8_Prime_Inertia_Engine_V2.py`) is the computational core of the **DULA Program**. It simulates the spectral properties of the prime number distribution by embedding them into the $E_8$ root lattice.

### **Key Mechanisms:**
1.  **The Modulo 6 Spinor Sieve:**
    - The code segregates the 240 roots of $E_8$ into "Vector" (Structural) and "Spinor" (Prime) channels.
    - It assigns a "Parity Charge" ($q = \pm 1$) based on the modular class ($p \equiv 1, 5 \pmod 6$).

2.  **Spectral Diffraction (The "Beam"):**
    - We compute the Structure Factor $I(\mathbf{k}) = |\sum q_j e^{i\mathbf{k}\cdot\mathbf{r}_j}|^2$.
    - **Result:** The output reveals "Bragg Peaks" (Constructive Interference / Sexy Primes) and **"Dark Voids" (Destructive Interference / Twin Primes)**.

3.  **The "Event Horizon" Dynamics:**
    - The simulation introduces a "Breathing Mode" and 8D Rotation.
    - It visualizes the **Twin Prime Void** as a topologically protected "Event Horizon" that traps the excess energy of the Split primes, preventing the lattice from diverging.

### **How to Run:**
```bash
python E8_Prime_Inertia_Engine_V2.py
