from mpmath import mp, zeta, bernpoly, quad, exp, pi

# Set higher precision
mp.dps = 200

# Euler-Maclaurin Formula with B2, B4, B6, B8, B10, B12 Terms
def euler_maclaurin_zeta(s, N, include_B2=True, include_B4=True, include_B6=True, include_B8=True, include_B10=True, include_B12=False):
    """
    Approximates zeta(s) using Euler-Maclaurin formula with optional B2, B4, B6, B8, B10, B12 terms.
    s: complex number
    N: number of terms in partial sum
    include_B2, include_B4, include_B6, include_B8, include_B10, include_B12: include respective Bernoulli terms
    """
    partial_sum = sum(mp.power(n, -s) for n in range(1, N+1))
    integral_term = mp.power(N, 1-s) / (s - 1)
    constant_term = mp.mpf('0.5')
    approx = partial_sum + integral_term + constant_term
    if include_B2:
        B2 = bernpoly(2, 1)  # B_2 = 1/6
        B2_term = B2 * s / (2 * mp.power(N, s+1))
        approx += B2_term
    if include_B4:
        B4 = bernpoly(4, 1)  # B_4 = -1/30
        B4_term = B4 * s * (s+1) * (s+2) / (4 * mp.power(N, s+3))
        approx += B4_term
    if include_B6:
        B6 = bernpoly(6, 1)  # B_6 = 1/42
        B6_term = B6 * s * (s+1) * (s+2) * (s+3) * (s+4) / (6 * mp.power(N, s+5))
        approx += B6_term
    if include_B8:
        B8 = bernpoly(8, 1)  # B_8 = -1/30
        B8_term = B8 * s * (s+1) * (s+2) * (s+3) * (s+4) * (s+5) * (s+6) / (8 * mp.power(N, s+7))
        approx += B8_term
    if include_B10:
        B10 = bernpoly(10, 1)  # B_10 = 5/66
        B10_term = B10 * s * (s+1) * (s+2) * (s+3) * (s+4) * (s+5) * (s+6) * (s+7) * (s+8) / (10 * mp.power(N, s+9))
        approx += B10_term
    if include_B12:
        B12 = bernpoly(12, 1)  # B_12 = -691/2730
        B12_term = B12 * s * (s+1) * (s+2) * (s+3) * (s+4) * (s+5) * (s+6) * (s+7) * (s+8) * (s+9) * (s+10) / (12 * mp.power(N, s+11))
        approx += B12_term
    return approx

# Perron’s Formula with Direct Zeta Computation
def perron_formula(s, N, c, T, segments=32):
    """
    Computes partial sum zeta_N(s) using Perron's formula with direct zeta computation.
    s: complex number
    N: number of terms
    c: real part of integration line (c > max(1, Re(s)))
    T: truncation parameter
    segments: number of integration segments
    """
    def integrand(z):
        return zeta(s + z, derivative=0) * mp.power(N, z) * mp.power(2 * pi, -z) / z
    segment_size = 2 * T / segments
    result = mp.mpf(0)
    for i in range(segments):
        t_start = -T + i * segment_size
        t_end = t_start + segment_size
        result += quad(lambda t: integrand(c + t * 1j), [t_start, t_end], method='tanh-sinh')
    return result / (2 * pi * 1j)

# Contour Integration (unchanged)
def contour_integral_unit_circle(T, segments=1000):
    def integrand(theta):
        z = mp.exp(1j * theta)
        dz = 1j * mp.exp(1j * theta)
        return (1/z) * dz
    return quad(integrand, [0, 2 * pi], method='tanh-sinh')

# Compute and print results
print("=== Euler-Maclaurin Formula ===")
s1 = mp.mpf('2')
N1_values = [100, 1000, 10000]
for N1 in N1_values:
    true_zeta_s1 = zeta(s1)
    approx_s1_partial = sum(mp.power(n, -s1) for n in range(1, N1+1))
    approx_s1 = euler_maclaurin_zeta(s1, N1, include_B2=False, include_B4=False, include_B6=False, include_B8=False, include_B10=False, include_B12=False)
    approx_s1_B2 = euler_maclaurin_zeta(s1, N1, include_B2=True, include_B4=False, include_B6=False, include_B8=False, include_B10=False, include_B12=False)
    approx_s1_B4 = euler_maclaurin_zeta(s1, N1, include_B2=True, include_B4=True, include_B6=False, include_B8=False, include_B10=False, include_B12=False)
    approx_s1_B6 = euler_maclaurin_zeta(s1, N1, include_B2=True, include_B4=True, include_B6=True, include_B8=False, include_B10=False, include_B12=False)
    approx_s1_B8 = euler_maclaurin_zeta(s1, N1, include_B2=True, include_B4=True, include_B6=True, include_B8=True, include_B10=False, include_B12=False)
    approx_s1_B10 = euler_maclaurin_zeta(s1, N1, include_B2=True, include_B4=True, include_B6=True, include_B8=True, include_B10=True, include_B12=False)
    approx_s1_B12 = euler_maclaurin_zeta(s1, N1, include_B2=True, include_B4=True, include_B6=True, include_B8=True, include_B10=True, include_B12=True)
    print(f"s = {s1}, N = {N1}:")
    print(f"True zeta(2): {true_zeta_s1}")
    print(f"Partial sum: {approx_s1_partial} (error: {abs(true_zeta_s1 - approx_s1_partial)})")
    print(f"With integral & 1/2 terms: {approx_s1} (error: {abs(true_zeta_s1 - approx_s1)})")
    print(f"With B2 term: {approx_s1_B2} (error: {abs(true_zeta_s1 - approx_s1_B2)})")
    print(f"With B2 & B4 terms: {approx_s1_B4} (error: {abs(true_zeta_s1 - approx_s1_B4)})")
    print(f"With B2, B4, B6 terms: {approx_s1_B6} (error: {abs(true_zeta_s1 - approx_s1_B6)})")
    print(f"With B2, B4, B6, B8 terms: {approx_s1_B8} (error: {abs(true_zeta_s1 - approx_s1_B8)})")
    print(f"With B2, B4, B6, B8, B10 terms: {approx_s1_B10} (error: {abs(true_zeta_s1 - approx_s1_B10)})")
    print(f"With B2, B4, B6, B8, B10, B12 terms: {approx_s1_B12} (error: {abs(true_zeta_s1 - approx_s1_B12)})")
s2 = mp.mpf('0.5') + mp.mpf('14.134725') * 1j  # Near a non-trivial zero
N2_values = [100, 1000, 10000, 100000, 1000000, 10000000]
true_zeta_s2 = zeta(s2)
print(f"\ns = {s2}:")
print(f"True zeta(s): {true_zeta_s2}")
for N2 in N2_values:
    approx_s2 = euler_maclaurin_zeta(s2, N2, include_B2=True, include_B4=False, include_B6=False, include_B8=False, include_B10=False, include_B12=False)
    print(f"N={N2}, Approximation: {approx_s2} (error: {abs(true_zeta_s2 - approx_s2)})")
print("\n=== Perron's Formula ===")
s_perron = mp.mpf('0.6') + mp.mpf('10') * 1j
N_perron = 1000
c = mp.mpf('1.1')
T = mp.mpf('500')
perron_result = perron_formula(s_perron, N_perron, c, T, segments=32)
actual_partial_sum = sum(mp.power(n, -s_perron) for n in range(1, N_perron+1))
print(f"s = {s_perron}, N = {N_perron}, c = {c}, T = {T}, segments = 32:")
print(f"Perron's integral: {perron_result}")
print(f"Actual partial sum: {actual_partial_sum}")
print("\n=== Contour Integration (1/z over unit circle) ===")
contour_result = contour_integral_unit_circle(T=50)
expected_contour = 2 * pi * 1j
print(f"Computed integral: {contour_result}")
print(f"Expected (2πi): {expected_contour} (error: {abs(contour_result - expected_contour)})")
