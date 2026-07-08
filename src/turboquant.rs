//! **TurboQuant** — online (data-oblivious) vector quantization with
//! near-optimal distortion rate, after Zandieh, Daliri, Hadian & Mirrokni,
//! *"TurboQuant: Online Vector Quantization with Near-optimal Distortion
//! Rate"* (arXiv:2504.19874).
//!
//! Two quantizers are provided:
//!
//! - [`TurboQuantMse`] (Algorithm 1): rotates the input by a random orthogonal
//!   matrix `Π`, which makes every coordinate of a unit vector follow a
//!   (scaled/shifted) Beta distribution that converges to `N(0, 1/d)` in high
//!   dimension, then applies the *optimal* Lloyd-Max scalar quantizer for that
//!   distribution to each coordinate independently. Its MSE is within a
//!   `√3π/2 ≈ 2.72` factor of the information-theoretic lower bound `4^-b` at
//!   every bit-width `b`.
//! - [`TurboQuantProd`] (Algorithm 2): MSE-optimal quantizers are *biased* for
//!   inner products (at 1 bit the bias factor is `2/π`), so the inner-product
//!   quantizer spends `b-1` bits on [`TurboQuantMse`] and 1 bit per coordinate
//!   on a Quantized Johnson-Lindenstrauss (QJL) sketch of the residual:
//!   `sign(S·r)` for a Gaussian matrix `S`, dequantized as
//!   `√(π/2)/D · ‖r‖ · Sᵀ·sign(S·r)`. The resulting inner-product estimator is
//!   exactly unbiased with distortion `≤ √3π²/d · 4^-b · ‖y‖²`.
//!
//! Both are *online*: no training pass over the data, no codebook learning —
//! a vector is quantized the moment it arrives, which is why indexing time is
//! essentially zero compared to product quantization.
//!
//! Deviations from the paper, chosen for practicality and noted here:
//!
//! - The random rotation `Π` is a 3-round randomized Hadamard transform
//!   (`H·D₃·H·D₂·H·D₁`, with `Dᵢ` random ±1 diagonals) instead of the QR
//!   decomposition of a Gaussian matrix. It is exactly orthogonal, needs no
//!   `d×d` storage, runs in `O(d log d)`, and is the standard
//!   accelerator-friendly substitute the paper's goals call for. Inputs are
//!   zero-padded to the next power of two.
//! - Vectors are not assumed unit-norm: the L2 norm is stored exactly
//!   (one f32) and the unit direction is quantized, exactly as the paper
//!   suggests for datasets that are not normalized.
//!
//! [`TurboQuantIndex`] wraps [`TurboQuantProd`] into a flat (brute-force scan)
//! ANN index with the same surface as the IVF backend: quantized codes only,
//! asymmetric distance estimation (`O(D²)` once per query to sketch the query,
//! then `O(D)` per stored vector), and the usual smaller-is-closer distance
//! convention for the `Cos` / `Ip` / `L2sq` metrics.

use parking_lot::RwLock;
use rustc_hash::FxHashMap;

use crate::ivf::Metric;

// ── deterministic RNG (no external deps) ───────────────────────────────────

/// SplitMix64: tiny, fast, seedable, and good enough for rotation signs and
/// Gaussian projection entries. Deterministic across platforms.
struct SplitMix64(u64);

impl SplitMix64 {
    const fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in the open interval (0, 1).
    fn next_unit(&mut self) -> f64 {
        (((self.next_u64() >> 11) as f64) + 0.5) * (1.0 / (1u64 << 53) as f64)
    }

    /// Standard normal via Box-Muller.
    fn next_gaussian(&mut self) -> f64 {
        let u1 = self.next_unit();
        let u2 = self.next_unit();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
}

/// Derive an independent sub-stream seed (used so the rotation, the QJL
/// projection, and each projection row draw from unrelated streams).
const fn mix_seed(seed: u64, salt: u64) -> u64 {
    // One SplitMix64 finalization round over seed ^ rotated salt.
    let mut z = seed ^ salt.rotate_left(17) ^ 0xA076_1D64_78BD_642F;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

// ── Gaussian math for the Lloyd-Max codebook ────────────────────────────────

const SQRT_2PI: f64 = 2.506_628_274_631_000_5;

/// Error function to near machine precision: Maclaurin series on `|x| < 2.5`,
/// the Laplace continued fraction for `erfc` (Abramowitz & Stegun 7.1.14)
/// beyond. High accuracy matters here: the optimal 8-bit codebook's MSE
/// (~4.1e-5) sits only ~1% under the distortion bound the tests verify, so an
/// erf error of the usual 1e-7-approximation order would be visible.
fn erf(x: f64) -> f64 {
    let ax = x.abs();
    let v = if ax < 2.5 {
        // erf(x) = 2/√π · Σ (−1)ⁿ x^{2n+1} / (n!·(2n+1))
        let mut term = ax;
        let mut sum = ax;
        let mut n = 1.0f64;
        loop {
            term *= -(ax * ax) / n;
            let add = term / (2.0 * n + 1.0);
            sum += add;
            // `<=` so an exactly-zero term terminates (erf(0): 0 ≤ 0); with
            // `<` the comparison 0 < 0 never fires and the loop spins forever.
            if add.abs() <= 1e-17 * sum.abs() {
                break;
            }
            n += 1.0;
        }
        sum * 2.0 / std::f64::consts::PI.sqrt()
    } else {
        // erfc(x) = e^{−x²}/√π · 1/(x + (1/2)/(x + 1/(x + (3/2)/(x + …))))
        let mut f = 0.0f64;
        for n in (1..=64u32).rev() {
            f = (f64::from(n) / 2.0) / (ax + f);
        }
        1.0 - (-ax * ax).exp() / std::f64::consts::PI.sqrt() / (ax + f)
    };
    if x < 0.0 { -v } else { v }
}

/// Standard normal density φ(x).
fn phi(x: f64) -> f64 {
    (-0.5 * x * x).exp() / SQRT_2PI
}

/// Standard normal CDF Φ(x). Handles ±∞.
fn big_phi(x: f64) -> f64 {
    if x == f64::NEG_INFINITY {
        return 0.0;
    }
    if x == f64::INFINITY {
        return 1.0;
    }
    0.5 * (1.0 + erf(x / std::f64::consts::SQRT_2))
}

/// `x·φ(x)` extended by its limit (0) at ±∞.
fn xphi(x: f64) -> f64 {
    if x.is_finite() { x * phi(x) } else { 0.0 }
}

/// Optimal (Lloyd-Max / 1-D k-means) codebook of `2^bits` centroids for the
/// standard normal distribution, ascending. This solves the continuous k-means
/// problem of Eq. (4) in the paper for the high-dimensional limit of the
/// coordinate distribution; centroids are later rescaled by `1/√D` to match
/// `N(0, 1/D)` coordinates. `bits = 0` yields the empty codebook (decode 0).
///
/// Reference values (Max, 1960): `b=1 → ±0.7979`, `b=2 → ±0.4528, ±1.5104`.
pub fn lloyd_max_gaussian(bits: u32) -> Vec<f64> {
    assert!(bits <= 8, "bits must be in 0..=8");
    static CODEBOOKS: [std::sync::OnceLock<Vec<f64>>; 9] =
        [const { std::sync::OnceLock::new() }; 9];
    CODEBOOKS[bits as usize]
        .get_or_init(|| lloyd_max_gaussian_uncached(bits))
        .clone()
}

fn lloyd_max_gaussian_uncached(bits: u32) -> Vec<f64> {
    if bits == 0 {
        return Vec::new();
    }
    let k = 1usize << bits;
    // Quantile init (the asymptotically optimal companding placement), found
    // by bisection on the CDF — it starts Lloyd close to the fixed point.
    let mut c: Vec<f64> = (0..k)
        .map(|i| {
            let target = (i as f64 + 0.5) / k as f64;
            let (mut lo, mut hi) = (-10.0f64, 10.0f64);
            for _ in 0..80 {
                let mid = 0.5 * (lo + hi);
                if big_phi(mid) < target {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            0.5 * (lo + hi)
        })
        .collect();

    // Over-relaxed Lloyd iterations. Plain Lloyd's convergence factor
    // approaches 1 as k grows — at k=256 it needs ~20x more rounds than
    // ω = 1.9 to reach the same excess MSE (<0.3% here). The density is
    // log-concave, so the fixed point is the unique optimum.
    const OMEGA: f64 = 1.9;
    let mut t = vec![0.0f64; k + 1];
    let mut means = vec![0.0f64; k];
    // Lloyd's linear convergence factor worsens with k, so the iteration
    // budget scales with it: k=256 needs ~30k rounds to land the ~1% below
    // the √3π/2·4⁻⁸ bound the tests verify, while k≤16 exits in hundreds
    // via the movement threshold. Computed once per bit-width (OnceLock).
    for _ in 0..(200 * k.max(16)) {
        t[0] = f64::NEG_INFINITY;
        t[k] = f64::INFINITY;
        for i in 1..k {
            t[i] = 0.5 * (c[i - 1] + c[i]);
        }
        let mut max_move = 0.0f64;
        for i in 0..k {
            let mass = big_phi(t[i + 1]) - big_phi(t[i]);
            means[i] = if mass > 0.0 {
                (phi_at(t[i]) - phi_at(t[i + 1])) / mass
            } else {
                c[i]
            };
            max_move = max_move.max((means[i] - c[i]).abs());
        }
        let relaxed: Vec<f64> = c
            .iter()
            .zip(&means)
            .map(|(&ci, &mi)| ci + OMEGA * (mi - ci))
            .collect();
        // Interval means are always ascending, so falling back to the plain
        // Lloyd step keeps the codebook monotone if over-relaxation overshot.
        if relaxed.windows(2).all(|w| w[0] < w[1]) {
            c = relaxed;
        } else {
            c.copy_from_slice(&means);
        }
        if max_move < 1e-12 {
            break;
        }
    }
    c
}

/// φ extended by its limit (0) at ±∞.
fn phi_at(x: f64) -> f64 {
    if x.is_finite() { phi(x) } else { 0.0 }
}

/// Exact MSE `E[(x − c(x))²]` of a scalar quantizer with centroids `c`
/// (ascending) under the standard normal — the cost `C(f_X, b)` of Eq. (4).
/// Reference values: 0.3634, 0.1175, 0.03454, 0.009497 for `b = 1..=4`.
pub fn gaussian_quantizer_mse(c: &[f64]) -> f64 {
    if c.is_empty() {
        return 1.0; // decode-to-zero: E[x²] = 1
    }
    let k = c.len();
    let mut total = 0.0;
    for i in 0..k {
        let a = if i == 0 {
            f64::NEG_INFINITY
        } else {
            0.5 * (c[i - 1] + c[i])
        };
        let b = if i == k - 1 {
            f64::INFINITY
        } else {
            0.5 * (c[i] + c[i + 1])
        };
        let mass = big_phi(b) - big_phi(a);
        // ∫(x-c)²φ over [a,b] = (1+c²)·mass − (bφ(b)−aφ(a)) − 2c(φ(a)−φ(b))
        total += (1.0 + c[i] * c[i]) * mass - (xphi(b) - xphi(a))
            - 2.0 * c[i] * (phi_at(a) - phi_at(b));
    }
    total
}

// ── random rotation (3-round randomized Hadamard) ───────────────────────────

/// In-place normalized fast Walsh-Hadamard transform. `v.len()` must be a
/// power of two. Orthogonal and self-inverse.
fn fwht(v: &mut [f32]) {
    let n = v.len();
    debug_assert!(n.is_power_of_two());
    let mut h = 1;
    while h < n {
        let mut i = 0;
        while i < n {
            for j in i..i + h {
                let x = v[j];
                let y = v[j + h];
                v[j] = x + y;
                v[j + h] = x - y;
            }
            i += h * 2;
        }
        h *= 2;
    }
    let s = 1.0 / (n as f32).sqrt();
    for x in v.iter_mut() {
        *x *= s;
    }
}

const ROTATION_ROUNDS: usize = 3;

/// A seeded random orthogonal transform `Π = H·D₃·H·D₂·H·D₁` on `R^D`
/// (`D` = input dimension padded to the next power of two), with `H` the
/// normalized Hadamard matrix and `Dᵢ` random ±1 diagonals. Exactly
/// norm-preserving; concentrates each coordinate of a rotated unit vector
/// around `N(0, 1/D)`, which is what the per-coordinate codebook assumes.
pub struct RandomRotation {
    dim: usize,
    padded: usize,
    /// ±1.0 per round, each of length `padded`.
    signs: Vec<Vec<f32>>,
}

impl RandomRotation {
    pub fn new(dim: usize, seed: u64) -> Self {
        assert!(dim > 0, "dimension must be positive");
        let padded = dim.next_power_of_two();
        let mut rng = SplitMix64::new(mix_seed(seed, 0x0517));
        let signs = (0..ROTATION_ROUNDS)
            .map(|_| {
                (0..padded)
                    .map(|_| if rng.next_u64() & 1 == 0 { 1.0 } else { -1.0 })
                    .collect()
            })
            .collect();
        Self { dim, padded, signs }
    }

    pub const fn dim(&self) -> usize {
        self.dim
    }

    /// The power-of-two dimension the rotation (and all codes) operate in.
    pub const fn padded_dim(&self) -> usize {
        self.padded
    }

    /// `Π·x`, zero-padding `x` to the padded dimension.
    pub fn forward(&self, x: &[f32]) -> Vec<f32> {
        debug_assert_eq!(x.len(), self.dim);
        let mut buf = vec![0.0f32; self.padded];
        buf[..x.len()].copy_from_slice(x);
        for signs in &self.signs {
            for (v, s) in buf.iter_mut().zip(signs) {
                *v *= s;
            }
            fwht(&mut buf);
        }
        buf
    }

    /// `Πᵀ·y`, truncated back to the original dimension.
    pub fn inverse(&self, y: &[f32]) -> Vec<f32> {
        debug_assert_eq!(y.len(), self.padded);
        let mut buf = y.to_vec();
        for signs in self.signs.iter().rev() {
            fwht(&mut buf);
            for (v, s) in buf.iter_mut().zip(signs) {
                *v *= s;
            }
        }
        buf.truncate(self.dim);
        buf
    }
}

// ── bit packing ─────────────────────────────────────────────────────────────

/// Bytes needed for `count` fields of `bits` bits each.
const fn packed_len(count: usize, bits: u32) -> usize {
    (count * bits as usize).div_ceil(8)
}

/// Pack `bits`-wide little-endian fields; `indices[j]` occupies bit range
/// `[j·bits, (j+1)·bits)`.
fn pack_indices(indices: &[u16], bits: u32, out: &mut Vec<u8>) {
    out.clear();
    out.resize(packed_len(indices.len(), bits), 0);
    for (j, &idx) in indices.iter().enumerate() {
        let bit = j * bits as usize;
        let (byte, off) = (bit / 8, (bit % 8) as u32);
        // bits ≤ 8, so a field spans at most two bytes.
        let v = idx << off;
        out[byte] |= v as u8;
        if off + bits > 8 {
            out[byte + 1] |= (v >> 8) as u8;
        }
    }
}

/// Read field `j` from a packed buffer.
fn unpack_index(packed: &[u8], j: usize, bits: u32) -> u16 {
    let bit = j * bits as usize;
    let (byte, off) = (bit / 8, (bit % 8) as u32);
    let lo = u16::from(packed[byte]);
    let hi = if off + bits > 8 {
        u16::from(packed[byte + 1]) << 8
    } else {
        0
    };
    ((lo | hi) >> off) & ((1u16 << bits) - 1)
}

// ── TurboQuant-MSE (Algorithm 1) ────────────────────────────────────────────

/// Quantized form produced by [`TurboQuantMse`]: `b`-bit centroid indices for
/// each rotated coordinate plus the exact L2 norm of the input.
#[derive(Clone, Debug, PartialEq)]
pub struct MseCode {
    /// Packed `b`-bit codebook indices, one per padded coordinate.
    pub idx: Vec<u8>,
    /// Exact L2 norm of the original vector.
    pub norm: f32,
}

/// A query prepared for asymmetric scoring against [`MseCode`]s: rotating the
/// query once (`O(D log D)`) makes each subsequent estimate `O(D)`.
pub struct MseQuery {
    /// `Π·q` (query in the rotated basis, un-normalized).
    q_rot: Vec<f32>,
}

/// MSE-optimal TurboQuant (Algorithm 1): `Quant(x) = nearest-centroid indices
/// of `Π·x/‖x‖``, `DeQuant = Πᵀ·centroids·‖x‖`.
pub struct TurboQuantMse {
    rotation: RandomRotation,
    bits: u32,
    /// Ascending centroids scaled by `1/√D` (i.e. for `N(0, 1/D)` coordinates).
    centroids: Vec<f32>,
    /// Voronoi boundaries: midpoints of consecutive centroids.
    boundaries: Vec<f32>,
}

impl TurboQuantMse {
    /// `bits` per coordinate in `1..=8` (use [`TurboQuantProd`] with
    /// `bits = 1` if you want the 0-bit degenerate case, where it is QJL).
    pub fn new(dim: usize, bits: u32, seed: u64) -> Self {
        assert!((1..=8).contains(&bits), "bits must be in 1..=8");
        Self::with_bits_allow_zero(dim, bits, seed)
    }

    /// Internal constructor that also allows `bits = 0` (empty codebook,
    /// decode-to-zero) — the MSE stage of a 1-bit [`TurboQuantProd`].
    fn with_bits_allow_zero(dim: usize, bits: u32, seed: u64) -> Self {
        assert!(bits <= 8, "bits must be in 0..=8");
        let rotation = RandomRotation::new(dim, seed);
        let scale = 1.0 / (rotation.padded_dim() as f64).sqrt();
        let centroids: Vec<f32> = lloyd_max_gaussian(bits)
            .into_iter()
            .map(|c| (c * scale) as f32)
            .collect();
        let boundaries = centroids
            .windows(2)
            .map(|w| 0.5 * (w[0] + w[1]))
            .collect();
        Self {
            rotation,
            bits,
            centroids,
            boundaries,
        }
    }

    pub const fn dim(&self) -> usize {
        self.rotation.dim()
    }

    pub const fn padded_dim(&self) -> usize {
        self.rotation.padded_dim()
    }

    pub const fn bits(&self) -> u32 {
        self.bits
    }

    /// Packed code size in bytes (norm not included).
    pub const fn code_len(&self) -> usize {
        packed_len(self.rotation.padded_dim(), self.bits)
    }

    /// Nearest-centroid index for one rotated coordinate.
    #[inline]
    fn quantize_coord(&self, y: f32) -> u16 {
        self.boundaries.partition_point(|&t| t < y) as u16
    }

    /// Quantize an already-rotated unit-norm vector into packed indices.
    fn quantize_rotated(&self, y: &[f32], out: &mut Vec<u8>) {
        if self.bits == 0 {
            out.clear();
            return;
        }
        let indices: Vec<u16> = y.iter().map(|&v| self.quantize_coord(v)).collect();
        pack_indices(&indices, self.bits, out);
    }

    /// Reconstruct the rotated unit-norm vector from packed indices.
    fn dequantize_rotated(&self, idx: &[u8]) -> Vec<f32> {
        let d = self.padded_dim();
        if self.bits == 0 {
            return vec![0.0; d];
        }
        (0..d)
            .map(|j| self.centroids[unpack_index(idx, j, self.bits) as usize])
            .collect()
    }

    /// `Quant_mse(x)`: rotate, quantize each coordinate to its nearest
    /// centroid, and record the exact norm.
    pub fn encode(&self, x: &[f32]) -> MseCode {
        assert_eq!(x.len(), self.dim(), "dimension mismatch");
        let norm = l2_norm(x);
        let mut idx = Vec::new();
        if norm > 0.0 {
            let unit: Vec<f32> = x.iter().map(|v| v / norm).collect();
            let y = self.rotation.forward(&unit);
            self.quantize_rotated(&y, &mut idx);
        } else {
            self.quantize_rotated(&vec![0.0; self.padded_dim()], &mut idx);
        }
        MseCode { idx, norm }
    }

    /// `DeQuant_mse`: centroids back through `Πᵀ`, rescaled by the stored norm.
    pub fn decode(&self, code: &MseCode) -> Vec<f32> {
        if code.norm == 0.0 {
            return vec![0.0; self.dim()];
        }
        let y = self.dequantize_rotated(&code.idx);
        let mut x = self.rotation.inverse(&y);
        for v in &mut x {
            *v *= code.norm;
        }
        x
    }

    /// Rotate a query once so it can be scored against many codes.
    pub fn prepare_query(&self, q: &[f32]) -> MseQuery {
        assert_eq!(q.len(), self.dim(), "dimension mismatch");
        MseQuery {
            q_rot: self.rotation.forward(q),
        }
    }

    /// Estimated `⟨q, x⟩` from the code alone (asymmetric: the query stays in
    /// full precision). Equals `⟨q, DeQuant(code)⟩` up to float rounding.
    pub fn dot(&self, q: &MseQuery, code: &MseCode) -> f32 {
        if code.norm == 0.0 || self.bits == 0 {
            return 0.0;
        }
        let mut acc = 0.0f32;
        for (j, &qv) in q.q_rot.iter().enumerate() {
            acc += qv * self.centroids[unpack_index(&code.idx, j, self.bits) as usize];
        }
        acc * code.norm
    }
}

// ── TurboQuant-Prod (Algorithm 2): MSE(b-1) + 1-bit QJL on the residual ────

/// Quantized form produced by [`TurboQuantProd`].
#[derive(Clone, Debug, PartialEq)]
pub struct ProdCode {
    /// Packed `(b-1)`-bit MSE indices (empty when `b = 1`).
    pub idx: Vec<u8>,
    /// Packed QJL sign bits of `S·r`, one per padded coordinate (bit set ⇔ +1).
    pub qjl: Vec<u8>,
    /// `‖r‖`: L2 norm of the residual of the *unit-norm* rotated vector.
    pub residual_norm: f32,
    /// Exact L2 norm of the original vector.
    pub norm: f32,
}

/// A query prepared for asymmetric scoring against [`ProdCode`]s. Building it
/// costs one rotation plus one `D×D` mat-vec (`S·q`); each estimate after that
/// is `O(D)`.
pub struct ProdQuery {
    /// `Π·q`.
    q_rot: Vec<f32>,
    /// `S·Π·q` — so that `⟨Π·q, Sᵀ·z⟩ = ⟨S·Π·q, z⟩` needs only a sign-sum.
    sq: Vec<f32>,
    /// Asymmetric-distance LUT: `mse_lut[j·K + c] = q_rot[j]·centroids[c]`
    /// (`K = 2^bits` centroids), so a scan does one load+add per coordinate
    /// instead of unpack+gather+multiply. Only built while it fits in L1
    /// (`K ≤ 8`, i.e. ≤ 64 KiB at D = 2048); empty otherwise, and the scan
    /// falls back to multiplying against `q_rot` directly.
    mse_lut: Vec<f32>,
}

/// Inner-product-optimal TurboQuant (Algorithm 2). Unbiased:
/// `E[⟨y, DeQuant(Quant(x))⟩] = ⟨y, x⟩` exactly, with near-optimal distortion
/// `≤ √3π²·4^{1-b}/d · ‖x‖²‖y‖²` — the estimator nearest-neighbor search wants.
pub struct TurboQuantProd {
    mse: TurboQuantMse,
    bits: u32,
    /// QJL projection `S`: `D×D` i.i.d. `N(0,1)`, row-major. Applied to the
    /// residual in the rotated basis (equivalent to the paper's original-basis
    /// formulation because `S·Πᵀ` is again i.i.d. Gaussian).
    proj: Vec<f32>,
    /// `√(π/2)/D` — the QJL dequantization constant.
    kappa: f32,
}

impl TurboQuantProd {
    /// `bits` per coordinate in `1..=8`; `bits - 1` go to the MSE stage and
    /// one to the QJL sign sketch (`bits = 1` is pure QJL).
    pub fn new(dim: usize, bits: u32, seed: u64) -> Self {
        assert!((1..=8).contains(&bits), "bits must be in 1..=8");
        let mse = TurboQuantMse::with_bits_allow_zero(dim, bits - 1, seed);
        let d = mse.padded_dim();
        let mut rng = SplitMix64::new(mix_seed(seed, 0x0951));
        let proj = (0..d * d).map(|_| rng.next_gaussian() as f32).collect();
        Self {
            mse,
            bits,
            proj,
            kappa: (std::f64::consts::PI / 2.0).sqrt() as f32 / d as f32,
        }
    }

    pub const fn dim(&self) -> usize {
        self.mse.dim()
    }

    pub const fn padded_dim(&self) -> usize {
        self.mse.padded_dim()
    }

    pub const fn bits(&self) -> u32 {
        self.bits
    }

    /// Packed byte sizes `(mse_idx, qjl_signs)` of one code (norms excluded).
    pub const fn code_len(&self) -> (usize, usize) {
        (self.mse.code_len(), packed_len(self.mse.padded_dim(), 1))
    }

    #[inline]
    fn proj_row(&self, i: usize) -> &[f32] {
        let d = self.padded_dim();
        &self.proj[i * d..(i + 1) * d]
    }

    /// `Quant_prod(x)`: MSE-quantize at `b-1` bits, then QJL-sketch the
    /// residual `r = y − ỹ` and record `‖r‖` (and the input norm).
    pub fn encode(&self, x: &[f32]) -> ProdCode {
        assert_eq!(x.len(), self.dim(), "dimension mismatch");
        let d = self.padded_dim();
        let norm = l2_norm(x);
        let (mut idx, mut qjl) = (Vec::new(), vec![0u8; packed_len(d, 1)]);
        if norm == 0.0 {
            self.mse.quantize_rotated(&vec![0.0; d], &mut idx);
            return ProdCode {
                idx,
                qjl,
                residual_norm: 0.0,
                norm: 0.0,
            };
        }

        let unit: Vec<f32> = x.iter().map(|v| v / norm).collect();
        let y = self.mse.rotation.forward(&unit);
        self.mse.quantize_rotated(&y, &mut idx);
        let y_hat = self.mse.dequantize_rotated(&idx);
        let r: Vec<f32> = y.iter().zip(&y_hat).map(|(a, b)| a - b).collect();
        let residual_norm = l2_norm(&r);

        if residual_norm > 0.0 {
            for i in 0..d {
                if dot(self.proj_row(i), &r) >= 0.0 {
                    qjl[i / 8] |= 1 << (i % 8);
                }
            }
        }
        ProdCode {
            idx,
            qjl,
            residual_norm,
            norm,
        }
    }

    /// `DeQuant_prod`: `Πᵀ·(ỹ + √(π/2)/D·‖r‖·Sᵀ·sign_bits)·‖x‖`.
    pub fn decode(&self, code: &ProdCode) -> Vec<f32> {
        if code.norm == 0.0 {
            return vec![0.0; self.dim()];
        }
        let d = self.padded_dim();
        let mut y = self.mse.dequantize_rotated(&code.idx);
        if code.residual_norm > 0.0 {
            let scale = self.kappa * code.residual_norm;
            for i in 0..d {
                let s = if code.qjl[i / 8] >> (i % 8) & 1 == 1 {
                    scale
                } else {
                    -scale
                };
                axpy(&mut y, s, self.proj_row(i));
            }
        }
        let mut x = self.mse.rotation.inverse(&y);
        for v in &mut x {
            *v *= code.norm;
        }
        x
    }

    /// Rotate and sketch a query once so it can be scored against many codes.
    pub fn prepare_query(&self, q: &[f32]) -> ProdQuery {
        assert_eq!(q.len(), self.dim(), "dimension mismatch");
        let q_rot = self.mse.rotation.forward(q);
        let d = self.padded_dim();
        let sq = (0..d).map(|i| dot(self.proj_row(i), &q_rot)).collect();
        let k = 1usize << self.mse.bits;
        let mse_lut = if self.mse.bits > 0 && k <= 8 {
            let mut lut = Vec::with_capacity(d * k);
            for &qv in &q_rot {
                for &c in &self.mse.centroids {
                    lut.push(qv * c);
                }
            }
            lut
        } else {
            Vec::new()
        };
        ProdQuery { q_rot, sq, mse_lut }
    }

    /// Unbiased estimate of `⟨q, x⟩` from the code alone:
    /// `(⟨Π·q, ỹ⟩ + √(π/2)/D·‖r‖·⟨S·Π·q, signs⟩)·‖x‖`.
    pub fn dot(&self, q: &ProdQuery, code: &ProdCode) -> f32 {
        self.score(q, &code.idx, &code.qjl, code.residual_norm, code.norm)
    }

    /// The scan kernel behind [`Self::dot`] and [`TurboQuantIndex::search`]:
    /// scores one packed code (borrowed slices, so flat index storage needs no
    /// copies) against a prepared query.
    fn score(&self, q: &ProdQuery, idx: &[u8], qjl: &[u8], residual_norm: f32, norm: f32) -> f32 {
        if norm == 0.0 {
            return 0.0;
        }
        let bits = self.mse.bits;
        let mut acc = 0.0f32;
        if bits > 0 {
            if q.mse_lut.is_empty() {
                for (j, &qv) in q.q_rot.iter().enumerate() {
                    acc += qv * self.mse.centroids[unpack_index(idx, j, bits) as usize];
                }
            } else {
                // ADC fast path: one L1 load + add per coordinate. The bit
                // cursor walks the packed fields without per-step multiplies;
                // a field spans at most two bytes (bits ≤ 8). Four
                // accumulators break the float-add latency chain, which is
                // what bounds this loop, not throughput.
                let mask = (1u16 << bits) - 1;
                let step = bits as usize;
                let extract = |bit: usize| -> usize {
                    let byte = bit >> 3;
                    let w = if byte + 1 < idx.len() {
                        u16::from_le_bytes([idx[byte], idx[byte + 1]])
                    } else {
                        u16::from(idx[byte])
                    };
                    usize::from((w >> (bit & 7)) & mask)
                };
                let d = q.q_rot.len();
                let lut = &q.mse_lut;
                let (mut a0, mut a1, mut a2, mut a3) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
                let mut bit = 0usize;
                let mut j = 0usize;
                while j + 4 <= d {
                    a0 += lut[(j << bits) + extract(bit)];
                    a1 += lut[((j + 1) << bits) + extract(bit + step)];
                    a2 += lut[((j + 2) << bits) + extract(bit + 2 * step)];
                    a3 += lut[((j + 3) << bits) + extract(bit + 3 * step)];
                    bit += 4 * step;
                    j += 4;
                }
                while j < d {
                    a0 += lut[(j << bits) + extract(bit)];
                    bit += step;
                    j += 1;
                }
                acc += (a0 + a1) + (a2 + a3);
            }
        }
        if residual_norm > 0.0 {
            acc += self.kappa * residual_norm * qjl_sign_dot(&q.sq, qjl);
        }
        acc * norm
    }
}

// ── SIMD kernels ────────────────────────────────────────────────────────────
//
// The QJL projection makes `dot` and `axpy` the only O(D²) inner loops in the
// crate (encode and prepare_query do a D×D mat-vec through `dot`; decode does
// the transpose through `axpy`), so they get explicit SIMD:
//
// - aarch64 (Apple Silicon / ARM64 Linux): NEON, guaranteed by the ABI.
// - x86_64 (Intel macOS / AMD64 Linux): AVX2+FMA when the CPU has it
//   (runtime-detected, cached by `is_x86_feature_detected!`), else SSE2,
//   which is baseline for the architecture.
// - anything else: an 8-lane unrolled scalar loop the compiler can
//   auto-vectorize.
//
// Lane sums are accumulated in independent registers, so results can differ
// from a naive sequential sum (and between platforms) by normal f32
// reassociation error — every consumer already tolerates that.

/// `Σ a[i]·b[i]`. Slices must be the same length.
#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { dot_neon(a, b) }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            unsafe { dot_avx2(a, b) }
        } else {
            unsafe { dot_sse2(a, b) }
        }
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        dot_scalar(a, b)
    }
}

/// `y[i] += s·p[i]`. Slices must be the same length.
#[inline]
fn axpy(y: &mut [f32], s: f32, p: &[f32]) {
    debug_assert_eq!(y.len(), p.len());
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { axpy_neon(y, s, p) }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            unsafe { axpy_avx2(y, s, p) }
        } else {
            unsafe { axpy_sse2(y, s, p) }
        }
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        axpy_scalar(y, s, p)
    }
}

/// `Σᵢ (bitᵢ ? +sq[i] : −sq[i])` over packed sign bits — the QJL half of every
/// scan estimate. Branchless by design: a cleared bit flips the float's sign
/// via XOR on bit 31, so there is no data-dependent branch (which mispredicts
/// ~50% of the time on QJL's random bits). SIMD applies the flips four lanes
/// at a time using a 16-entry nibble→sign-mask table; SSE2 is x86-64 baseline,
/// so no runtime dispatch is needed.
#[inline]
fn qjl_sign_dot(sq: &[f32], bits: &[u8]) -> f32 {
    let d = sq.len();
    let full = d / 8;
    let mut sum;
    #[cfg(target_arch = "aarch64")]
    {
        sum = unsafe { qjl_sign_dot_neon(sq, bits, full) };
    }
    #[cfg(target_arch = "x86_64")]
    {
        sum = unsafe { qjl_sign_dot_sse2(sq, bits, full) };
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        sum = qjl_sign_dot_scalar(&sq[..full * 8], bits);
    }
    // Tail (padded dims < 8 only): the trailing bits of the last byte belong
    // to zero-padded coordinates and must not be read.
    for i in full * 8..d {
        let mask = (u32::from(bits[i >> 3] >> (i & 7) & 1) ^ 1) << 31;
        sum += f32::from_bits(sq[i].to_bits() ^ mask);
    }
    sum
}

/// Lane `l` of entry `v`: `0` when bit `l` of `v` is set (keep sign), `1<<31`
/// when cleared (flip sign).
static QJL_NIBBLE_MASKS: [[u32; 4]; 16] = qjl_nibble_masks();

const fn qjl_nibble_masks() -> [[u32; 4]; 16] {
    let mut t = [[0u32; 4]; 16];
    let mut v = 0;
    while v < 16 {
        let mut l = 0;
        while l < 4 {
            t[v][l] = (((v as u32 >> l) & 1) ^ 1) << 31;
            l += 1;
        }
        v += 1;
    }
    t
}

#[cfg_attr(any(target_arch = "aarch64", target_arch = "x86_64"), allow(dead_code))]
#[inline]
fn qjl_sign_dot_scalar(sq: &[f32], bits: &[u8]) -> f32 {
    let mut sum = 0.0f32;
    for (i, &s) in sq.iter().enumerate() {
        let mask = (u32::from(bits[i >> 3] >> (i & 7) & 1) ^ 1) << 31;
        sum += f32::from_bits(s.to_bits() ^ mask);
    }
    sum
}

#[cfg(target_arch = "aarch64")]
#[allow(unsafe_op_in_unsafe_fn)] // body is one unsafe region of intrinsics
unsafe fn qjl_sign_dot_neon(sq: &[f32], bits: &[u8], full_bytes: usize) -> f32 {
    use std::arch::aarch64::*;
    let p = sq.as_ptr();
    let mut acc = vdupq_n_f32(0.0);
    for (b, &byte) in bits[..full_bytes].iter().enumerate() {
        let i = b * 8;
        let m0 = vld1q_u32(QJL_NIBBLE_MASKS[usize::from(byte & 15)].as_ptr());
        let m1 = vld1q_u32(QJL_NIBBLE_MASKS[usize::from(byte >> 4)].as_ptr());
        let v0 = veorq_u32(vreinterpretq_u32_f32(vld1q_f32(p.add(i))), m0);
        let v1 = veorq_u32(vreinterpretq_u32_f32(vld1q_f32(p.add(i + 4))), m1);
        acc = vaddq_f32(acc, vreinterpretq_f32_u32(v0));
        acc = vaddq_f32(acc, vreinterpretq_f32_u32(v1));
    }
    vaddvq_f32(acc)
}

#[cfg(target_arch = "x86_64")]
#[allow(unsafe_op_in_unsafe_fn)] // body is one unsafe region of intrinsics
unsafe fn qjl_sign_dot_sse2(sq: &[f32], bits: &[u8], full_bytes: usize) -> f32 {
    use std::arch::x86_64::*;
    let p = sq.as_ptr();
    let mut acc = _mm_setzero_ps();
    for (b, &byte) in bits[..full_bytes].iter().enumerate() {
        let i = b * 8;
        let m0 = _mm_loadu_ps(QJL_NIBBLE_MASKS[usize::from(byte & 15)].as_ptr().cast());
        let m1 = _mm_loadu_ps(QJL_NIBBLE_MASKS[usize::from(byte >> 4)].as_ptr().cast());
        acc = _mm_add_ps(acc, _mm_xor_ps(_mm_loadu_ps(p.add(i)), m0));
        acc = _mm_add_ps(acc, _mm_xor_ps(_mm_loadu_ps(p.add(i + 4)), m1));
    }
    let s2 = _mm_add_ps(acc, _mm_movehl_ps(acc, acc));
    let s1 = _mm_add_ss(s2, _mm_shuffle_ps(s2, s2, 1));
    _mm_cvtss_f32(s1)
}

#[cfg_attr(any(target_arch = "aarch64", target_arch = "x86_64"), allow(dead_code))]
#[inline]
fn dot_scalar(a: &[f32], b: &[f32]) -> f32 {
    // 8 independent accumulators: breaks the sequential dependency chain so
    // the loop auto-vectorizes / pipelines at opt-level ≥ 1.
    let mut acc = [0.0f32; 8];
    let chunks = a.len() / 8;
    for c in 0..chunks {
        let (x, y) = (&a[c * 8..c * 8 + 8], &b[c * 8..c * 8 + 8]);
        for l in 0..8 {
            acc[l] += x[l] * y[l];
        }
    }
    let mut sum = (acc[0] + acc[1]) + (acc[2] + acc[3]) + ((acc[4] + acc[5]) + (acc[6] + acc[7]));
    for i in chunks * 8..a.len() {
        sum += a[i] * b[i];
    }
    sum
}

#[cfg_attr(any(target_arch = "aarch64", target_arch = "x86_64"), allow(dead_code))]
#[inline]
fn axpy_scalar(y: &mut [f32], s: f32, p: &[f32]) {
    for (v, x) in y.iter_mut().zip(p) {
        *v += s * x;
    }
}

#[cfg(target_arch = "aarch64")]
#[allow(unsafe_op_in_unsafe_fn)] // body is one unsafe region of intrinsics
unsafe fn dot_neon(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::aarch64::*;
    let n = a.len();
    let (pa, pb) = (a.as_ptr(), b.as_ptr());
    let (mut a0, mut a1, mut a2, mut a3) = (
        vdupq_n_f32(0.0),
        vdupq_n_f32(0.0),
        vdupq_n_f32(0.0),
        vdupq_n_f32(0.0),
    );
    let mut i = 0usize;
    while i + 16 <= n {
        a0 = vfmaq_f32(a0, vld1q_f32(pa.add(i)), vld1q_f32(pb.add(i)));
        a1 = vfmaq_f32(a1, vld1q_f32(pa.add(i + 4)), vld1q_f32(pb.add(i + 4)));
        a2 = vfmaq_f32(a2, vld1q_f32(pa.add(i + 8)), vld1q_f32(pb.add(i + 8)));
        a3 = vfmaq_f32(a3, vld1q_f32(pa.add(i + 12)), vld1q_f32(pb.add(i + 12)));
        i += 16;
    }
    let mut acc = vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3));
    while i + 4 <= n {
        acc = vfmaq_f32(acc, vld1q_f32(pa.add(i)), vld1q_f32(pb.add(i)));
        i += 4;
    }
    let mut sum = vaddvq_f32(acc);
    while i < n {
        sum += a[i] * b[i];
        i += 1;
    }
    sum
}

#[cfg(target_arch = "aarch64")]
#[allow(unsafe_op_in_unsafe_fn)] // body is one unsafe region of intrinsics
unsafe fn axpy_neon(y: &mut [f32], s: f32, p: &[f32]) {
    use std::arch::aarch64::*;
    let n = y.len();
    let py = y.as_mut_ptr();
    let pp = p.as_ptr();
    let vs = vdupq_n_f32(s);
    let mut i = 0usize;
    while i + 4 <= n {
        vst1q_f32(py.add(i), vfmaq_f32(vld1q_f32(py.add(i)), vs, vld1q_f32(pp.add(i))));
        i += 4;
    }
    while i < n {
        y[i] += s * p[i];
        i += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
#[allow(unsafe_op_in_unsafe_fn)] // body is one unsafe region of intrinsics
unsafe fn dot_avx2(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let n = a.len();
    let (pa, pb) = (a.as_ptr(), b.as_ptr());
    let mut a0 = _mm256_setzero_ps();
    let mut a1 = _mm256_setzero_ps();
    let mut i = 0usize;
    while i + 16 <= n {
        a0 = _mm256_fmadd_ps(_mm256_loadu_ps(pa.add(i)), _mm256_loadu_ps(pb.add(i)), a0);
        a1 = _mm256_fmadd_ps(_mm256_loadu_ps(pa.add(i + 8)), _mm256_loadu_ps(pb.add(i + 8)), a1);
        i += 16;
    }
    while i + 8 <= n {
        a0 = _mm256_fmadd_ps(_mm256_loadu_ps(pa.add(i)), _mm256_loadu_ps(pb.add(i)), a0);
        i += 8;
    }
    let acc = _mm256_add_ps(a0, a1);
    let s4 = _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1));
    let s2 = _mm_add_ps(s4, _mm_movehl_ps(s4, s4));
    let s1 = _mm_add_ss(s2, _mm_shuffle_ps(s2, s2, 1));
    let mut sum = _mm_cvtss_f32(s1);
    while i < n {
        sum += a[i] * b[i];
        i += 1;
    }
    sum
}

#[cfg(target_arch = "x86_64")]
#[allow(unsafe_op_in_unsafe_fn)] // body is one unsafe region of intrinsics
unsafe fn dot_sse2(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let n = a.len();
    let (pa, pb) = (a.as_ptr(), b.as_ptr());
    let mut a0 = _mm_setzero_ps();
    let mut a1 = _mm_setzero_ps();
    let mut i = 0usize;
    while i + 8 <= n {
        a0 = _mm_add_ps(a0, _mm_mul_ps(_mm_loadu_ps(pa.add(i)), _mm_loadu_ps(pb.add(i))));
        a1 = _mm_add_ps(
            a1,
            _mm_mul_ps(_mm_loadu_ps(pa.add(i + 4)), _mm_loadu_ps(pb.add(i + 4))),
        );
        i += 8;
    }
    let acc = _mm_add_ps(a0, a1);
    let s2 = _mm_add_ps(acc, _mm_movehl_ps(acc, acc));
    let s1 = _mm_add_ss(s2, _mm_shuffle_ps(s2, s2, 1));
    let mut sum = _mm_cvtss_f32(s1);
    while i < n {
        sum += a[i] * b[i];
        i += 1;
    }
    sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
#[allow(unsafe_op_in_unsafe_fn)] // body is one unsafe region of intrinsics
unsafe fn axpy_avx2(y: &mut [f32], s: f32, p: &[f32]) {
    use std::arch::x86_64::*;
    let n = y.len();
    let py = y.as_mut_ptr();
    let pp = p.as_ptr();
    let vs = _mm256_set1_ps(s);
    let mut i = 0usize;
    while i + 8 <= n {
        _mm256_storeu_ps(
            py.add(i),
            _mm256_fmadd_ps(vs, _mm256_loadu_ps(pp.add(i)), _mm256_loadu_ps(py.add(i))),
        );
        i += 8;
    }
    while i < n {
        y[i] += s * p[i];
        i += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[allow(unsafe_op_in_unsafe_fn)] // body is one unsafe region of intrinsics
unsafe fn axpy_sse2(y: &mut [f32], s: f32, p: &[f32]) {
    use std::arch::x86_64::*;
    let n = y.len();
    let py = y.as_mut_ptr();
    let pp = p.as_ptr();
    let vs = _mm_set1_ps(s);
    let mut i = 0usize;
    while i + 4 <= n {
        _mm_storeu_ps(
            py.add(i),
            _mm_add_ps(_mm_loadu_ps(py.add(i)), _mm_mul_ps(vs, _mm_loadu_ps(pp.add(i)))),
        );
        i += 4;
    }
    while i < n {
        y[i] += s * p[i];
        i += 1;
    }
}

#[inline]
fn l2_norm(a: &[f32]) -> f32 {
    dot(a, a).sqrt()
}

// ── flat quantized index over TurboQuant-Prod codes ─────────────────────────

struct IndexInner {
    /// Entity ids, parallel to the per-row arrays below.
    ids: Vec<u64>,
    /// Flat packed codes, `code_stride` bytes per row (MSE indices ++ QJL bits).
    codes: Vec<u8>,
    /// Exact vector norm per row.
    norms: Vec<f32>,
    /// Residual norm `‖r‖` per row.
    residual_norms: Vec<f32>,
    /// `id -> row position` for O(1) upsert/remove.
    id_pos: FxHashMap<u64, usize>,
}

/// A flat ANN index that stores only TurboQuant-Prod codes (`b` bits per padded
/// coordinate + 8 bytes of norms per vector) and searches by exhaustive
/// asymmetric scan: `O(D²)` once to sketch the query, then `O(D)` per stored
/// vector. Distances follow usearch's smaller-is-closer convention, so it is
/// interchangeable with the HNSW/IVF backends. Quantization is data-oblivious:
/// there is nothing to train, and codes never need rebuilding.
pub struct TurboQuantIndex {
    dims: usize,
    metric: Metric,
    quant: TurboQuantProd,
    /// Bytes per row in `codes`.
    code_stride: usize,
    /// Split point inside a row: `[0, idx_len)` MSE indices, the rest QJL bits.
    idx_len: usize,
    inner: RwLock<IndexInner>,
}

impl TurboQuantIndex {
    /// `bits` is clamped to `1..=8` (papers' sweet spot for NN search is 2-4).
    pub fn new(dims: usize, metric: Metric, bits: u32, seed: u64) -> Self {
        let bits = bits.clamp(1, 8);
        let quant = TurboQuantProd::new(dims, bits, seed);
        let (idx_len, qjl_len) = quant.code_len();
        Self {
            dims,
            metric,
            quant,
            code_stride: idx_len + qjl_len,
            idx_len,
            inner: RwLock::new(IndexInner {
                ids: Vec::new(),
                codes: Vec::new(),
                norms: Vec::new(),
                residual_norms: Vec::new(),
                id_pos: FxHashMap::default(),
            }),
        }
    }

    pub fn len(&self) -> usize {
        self.inner.read().ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub const fn metric(&self) -> Metric {
        self.metric
    }

    pub const fn bits(&self) -> u32 {
        self.quant.bits()
    }

    /// The quantizer behind the index (for direct encode/decode access).
    pub const fn quantizer(&self) -> &TurboQuantProd {
        &self.quant
    }

    /// Approximate resident bytes: packed codes + norms + bookkeeping + the
    /// one-off `D×D` QJL projection and rotation signs.
    pub fn memory_bytes(&self) -> usize {
        let g = self.inner.read();
        let fixed = self.quant.proj.len() * 4
            + self.quant.mse.rotation.signs.iter().map(|s| s.len() * 4).sum::<usize>()
            + self.quant.mse.centroids.len() * 4;
        fixed
            + g.codes.len()
            + (g.norms.len() + g.residual_norms.len()) * 4
            + g.ids.len() * 8
            + g.id_pos.len() * 16
    }

    /// Insert or replace the vector for `id`. Returns `true` if it replaced an
    /// existing entry. The vector itself is not retained — only its code.
    pub fn upsert(&self, id: u64, v: &[f32]) -> Result<bool, String> {
        if v.len() != self.dims {
            return Err(format!(
                "dimension mismatch: got {}, expected {}",
                v.len(),
                self.dims
            ));
        }
        let code = self.quant.encode(v);
        let mut g = self.inner.write();
        let existed = g.id_pos.contains_key(&id);
        if existed {
            remove_row(&mut g, id, self.code_stride);
        }
        let pos = g.ids.len();
        g.ids.push(id);
        g.codes.extend_from_slice(&code.idx);
        g.codes.extend_from_slice(&code.qjl);
        debug_assert_eq!(g.codes.len(), (pos + 1) * self.code_stride);
        g.norms.push(code.norm);
        g.residual_norms.push(code.residual_norm);
        g.id_pos.insert(id, pos);
        Ok(existed)
    }

    /// Remove the vector for `id`. Returns `true` if it existed.
    pub fn remove(&self, id: u64) -> bool {
        let mut g = self.inner.write();
        if !g.id_pos.contains_key(&id) {
            return false;
        }
        remove_row(&mut g, id, self.code_stride);
        true
    }

    /// Return the `top_k` nearest ids with their distances (ascending),
    /// estimated from the quantized codes.
    pub fn search(&self, query: &[f32], top_k: usize) -> Result<Vec<(u64, f32)>, String> {
        if query.len() != self.dims {
            return Err(format!(
                "dimension mismatch: got {}, expected {}",
                query.len(),
                self.dims
            ));
        }
        if top_k == 0 {
            return Ok(Vec::new());
        }
        let g = self.inner.read();
        if g.ids.is_empty() {
            return Ok(Vec::new());
        }
        let prepared = self.quant.prepare_query(query);
        let q_sq = dot(query, query);
        let q_norm = q_sq.sqrt();

        let mut scored: Vec<(u64, f32)> = (0..g.ids.len())
            .map(|pos| {
                let row = &g.codes[pos * self.code_stride..(pos + 1) * self.code_stride];
                let est_ip = self.quant.score(
                    &prepared,
                    &row[..self.idx_len],
                    &row[self.idx_len..],
                    g.residual_norms[pos],
                    g.norms[pos],
                );
                let vnorm = g.norms[pos];
                let dist = match self.metric {
                    Metric::Cos => {
                        let denom = q_norm * vnorm;
                        if denom == 0.0 {
                            1.0
                        } else {
                            1.0 - est_ip / denom
                        }
                    }
                    Metric::Ip => 1.0 - est_ip,
                    Metric::L2sq => q_sq + vnorm * vnorm - 2.0 * est_ip,
                };
                (g.ids[pos], dist)
            })
            .collect();
        let cmp =
            |a: &(u64, f32), b: &(u64, f32)| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal);
        // Partial selection: O(n) to isolate the top_k, then sort only those.
        if scored.len() > top_k {
            scored.select_nth_unstable_by(top_k - 1, cmp);
            scored.truncate(top_k);
        }
        scored.sort_unstable_by(cmp);
        Ok(scored)
    }
}

/// Remove `id` from a locked [`IndexInner`] via swap-remove, keeping all
/// parallel arrays and `id_pos` consistent. Caller guarantees `id` is present.
fn remove_row(g: &mut IndexInner, id: u64, stride: usize) {
    let pos = g.id_pos[&id];
    let last = g.ids.len() - 1;
    if pos != last {
        let moved_id = g.ids[last];
        g.ids.swap_remove(pos);
        g.norms.swap_remove(pos);
        g.residual_norms.swap_remove(pos);
        let (head, tail) = g.codes.split_at_mut(last * stride);
        head[pos * stride..(pos + 1) * stride].copy_from_slice(&tail[..stride]);
        g.id_pos.insert(moved_id, pos);
    } else {
        g.ids.pop();
        g.norms.pop();
        g.residual_norms.pop();
    }
    g.codes.truncate(last * stride);
    g.id_pos.remove(&id);
}

#[cfg(test)]
mod tests {
    use super::*;

    const SEED: u64 = 0xC0FFEE;

    fn rand_unit(dim: usize, rng: &mut SplitMix64) -> Vec<f32> {
        let mut v: Vec<f32> = (0..dim).map(|_| rng.next_gaussian() as f32).collect();
        let n = l2_norm(&v);
        for x in &mut v {
            *x /= n;
        }
        v
    }

    // ── Gaussian math + Lloyd-Max codebook ─────────────────────────────

    #[test]
    fn erf_matches_known_values() {
        // Reference values to 13+ digits, exercising both branches (series
        // below 2.5, continued fraction above).
        assert!(erf(0.0).abs() < 1e-15);
        assert!((erf(0.5) - 0.520_499_877_813_046_5).abs() < 1e-12);
        assert!((erf(1.0) - 0.842_700_792_949_714_9).abs() < 1e-12);
        assert!((erf(2.0) - 0.995_322_265_018_952_7).abs() < 1e-12);
        assert!((erf(3.0) - 0.999_977_909_503_001_4).abs() < 1e-12);
        assert!((erf(4.0) - 0.999_999_984_582_742_1).abs() < 1e-13);
        assert!((erf(-1.5) + erf(1.5)).abs() < 1e-15);
    }

    #[test]
    fn normal_cdf_endpoints_and_median() {
        assert_eq!(big_phi(f64::NEG_INFINITY), 0.0);
        assert_eq!(big_phi(f64::INFINITY), 1.0);
        assert!((big_phi(0.0) - 0.5).abs() < 1e-12);
        // Φ(-5) = 2.8665157e-7: the far tail must stay accurate — it weights
        // the outermost cells of the 8-bit codebook.
        assert!((big_phi(-5.0) - 2.866_515_719e-7).abs() < 1e-13);
        assert!((big_phi(1.96) - 0.975_002_104_851_78).abs() < 1e-10);
    }

    #[test]
    fn lloyd_max_1bit_matches_paper() {
        // Paper §3.1: optimal 1-bit centroids are ±√(2/π) ≈ ±0.7979.
        let c = lloyd_max_gaussian(1);
        assert_eq!(c.len(), 2);
        let expect = (2.0 / std::f64::consts::PI).sqrt();
        assert!((c[0] + expect).abs() < 1e-4, "got {c:?}");
        assert!((c[1] - expect).abs() < 1e-4, "got {c:?}");
    }

    #[test]
    fn lloyd_max_2bit_matches_paper() {
        // Paper §3.1: optimal 2-bit centroids are ±0.453, ±1.51.
        let c = lloyd_max_gaussian(2);
        assert_eq!(c.len(), 4);
        assert!((c[0] + 1.510).abs() < 2e-3, "got {c:?}");
        assert!((c[1] + 0.4528).abs() < 2e-3, "got {c:?}");
        assert!((c[2] - 0.4528).abs() < 2e-3, "got {c:?}");
        assert!((c[3] - 1.510).abs() < 2e-3, "got {c:?}");
    }

    #[test]
    fn lloyd_max_codebooks_sorted_and_symmetric() {
        for bits in 1..=8u32 {
            let c = lloyd_max_gaussian(bits);
            assert_eq!(c.len(), 1 << bits);
            for w in c.windows(2) {
                assert!(w[0] < w[1], "b={bits}: not ascending: {w:?}");
            }
            let k = c.len();
            for i in 0..k {
                assert!(
                    (c[i] + c[k - 1 - i]).abs() < 1e-6,
                    "b={bits}: not symmetric at {i}: {} vs {}",
                    c[i],
                    c[k - 1 - i]
                );
            }
        }
    }

    #[test]
    fn codebook_mse_matches_paper_distortions() {
        // Theorem 1: C(f_X, b) ≈ 0.36, 0.117, 0.03(45), 0.009(5) for b = 1..4
        // (classic Lloyd-Max distortions of the standard normal).
        let expect = [0.3634, 0.1175, 0.03454, 0.009497];
        for (b, &e) in (1..=4u32).zip(&expect) {
            let mse = gaussian_quantizer_mse(&lloyd_max_gaussian(b));
            assert!(
                (mse - e).abs() / e < 0.01,
                "b={b}: mse {mse} vs expected {e}"
            );
        }
    }

    #[test]
    fn codebook_mse_beats_shannon_lower_bound() {
        // Theorem 3 lower bound: no b-bit quantizer beats 4^-b. The optimal
        // scalar codebook must sit between that and the √3π/2·4^-b upper bound.
        for b in 1..=8u32 {
            let mse = gaussian_quantizer_mse(&lloyd_max_gaussian(b));
            let lower = 0.25f64.powi(b as i32);
            let upper = 3.0f64.sqrt() * std::f64::consts::PI / 2.0 * lower;
            assert!(mse > lower, "b={b}: {mse} below Shannon bound {lower}");
            assert!(mse < upper, "b={b}: {mse} above paper bound {upper}");
        }
    }

    #[test]
    fn zero_bit_codebook_is_empty_and_unit_mse() {
        assert!(lloyd_max_gaussian(0).is_empty());
        assert_eq!(gaussian_quantizer_mse(&[]), 1.0);
    }

    // ── RNG ────────────────────────────────────────────────────────────

    #[test]
    fn rng_is_deterministic_and_seed_sensitive() {
        let a: Vec<u64> = {
            let mut r = SplitMix64::new(7);
            (0..8).map(|_| r.next_u64()).collect()
        };
        let b: Vec<u64> = {
            let mut r = SplitMix64::new(7);
            (0..8).map(|_| r.next_u64()).collect()
        };
        let c: Vec<u64> = {
            let mut r = SplitMix64::new(8);
            (0..8).map(|_| r.next_u64()).collect()
        };
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn rng_gaussian_moments() {
        let mut r = SplitMix64::new(42);
        let n = 50_000;
        let (mut sum, mut sum2) = (0.0f64, 0.0f64);
        for _ in 0..n {
            let g = r.next_gaussian();
            sum += g;
            sum2 += g * g;
        }
        let mean = sum / n as f64;
        let var = sum2 / n as f64 - mean * mean;
        assert!(mean.abs() < 0.02, "gaussian mean {mean}");
        assert!((var - 1.0).abs() < 0.03, "gaussian variance {var}");
    }

    // ── SIMD kernels ───────────────────────────────────────────────────

    /// Naive sequential f64 reference the SIMD paths are checked against.
    fn dot_ref(a: &[f32], b: &[f32]) -> f64 {
        a.iter()
            .zip(b)
            .map(|(&x, &y)| f64::from(x) * f64::from(y))
            .sum()
    }

    #[test]
    fn simd_dot_matches_reference_all_remainders() {
        // Lengths straddle every unroll boundary (16-wide body, 4/8-wide
        // tail, scalar remainder) so each code path is exercised.
        let mut rng = SplitMix64::new(0x51D);
        for n in [0usize, 1, 3, 4, 5, 7, 8, 15, 16, 17, 31, 33, 64, 100, 257] {
            let a: Vec<f32> = (0..n).map(|_| rng.next_gaussian() as f32).collect();
            let b: Vec<f32> = (0..n).map(|_| rng.next_gaussian() as f32).collect();
            let got = f64::from(dot(&a, &b));
            let want = dot_ref(&a, &b);
            assert!(
                (got - want).abs() <= 1e-4 * want.abs().max(1.0),
                "n={n}: simd dot {got} vs reference {want}"
            );
            let scalar = f64::from(dot_scalar(&a, &b));
            assert!(
                (scalar - want).abs() <= 1e-4 * want.abs().max(1.0),
                "n={n}: scalar-fallback dot {scalar} vs reference {want}"
            );
        }
    }

    #[test]
    fn simd_axpy_matches_reference_all_remainders() {
        let mut rng = SplitMix64::new(0xA9);
        for n in [0usize, 1, 3, 4, 7, 8, 9, 16, 31, 100] {
            let p: Vec<f32> = (0..n).map(|_| rng.next_gaussian() as f32).collect();
            let mut y: Vec<f32> = (0..n).map(|_| rng.next_gaussian() as f32).collect();
            let mut want = y.clone();
            let s = 0.37f32;
            axpy_scalar(&mut want, s, &p);
            axpy(&mut y, s, &p);
            for (i, (g, w)) in y.iter().zip(&want).enumerate() {
                assert!(
                    (g - w).abs() <= 1e-5 * w.abs().max(1.0),
                    "n={n} i={i}: simd axpy {g} vs scalar {w}"
                );
            }
        }
    }

    #[test]
    fn qjl_sign_dot_matches_branchy_reference() {
        let mut rng = SplitMix64::new(0x9B1);
        // Covers: multiple of 8 (SIMD body only), sub-8 padded dims (pure
        // tail), and byte patterns 0x00/0xFF/random.
        for d in [1usize, 2, 4, 8, 16, 64, 200] {
            let sq: Vec<f32> = (0..d).map(|_| rng.next_gaussian() as f32).collect();
            for pattern in [0x00u8, 0xFF, 0xA7, 0x31] {
                let mut bits = vec![pattern; d.div_ceil(8)];
                for b in &mut bits {
                    *b ^= rng.next_u64() as u8;
                }
                let want: f32 = sq
                    .iter()
                    .enumerate()
                    .map(|(i, &s)| if bits[i / 8] >> (i % 8) & 1 == 1 { s } else { -s })
                    .sum();
                let got = qjl_sign_dot(&sq, &bits);
                assert!(
                    (got - want).abs() < 1e-4 * want.abs().max(1.0),
                    "d={d} pattern={pattern:#x}: {got} vs {want}"
                );
            }
        }
    }

    #[test]
    fn score_lut_path_matches_multiply_path() {
        // bits 2..=4 build the LUT (K ≤ 8); bits 5+ take the multiply path;
        // both must agree with the decoded dot product.
        let mut rng = SplitMix64::new(0x1DE);
        let dim = 96;
        let x: Vec<f32> = (0..dim).map(|_| rng.next_gaussian() as f32).collect();
        let y: Vec<f32> = (0..dim).map(|_| rng.next_gaussian() as f32).collect();
        for bits in [2u32, 3, 4, 5, 8] {
            let q = TurboQuantProd::new(dim, bits, 3 + u64::from(bits));
            let prepared = q.prepare_query(&y);
            assert_eq!(
                prepared.mse_lut.is_empty(),
                bits > 4,
                "LUT presence rule changed for bits={bits}"
            );
            let code = q.encode(&x);
            let est = q.dot(&prepared, &code);
            let explicit = dot(&y, &q.decode(&code));
            assert!(
                (est - explicit).abs() < 1e-2 * explicit.abs().max(1.0),
                "bits={bits}: score {est} vs decoded dot {explicit}"
            );
        }
    }

    // ── rotation ───────────────────────────────────────────────────────

    #[test]
    fn fwht_is_orthonormal_and_self_inverse() {
        let mut v = vec![3.0, -1.0, 2.0, 0.5, -2.5, 4.0, 0.0, 1.0];
        let orig = v.clone();
        let n0 = l2_norm(&v);
        fwht(&mut v);
        assert!((l2_norm(&v) - n0).abs() < 1e-5, "norm not preserved");
        fwht(&mut v);
        for (a, b) in v.iter().zip(&orig) {
            assert!((a - b).abs() < 1e-5, "not self-inverse: {v:?} vs {orig:?}");
        }
    }

    #[test]
    fn rotation_preserves_norms_and_inner_products() {
        let mut rng = SplitMix64::new(SEED);
        for dim in [1usize, 2, 5, 48, 100, 128] {
            let rot = RandomRotation::new(dim, SEED);
            assert_eq!(rot.padded_dim(), dim.next_power_of_two());
            let x: Vec<f32> = (0..dim).map(|_| rng.next_gaussian() as f32).collect();
            let y: Vec<f32> = (0..dim).map(|_| rng.next_gaussian() as f32).collect();
            let (rx, ry) = (rot.forward(&x), rot.forward(&y));
            assert!(
                (l2_norm(&rx) - l2_norm(&x)).abs() < 1e-4 * l2_norm(&x).max(1.0),
                "d={dim}: norm changed"
            );
            assert!(
                (dot(&rx, &ry) - dot(&x, &y)).abs() < 1e-3 * (l2_norm(&x) * l2_norm(&y)).max(1.0),
                "d={dim}: inner product changed"
            );
        }
    }

    #[test]
    fn rotation_roundtrips_exactly() {
        let mut rng = SplitMix64::new(SEED);
        for dim in [1usize, 3, 37, 64, 200] {
            let rot = RandomRotation::new(dim, SEED);
            let x: Vec<f32> = (0..dim).map(|_| rng.next_gaussian() as f32).collect();
            let back = rot.inverse(&rot.forward(&x));
            assert_eq!(back.len(), dim);
            for (a, b) in back.iter().zip(&x) {
                assert!((a - b).abs() < 1e-4, "d={dim}: roundtrip off: {a} vs {b}");
            }
        }
    }

    #[test]
    fn rotation_is_seed_deterministic() {
        let x: Vec<f32> = (0..33).map(|i| (i as f32).sin()).collect();
        let a = RandomRotation::new(33, 5).forward(&x);
        let b = RandomRotation::new(33, 5).forward(&x);
        let c = RandomRotation::new(33, 6).forward(&x);
        assert_eq!(a, b, "same seed must give the same rotation");
        assert_ne!(a, c, "different seeds must give different rotations");
    }

    #[test]
    fn rotation_spreads_a_basis_vector() {
        // The worst input for per-coordinate quantization is a spike; the
        // rotation must smear it so no coordinate dominates.
        let dim = 256;
        let rot = RandomRotation::new(dim, SEED);
        let mut e0 = vec![0.0f32; dim];
        e0[0] = 1.0;
        let y = rot.forward(&e0);
        let max = y.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        // A perfectly Gaussian coordinate profile would put the max around
        // √(2 ln D / D) ≈ 0.21 for D=256; allow generous slack.
        assert!(max < 0.4, "rotation left a spike: max |coord| = {max}");
    }

    // ── bit packing ────────────────────────────────────────────────────

    #[test]
    fn pack_unpack_roundtrip_all_widths() {
        let mut rng = SplitMix64::new(3);
        for bits in 1..=8u32 {
            for count in [1usize, 7, 8, 9, 64, 129] {
                let indices: Vec<u16> = (0..count)
                    .map(|_| (rng.next_u64() % (1 << bits)) as u16)
                    .collect();
                let mut packed = Vec::new();
                pack_indices(&indices, bits, &mut packed);
                assert_eq!(packed.len(), packed_len(count, bits));
                for (j, &want) in indices.iter().enumerate() {
                    assert_eq!(
                        unpack_index(&packed, j, bits),
                        want,
                        "b={bits} count={count} j={j}"
                    );
                }
            }
        }
    }

    #[test]
    fn packed_len_is_tight() {
        assert_eq!(packed_len(8, 1), 1);
        assert_eq!(packed_len(9, 1), 2);
        assert_eq!(packed_len(4, 2), 1);
        assert_eq!(packed_len(3, 3), 2);
        assert_eq!(packed_len(128, 4), 64);
        assert_eq!(packed_len(0, 5), 0);
    }

    // ── TurboQuant-MSE ─────────────────────────────────────────────────

    #[test]
    fn mse_rejects_bad_bits_and_dims() {
        let q = TurboQuantMse::new(8, 4, SEED);
        assert_eq!(q.dim(), 8);
        assert_eq!(q.bits(), 4);
        let r = std::panic::catch_unwind(|| TurboQuantMse::new(8, 0, SEED));
        assert!(r.is_err(), "bits=0 must be rejected publicly");
        let r = std::panic::catch_unwind(|| TurboQuantMse::new(8, 9, SEED));
        assert!(r.is_err(), "bits=9 must be rejected");
    }

    #[test]
    fn mse_code_size_matches_bit_budget() {
        for bits in 1..=8u32 {
            let q = TurboQuantMse::new(100, bits, SEED); // pads to 128
            assert_eq!(q.code_len(), (128 * bits as usize).div_ceil(8));
        }
    }

    #[test]
    fn mse_encode_decode_preserves_norm_exactly() {
        let mut rng = SplitMix64::new(SEED);
        let q = TurboQuantMse::new(64, 4, SEED);
        let x: Vec<f32> = (0..64).map(|_| 3.5 * rng.next_gaussian() as f32).collect();
        let code = q.encode(&x);
        assert!((code.norm - l2_norm(&x)).abs() < 1e-4);
        let x_hat = q.decode(&code);
        assert_eq!(x_hat.len(), 64);
        // The decoded direction is approximate but the codebook keeps the
        // rotated unit vector's coordinates near their centroids, so the
        // reconstruction's norm stays within the quantizer's MSE budget.
        let rel = (l2_norm(&x_hat) - code.norm).abs() / code.norm;
        assert!(rel < 0.15, "decoded norm off by {rel}");
    }

    #[test]
    fn mse_zero_vector_roundtrips_to_zero() {
        let q = TurboQuantMse::new(16, 3, SEED);
        let code = q.encode(&vec![0.0; 16]);
        assert_eq!(code.norm, 0.0);
        assert!(q.decode(&code).iter().all(|&v| v == 0.0));
        let prep = q.prepare_query(&vec![1.0; 16]);
        assert_eq!(q.dot(&prep, &code), 0.0);
    }

    #[test]
    fn mse_reconstruction_improves_with_bits() {
        let mut rng = SplitMix64::new(SEED);
        let x = rand_unit(128, &mut rng);
        let mut prev_err = f32::INFINITY;
        for bits in 1..=6u32 {
            let q = TurboQuantMse::new(128, bits, SEED);
            let x_hat = q.decode(&q.encode(&x));
            let err: f32 = x
                .iter()
                .zip(&x_hat)
                .map(|(a, b)| (a - b) * (a - b))
                .sum();
            assert!(
                err < prev_err,
                "bits={bits}: error {err} did not improve on {prev_err}"
            );
            prev_err = err;
        }
        // 6 bits should reconstruct a unit vector almost perfectly.
        assert!(prev_err < 5e-3, "6-bit reconstruction error {prev_err}");
    }

    #[test]
    fn mse_asymmetric_dot_equals_decoded_dot() {
        let mut rng = SplitMix64::new(SEED);
        let q = TurboQuantMse::new(96, 3, SEED);
        let x: Vec<f32> = (0..96).map(|_| rng.next_gaussian() as f32).collect();
        let y: Vec<f32> = (0..96).map(|_| rng.next_gaussian() as f32).collect();
        let code = q.encode(&x);
        let est = q.dot(&q.prepare_query(&y), &code);
        let explicit = dot(&y, &q.decode(&code));
        assert!(
            (est - explicit).abs() < 1e-2 * explicit.abs().max(1.0),
            "asymmetric {est} vs explicit {explicit}"
        );
    }

    #[test]
    fn mse_encoding_is_deterministic() {
        let mut rng = SplitMix64::new(SEED);
        let x: Vec<f32> = (0..40).map(|_| rng.next_gaussian() as f32).collect();
        let a = TurboQuantMse::new(40, 2, 11).encode(&x);
        let b = TurboQuantMse::new(40, 2, 11).encode(&x);
        assert_eq!(a, b);
        let c = TurboQuantMse::new(40, 2, 12).encode(&x);
        assert!(a.idx != c.idx, "different seed should permute codes");
    }

    #[test]
    fn quantize_coord_picks_nearest_centroid() {
        let q = TurboQuantMse::new(64, 2, SEED);
        // Exhaustively verify the boundary search against brute force.
        for step in -300..=300 {
            let y = step as f32 * 0.001;
            let idx = q.quantize_coord(y) as usize;
            let best = q
                .centroids
                .iter()
                .enumerate()
                .min_by(|a, b| {
                    (a.1 - y).abs().partial_cmp(&(b.1 - y).abs()).unwrap()
                })
                .unwrap()
                .0;
            let tie = (q.centroids[idx] - y).abs() - (q.centroids[best] - y).abs();
            assert!(
                tie.abs() < 1e-7,
                "y={y}: picked {idx} but nearest is {best}"
            );
        }
    }

    // ── TurboQuant-Prod ────────────────────────────────────────────────

    #[test]
    fn prod_rejects_bad_bits() {
        assert!(std::panic::catch_unwind(|| TurboQuantProd::new(8, 0, SEED)).is_err());
        assert!(std::panic::catch_unwind(|| TurboQuantProd::new(8, 9, SEED)).is_err());
    }

    #[test]
    fn prod_1bit_is_pure_qjl() {
        // b=1 → the MSE stage gets 0 bits: no indices, residual = the unit
        // vector itself (norm exactly 1 in the rotated basis).
        let q = TurboQuantProd::new(32, 1, SEED);
        let mut rng = SplitMix64::new(SEED);
        let x = rand_unit(32, &mut rng);
        let code = q.encode(&x);
        assert!(code.idx.is_empty());
        assert_eq!(code.qjl.len(), 32 / 8);
        assert!(
            (code.residual_norm - 1.0).abs() < 1e-4,
            "pure-QJL residual norm {} should be 1",
            code.residual_norm
        );
    }

    #[test]
    fn prod_code_sizes_match_bit_budget() {
        for bits in 1..=8u32 {
            let q = TurboQuantProd::new(100, bits, SEED); // pads to 128
            let (idx, qjl) = q.code_len();
            assert_eq!(idx, (128 * (bits as usize - 1)).div_ceil(8));
            assert_eq!(qjl, 16);
        }
    }

    #[test]
    fn prod_zero_vector_roundtrips_to_zero() {
        let q = TurboQuantProd::new(24, 3, SEED);
        let code = q.encode(&vec![0.0; 24]);
        assert_eq!(code.norm, 0.0);
        assert_eq!(code.residual_norm, 0.0);
        assert!(q.decode(&code).iter().all(|&v| v == 0.0));
        assert_eq!(q.dot(&q.prepare_query(&vec![1.0; 24]), &code), 0.0);
    }

    #[test]
    fn prod_asymmetric_dot_equals_decoded_dot() {
        let mut rng = SplitMix64::new(SEED);
        let q = TurboQuantProd::new(64, 3, SEED);
        let x: Vec<f32> = (0..64).map(|_| 2.0 * rng.next_gaussian() as f32).collect();
        let y: Vec<f32> = (0..64).map(|_| rng.next_gaussian() as f32).collect();
        let code = q.encode(&x);
        let est = q.dot(&q.prepare_query(&y), &code);
        let explicit = dot(&y, &q.decode(&code));
        assert!(
            (est - explicit).abs() < 1e-2 * explicit.abs().max(1.0),
            "asymmetric {est} vs explicit {explicit}"
        );
    }

    #[test]
    fn prod_residual_shrinks_with_more_bits() {
        // More MSE bits → smaller residual for QJL to mop up.
        let mut rng = SplitMix64::new(SEED);
        let x = rand_unit(128, &mut rng);
        let mut prev = f32::INFINITY;
        for bits in 1..=5u32 {
            let q = TurboQuantProd::new(128, bits, SEED);
            let rn = q.encode(&x).residual_norm;
            assert!(rn < prev, "bits={bits}: residual {rn} !< {prev}");
            prev = rn;
        }
    }

    #[test]
    fn prod_scales_linearly_with_input_norm() {
        // Norms are stored exactly, so scaling the input scales the estimate.
        let mut rng = SplitMix64::new(SEED);
        let q = TurboQuantProd::new(48, 2, SEED);
        let x = rand_unit(48, &mut rng);
        let x5: Vec<f32> = x.iter().map(|v| v * 5.0).collect();
        let y = rand_unit(48, &mut rng);
        let prep = q.prepare_query(&y);
        let e1 = q.dot(&prep, &q.encode(&x));
        let e5 = q.dot(&prep, &q.encode(&x5));
        assert!(
            (e5 - 5.0 * e1).abs() < 1e-3 * e1.abs().max(1.0),
            "estimate must scale with the norm: {e5} vs 5·{e1}"
        );
    }

    // ── TurboQuantIndex ────────────────────────────────────────────────

    #[test]
    fn index_empty_search_returns_nothing() {
        let idx = TurboQuantIndex::new(8, Metric::Cos, 4, SEED);
        assert!(idx.is_empty());
        assert!(idx.search(&[0.5; 8], 5).unwrap().is_empty());
        assert!(idx.search(&[0.5; 8], 0).unwrap().is_empty());
    }

    #[test]
    fn index_dimension_mismatch_errors() {
        let idx = TurboQuantIndex::new(8, Metric::Cos, 4, SEED);
        assert!(idx.upsert(1, &[1.0; 4]).is_err());
        assert!(idx.search(&[1.0; 4], 1).is_err());
    }

    #[test]
    fn index_bits_are_clamped() {
        assert_eq!(TurboQuantIndex::new(8, Metric::Cos, 0, SEED).bits(), 1);
        assert_eq!(TurboQuantIndex::new(8, Metric::Cos, 99, SEED).bits(), 8);
    }

    #[test]
    fn index_upsert_replaces_and_counts() {
        let mut rng = SplitMix64::new(SEED);
        let idx = TurboQuantIndex::new(16, Metric::L2sq, 4, SEED);
        let a = rand_unit(16, &mut rng);
        let b = rand_unit(16, &mut rng);
        assert!(!idx.upsert(1, &a).unwrap());
        assert!(idx.upsert(1, &b).unwrap()); // replaced
        assert_eq!(idx.len(), 1);
        let r = idx.search(&b, 1).unwrap();
        assert_eq!(r[0].0, 1);
        assert!(r[0].1 < 0.1, "distance to (quantized) self should be ~0, got {}", r[0].1);
    }

    #[test]
    fn index_remove_keeps_storage_consistent() {
        let mut rng = SplitMix64::new(SEED);
        let idx = TurboQuantIndex::new(16, Metric::Cos, 4, SEED);
        let vecs: Vec<Vec<f32>> = (0..6).map(|_| rand_unit(16, &mut rng)).collect();
        for (i, v) in vecs.iter().enumerate() {
            idx.upsert(i as u64, v).unwrap();
        }
        assert!(idx.remove(2));
        assert!(!idx.remove(2)); // already gone
        assert_eq!(idx.len(), 5);
        let r = idx.search(&vecs[5], 6).unwrap();
        let ids: Vec<u64> = r.iter().map(|(id, _)| *id).collect();
        assert!(!ids.contains(&2));
        assert_eq!(ids.len(), 5);
        // The swap-removed row (previously last) must still score correctly:
        // vector 5 should be its own nearest neighbor.
        assert_eq!(r[0].0, 5, "swap-remove corrupted a row: {r:?}");
    }

    #[test]
    fn index_finds_nearest_in_clusters() {
        // Two well-separated direction clusters; every query from cluster A
        // must retrieve only cluster-A members first.
        let dims = 32;
        let idx = TurboQuantIndex::new(dims, Metric::Cos, 4, SEED);
        let mut rng = SplitMix64::new(1);
        let mut base_a = rand_unit(dims, &mut rng);
        let mut base_b: Vec<f32> = base_a.iter().map(|v| -v).collect();
        base_a[0] += 0.1;
        base_b[0] -= 0.1;
        let jitter = |base: &[f32], rng: &mut SplitMix64| -> Vec<f32> {
            base.iter()
                .map(|v| v + 0.05 * rng.next_gaussian() as f32)
                .collect()
        };
        for i in 0..20u64 {
            idx.upsert(i, &jitter(&base_a, &mut rng)).unwrap();
            idx.upsert(100 + i, &jitter(&base_b, &mut rng)).unwrap();
        }
        let r = idx.search(&base_a, 10).unwrap();
        for (id, d) in &r {
            assert!(*id < 100, "cluster-B id {id} (dist {d}) leaked into top-10");
        }
    }

    #[test]
    fn index_ip_metric_ranks_by_inner_product() {
        let dims = 16;
        let idx = TurboQuantIndex::new(dims, Metric::Ip, 6, SEED);
        let mut base = vec![0.0f32; dims];
        base[0] = 1.0;
        // Same direction, growing magnitude: bigger IP = closer under Ip.
        for (i, scale) in [0.5f32, 1.0, 2.0, 4.0].iter().enumerate() {
            let v: Vec<f32> = base.iter().map(|x| x * scale).collect();
            idx.upsert(i as u64, &v).unwrap();
        }
        let r = idx.search(&base, 4).unwrap();
        assert_eq!(r[0].0, 3, "largest inner product should rank first: {r:?}");
        assert_eq!(r[3].0, 0, "smallest inner product should rank last: {r:?}");
    }

    #[test]
    fn index_l2_metric_uses_exact_norms() {
        let dims = 16;
        let idx = TurboQuantIndex::new(dims, Metric::L2sq, 6, SEED);
        let mut rng = SplitMix64::new(SEED);
        let q = rand_unit(dims, &mut rng);
        // Same direction as q at distance 0, plus a scaled copy at distance 1.
        let far: Vec<f32> = q.iter().map(|v| v * 2.0).collect();
        idx.upsert(1, &q).unwrap();
        idx.upsert(2, &far).unwrap();
        let r = idx.search(&q, 2).unwrap();
        assert_eq!(r[0].0, 1);
        assert!(r[0].1.abs() < 0.05, "self distance should be ~0: {}", r[0].1);
        assert!((r[1].1 - 1.0).abs() < 0.1, "‖2q−q‖² should be ~1: {}", r[1].1);
    }

    #[test]
    fn index_zero_vector_is_storable_and_searchable() {
        let idx = TurboQuantIndex::new(8, Metric::Cos, 2, SEED);
        idx.upsert(1, &[0.0; 8]).unwrap();
        idx.upsert(2, &[1.0; 8]).unwrap();
        let r = idx.search(&[1.0; 8], 2).unwrap();
        assert_eq!(r.len(), 2);
        assert_eq!(r[0].0, 2, "the non-zero vector must rank above the zero vector");
    }

    #[test]
    fn index_codes_compress_versus_f32() {
        // 4 bits/coord + 8 bytes of norms per vector must undercut 32-bit
        // floats by a wide margin once the fixed projection is amortized.
        let dims = 128;
        let n = 500usize;
        let idx = TurboQuantIndex::new(dims, Metric::Cos, 4, SEED);
        let mut rng = SplitMix64::new(SEED);
        let empty_bytes = idx.memory_bytes(); // fixed: projection + signs
        for i in 0..n {
            idx.upsert(i as u64, &rand_unit(dims, &mut rng)).unwrap();
        }
        let per_vector = (idx.memory_bytes() - empty_bytes) / n;
        let flat = dims * 4;
        assert!(
            per_vector * 3 < flat,
            "per-vector {per_vector}B should be <1/3 of flat {flat}B"
        );
    }

    #[test]
    fn index_search_is_deterministic_across_instances() {
        let dims = 24;
        let mut rng = SplitMix64::new(9);
        let vecs: Vec<Vec<f32>> = (0..30).map(|_| rand_unit(dims, &mut rng)).collect();
        let q = rand_unit(dims, &mut rng);
        let run = || {
            let idx = TurboQuantIndex::new(dims, Metric::Cos, 3, 77);
            for (i, v) in vecs.iter().enumerate() {
                idx.upsert(i as u64, v).unwrap();
            }
            idx.search(&q, 5).unwrap()
        };
        assert_eq!(run(), run(), "same seed + data must reproduce results");
    }

    #[test]
    fn index_top_k_truncates_and_sorts_ascending() {
        let mut rng = SplitMix64::new(SEED);
        let idx = TurboQuantIndex::new(16, Metric::Cos, 4, SEED);
        for i in 0..25u64 {
            idx.upsert(i, &rand_unit(16, &mut rng)).unwrap();
        }
        let r = idx.search(&rand_unit(16, &mut rng), 7).unwrap();
        assert_eq!(r.len(), 7);
        for w in r.windows(2) {
            assert!(w[0].1 <= w[1].1, "distances not ascending: {r:?}");
        }
    }

    #[test]
    fn index_searches_at_extreme_bit_widths() {
        // bits=1: the MSE stage is empty (idx_len = 0), a row is QJL bits only.
        // bits=8: widest packed fields (byte-aligned). Both must self-recall.
        let dims = 64;
        let mut rng = SplitMix64::new(SEED);
        let vecs: Vec<Vec<f32>> = (0..20).map(|_| rand_unit(dims, &mut rng)).collect();
        for bits in [1u32, 8] {
            let idx = TurboQuantIndex::new(dims, Metric::Cos, bits, SEED);
            for (i, v) in vecs.iter().enumerate() {
                idx.upsert(i as u64, v).unwrap();
            }
            for (i, v) in vecs.iter().enumerate() {
                let r = idx.search(v, 1).unwrap();
                assert_eq!(
                    r[0].0, i as u64,
                    "bits={bits}: vector {i} did not retrieve itself: {r:?}"
                );
            }
        }
    }

    #[test]
    fn index_is_send_sync_and_concurrent() {
        use std::sync::Arc;
        let idx = Arc::new(TurboQuantIndex::new(16, Metric::Cos, 2, SEED));
        let mut handles = Vec::new();
        for t in 0..4u64 {
            let idx = Arc::clone(&idx);
            handles.push(std::thread::spawn(move || {
                let mut rng = SplitMix64::new(t);
                for i in 0..25u64 {
                    idx.upsert(t * 1000 + i, &rand_unit(16, &mut rng)).unwrap();
                    idx.search(&rand_unit(16, &mut rng), 3).unwrap();
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
        assert_eq!(idx.len(), 100);
    }
}
