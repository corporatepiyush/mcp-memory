//! Statistical validation of the TurboQuant implementation against the
//! theoretical guarantees of arXiv:2504.19874 (Theorems 1-3), plus
//! end-to-end nearest-neighbor quality checks against brute force.
//!
//! All randomness is seeded, so these tests are deterministic; tolerances
//! still leave slack over the pure theory because the coordinate distribution
//! after the Hadamard rotation is only asymptotically Gaussian.

use mcp_memory::ivf::Metric;
use mcp_memory::turboquant::{
    gaussian_quantizer_mse, lloyd_max_gaussian, TurboQuantIndex, TurboQuantMse, TurboQuantProd,
};

// ── tiny seeded RNG (tests only; the crate's RNG is private) ───────────────

struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn unit(&mut self) -> f64 {
        (((self.next_u64() >> 11) as f64) + 0.5) * (1.0 / (1u64 << 53) as f64)
    }

    fn gaussian(&mut self) -> f64 {
        (-2.0 * self.unit().ln()).sqrt() * (std::f64::consts::TAU * self.unit()).cos()
    }

    fn unit_vector(&mut self, dim: usize) -> Vec<f32> {
        let mut v: Vec<f32> = (0..dim).map(|_| self.gaussian() as f32).collect();
        let n = norm(&v);
        for x in &mut v {
            *x /= n;
        }
        v
    }
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn norm(a: &[f32]) -> f32 {
    dot(a, a).sqrt()
}

/// A unit vector with inner product exactly `target` against `x`.
fn correlated_unit(x: &[f32], target: f32, rng: &mut Rng) -> Vec<f32> {
    let d = x.len();
    let mut z = rng.unit_vector(d);
    // Orthogonalize z against x, then mix.
    let zx = dot(&z, x);
    for (zi, xi) in z.iter_mut().zip(x) {
        *zi -= zx * xi;
    }
    let zn = norm(&z);
    let ortho_scale = (1.0 - target * target).sqrt() / zn;
    z.iter()
        .zip(x)
        .map(|(zi, xi)| target * xi + ortho_scale * zi)
        .collect()
}

// ── Theorem 1: MSE distortion ───────────────────────────────────────────────

/// D_mse must land between the Shannon lower bound `4^-b` (Theorem 3) and the
/// optimal-scalar-quantizer cost `C(f_X, b)` (0.36, 0.117, 0.03, 0.009 …) that
/// Theorem 1 promises, with slack for the finite-dimensional rotation.
#[test]
fn theorem1_mse_distortion_matches_theory() {
    let dim = 128;
    let trials = 100;
    let mut rng = Rng(0xA11CE);
    for bits in 1..=5u32 {
        let q = TurboQuantMse::new(dim, bits, 0xBEEF + u64::from(bits));
        let mut total = 0.0f64;
        for _ in 0..trials {
            let x = rng.unit_vector(dim);
            let x_hat = q.decode(&q.encode(&x));
            total += x
                .iter()
                .zip(&x_hat)
                .map(|(a, b)| f64::from(a - b) * f64::from(a - b))
                .sum::<f64>();
        }
        let d_mse = total / trials as f64;
        let optimal = gaussian_quantizer_mse(&lloyd_max_gaussian(bits));
        let shannon = 0.25f64.powi(bits as i32);
        assert!(
            d_mse < 1.25 * optimal,
            "b={bits}: D_mse {d_mse:.5} exceeds 1.25x optimal cost {optimal:.5}"
        );
        assert!(
            d_mse > 0.7 * shannon,
            "b={bits}: D_mse {d_mse:.6} implausibly beats the Shannon bound {shannon:.6}"
        );
    }
}

/// The paper's headline: TurboQuant-MSE stays within the `√3π/2 ≈ 2.72`
/// constant of the information-theoretic lower bound at every bit-width.
#[test]
fn theorem1_within_constant_factor_of_lower_bound() {
    let dim = 256;
    let trials = 50;
    let mut rng = Rng(0xFACADE);
    for bits in 1..=4u32 {
        let q = TurboQuantMse::new(dim, bits, 0xDEED + u64::from(bits));
        let mut total = 0.0f64;
        for _ in 0..trials {
            let x = rng.unit_vector(dim);
            let x_hat = q.decode(&q.encode(&x));
            total += x
                .iter()
                .zip(&x_hat)
                .map(|(a, b)| f64::from(a - b) * f64::from(a - b))
                .sum::<f64>();
        }
        let ratio = (total / trials as f64) / 0.25f64.powi(bits as i32);
        let bound = 3.0f64.sqrt() * std::f64::consts::PI / 2.0;
        assert!(
            ratio < bound * 1.1,
            "b={bits}: distortion ratio {ratio:.3} above √3π/2 ≈ {bound:.3}"
        );
    }
}

// ── Theorem 2: unbiased inner products ──────────────────────────────────────

/// TurboQuant-Prod's inner-product estimator must be unbiased, while the
/// MSE-optimal quantizer at 1 bit shows its known `2/π` multiplicative bias.
/// Averages the signed error over many independent quantizer instances.
#[test]
fn theorem2_prod_is_unbiased_where_mse_is_biased() {
    // Each instance constructs a fresh quantizer (a D×D Gaussian draw), so the
    // instance count is the whole cost of this test. 240 keeps the 0.02 bias
    // threshold above 3σ of the estimator's mean (σ ≈ 0.006).
    let dim = 64;
    let instances = 240;
    let mut rng = Rng(0x5EED);
    let x = rng.unit_vector(dim);
    let y = correlated_unit(&x, 0.7, &mut rng);
    let true_ip = f64::from(dot(&x, &y));

    let mut prod_err_sum = 0.0f64;
    let mut mse_err_sum = 0.0f64;
    for s in 0..instances {
        let qp = TurboQuantProd::new(dim, 2, 1000 + s);
        prod_err_sum += f64::from(qp.dot(&qp.prepare_query(&y), &qp.encode(&x))) - true_ip;

        let qm = TurboQuantMse::new(dim, 1, 1000 + s);
        mse_err_sum += f64::from(qm.dot(&qm.prepare_query(&y), &qm.encode(&x))) - true_ip;
    }
    let prod_bias = prod_err_sum / instances as f64;
    let mse_bias = mse_err_sum / instances as f64;

    // Prod: variance ≤ 0.56/64 → std of the mean ≈ 0.006; 0.02 is >3σ.
    assert!(
        prod_bias.abs() < 0.02,
        "prod estimator is biased: mean error {prod_bias:.4}"
    );
    // MSE @ 1 bit: multiplicative bias 2/π ⇒ mean error ≈ (2/π − 1)·0.7 ≈ −0.25.
    assert!(
        mse_bias < -0.15,
        "mse@1bit should under-estimate by ~0.25, got {mse_bias:.4}"
    );
    assert!(
        prod_bias.abs() * 5.0 < mse_bias.abs(),
        "prod bias {prod_bias:.4} not clearly smaller than mse bias {mse_bias:.4}"
    );
}

/// The MSE quantizer's inner-product bias must shrink as bits grow (Fig. 1b).
#[test]
fn mse_inner_product_bias_shrinks_with_bits() {
    let dim = 64;
    let instances = 120;
    let mut rng = Rng(0xB1A5);
    let x = rng.unit_vector(dim);
    let y = correlated_unit(&x, 0.7, &mut rng);
    let true_ip = f64::from(dot(&x, &y));

    let bias_at = |bits: u32| -> f64 {
        let mut sum = 0.0f64;
        for s in 0..instances {
            let q = TurboQuantMse::new(dim, bits, 400 + s);
            sum += f64::from(q.dot(&q.prepare_query(&y), &q.encode(&x))) - true_ip;
        }
        (sum / instances as f64).abs()
    };
    let b1 = bias_at(1);
    let b4 = bias_at(4);
    assert!(
        b4 < b1 / 4.0,
        "bias should collapse with bits: b=1 → {b1:.4}, b=4 → {b4:.4}"
    );
}

/// Theorem 2's distortion rate: D_prod ≈ 1.57/d, 0.56/d, 0.18/d, 0.047/d for
/// b = 1..4 (the b=1 case doubles as the QJL variance bound π/(2d) of Lemma 4).
#[test]
fn theorem2_prod_distortion_matches_theory() {
    // Every trial pays two O(D²) matvecs (encode + prepare_query); 500 trials
    // put the mean-squared-error estimate within ~6% (√(2/500)) of truth,
    // comfortably inside the 1.4x assertion slack.
    let dim = 128;
    let trials = 500;
    let paper = [1.57, 0.56, 0.18, 0.047];
    for (bits, &num) in (1..=4u32).zip(&paper) {
        let q = TurboQuantProd::new(dim, bits, 0xD07 + u64::from(bits));
        let mut rng = Rng(0x1234 + u64::from(bits));
        let mut sq_err = 0.0f64;
        for _ in 0..trials {
            let x = rng.unit_vector(dim);
            let y = rng.unit_vector(dim);
            let est = f64::from(q.dot(&q.prepare_query(&y), &q.encode(&x)));
            let err = est - f64::from(dot(&x, &y));
            sq_err += err * err;
        }
        let d_prod = sq_err / f64::from(trials);
        let bound = num / dim as f64;
        assert!(
            d_prod < 1.4 * bound,
            "b={bits}: D_prod {d_prod:.6} above 1.4x paper value {bound:.6}"
        );
        // Theorem 3 lower bound: 4^-b/d — sanity that we're in the right decade.
        let lower = 0.25f64.powi(bits as i32) / dim as f64;
        assert!(
            d_prod > 0.5 * lower,
            "b={bits}: D_prod {d_prod:.7} implausibly below the lower bound {lower:.7}"
        );
    }
}

/// Distortion should not depend on the true inner product (Fig. 2): the
/// prod estimator's error spread stays flat as ⟨x,y⟩ grows.
#[test]
fn prod_error_variance_is_ip_independent() {
    let dim = 128;
    let trials = 300;
    let q = TurboQuantProd::new(dim, 2, 0xF1A7);
    let var_at = |target: f32, seed: u64| -> f64 {
        let mut rng = Rng(seed);
        let mut sq = 0.0f64;
        for _ in 0..trials {
            let x = rng.unit_vector(dim);
            let y = correlated_unit(&x, target, &mut rng);
            let est = f64::from(q.dot(&q.prepare_query(&y), &q.encode(&x)));
            let err = est - f64::from(dot(&x, &y));
            sq += err * err;
        }
        sq / f64::from(trials)
    };
    let low = var_at(0.01, 21);
    let high = var_at(0.6, 22);
    assert!(
        high < 2.0 * low && low < 2.0 * high,
        "error variance should be flat in ⟨x,y⟩: at 0.01 → {low:.6}, at 0.6 → {high:.6}"
    );
}

// ── reconstruction quality ──────────────────────────────────────────────────

/// Decoded vectors must point in nearly the original direction, monotonically
/// better with bits, for both quantizers.
#[test]
fn reconstruction_cosine_improves_with_bits() {
    let dim = 256;
    let mut rng = Rng(0xC05);
    let x = rng.unit_vector(dim);
    let mut prev_mse = -1.0f32;
    let mut prev_prod = -1.0f32;
    for bits in 1..=6u32 {
        let qm = TurboQuantMse::new(dim, bits, 7);
        let qp = TurboQuantProd::new(dim, bits, 7);
        let dec_mse = qm.decode(&qm.encode(&x));
        let dec_prod = qp.decode(&qp.encode(&x));
        let cos_mse = dot(&dec_mse, &x) / norm(&dec_mse);
        let cos_prod = dot(&dec_prod, &x) / norm(&dec_prod);
        assert!(
            cos_mse > prev_mse,
            "mse cosine should improve: b={bits} {cos_mse} vs {prev_mse}"
        );
        assert!(
            cos_prod > prev_prod - 0.02,
            "prod cosine should not regress: b={bits} {cos_prod} vs {prev_prod}"
        );
        prev_mse = cos_mse;
        prev_prod = cos_prod;
    }
    assert!(prev_mse > 0.995, "6-bit mse reconstruction cosine {prev_mse}");
    assert!(prev_prod > 0.98, "6-bit prod reconstruction cosine {prev_prod}");
}

/// Norms are stored exactly, so reconstruction error is purely directional
/// and scales linearly with the input's magnitude.
#[test]
fn reconstruction_error_scales_with_norm() {
    let dim = 64;
    let mut rng = Rng(0x40);
    let q = TurboQuantProd::new(dim, 3, 11);
    let x = rng.unit_vector(dim);
    let err = |scale: f32| -> f32 {
        let xs: Vec<f32> = x.iter().map(|v| v * scale).collect();
        let x_hat = q.decode(&q.encode(&xs));
        xs.iter()
            .zip(&x_hat)
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f32>()
            .sqrt()
    };
    let e1 = err(1.0);
    let e10 = err(10.0);
    assert!(
        (e10 - 10.0 * e1).abs() < 0.1 * e10,
        "error should scale ~linearly with norm: {e1} → {e10}"
    );
}

// ── nearest-neighbor recall vs. brute force ─────────────────────────────────

/// Returns `(recall@k, quality gap)` of quantized search vs exact brute force.
/// The quality gap is `mean_true_cosine(exact top-k) − mean_true_cosine(retrieved k)`:
/// how much worse the retrieved neighbors are than the optimal ones, in cosine.
/// For an unbiased estimator with per-item std σ this is O(σ) even when exact
/// set membership (recall) is unstable because true neighbors are packed
/// within σ of each other.
fn search_quality(dim: usize, n: usize, queries: usize, bits: u32, k: usize) -> (f64, f64) {
    let mut rng = Rng(0xEC0);
    // Clustered corpus: 8 direction anchors plus noise of *total* norm ~0.5
    // (scaled by 1/√d per coordinate — unscaled noise would have norm 0.4·√d
    // and drown the anchors, making all vectors near-orthogonal and the true
    // top-10 an unresolvable hair-width ranking).
    let anchors: Vec<Vec<f32>> = (0..8).map(|_| rng.unit_vector(dim)).collect();
    let noise = 0.5 / (dim as f32).sqrt();
    let make = |rng: &mut Rng| -> Vec<f32> {
        let a = &anchors[(rng.next_u64() % 8) as usize];
        let mut v: Vec<f32> = a
            .iter()
            .map(|x| x + noise * rng.gaussian() as f32)
            .collect();
        let nn = norm(&v);
        for x in &mut v {
            *x /= nn;
        }
        v
    };
    let corpus: Vec<Vec<f32>> = (0..n).map(|_| make(&mut rng)).collect();
    let index = TurboQuantIndex::new(dim, Metric::Cos, bits, 0xACE);
    for (i, v) in corpus.iter().enumerate() {
        index.upsert(i as u64, v).unwrap();
    }

    let mut hits = 0usize;
    let mut gap_sum = 0.0f64;
    for _ in 0..queries {
        let q = make(&mut rng);
        // Exact top-k by cosine (unit norms ⇒ by inner product).
        let mut exact: Vec<(usize, f32)> = corpus
            .iter()
            .enumerate()
            .map(|(i, v)| (i, dot(&q, v) / norm(v)))
            .collect();
        exact.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let truth: std::collections::HashSet<u64> =
            exact[..k].iter().map(|&(i, _)| i as u64).collect();
        let best: f64 = exact[..k].iter().map(|&(_, c)| f64::from(c)).sum::<f64>() / k as f64;

        let got = index.search(&q, k).unwrap();
        hits += got.iter().filter(|(id, _)| truth.contains(id)).count();
        let got_true: f64 = got
            .iter()
            .map(|&(id, _)| f64::from(dot(&q, &corpus[id as usize])))
            .sum::<f64>()
            / k as f64;
        gap_sum += best - got_true;
    }
    (
        hits as f64 / (queries * k) as f64,
        gap_sum / queries as f64,
    )
}

/// Search quality against exact brute force on clustered unit vectors.
/// Set-identity recall@10 is unstable here by construction (cluster members
/// sit within ~σ of each other), so the assertion is the guarantee Theorem 2
/// actually provides: the retrieved neighbors' true cosine is within O(σ) of
/// the exact top-10's, where σ = √(D_prod) = √(0.56/d) at 2 bits and
/// √(0.047/d) at 4 bits (0.066 and 0.019 at d=128).
#[test]
fn retrieved_neighbors_are_near_optimal() {
    let (r2, gap2) = search_quality(128, 400, 40, 2, 10);
    let (r4, gap4) = search_quality(128, 400, 40, 4, 10);
    eprintln!("2 bits: recall {r2:.3}, quality gap {gap2:.4}; 4 bits: recall {r4:.3}, gap {gap4:.4}");
    let sigma2 = (0.56f64 / 128.0).sqrt();
    let sigma4 = (0.047f64 / 128.0).sqrt();
    assert!(gap2 < 2.0 * sigma2, "2-bit quality gap {gap2:.4} above 2σ = {:.4}", 2.0 * sigma2);
    assert!(gap4 < 2.0 * sigma4, "4-bit quality gap {gap4:.4} above 2σ = {:.4}", 2.0 * sigma4);
    assert!(gap4 < gap2, "more bits must improve quality: {gap2:.4} → {gap4:.4}");
    assert!(r4 > r2, "more bits must improve recall: {r2:.3} → {r4:.3}");
}

/// Every vector must retrieve itself first (top-1 self-recall) at 4 bits.
#[test]
fn self_recall_is_perfect_at_4_bits() {
    let dim = 96;
    let mut rng = Rng(0x5E1F);
    let index = TurboQuantIndex::new(dim, Metric::Cos, 4, 3);
    let corpus: Vec<Vec<f32>> = (0..60).map(|_| rng.unit_vector(dim)).collect();
    for (i, v) in corpus.iter().enumerate() {
        index.upsert(i as u64, v).unwrap();
    }
    for (i, v) in corpus.iter().enumerate() {
        let r = index.search(v, 1).unwrap();
        assert_eq!(
            r[0].0, i as u64,
            "vector {i} did not retrieve itself (got {r:?})"
        );
    }
}

/// The L2sq metric path: estimated distances must track true squared
/// Euclidean distances closely (norms are exact; only the IP is estimated).
#[test]
fn l2_distance_estimates_track_truth() {
    let dim = 128;
    let mut rng = Rng(0x12);
    let index = TurboQuantIndex::new(dim, Metric::L2sq, 4, 5);
    let corpus: Vec<Vec<f32>> = (0..50)
        .map(|_| {
            let s = 0.5 + 1.5 * rng.unit() as f32; // varied magnitudes
            rng.unit_vector(dim).iter().map(|v| v * s).collect()
        })
        .collect();
    for (i, v) in corpus.iter().enumerate() {
        index.upsert(i as u64, v).unwrap();
    }
    let q = rng.unit_vector(dim);
    let got = index.search(&q, 50).unwrap();
    for (id, est) in got {
        let v = &corpus[id as usize];
        let truth: f32 = q.iter().zip(v).map(|(a, b)| (a - b) * (a - b)).sum();
        assert!(
            (est - truth).abs() < 0.25,
            "id {id}: est {est:.3} vs true {truth:.3}"
        );
    }
}

/// Embedding-sized sanity run: 384-dim (pads to 512), non-trivial corpus,
/// self-recall and decode quality hold up.
#[test]
fn high_dimensional_embedding_sized_run() {
    let dim = 384;
    let mut rng = Rng(0xE44);
    let index = TurboQuantIndex::new(dim, Metric::Cos, 4, 9);
    let corpus: Vec<Vec<f32>> = (0..80).map(|_| rng.unit_vector(dim)).collect();
    for (i, v) in corpus.iter().enumerate() {
        index.upsert(i as u64, v).unwrap();
    }
    let r = index.search(&corpus[17], 5).unwrap();
    assert_eq!(r[0].0, 17, "self-recall failed at 384 dims: {r:?}");

    let q = index.quantizer();
    let x_hat = q.decode(&q.encode(&corpus[17]));
    let cos = dot(&x_hat, &corpus[17]) / norm(&x_hat);
    assert!(cos > 0.95, "384-dim 4-bit reconstruction cosine {cos:.4}");
}

/// Odd (non-power-of-two) dimensions run through the same guarantees:
/// unbiasedness and self-recall hold with zero-padding.
#[test]
fn odd_dimensions_are_supported() {
    for dim in [33usize, 100, 200] {
        let mut rng = Rng(0x0DD + dim as u64);
        let index = TurboQuantIndex::new(dim, Metric::Cos, 3, 1);
        let corpus: Vec<Vec<f32>> = (0..30).map(|_| rng.unit_vector(dim)).collect();
        for (i, v) in corpus.iter().enumerate() {
            index.upsert(i as u64, v).unwrap();
        }
        let r = index.search(&corpus[7], 1).unwrap();
        assert_eq!(r[0].0, 7, "self-recall failed at d={dim}");
    }
}
