//! A self-contained **IVF-Flat** (inverted-file, flat-storage) approximate
//! nearest-neighbour index.
//!
//! IVF-Flat partitions the vector space into `nlist` Voronoi cells via k-means.
//! Each vector is stored verbatim (no quantization — the "flat" part) in the
//! cell of its nearest centroid. A query scans only the `nprobe` cells whose
//! centroids are closest to it, trading a little recall for a large speed-up
//! over brute force on big collections. It complements the HNSW (usearch)
//! backend: IVF trains/builds far faster and uses less memory per vector, which
//! suits large, batch-ingested, periodically-rebuilt corpora typical of RAG.
//!
//! All vectors live in RAM (rebuilt from the SQLite `vector_embedding` table on
//! open, exactly like the HNSW backend), so this type owns no persistence.
//! Until the index is trained — or when the collection is smaller than `nlist`
//! — search transparently falls back to an exact brute-force scan, so results
//! are always correct, just not always sub-linear.

use parking_lot::RwLock;
use rustc_hash::FxHashMap;
use usearch::MetricKind;

/// Distance functions supported by the IVF index. Smaller is always "closer",
/// matching the convention usearch uses, so the two backends are interchangeable
/// to the rest of the store.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Metric {
    /// `1 - cosine_similarity` (range `[0, 2]`).
    Cos,
    /// `1 - inner_product` (raw dot product; assumes caller-normalized vectors).
    Ip,
    /// Squared Euclidean distance.
    L2sq,
}

impl Metric {
    /// Map a usearch metric onto the IVF metric, falling back to cosine for the
    /// metrics IVF does not model.
    pub const fn from_usearch(m: MetricKind) -> Self {
        match m {
            MetricKind::IP => Metric::Ip,
            MetricKind::L2sq => Metric::L2sq,
            _ => Metric::Cos,
        }
    }
}

#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

#[inline]
fn l2sq(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
}

#[inline]
fn norm(a: &[f32]) -> f32 {
    dot(a, a).sqrt()
}

struct Inner {
    dims: usize,
    metric: Metric,
    /// Entity ids, parallel to `vecs` rows, `norms` and `assign`.
    ids: Vec<u64>,
    /// Flat row-major vectors: `ids.len() * dims` floats.
    vecs: Vec<f32>,
    /// Cached L2 norm per stored vector (used by the cosine metric).
    norms: Vec<f32>,
    /// Centroid index each vector belongs to, or `-1` when not yet assigned.
    assign: Vec<i32>,
    /// `id -> row position` for O(1) upsert/remove.
    id_pos: FxHashMap<u64, usize>,
    /// Trained centroids, flat row-major: `centroid_count * dims`. Empty until trained.
    centroids: Vec<f32>,
    /// Inverted lists: for each centroid, the row positions assigned to it.
    lists: Vec<Vec<usize>>,
}

impl Inner {
    #[inline]
    fn row(&self, pos: usize) -> &[f32] {
        &self.vecs[pos * self.dims..(pos + 1) * self.dims]
    }

    #[inline]
    fn centroid(&self, c: usize) -> &[f32] {
        &self.centroids[c * self.dims..(c + 1) * self.dims]
    }

    /// Distance between a query (with precomputed norm for cosine) and stored row.
    #[inline]
    fn dist_to_row(&self, q: &[f32], q_norm: f32, pos: usize) -> f32 {
        let v = self.row(pos);
        match self.metric {
            Metric::Cos => {
                let denom = q_norm * self.norms[pos];
                if denom == 0.0 {
                    1.0
                } else {
                    1.0 - dot(q, v) / denom
                }
            }
            Metric::Ip => 1.0 - dot(q, v),
            Metric::L2sq => l2sq(q, v),
        }
    }

    /// Distance between a query and a centroid (centroid norm computed on the fly
    /// — there are far fewer centroids than vectors, so this stays cheap).
    #[inline]
    fn dist_to_centroid(&self, q: &[f32], q_norm: f32, c: usize) -> f32 {
        let v = self.centroid(c);
        match self.metric {
            Metric::Cos => {
                let denom = q_norm * norm(v);
                if denom == 0.0 {
                    1.0
                } else {
                    1.0 - dot(q, v) / denom
                }
            }
            Metric::Ip => 1.0 - dot(q, v),
            Metric::L2sq => l2sq(q, v),
        }
    }

    fn nearest_centroid(&self, v: &[f32], v_norm: f32) -> i32 {
        let mut best = -1i32;
        let mut best_d = f32::INFINITY;
        for c in 0..self.lists.len() {
            let d = self.dist_to_centroid(v, v_norm, c);
            if d < best_d {
                best_d = d;
                best = c as i32;
            }
        }
        best
    }
}

/// An IVF-Flat index. All methods take `&self`; internal mutable state is guarded
/// by a single `RwLock`, so the index is `Send + Sync` and safe to share behind
/// an `Arc` like the usearch backend.
pub struct IvfFlatIndex {
    dims: usize,
    metric: Metric,
    /// Target number of Voronoi cells (centroids). Actual count is capped at the
    /// number of stored vectors when training.
    nlist: usize,
    /// Default number of cells probed per query (clamped to the trained count).
    nprobe: usize,
    inner: RwLock<Inner>,
}

impl IvfFlatIndex {
    pub fn new(dims: usize, metric: Metric, nlist: usize, nprobe: usize) -> Self {
        let nlist = nlist.max(1);
        let nprobe = nprobe.clamp(1, nlist);
        Self {
            dims,
            metric,
            nlist,
            nprobe,
            inner: RwLock::new(Inner {
                dims,
                metric,
                ids: Vec::new(),
                vecs: Vec::new(),
                norms: Vec::new(),
                assign: Vec::new(),
                id_pos: FxHashMap::default(),
                centroids: Vec::new(),
                lists: Vec::new(),
            }),
        }
    }

    pub fn len(&self) -> usize {
        self.inner.read().ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn is_trained(&self) -> bool {
        !self.inner.read().centroids.is_empty()
    }

    pub const fn metric(&self) -> Metric {
        self.metric
    }

    pub const fn nlist(&self) -> usize {
        self.nlist
    }

    pub const fn nprobe(&self) -> usize {
        self.nprobe
    }

    /// The number of trained centroids (0 until [`IvfFlatIndex::train`] runs).
    pub fn centroid_count(&self) -> usize {
        self.inner.read().lists.len()
    }

    /// Approximate resident bytes: stored vectors + norms + centroids + bookkeeping.
    pub fn memory_bytes(&self) -> usize {
        let g = self.inner.read();
        g.vecs.len() * 4
            + g.norms.len() * 4
            + g.centroids.len() * 4
            + g.assign.len() * 4
            + g.ids.len() * 8
            + g.id_pos.len() * 16
            + g.lists.iter().map(|l| l.len() * 8).sum::<usize>()
    }

    /// Insert or replace the vector for `id`. Returns `true` if it replaced an
    /// existing entry.
    pub fn upsert(&self, id: u64, v: &[f32]) -> Result<bool, String> {
        if v.len() != self.dims {
            return Err(format!(
                "dimension mismatch: got {}, expected {}",
                v.len(),
                self.dims
            ));
        }
        let mut g = self.inner.write();
        let existed = g.id_pos.contains_key(&id);
        if existed {
            remove_locked(&mut g, id);
        }

        let pos = g.ids.len();
        g.ids.push(id);
        g.vecs.extend_from_slice(v);
        g.norms.push(norm(v));
        g.id_pos.insert(id, pos);

        // Assign into a cell when the index is already trained, so the new vector
        // is reachable by probe-limited search; otherwise leave it unassigned
        // (it is still found by the brute-force fallback).
        if !g.centroids.is_empty() {
            let n = g.norms[pos];
            let c = g.nearest_centroid(v, n);
            g.assign.push(c);
            if c >= 0 {
                g.lists[c as usize].push(pos);
            }
        } else {
            g.assign.push(-1);
        }
        Ok(existed)
    }

    /// Remove the vector for `id`. Returns `true` if it existed.
    pub fn remove(&self, id: u64) -> bool {
        let mut g = self.inner.write();
        if !g.id_pos.contains_key(&id) {
            return false;
        }
        remove_locked(&mut g, id);
        true
    }

    /// Return the `top_k` nearest ids with their distances (ascending). Uses
    /// `nprobe_override` cells when given, else the configured default; falls back
    /// to an exact scan when untrained.
    pub fn search(
        &self,
        query: &[f32],
        top_k: usize,
        nprobe_override: Option<usize>,
    ) -> Result<Vec<(u64, f32)>, String> {
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
        let q_norm = norm(query);

        // Gather candidate row positions: either from the probed lists, or all
        // rows when the index has not been trained yet.
        let candidates: Vec<usize> = if g.centroids.is_empty() {
            (0..g.ids.len()).collect()
        } else {
            let nprobe = nprobe_override.unwrap_or(self.nprobe).clamp(1, g.lists.len());
            // Rank centroids by distance, take the nearest `nprobe`.
            let mut cd: Vec<(usize, f32)> = (0..g.lists.len())
                .map(|c| (c, g.dist_to_centroid(query, q_norm, c)))
                .collect();
            cd.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            let mut cand = Vec::new();
            for &(c, _) in cd.iter().take(nprobe) {
                cand.extend_from_slice(&g.lists[c]);
            }
            cand
        };

        let mut scored: Vec<(u64, f32)> = candidates
            .into_iter()
            .map(|pos| (g.ids[pos], g.dist_to_row(query, q_norm, pos)))
            .collect();
        let cmp = |a: &(u64, f32), b: &(u64, f32)| {
            a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
        };
        // Partial selection: O(n) to isolate the top_k, then sort only those.
        if scored.len() > top_k {
            scored.select_nth_unstable_by(top_k - 1, cmp);
            scored.truncate(top_k);
        }
        scored.sort_unstable_by(cmp);
        Ok(scored)
    }

    /// (Re)train centroids over the currently stored vectors via k-means, then
    /// rebuild the inverted lists. Cheap no-op when empty. The centroid count is
    /// `min(nlist, n)`.
    pub fn train(&self) -> Result<(), String> {
        let mut g = self.inner.write();
        let n = g.ids.len();
        if n == 0 {
            g.centroids.clear();
            g.lists.clear();
            return Ok(());
        }
        let k = self.nlist.min(n);
        let dims = self.dims;

        // k-means++ style seeding: first centroid random-ish (row 0), each
        // subsequent centroid the row farthest from its nearest chosen centroid.
        let mut centroids: Vec<f32> = Vec::with_capacity(k * dims);
        centroids.extend_from_slice(g.row(0));
        let mut min_d: Vec<f32> = (0..n)
            .map(|p| dist_rows(&g, g.row(p), &centroids[0..dims]))
            .collect();
        while centroids.len() / dims < k {
            // Pick the row with the largest distance to its nearest centroid.
            let mut far = 0usize;
            let mut far_d = -1.0f32;
            for (p, &d) in min_d.iter().enumerate() {
                if d > far_d {
                    far_d = d;
                    far = p;
                }
            }
            let start = centroids.len();
            centroids.extend_from_slice(g.row(far));
            let new_c = &centroids[start..start + dims];
            for (p, slot) in min_d.iter_mut().enumerate() {
                let d = dist_rows(&g, g.row(p), new_c);
                if d < *slot {
                    *slot = d;
                }
            }
        }

        // Lloyd iterations.
        let mut assign = vec![0i32; n];
        for _ in 0..IVF_KMEANS_ITERS {
            // Assignment step.
            let mut changed = false;
            for (p, a) in assign.iter_mut().enumerate() {
                let row = g.row(p);
                let mut best = 0usize;
                let mut best_d = f32::INFINITY;
                for c in 0..k {
                    let d = dist_rows(&g, row, &centroids[c * dims..(c + 1) * dims]);
                    if d < best_d {
                        best_d = d;
                        best = c;
                    }
                }
                if *a != best as i32 {
                    *a = best as i32;
                    changed = true;
                }
            }
            // Update step: centroid = mean of members; keep old centroid if empty.
            let mut sums = vec![0f32; k * dims];
            let mut counts = vec![0usize; k];
            for (p, &c_raw) in assign.iter().enumerate() {
                let c = c_raw as usize;
                counts[c] += 1;
                let row = g.row(p);
                let base = c * dims;
                for (j, &x) in row.iter().enumerate() {
                    sums[base + j] += x;
                }
            }
            for (c, &cnt) in counts.iter().enumerate() {
                if cnt == 0 {
                    continue;
                }
                let inv = 1.0 / cnt as f32;
                let base = c * dims;
                for (j, slot) in centroids[base..base + dims].iter_mut().enumerate() {
                    *slot = sums[base + j] * inv;
                }
            }
            if !changed {
                break;
            }
        }

        // Commit centroids + inverted lists + per-row assignment.
        let mut lists: Vec<Vec<usize>> = vec![Vec::new(); k];
        for (p, &c) in assign.iter().enumerate() {
            lists[c as usize].push(p);
        }
        g.centroids = centroids;
        g.lists = lists;
        g.assign = assign;
        Ok(())
    }

    /// Replace the entire contents in one shot (used for the initial bulk load).
    /// Does not train; call [`IvfFlatIndex::train`] afterwards.
    pub fn bulk_load(&self, items: impl IntoIterator<Item = (u64, Vec<f32>)>) -> Result<(), String> {
        let mut g = self.inner.write();
        for (id, v) in items {
            if v.len() != self.dims {
                return Err(format!(
                    "dimension mismatch: got {}, expected {}",
                    v.len(),
                    self.dims
                ));
            }
            let pos = g.ids.len();
            g.ids.push(id);
            g.vecs.extend_from_slice(&v);
            g.norms.push(norm(&v));
            g.assign.push(-1);
            g.id_pos.insert(id, pos);
        }
        Ok(())
    }
}

/// Distance between two raw rows under the inner metric (used during training).
#[inline]
fn dist_rows(inner: &Inner, a: &[f32], b: &[f32]) -> f32 {
    match inner.metric {
        Metric::Cos => {
            let denom = norm(a) * norm(b);
            if denom == 0.0 {
                1.0
            } else {
                1.0 - dot(a, b) / denom
            }
        }
        Metric::Ip => 1.0 - dot(a, b),
        Metric::L2sq => l2sq(a, b),
    }
}

/// Remove `id` from a locked `Inner` via swap-remove, keeping `id_pos`, the row
/// arrays and the inverted lists consistent. Caller guarantees `id` is present.
fn remove_locked(g: &mut Inner, id: u64) {
    let dims = g.dims;
    let pos = g.id_pos[&id];
    let last = g.ids.len() - 1;

    // Detach `pos` from its inverted list (if assigned).
    let c_pos = g.assign[pos];
    if c_pos >= 0 {
        let list = &mut g.lists[c_pos as usize];
        if let Some(i) = list.iter().position(|&p| p == pos) {
            list.swap_remove(i);
        }
    }

    if pos != last {
        // Move the last row into `pos`.
        let moved_id = g.ids[last];
        let moved_c = g.assign[last];
        g.ids.swap_remove(pos);
        g.assign.swap_remove(pos);
        g.norms.swap_remove(pos);
        // vecs is flat: copy the last row over `pos`, then truncate.
        let (head, tail) = g.vecs.split_at_mut(last * dims);
        head[pos * dims..(pos + 1) * dims].copy_from_slice(&tail[..dims]);
        g.vecs.truncate(last * dims);

        g.id_pos.insert(moved_id, pos);
        // Repoint the moved row in its list from `last` to `pos`.
        if moved_c >= 0 {
            let list = &mut g.lists[moved_c as usize];
            if let Some(i) = list.iter().position(|&p| p == last) {
                list[i] = pos;
            }
        }
    } else {
        g.ids.pop();
        g.assign.pop();
        g.norms.pop();
        g.vecs.truncate(last * dims);
    }
    g.id_pos.remove(&id);
}

/// Lloyd iterations during training — bounded so a large collection cannot stall
/// startup/reindex.
const IVF_KMEANS_ITERS: usize = 15;

#[cfg(test)]
mod tests {
    use super::*;

    fn v(xs: &[f32]) -> Vec<f32> {
        xs.to_vec()
    }

    #[test]
    fn empty_search_returns_nothing() {
        let idx = IvfFlatIndex::new(3, Metric::L2sq, 4, 2);
        assert!(idx.search(&[1.0, 0.0, 0.0], 5, None).unwrap().is_empty());
        assert_eq!(idx.len(), 0);
        assert!(!idx.is_trained());
    }

    #[test]
    fn brute_force_before_training_is_exact() {
        let idx = IvfFlatIndex::new(2, Metric::L2sq, 8, 2);
        idx.upsert(1, &v(&[0.0, 0.0])).unwrap();
        idx.upsert(2, &v(&[10.0, 10.0])).unwrap();
        idx.upsert(3, &v(&[1.0, 1.0])).unwrap();
        // Untrained: still returns the exact nearest.
        let r = idx.search(&[0.0, 0.0], 2, None).unwrap();
        assert_eq!(r[0].0, 1);
        assert_eq!(r[1].0, 3);
    }

    #[test]
    fn trained_search_finds_cluster_members() {
        let idx = IvfFlatIndex::new(2, Metric::L2sq, 2, 2);
        // Two well-separated clusters.
        for i in 0..10 {
            idx.upsert(i, &v(&[i as f32 * 0.01, 0.0])).unwrap();
        }
        for i in 10..20 {
            idx.upsert(i, &v(&[100.0 + i as f32 * 0.01, 100.0])).unwrap();
        }
        idx.train().unwrap();
        assert!(idx.is_trained());
        assert_eq!(idx.centroid_count(), 2);
        let r = idx.search(&[0.0, 0.0], 3, None).unwrap();
        // All three nearest should come from the first cluster (ids < 10).
        for (id, _) in &r {
            assert!(*id < 10, "unexpected id {id} from far cluster");
        }
    }

    #[test]
    fn upsert_replaces_and_counts() {
        let idx = IvfFlatIndex::new(2, Metric::L2sq, 4, 2);
        assert!(!idx.upsert(1, &v(&[0.0, 0.0])).unwrap());
        assert!(idx.upsert(1, &v(&[5.0, 5.0])).unwrap()); // replaced
        assert_eq!(idx.len(), 1);
        let r = idx.search(&[5.0, 5.0], 1, None).unwrap();
        assert_eq!(r[0].0, 1);
        assert!(r[0].1 < 0.001, "distance to exact match should be ~0");
    }

    #[test]
    fn remove_keeps_index_consistent() {
        let idx = IvfFlatIndex::new(2, Metric::L2sq, 3, 3);
        for i in 0..6 {
            idx.upsert(i, &v(&[i as f32, 0.0])).unwrap();
        }
        idx.train().unwrap();
        assert!(idx.remove(2));
        assert!(!idx.remove(2)); // already gone
        assert_eq!(idx.len(), 5);
        // The removed id must not appear; remaining ids must still be searchable.
        let r = idx.search(&[5.0, 0.0], 6, None).unwrap();
        let ids: Vec<u64> = r.iter().map(|(id, _)| *id).collect();
        assert!(!ids.contains(&2));
        assert!(ids.contains(&5));
        assert_eq!(ids.len(), 5);
    }

    #[test]
    fn add_after_training_is_findable() {
        let idx = IvfFlatIndex::new(2, Metric::L2sq, 2, 2);
        for i in 0..8 {
            idx.upsert(i, &v(&[i as f32, 0.0])).unwrap();
        }
        idx.train().unwrap();
        idx.upsert(99, &v(&[3.5, 0.0])).unwrap();
        let r = idx.search(&[3.5, 0.0], 1, None).unwrap();
        assert_eq!(r[0].0, 99);
    }

    #[test]
    fn cosine_metric_ranks_by_direction() {
        let idx = IvfFlatIndex::new(2, Metric::Cos, 4, 4);
        idx.upsert(1, &v(&[1.0, 0.0])).unwrap();
        idx.upsert(2, &v(&[0.0, 1.0])).unwrap();
        idx.upsert(3, &v(&[10.0, 0.0])).unwrap(); // same direction as id 1, bigger magnitude
        let r = idx.search(&[2.0, 0.0], 3, None).unwrap();
        // Cosine ignores magnitude: ids 1 and 3 tie at distance ~0, id 2 is far.
        assert!(r[0].0 == 1 || r[0].0 == 3);
        assert!(r[1].0 == 1 || r[1].0 == 3);
        assert_eq!(r[2].0, 2);
    }

    #[test]
    fn dimension_mismatch_errors() {
        let idx = IvfFlatIndex::new(3, Metric::L2sq, 2, 2);
        assert!(idx.upsert(1, &v(&[1.0, 2.0])).is_err());
        assert!(idx.search(&[1.0, 2.0], 1, None).is_err());
    }

    #[test]
    fn retrain_after_many_inserts() {
        let idx = IvfFlatIndex::new(4, Metric::L2sq, 4, 4);
        for i in 0..50 {
            idx.upsert(i, &v(&[i as f32, 0.0, 0.0, 0.0])).unwrap();
        }
        idx.train().unwrap();
        let c1 = idx.centroid_count();
        for i in 50..100 {
            idx.upsert(i, &v(&[i as f32, 0.0, 0.0, 0.0])).unwrap();
        }
        idx.train().unwrap(); // retrain over the larger set
        assert_eq!(idx.len(), 100);
        assert_eq!(c1, 4);
        // Exact nearest still correct after retrain.
        let r = idx.search(&[75.0, 0.0, 0.0, 0.0], 1, None).unwrap();
        assert_eq!(r[0].0, 75);
    }
}
