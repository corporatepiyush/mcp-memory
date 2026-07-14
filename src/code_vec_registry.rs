//! Per-project registry of code-symbol vector indexes (HNSW ANN).
//!
//! Semantic code search reuses the [`VectorStore`](crate::vector_store::VectorStore)
//! (usearch HNSW) directly on top of each project's code database: code symbols
//! are already stored as entities in `<memory_file>.code/<project>.code.db`, and
//! `VectorStore` keys embeddings by the `entity` row id in that same file. So an
//! HNSW index opened on the project DB indexes exactly the code symbols, with no
//! separate identifier space.
//!
//! Like [`crate::code_registry`], there must be **at most one live
//! [`VectorStore`] per project file** in the process (the in-memory HNSW graph
//! must not diverge from a second instance's). The same `Weak` + warm-LRU scheme
//! upholds that invariant. [`resolve`] first ensures the project's
//! knowledge-graph schema exists (via [`crate::code_registry::resolve`]) so the
//! vector store can read the `entity` table on open.

#![cfg(feature = "code")]

use std::collections::HashMap;
use std::num::NonZeroUsize;
use std::path::PathBuf;
use std::sync::{Arc, OnceLock, Weak};

use lru::LruCache;
use parking_lot::Mutex;

use crate::errors::{MCSError, Result};
use crate::vector_store::{VectorConfig, VectorStore};

/// Default embedding dimension for code semantic search. Matches common
/// open-weight code/text embedders (e.g. bge-base, nomic-embed, all-mpnet).
pub const DEFAULT_CODE_EMBEDDING_DIMS: u32 = 768;

/// How many idle vector indexes stay warm before the LRU evicts (and frees the
/// HNSW graph) them.
const MAX_WARM_HANDLES: usize = 8;

struct RegistryConfig {
    base: PathBuf,
    dims: u32,
}

struct Inner {
    /// Canonical instance per project — `Weak` so the index is freed once no
    /// caller and no warm slot hold it.
    live: HashMap<String, Weak<VectorStore>>,
    /// Recently-used indexes kept alive to avoid reopen churn.
    warm: LruCache<String, Arc<VectorStore>>,
}

static CONFIG: OnceLock<RegistryConfig> = OnceLock::new();
static INNER: OnceLock<Mutex<Inner>> = OnceLock::new();

/// Initialize the registry. Idempotent; called once at startup when code
/// indexing is enabled. `base` is the directory holding per-project databases
/// (the same one passed to [`crate::code_registry::init`]).
pub fn init(base: PathBuf, dims: u32) {
    let _ = CONFIG.set(RegistryConfig { base, dims });
    let warm = LruCache::new(NonZeroUsize::new(MAX_WARM_HANDLES).expect("MAX_WARM_HANDLES > 0"));
    let _ = INNER.set(Mutex::new(Inner {
        live: HashMap::new(),
        warm,
    }));
}

/// The configured embedding dimension, or the default if the registry is not
/// initialized (e.g. code disabled).
pub fn embedding_dims() -> u32 {
    CONFIG.get().map(|c| c.dims).unwrap_or(DEFAULT_CODE_EMBEDDING_DIMS)
}

/// Resolve the (lazily opened) HNSW vector index for `project`, opening it if
/// necessary. Returns the single canonical instance so callers share one index.
pub fn resolve(project: &str) -> Result<Arc<VectorStore>> {
    crate::code_registry::validate_project(project)?;
    let cfg = CONFIG.get().ok_or_else(|| {
        MCSError::InvalidParams(
            "code vector registry not initialized (start the server with --enable-code)".into(),
        )
    })?;
    let inner = INNER.get().expect("registry inner set alongside config");

    // Ensure the project's knowledge-graph schema (the `entity` table) exists
    // before the vector store opens the file and tries to read it.
    let _ = crate::code_registry::resolve(project)?;

    let mut g = inner.lock();
    if let Some(existing) = g.live.get(project).and_then(Weak::upgrade) {
        g.warm.put(project.to_string(), Arc::clone(&existing));
        return Ok(existing);
    }

    g.live.retain(|_, w| w.strong_count() > 0);

    let path = cfg.base.join(format!("{project}.code.db"));
    let store = Arc::new(VectorStore::with_config(&path, &VectorConfig::new(cfg.dims))?);
    g.live.insert(project.to_string(), Arc::downgrade(&store));
    g.warm.put(project.to_string(), Arc::clone(&store));
    Ok(store)
}
