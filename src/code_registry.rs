//! Per-project registry of code-index database handles.
//!
//! Code search is physically partitioned one SQLite file per project
//! (`<memory_file>.code/<project>.code.db`). This keeps each project's FTS
//! index small and independent, makes dropping/re-indexing a project trivial,
//! and guarantees the regular knowledge-graph tools (which only ever touch the
//! main memory database) can never see or mutate code-symbol data.
//!
//! [`init`] is called once at server startup; [`resolve`] lazily opens (and
//! caches) the handle for a given project. A project handle owns an in-memory
//! entity cache, so to keep cache coherence there must be **at most one live
//! [`GraphHandle`] per project file** in the process. That invariant is upheld
//! by tracking each handle with a `Weak`: as long as any caller (e.g. a running
//! watcher) holds a strong reference, `resolve` hands back that same instance.
//! A small LRU of strong references keeps recently-used, otherwise-idle handles
//! warm to avoid reopen churn.

#![cfg(feature = "code")]

use std::collections::HashMap;
use std::num::NonZeroUsize;
use std::path::PathBuf;
use std::sync::{Arc, OnceLock, Weak};

use lru::LruCache;
use parking_lot::Mutex;

use crate::config::{Durability, SqliteTuning};
use crate::errors::{MCSError, Result};
use crate::kg::GraphHandle;

/// Default project when a caller omits the `project` argument.
pub const DEFAULT_PROJECT: &str = "default";
/// Upper bound on project-name length (also keeps the derived filename sane).
const MAX_PROJECT_LEN: usize = 64;
/// How many idle project handles stay warm before the LRU evicts (and closes)
/// them. A watcher's own strong reference keeps a project open regardless.
const MAX_WARM_HANDLES: usize = 16;

/// Construction parameters captured at startup so [`resolve`] can build a
/// [`GraphHandle`] per project on demand.
struct RegistryConfig {
    base: PathBuf,
    durability: Durability,
    tuning: SqliteTuning,
    lru_cache: NonZeroUsize,
    read_pool_size: usize,
}

struct Inner {
    /// Canonical instance per project — `Weak` so a handle is dropped (and the
    /// SQLite connections closed) once no caller and no warm slot hold it.
    live: HashMap<String, Weak<GraphHandle>>,
    /// Recently-used handles kept alive to avoid reopen churn.
    warm: LruCache<String, Arc<GraphHandle>>,
}

static CONFIG: OnceLock<RegistryConfig> = OnceLock::new();
static INNER: OnceLock<Mutex<Inner>> = OnceLock::new();

/// Initialize the registry. Idempotent; safe to call once at startup. `base` is
/// the directory under which per-project databases are created.
pub fn init(
    base: PathBuf,
    durability: Durability,
    tuning: SqliteTuning,
    lru_cache: NonZeroUsize,
    read_pool_size: usize,
) {
    // Best-effort: a failure here surfaces later as an open error from `resolve`.
    let _ = std::fs::create_dir_all(&base);
    let _ = CONFIG.set(RegistryConfig {
        base,
        durability,
        tuning,
        lru_cache,
        read_pool_size,
    });
    let warm = LruCache::new(NonZeroUsize::new(MAX_WARM_HANDLES).expect("MAX_WARM_HANDLES > 0"));
    let _ = INNER.set(Mutex::new(Inner {
        live: HashMap::new(),
        warm,
    }));
}

/// Validate a project identifier. It is used verbatim as a filename component,
/// so restrict it to a safe, traversal-free character set.
pub fn validate_project(project: &str) -> Result<()> {
    let ok = !project.is_empty()
        && project.len() <= MAX_PROJECT_LEN
        && project
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-');
    if ok {
        Ok(())
    } else {
        Err(MCSError::InvalidParams(format!(
            "invalid project '{project}': use 1-{MAX_PROJECT_LEN} chars of [A-Za-z0-9_-]"
        )))
    }
}

/// Resolve the (lazily opened) database handle for `project`, opening it if
/// necessary. Returns the single canonical instance for that project so callers
/// share one entity cache.
pub fn resolve(project: &str) -> Result<Arc<GraphHandle>> {
    validate_project(project)?;
    let cfg = CONFIG.get().ok_or_else(|| {
        MCSError::InvalidParams("code registry not initialized (start the server with --code)".into())
    })?;
    let inner = INNER.get().expect("registry inner set alongside config");

    let mut g = inner.lock();
    // Reuse the canonical instance if it is still alive anywhere.
    if let Some(existing) = g.live.get(project).and_then(Weak::upgrade) {
        g.warm.put(project.to_string(), Arc::clone(&existing));
        return Ok(existing);
    }

    // Cold path (rare): opening a project. Drop any `Weak`s whose handles have
    // been closed so `live` stays bounded by the live project count, not by the
    // number of projects ever touched.
    g.live.retain(|_, w| w.strong_count() > 0);

    let path = cfg.base.join(format!("{project}.code.db"));
    let handle = Arc::new(GraphHandle::new(
        &path,
        cfg.durability,
        cfg.tuning,
        cfg.lru_cache,
        cfg.read_pool_size,
    )?);
    g.live.insert(project.to_string(), Arc::downgrade(&handle));
    g.warm.put(project.to_string(), Arc::clone(&handle));
    Ok(handle)
}
