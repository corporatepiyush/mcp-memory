//! Server-side TLS (HTTPS) for the Streamable HTTP transport.
//!
//! TLS is opt-in: the HTTP transport only serves over TLS when both a
//! `--tls-cert` and `--tls-key` are configured, otherwise it stays plaintext
//! (the default). Backed by rustls with the `ring` crypto provider — no
//! OpenSSL/aws-lc.

use std::path::Path;

/// Install the rustls `ring` crypto provider as the process default.
///
/// Idempotent — only the first install in the process wins. The rustls
/// `ServerConfig` builder reads this process default, so it must be installed
/// before any TLS config is built.
pub fn ensure_crypto_provider() {
    let _ = rustls::crypto::ring::default_provider().install_default();
}

/// Build an axum-server rustls config from a PEM certificate chain and private
/// key. Installs the `ring` provider on first call. The returned `io::Error`
/// maps onto `MCSError::IoError`.
pub async fn server_config(
    cert_path: &Path,
    key_path: &Path,
) -> std::io::Result<axum_server::tls_rustls::RustlsConfig> {
    ensure_crypto_provider();
    axum_server::tls_rustls::RustlsConfig::from_pem_file(cert_path, key_path).await
}
