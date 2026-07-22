//! A tiny, dependency-free HTTP/1.1 server that exposes the [`Inspector`] as a
//! simple web UI plus a JSON API.
//!
//! It intentionally avoids pulling in a web framework: it speaks just enough
//! HTTP to serve three `GET` routes over `tokio`'s TCP listener:
//!
//! - `GET /`            — the HTML dashboard (auto-refreshing).
//! - `GET /api/sends`   — every captured request as JSON, newest first.
//! - `GET /api/state`   — the latest persisted state per session, as JSON.
//! - anything else      — `404`.

use std::net::SocketAddr;

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

use crate::store::Inspector;

/// The embedded HTML dashboard. It fetches `/api/sends` and renders the
/// captured requests grouped by session, refreshing periodically.
const INDEX_HTML: &str = include_str!("index.html");

/// Start the inspector HTTP server, binding to `addr`.
///
/// The accept loop runs on a background task; the returned [`SocketAddr`] is the
/// actual bound address (useful when binding to port `0` in tests). Individual
/// connection errors are logged and never crash the server.
///
/// # Errors
///
/// Returns an error only if the initial bind fails.
pub async fn serve(inspector: Inspector, addr: SocketAddr) -> std::io::Result<SocketAddr> {
    let listener = TcpListener::bind(addr).await?;
    let local = listener.local_addr()?;
    tokio::spawn(async move {
        loop {
            match listener.accept().await {
                Ok((stream, _peer)) => {
                    let inspector = inspector.clone();
                    tokio::spawn(async move {
                        if let Err(err) = handle_connection(stream, inspector).await {
                            tracing::debug!(%err, "llm-inspector: connection error");
                        }
                    });
                }
                Err(err) => {
                    tracing::warn!(%err, "llm-inspector: accept failed");
                }
            }
        }
    });
    Ok(local)
}

/// Start the inspector on `preferred_addr`, falling back to an ephemeral port
/// on the same interface when that specific port is already occupied.
///
/// ACP clients may keep a pool of agent processes alive. Each process owns an
/// independent in-memory inspector, so sharing the first process's fixed URL
/// would show the wrong sessions. Falling back to port `0` gives every process
/// its own reachable server and lets callers publish the actual bound address.
/// Errors other than [`std::io::ErrorKind::AddrInUse`] are returned unchanged.
pub async fn serve_with_fallback(
    inspector: Inspector,
    preferred_addr: SocketAddr,
) -> std::io::Result<SocketAddr> {
    match serve(inspector.clone(), preferred_addr).await {
        Ok(bound) => Ok(bound),
        Err(err) if err.kind() == std::io::ErrorKind::AddrInUse && preferred_addr.port() != 0 => {
            let mut fallback_addr = preferred_addr;
            fallback_addr.set_port(0);
            tracing::warn!(
                %preferred_addr,
                "llm-inspector: preferred address is occupied; using a process-local port"
            );
            serve(inspector, fallback_addr).await
        }
        Err(err) => Err(err),
    }
}

/// Read one request from `stream`, route it, and write the response.
async fn handle_connection(mut stream: TcpStream, inspector: Inspector) -> std::io::Result<()> {
    let path = match read_request_path(&mut stream).await? {
        Some(path) => path,
        None => return Ok(()),
    };

    let response = route(&path, &inspector);
    stream.write_all(response.as_bytes()).await?;
    stream.flush().await
}

/// Read bytes until the end of the HTTP request headers and return the request
/// target (path) from the request line, if the request looks well-formed.
async fn read_request_path(stream: &mut TcpStream) -> std::io::Result<Option<String>> {
    let mut buf = Vec::with_capacity(1024);
    let mut chunk = [0u8; 1024];
    // Read until we see the end-of-headers marker, with a small cap so a
    // misbehaving client cannot make us buffer unbounded data.
    loop {
        let n = stream.read(&mut chunk).await?;
        if n == 0 {
            break;
        }
        buf.extend_from_slice(&chunk[..n]);
        if buf.windows(4).any(|w| w == b"\r\n\r\n") || buf.len() > 64 * 1024 {
            break;
        }
    }
    let text = String::from_utf8_lossy(&buf);
    let request_line = text.lines().next().unwrap_or("");
    let mut parts = request_line.split_whitespace();
    let method = parts.next().unwrap_or("");
    let target = parts.next().unwrap_or("");
    if method != "GET" || target.is_empty() {
        return Ok(Some(String::new()));
    }
    Ok(Some(target.to_string()))
}

/// Map a request path to a complete HTTP response string.
fn route(target: &str, inspector: &Inspector) -> String {
    // Split the request target into a path and an optional query string.
    let (path, query) = match target.split_once('?') {
        Some((p, q)) => (p, q),
        None => (target, ""),
    };
    match path {
        "" => http_response(
            "400 Bad Request",
            "text/plain; charset=utf-8",
            "bad request",
        ),
        "/" | "/index.html" => http_response("200 OK", "text/html; charset=utf-8", INDEX_HTML),
        "/api/sends" => {
            let mut sends = inspector.snapshot();
            // Optional `?session=<id>` filter so a per-session UI link shows
            // only that session's requests.
            if let Some(session) = query_param(query, "session") {
                sends.retain(|s| s.session_id == session);
            }
            let body = serde_json::to_string(&sends).unwrap_or_else(|_| "[]".to_string());
            http_response("200 OK", "application/json; charset=utf-8", &body)
        }
        "/api/state" => {
            let mut states = inspector.session_states();
            // Optional `?session=<id>` filter so a per-session UI link shows
            // only that session's state.
            if let Some(session) = query_param(query, "session") {
                states.retain(|s| s.session_id == session);
            }
            let body = serde_json::to_string(&states).unwrap_or_else(|_| "[]".to_string());
            http_response("200 OK", "application/json; charset=utf-8", &body)
        }
        _ => http_response("404 Not Found", "text/plain; charset=utf-8", "not found"),
    }
}

/// Extract a query parameter value by key from a raw query string, applying
/// minimal `%`-decoding for the few characters session ids can contain.
fn query_param(query: &str, key: &str) -> Option<String> {
    query.split('&').find_map(|pair| {
        let (k, v) = pair.split_once('=')?;
        (k == key).then(|| percent_decode(v))
    })
}

/// Minimal percent-decoder: turns `%XX` escapes and `+` back into bytes.
fn percent_decode(input: &str) -> String {
    let bytes = input.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'%' if i + 2 < bytes.len() => {
                let hi = (bytes[i + 1] as char).to_digit(16);
                let lo = (bytes[i + 2] as char).to_digit(16);
                if let (Some(hi), Some(lo)) = (hi, lo) {
                    out.push((hi * 16 + lo) as u8);
                    i += 3;
                    continue;
                }
                out.push(bytes[i]);
                i += 1;
            }
            b'+' => {
                out.push(b' ');
                i += 1;
            }
            b => {
                out.push(b);
                i += 1;
            }
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// Build a minimal HTTP/1.1 response with the given status, content type, body.
fn http_response(status: &str, content_type: &str, body: &str) -> String {
    format!(
        "HTTP/1.1 {status}\r\n\
         Content-Type: {content_type}\r\n\
         Content-Length: {len}\r\n\
         Access-Control-Allow-Origin: *\r\n\
         Connection: close\r\n\
         \r\n\
         {body}",
        len = body.len(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn routes_index_and_api() {
        let insp = Inspector::new();
        insp.record_provider(
            "anthropic",
            Some("s1"),
            "m1",
            &serde_json::json!({"messages": []}),
        );
        insp.record_provider(
            "anthropic",
            Some("s2"),
            "m2",
            &serde_json::json!({"messages": []}),
        );

        let index = route("/", &insp);
        assert!(index.contains("200 OK"));
        assert!(index.contains("text/html"));
        assert!(index.contains("todo list ("));
        assert!(index.contains("item.status"));

        let api = route("/api/sends?x=1", &insp);
        assert!(api.contains("200 OK"));
        assert!(api.contains("application/json"));
        assert!(api.contains("\"session_id\":\"s1\""));

        // The `?session=` filter narrows to one session.
        let filtered = route("/api/sends?session=s2", &insp);
        assert!(filtered.contains("\"session_id\":\"s2\""));
        assert!(!filtered.contains("\"session_id\":\"s1\""));

        let missing = route("/nope", &insp);
        assert!(missing.contains("404 Not Found"));
    }

    #[test]
    fn routes_session_state_with_filter() {
        use agent_core::SessionState;
        let insp = Inspector::new();
        let mut a = SessionState::new("s1");
        a.push_user_text("hi");
        let mut b = SessionState::new("s2");
        b.push_user_text("yo");
        insp.record_session_state(&agent_core::SessionStateSnapshot::from_state(&a));
        insp.record_session_state(&agent_core::SessionStateSnapshot::from_state(&b));

        let all = route("/api/state", &insp);
        assert!(all.contains("200 OK"));
        assert!(all.contains("application/json"));
        assert!(all.contains("\"session_id\":\"s1\""));
        assert!(all.contains("\"session_id\":\"s2\""));
        assert!(all.contains("\"todo_list\""));

        let filtered = route("/api/state?session=s2", &insp);
        assert!(filtered.contains("\"session_id\":\"s2\""));
        assert!(!filtered.contains("\"session_id\":\"s1\""));
    }

    #[tokio::test]
    async fn serves_over_tcp_end_to_end() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        let insp = Inspector::new();
        insp.record_provider(
            "anthropic",
            Some("sess-42"),
            "claude",
            &serde_json::json!({"messages": [1, 2]}),
        );
        let addr = serve(insp, "127.0.0.1:0".parse().unwrap()).await.unwrap();

        let mut stream = TcpStream::connect(addr).await.unwrap();
        stream
            .write_all(b"GET /api/sends HTTP/1.1\r\nHost: x\r\n\r\n")
            .await
            .unwrap();
        let mut resp = Vec::new();
        stream.read_to_end(&mut resp).await.unwrap();
        let resp = String::from_utf8_lossy(&resp);
        assert!(resp.contains("200 OK"));
        assert!(resp.contains("sess-42"));
    }

    #[tokio::test]
    async fn occupied_preferred_port_falls_back_to_process_local_port() {
        let occupied = serve(Inspector::new(), "127.0.0.1:0".parse().unwrap())
            .await
            .unwrap();

        let bound = serve_with_fallback(Inspector::new(), occupied)
            .await
            .expect("a pooled agent process must still start its own inspector");

        assert_ne!(bound, occupied);
        assert_eq!(bound.ip(), occupied.ip());
        let _stream = TcpStream::connect(bound)
            .await
            .expect("fallback inspector must be reachable");
    }
}
