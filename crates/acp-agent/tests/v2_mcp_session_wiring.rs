#![cfg(feature = "unstable_protocol_v2")]

//! End-to-end test proving MCP tools become available through *real* runtime
//! wiring: `session/new`'s `mcp_servers` connects an actual child-process MCP
//! server (over stdio), registers its tools into that session's registry, and
//! a subsequent `session/prompt` can dispatch a tool call against it -- not
//! just the seam-level unit tests in `mcp-client` that exercise an in-memory
//! stub or a directly-constructed [`StdioMcpConnection`].

use std::io::Write;
use std::os::unix::fs::PermissionsExt;
use std::sync::{Arc, Mutex};

use agent_client_protocol::schema::v2::{
    ContentBlock, InitializeRequest, McpServer, McpServerStdio, NewSessionRequest, PromptRequest,
    SessionUpdate, StopReason, ToolCallStatus, UpdateSessionNotification,
};
use agent_client_protocol::schema::ProtocolVersion;
use agent_client_protocol::{on_receive_notification, Client};
use agent_core::{
    NextTurnService, StopReason as CoreStopReason, ToolCallId, ToolRegistry, TurnEvent,
};
use serde_json::json;

use acp_agent::{build_agent, AgentConfig, AgentDeps, NextTurnFactory};

mod common;

/// A tiny, self-contained MCP server implemented as a POSIX shell script; see
/// `mcp-client/src/stdio.rs`'s own tests for the same fixture used to exercise
/// [`StdioMcpConnection`] directly. It answers `initialize`, `tools/list`
/// (advertising a single `ping` tool), and `tools/call`.
const MOCK_SERVER: &str = r#"#!/bin/sh
while IFS= read -r line; do
  case "$line" in
    *'"method":"initialize"'*)
      id=$(printf '%s' "$line" | sed -n 's/.*"id":\([0-9]*\).*/\1/p')
      printf '{"jsonrpc":"2.0","id":%s,"result":{"protocolVersion":"2024-11-05","capabilities":{},"serverInfo":{"name":"mock","version":"0"}}}\n' "$id"
      ;;
    *'"method":"notifications/initialized"'*)
      : ;;
    *'"method":"tools/list"'*)
      id=$(printf '%s' "$line" | sed -n 's/.*"id":\([0-9]*\).*/\1/p')
      printf '{"jsonrpc":"2.0","id":%s,"result":{"tools":[{"name":"ping","description":"pong back","inputSchema":{"type":"object"}}]}}\n' "$id"
      ;;
    *'"method":"tools/call"'*)
      id=$(printf '%s' "$line" | sed -n 's/.*"id":\([0-9]*\).*/\1/p')
      printf '{"jsonrpc":"2.0","id":%s,"result":{"content":[{"type":"text","text":"pong"}],"isError":false}}\n' "$id"
      ;;
  esac
done
"#;

/// Write the mock server script to a fresh temp file and mark it executable.
fn write_mock_server() -> std::path::PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let path = std::env::temp_dir().join(format!(
        "acp-agent-mock-mcp-{}-{nanos}.sh",
        std::process::id()
    ));
    let mut file = std::fs::File::create(&path).unwrap();
    file.write_all(MOCK_SERVER.as_bytes()).unwrap();
    file.flush().unwrap();
    let mut perms = std::fs::metadata(&path).unwrap().permissions();
    perms.set_mode(0o755);
    std::fs::set_permissions(&path, perms).unwrap();
    path
}

/// Build [`AgentDeps`] whose decision layer replays the given scripted rounds,
/// with an otherwise-empty base tool registry (any tools available during the
/// turn must come from the session's own `mcp_servers`, proving the runtime
/// wiring path rather than a pre-populated base registry).
fn deps_with_script(script: Vec<Vec<TurnEvent>>) -> AgentDeps {
    let factory: NextTurnFactory = Arc::new(move |_selection| {
        let service: Arc<dyn NextTurnService> =
            Arc::new(turn_replay::ReplayTurnService::new(script.clone()));
        service
    });
    AgentDeps {
        next_turn_factory: factory,
        tools: ToolRegistry::new(),
        config: AgentConfig {
            enable_subagents: false,
            require_submit_result: false,
            ..AgentConfig::default()
        },
        store: acp_agent::default_store(),
        turn_observer: None,
        inspector_base_url: None,
    }
}

/// A two-round script: request the MCP-provided `ping` tool, then finish.
fn mcp_tool_call_script() -> Vec<Vec<TurnEvent>> {
    vec![
        vec![
            TurnEvent::ToolCallRequested {
                id: ToolCallId::new("call-1"),
                name: "mcp__mockserver__ping".to_string(),
                arguments: json!({}),
            },
            TurnEvent::TurnFinished {
                stop_reason: CoreStopReason::ToolUse,
            },
        ],
        vec![
            TurnEvent::TextDelta("done".to_string()),
            TurnEvent::TurnFinished {
                stop_reason: CoreStopReason::EndTurn,
            },
        ],
    ]
}

#[tokio::test]
async fn mcp_stdio_server_from_session_new_is_callable_in_prompt_turn() {
    let server_path = write_mock_server();

    let deps = deps_with_script(mcp_tool_call_script());
    let agent = build_agent(deps);

    let updates: Arc<Mutex<Vec<SessionUpdate>>> = Arc::new(Mutex::new(Vec::new()));
    let collected = updates.clone();
    let completion = updates.clone();

    let server_path_for_request = server_path.clone();
    let stop = Client
        .v2()
        .name("test-client")
        .on_receive_notification(
            async move |notif: UpdateSessionNotification, _cx| {
                collected.lock().unwrap().push(notif.update);
                Ok(())
            },
            on_receive_notification!(),
        )
        .connect_with(agent, async move |cx| {
            cx.send_request(InitializeRequest::new(
                ProtocolVersion::V2,
                agent_client_protocol::schema::v2::Implementation::new("test-client", "0.1"),
            ))
            .block_task()
            .await?;

            // This is the ACP-native path: the client declares the MCP server
            // it wants for this session directly in `session/new`, exactly as
            // a real editor would.
            let new_session =
                cx.send_request(NewSessionRequest::new("/tmp").mcp_servers(vec![
                    McpServer::Stdio(McpServerStdio::new("mockserver", server_path_for_request)),
                ]))
                .block_task()
                .await?;

            cx.send_request(PromptRequest::new(
                new_session.session_id,
                vec![ContentBlock::from("ping the mcp server")],
            ))
            .block_task()
            .await?;
            Ok(common::wait_for_idle(&completion).await)
        })
        .await
        .expect("connection completed");

    let _ = std::fs::remove_file(&server_path);

    assert_eq!(stop, StopReason::EndTurn);

    let statuses: Vec<ToolCallStatus> = common::tool_statuses(&updates.lock().unwrap());
    // The MCP tool was genuinely connected and dispatched at runtime: pending
    // (creation) -> in_progress -> completed, never failed.
    assert!(statuses.contains(&ToolCallStatus::InProgress));
    assert!(statuses.contains(&ToolCallStatus::Completed));
    assert!(!statuses.contains(&ToolCallStatus::Failed));
}

#[tokio::test]
async fn unknown_mcp_transport_is_skipped_without_failing_session_new() {
    // A server with no matching tool still lets session/new succeed and the
    // turn proceed (unresolved tool calls fail gracefully rather than
    // panicking or blocking session creation).
    let deps = deps_with_script(vec![vec![
        TurnEvent::TextDelta("hi".to_string()),
        TurnEvent::TurnFinished {
            stop_reason: CoreStopReason::EndTurn,
        },
    ]]);
    let agent = build_agent(deps);
    let updates: common::Updates = Arc::new(Mutex::new(Vec::new()));
    let collected = updates.clone();
    let completion = updates.clone();

    let stop =
        Client
            .v2()
            .name("test-client")
            .on_receive_notification(
                async move |notif: UpdateSessionNotification, _cx| {
                    collected.lock().unwrap().push(notif.update);
                    Ok(())
                },
                on_receive_notification!(),
            )
            .connect_with(agent, async move |cx| {
                cx.send_request(InitializeRequest::new(
                    ProtocolVersion::V2,
                    agent_client_protocol::schema::v2::Implementation::new("test-client", "0.1"),
                ))
                .block_task()
                .await?;
                let new_session = cx
                    .send_request(NewSessionRequest::new("/tmp").mcp_servers(vec![
                        McpServer::Http(agent_client_protocol::schema::v2::McpServerHttp::new(
                            "unsupported-http",
                            "https://example.invalid/mcp",
                        )),
                    ]))
                    .block_task()
                    .await?;
                cx.send_request(PromptRequest::new(
                    new_session.session_id,
                    vec![ContentBlock::from("hi")],
                ))
                .block_task()
                .await?;
                Ok(common::wait_for_idle(&completion).await)
            })
            .await
            .expect("connection completed");

    assert_eq!(stop, StopReason::EndTurn);
}
