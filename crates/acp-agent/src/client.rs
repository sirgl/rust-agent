//! [`AcpClientAccess`]: the concrete [`agent_core::ClientAccess`] implementation
//! backed by a live ACP [`ConnectionTo<Client>`].
//!
//! This is the ACP dialect of the client seam: it translates the
//! provider-independent [`agent_core::ClientAccess`] calls (used by the shared
//! built-in tools) into ACP `fs/*` and `terminal/*` requests. The built-in
//! tools themselves never see this type — they only depend on the
//! [`agent_core::ClientAccess`] trait, so they are written once and reused by
//! every decision-layer provider.

use agent_client_protocol::schema::v1::{
    CreateElicitationRequest, CreateTerminalRequest, ElicitationAction, ElicitationFormMode,
    ElicitationSchema, ElicitationSessionScope, ReadTextFileRequest, ReleaseTerminalRequest,
    SessionId, TerminalOutputRequest, WaitForTerminalExitRequest, WriteTextFileRequest,
};
use agent_client_protocol::{Client, ConnectionTo};
use agent_core::{AgentError, ClientAccess, ElicitationOutcome, Result, TerminalOutcome};
use async_trait::async_trait;
use tracing::debug;

/// ACP-backed [`ClientAccess`] bound to a single session's connection.
pub struct AcpClientAccess {
    cx: ConnectionTo<Client>,
    session_id: SessionId,
}

impl AcpClientAccess {
    /// Create a client-access handle for the given connection and session.
    #[must_use]
    pub fn new(cx: ConnectionTo<Client>, session_id: SessionId) -> Self {
        Self { cx, session_id }
    }
}

/// Map an ACP transport error into a core error.
fn map_err(context: &str, err: impl std::fmt::Display) -> AgentError {
    AgentError::Other(format!("{context}: {err}"))
}

#[async_trait]
impl ClientAccess for AcpClientAccess {
    async fn read_text_file(&self, path: &str) -> Result<String> {
        debug!(path, "fs/read_text_file");
        let request = ReadTextFileRequest::new(self.session_id.clone(), path);
        let response = self
            .cx
            .send_request(request)
            .block_task()
            .await
            .map_err(|err| map_err("fs/read_text_file failed", err))?;
        Ok(response.content)
    }

    async fn write_text_file(&self, path: &str, content: &str) -> Result<()> {
        debug!(path, "fs/write_text_file");
        let request = WriteTextFileRequest::new(self.session_id.clone(), path, content);
        self.cx
            .send_request(request)
            .block_task()
            .await
            .map_err(|err| map_err("fs/write_text_file failed", err))?;
        Ok(())
    }

    async fn run_terminal(&self, command: &str, args: &[String]) -> Result<TerminalOutcome> {
        debug!(command, "terminal/create");
        // 1. Create the terminal and start the command.
        let create = CreateTerminalRequest::new(self.session_id.clone(), command)
            .args(args.to_vec());
        let created = self
            .cx
            .send_request(create)
            .block_task()
            .await
            .map_err(|err| map_err("terminal/create failed", err))?;
        let terminal_id = created.terminal_id;

        // 2. Wait for the command to exit.
        let exit = self
            .cx
            .send_request(WaitForTerminalExitRequest::new(
                self.session_id.clone(),
                terminal_id.clone(),
            ))
            .block_task()
            .await
            .map_err(|err| map_err("terminal/wait_for_exit failed", err))?;

        // 3. Collect the captured output.
        let output = self
            .cx
            .send_request(TerminalOutputRequest::new(
                self.session_id.clone(),
                terminal_id.clone(),
            ))
            .block_task()
            .await
            .map_err(|err| map_err("terminal/output failed", err))?;

        // 4. Release the terminal (best-effort; ignore release errors).
        if let Err(err) = self
            .cx
            .send_request(ReleaseTerminalRequest::new(
                self.session_id.clone(),
                terminal_id,
            ))
            .block_task()
            .await
        {
            debug!(%err, "terminal/release failed (ignored)");
        }

        Ok(TerminalOutcome {
            output: output.output,
            truncated: output.truncated,
            exit_code: exit.exit_status.exit_code,
            signal: exit.exit_status.signal,
        })
    }

    async fn request_elicitation(
        &self,
        message: &str,
        requested_schema: serde_json::Value,
    ) -> Result<ElicitationOutcome> {
        debug!(message, "elicitation/create");
        // Map the provider-independent JSON schema into the ACP form schema.
        let schema: ElicitationSchema = serde_json::from_value(requested_schema)
            .map_err(|err| map_err("invalid elicitation schema", err))?;
        let scope = ElicitationSessionScope::new(self.session_id.clone());
        let mode = ElicitationFormMode::new(scope, schema);
        let request = CreateElicitationRequest::new(mode, message);

        let response = self
            .cx
            .send_request(request)
            .block_task()
            .await
            .map_err(|err| map_err("elicitation/create failed", err))?;

        match response.action {
            ElicitationAction::Accept(accept) => {
                let content = accept.content.unwrap_or_default();
                let value = serde_json::to_value(content)
                    .map_err(|err| map_err("invalid elicitation content", err))?;
                Ok(ElicitationOutcome::Accepted(value))
            }
            ElicitationAction::Decline => Ok(ElicitationOutcome::Declined),
            ElicitationAction::Cancel => Ok(ElicitationOutcome::Cancelled),
            other => Err(AgentError::Other(format!(
                "unsupported elicitation action: {other:?}"
            ))),
        }
    }
}
