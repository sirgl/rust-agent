//! Integration tests for ACP configuration options (model and effort selection).

use std::sync::{Arc, Mutex};

use agent_client_protocol::schema::v1::{
    ContentBlock, InitializeRequest, NewSessionRequest, PromptRequest,
    SessionConfigId, SessionConfigKind, SessionConfigValueId,
    SetSessionConfigOptionRequest,
};
use agent_client_protocol::schema::ProtocolVersion;
use agent_client_protocol::Client;
use agent_core::{NextTurnService, StopReason as CoreStopReason, ToolRegistry, TurnEvent};
use acp_agent::{build_agent, AgentConfig, AgentDeps, NextTurnFactory};
use turn_anthropic::Effort;

#[tokio::test]
async fn test_config_options_e2e() {
    // A spy factory that records the selection passed to it for each turn.
    let last_selection = Arc::new(Mutex::new(None));
    let last_selection_clone = last_selection.clone();

    let factory: NextTurnFactory = Arc::new(move |selection| {
        *last_selection_clone.lock().unwrap() = Some(selection.clone());
        let service: Arc<dyn NextTurnService> = Arc::new(turn_replay::ReplayTurnService::new(vec![vec![
            TurnEvent::TextDelta("Hello".to_string()),
            TurnEvent::TurnFinished {
                stop_reason: CoreStopReason::EndTurn,
            },
        ]]));
        service
    });

    let deps = AgentDeps {
        next_turn_factory: factory,
        tools: ToolRegistry::new(),
        config: AgentConfig::default(),
    };
    let agent = build_agent(deps);

    Client
        .builder()
        .name("test-client")
        .connect_with(agent, async move |cx| {
            cx.send_request(InitializeRequest::new(ProtocolVersion::V1))
                .block_task()
                .await?;

            // 1. session/new returns expected config options
            let new_session = cx
                .send_request(NewSessionRequest::new("/tmp"))
                .block_task()
                .await?;
            
            let options = new_session.config_options.expect("missing config options");
            
            let model_opt = options.iter().find(|o| &*o.id.0 == "model").expect("missing model option");
            let effort_opt = options.iter().find(|o| &*o.id.0 == "effort").expect("missing effort option");

            // Default model should be sonnet, default effort should be high
            if let SessionConfigKind::Select(select) = &model_opt.kind {
                assert_eq!(&*select.current_value.0, "claude-3-5-sonnet-latest");
            } else {
                panic!("model option is not a select");
            }

            if let SessionConfigKind::Select(select) = &effort_opt.kind {
                assert_eq!(&*select.current_value.0, "high");
            } else {
                panic!("effort option is not a select");
            }

            // 2. session/set_config_option updates the session state and returns the current options
            let update_res = cx.send_request(SetSessionConfigOptionRequest::new(
                new_session.session_id.clone(),
                SessionConfigId::new("effort"),
                SessionConfigValueId::new("low")
            ))
            .block_task()
            .await?;

            let effort_opt = update_res.config_options.iter().find(|o| &*o.id.0 == "effort").expect("missing effort option");
            if let SessionConfigKind::Select(select) = &effort_opt.kind {
                assert_eq!(&*select.current_value.0, "low");
            } else {
                panic!("effort option is not a select");
            }

            // 3. session/prompt uses the updated selection
            cx.send_request(PromptRequest::new(
                new_session.session_id.clone(),
                vec![ContentBlock::from("hi")],
            ))
            .block_task()
            .await?;
            
            {
                let selection = last_selection.lock().unwrap().take().expect("no selection recorded");
                assert_eq!(selection.effort, Effort::Low);
                assert_eq!(selection.model, "claude-3-5-sonnet-latest");
            }

            // 4. Update model and verify next prompt uses it
            cx.send_request(SetSessionConfigOptionRequest::new(
                new_session.session_id.clone(),
                SessionConfigId::new("model"),
                SessionConfigValueId::new("claude-3-5-haiku-latest")
            ))
            .block_task()
            .await?;

            cx.send_request(PromptRequest::new(
                new_session.session_id.clone(),
                vec![ContentBlock::from("hi again")],
            ))
            .block_task()
            .await?;

            {
                let selection = last_selection.lock().unwrap().take().expect("no selection recorded");
                assert_eq!(selection.model, "claude-3-5-haiku-latest");
                assert_eq!(selection.effort, Effort::Low); // Should persist from previous update
            }

            // 5. Invalid update should be ignored and return previous valid state
            let update_res = cx.send_request(SetSessionConfigOptionRequest::new(
                new_session.session_id.clone(),
                SessionConfigId::new("model"),
                SessionConfigValueId::new("non-existent-model")
            ))
            .block_task()
            .await?;

            let model_opt = update_res.config_options.iter().find(|o| &*o.id.0 == "model").expect("missing model option");
            if let SessionConfigKind::Select(select) = &model_opt.kind {
                assert_eq!(&*select.current_value.0, "claude-3-5-haiku-latest");
            }

            Ok(())
        })
        .await
        .expect("connection completed");
}
