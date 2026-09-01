#![allow(dead_code)]

use std::sync::{Arc, Mutex};

use agent_client_protocol::schema::v2::{
    PlanEntry, PlanUpdate, PlanUpdateContent, SessionUpdate, StateUpdate, StopReason,
    ToolCallStatus,
};

pub type Updates = Arc<Mutex<Vec<SessionUpdate>>>;

pub async fn wait_for_idle(updates: &Updates) -> StopReason {
    tokio::time::timeout(std::time::Duration::from_secs(5), async {
        loop {
            if let Some(reason) =
                updates
                    .lock()
                    .unwrap()
                    .iter()
                    .rev()
                    .find_map(|update| match update {
                        SessionUpdate::StateUpdate(StateUpdate::Idle(idle)) => {
                            idle.stop_reason.clone()
                        }
                        _ => None,
                    })
            {
                return reason;
            }
            tokio::time::sleep(std::time::Duration::from_millis(5)).await;
        }
    })
    .await
    .expect("agent did not publish an idle state")
}

pub fn tool_statuses(updates: &[SessionUpdate]) -> Vec<ToolCallStatus> {
    updates
        .iter()
        .filter_map(|update| match update {
            SessionUpdate::ToolCallUpdate(update) => update.status.value().cloned(),
            _ => None,
        })
        .collect()
}

pub fn plan_entries(plan: &PlanUpdate) -> &[PlanEntry] {
    match &plan.plan {
        PlanUpdateContent::Items(items) => &items.entries,
        _ => panic!("expected itemized ACP plan"),
    }
}
