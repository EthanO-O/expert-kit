//! Reject the retired graph-placement authority without reading files or services.

use std::path::PathBuf;

use ek_base::error::{EKError, EKResult};

use super::RETIRED_GRAPH_SCHEDULE_MESSAGE;

/// Explain migration before accessing configuration, files, Controller or database.
pub(super) async fn execute_graph_schedule(
    _graph_config: PathBuf,
    _wait_ready_seconds: u64,
) -> EKResult<()> {
    Err(EKError::RuntimeError(RETIRED_GRAPH_SCHEDULE_MESSAGE.into()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn retired_command_needs_no_settings_or_existing_document() {
        let error = execute_graph_schedule(PathBuf::from("/nonexistent/graph.json"), 1800)
            .await
            .unwrap_err();
        assert!(error.to_string().contains(RETIRED_GRAPH_SCHEDULE_MESSAGE));
    }
}
