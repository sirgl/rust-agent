# ACP v2 feature and version negotiation

ACP v1 remains the default protocol exposed by `acp-agent`. ACP v2 support is
opt-in at compile time:

```bash
cargo run -p acp-agent --features unstable_protocol_v2
```

The feature forwards to
`agent-client-protocol/unstable_protocol_v2`. When disabled, `build_agent`
constructs the existing ACP v1 implementation directly. When enabled, it uses
the SDK's `AgentProtocolRouter` with independent v1 and v2 implementations.
The router inspects the initial `initialize.protocolVersion` and chooses the
highest compatible implementation, so enabling v2 does not remove v1 support.

The adapters intentionally share `AgentDeps`, `ModelSelection`, local
`ClientAccess`, and the protocol-neutral turn engine. Wire-schema types and
version-specific lifecycle behavior remain separate:

- v1 keeps `authenticate`, `session/load`, `session/set_mode`, synchronous
  prompt completion, and the original update shapes;
- v2 uses the nested capability objects, `session/list` / `session/resume` /
  `session/close`, mode config options, immediate prompt acknowledgement,
  message IDs, upsert-based tool and plan updates, and running/requires-action/
  idle state updates.

Both configurations are covered independently:

```bash
cargo test -p acp-agent
cargo test -p acp-agent --features unstable_protocol_v2
```

The second command runs the unchanged v1 integration suite through the router
as well as the v2-specific suite.
