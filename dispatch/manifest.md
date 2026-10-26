# Dispatch manifest
**TLDR:** Owned SNAP worker for 07_calibrated_conditional_sft_zipmix diagnosis.
Client: Codex 0.154.0; model gpt-6-astra; effort ultra (subtle scientific diagnosis).
Permissions: --dangerously-bypass-approvals-and-sandbox, full access/no approval waits.
Account: Codex ChatGPT personal cached login verified; allowance unknown. Fallback codex-vals cached ChatGPT login verified, not yet model entitlement.
Budget: 2 hours executor, one diagnostic at most 30 GPU-minutes, no purchased/provider-key calls. No main edits/push/merge.
Source: local Git bundle HEAD plus original untracked experiment folder; input.sha256 is immutable transfer manifest.
Health: canonical CLI/shell and Kerberos/AFS pass. Disk percentage checks fail but 73 GiB local / 1251 GiB shared available; task working set capped 15 GiB and worker must verify actual space before training. Claude profile symlink warning irrelevant to selected Codex. Stale binaries later in PATH not selected.
Recovery owner: parent Codex chat. Host watcher records state every 60 seconds and terminal failure; no automatic provider switch claimed. Reboot persistence unavailable.

Attempt1 personal Codex failed refresh-token validation before useful action. Attempt2 selects verified cached Vals Codex profile; same model/effort/full permissions and remaining deadline.

Attempt2 Vals Codex refresh-token failure, before useful action. Attempt3 personal Claude authenticated oauth_token profile /lfs/skampere2/0/brando9/.claude-personal-node, model claude-fable-5-1 effort max, full bypass, same remaining deadline and immutable caps.

Email authorization updated 10-21-2026 by direct user. See EMAIL_POLICY.md; original PROMPT.md hash superseded, source bundle/archive unchanged. Existing worker not restarted; email-watch observer added.
