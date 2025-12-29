AGENT changelog POLICY for Oelala
================================

Doel
----
Elke geautomatiseerde agent of bijdrager die wijzigingen aanbrengt in This project moet een duidelijke, datumgestuurde log-entry toevoegen to `changelog.md` in the projectroot. This maakt wijzigingen traceerbaar and helpt debugging and audits.

Verplichte velden per entry
--------------------------
- Date: YYYY-MM-DD
- Agent: volledige agentnaam (bijv. "GitHub Copilot")
- AgentTag: korte tag max 5 tekens (bijv. `GCOP`)
- ModelTag: korte model/feature tag (max 8 tekens aanbevolen) (bijv. `WAN2`, `I2V`, `LORA`)
- Details: zo gedetailleerd mogelijk; beschrijf which er is aangepast, waarom, and eventuele impact.
- FilesChanged: korte lijst of paden of glob patterns
- FollowUp: optioneel - suggesties of open tasks

Format and conventies
---------------------
- Het bestand `changelog.md` is UTF-8, append-only per wijziging.
- Voeg een volledige entry toe as nieuw blok bovenaan (chronologisch aflopend).
- use korte agent tags (≤5 tekens) for snelle filtering.
- use model tags to wijzigingen die modelgedrag beïnvloeden snel te vinden.
- Voeg links toe to PR/issue indien available.

Voorbeeld entry
----------------
2025-09-07
Agent: GitHub Copilot
AgentTag: GCOP
ModelTag: WAN2
Details:
- Updated multiple README files to change project LAN IP to 192.168.1.2.
- added network notes in WAN2_README.md and PROJECT_PLAN.md.
FilesChanged:
- WEB_INTERFACE_README.md
- WAN2_README.md
- PROJECT_PLAN.md
FollowUp:
- (Optional) propagate IP to runtime config files and run smoke-test.

Handhaving
---------
- CI checks may later validate that `changelog.md` was updated for code/documentation changes. Voeg entries toe at elke agent-run die substantiele wijzigingen maakt.

Special note: Copilot agents
---------------------------
This policy applies to all automated agents, including Copilot-style agents. For clarity when multiple Copilot instances or versions operate in the repo, Copilot agents should follow these extra conventions:

- The `Agent` field should contain the full agent name (for example: `GitHub Copilot`).
- The `AgentTag` should be short (<=5 chars). It's recommended — but not strictly required — that Copilot AgentTags start with `CP` or `GC` for quick filtering (examples: `CP01`, `GCOP`, `CPX`).
- Add an optional `AgentVersion` field if the copilot agent run used a specific tool/version (e.g., `v0.10.1`).

These additions make it easy to filter changelogs by Copilot runs and debug which automated helper made a specific change.
