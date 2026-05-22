# l3-settlement-smoke

Use this skill when testing the L3 long-term memory settlement path.

## Verified Context
- Target skill path: `skills/l3-settlement-smoke/SKILL.md`

## Reusable Pattern
1. Search for this existing skill before fetching or editing it.
2. Fetch this skill body before patching it.
3. Patch this existing `SKILL.md` rather than creating a duplicate skill.
4. Call `complete_long_term_update` after patching so the settlement is auditable.

## Memory Rule
Only carry forward facts that were verified by successful tool calls.
