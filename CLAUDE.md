## Project Architecture

This project has multiple interconnected systems. Always confirm which subsystem you're working in before making changes:

* **Nowcast Engine** — powers the dashboard temperature predictions (NOT the physics engine)
* **Physics Engine** — theoretical/model-based predictions, separate from dashboard
* **Dashboard UI** — HTML/JS frontend displaying predictions, METAR data, timers
* **METAR Data Fetching** — weather station data ingestion with deduplication
* **PWS Data Fetching** — personal weather station data
* **Backtest/Replay Modules** — historical simulation and replay infrastructure
* **NDJSON Tape Logging** — append-only structured log for all predictions

## Code Audits & Spec Compliance

When auditing code against a spec or plan document, always do TWO passes:

1. **First pass**: implement all missing pieces
2. **Second pass**: re-read the spec line-by-line, grep each requirement against actual code to verify nothing was missed

Never mark an audit as complete after a single pass. Output a checklist with file:line references for each requirement.

## Bug Fixing Protocol

Before fixing a bug, always trace the FULL data flow from user-facing symptom back to root cause:

1. Ask: "Which component does the dashboard/UI actually call?"
2. Trace the code path from symptom → rendering → data source → engine
3. Confirm the correct subsystem before editing ANY code
4. If unclear, use parallel subagents to investigate each candidate system

Never assume which engine or module is responsible — verify it.

## UI & Display Conventions

* **Time/duration displays**: use the most human-readable granularity
  - Under 2 hours: show exact minutes ("47 minutes ago", "3m ago")
  - Over 2 hours: show hours+minutes ("1h 23m ago")
  - NEVER round to decimal hours ("1.0h ago" is forbidden)
* **Polymarket labels**: always use Polymarket's exact format as source of truth

## Workflow Priorities

* When a session involves both a bug fix AND a new feature, complete and verify the bug fix FIRST
* Do not context-switch to features until the user confirms the fix works
* For multi-goal sessions, prioritize: bugs → features → docs → cleanup

## Workflow Orchestration

### 1. Plan Mode Default

* Enter plan mode for ANY non-trivial task (3+ steps or architectural decisions)
* If something goes sideways, STOP and re-plan immediately – don't keep pushing
* Use plan mode for verification steps, not just building
* Write detailed specs upfront to reduce ambiguity

### 2. Subagent Strategy

* Use subagents liberally to keep main context window clean
* Offload research, exploration, and parallel analysis to subagents
* For complex problems, throw more compute at it via subagents
* One task per subagent for focused execution

### 3. Self-Improvement Loop

* After ANY correction from the user: update `tasks/lessons.md` with the pattern
* Write rules for yourself that prevent the same mistake
* Ruthlessly iterate on these lessons until mistake rate drops
* Review lessons at session start for relevant project

### 4. Verification Before Done

* Never mark a task complete without proving it works
* Diff behavior between main and your changes when relevant
* Ask yourself: "Would a staff engineer approve this?"
* Run tests, check logs, demonstrate correctness

### 5. Demand Elegance (Balanced)

* For non-trivial changes: pause and ask "is there a more elegant way?"
* If a fix feels hacky: "Knowing everything I know now, implement the elegant solution"
* Skip this for simple, obvious fixes – don't over-engineer
* Challenge your own work before presenting it

### 6. Autonomous Bug Fixing

* When given a bug report: just fix it. Don't ask for hand-holding
* Point at logs, errors, failing tests – then resolve them
* Zero context switching required from the user
* Go fix failing CI tests without being told how

## Task Management

1. **Plan First**: Write plan to `tasks/todo.md` with checkable items
2. **Verify Plan**: Check in before starting implementation
3. **Track Progress**: Mark items complete as you go
4. **Explain Changes**: High-level summary at each step
5. **Document Results**: Add review section to `tasks/todo.md`
6. **Capture Lessons**: Update `tasks/lessons.md` after corrections

## Core Principles

* **Simplicity First**: Make every change as simple as possible. Impact minimal code.
* **No Laziness**: Find root causes. No temporary fixes. Senior developer standards.
* **Minimal Impact**: Changes should only touch what's necessary. Avoid introducing bugs.