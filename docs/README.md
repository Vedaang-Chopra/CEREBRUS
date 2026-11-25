# docs/ — index

**Start here:** `IMPLEMENTATION.md` is the master reference: steps, gates, architecture, agent prompt. It wins on any conflict.

| Path | What it is | Read when |
|---|---|---|
| `STATUS.md` | **Progress tracker:** stage table, your next action, open decisions | Every day
| `FIGURES.md` | Paper figures and tables: which notebook makes each, plotting rules | Before writing any results notebook
| `DEC3_MVP.md` | **Course deadline build (Dec 3):** minimal version, stages S0–S17, one results notebook, agent prompt. Wins over other docs until Dec 3 | Now
| `TEAM_PLAN.md` | Who does what: roles, step ownership, code ownership, handoffs, timeline, per-person agent prompts | Before assigning or starting work |
| `IMPLEMENTATION.md` | Master to-do list, architecture, data contracts, gates, ablations, agent prompt | Always first |
| `decisions.md` | Gate outcomes and design decisions, one entry each | After every gate |
| `ai_context/CODEBASE_MAP.md` | Public entry points and module ownership (kept current by the agent) | Before editing code |
| `notes/` | Agent/Vedaang notes, e.g. `modeling_code.md` (Inkling2Loop internals) | When a step asks |
| **01_research/** | | |
| `01_research/vision_and_findings.md` | Vision, hypotheses, decision history, gap, prior-work map | Any new design session |
| `01_research/reading_brief.md` | Paper cards with page anchors, novelty verdict, objections | Writing related work; checking a claim |
| **02_loop/** (AdaLoop: loop + controller) | | |
| `02_loop/adaloop_build_spec.md` | Detailed build spec: models, eval, safety tests, recipes, reward, controller, gates | Phases 1–6, 8, 9 |
| `02_loop/adaloop_execution_plan.md` | Original phase plan A1–C4 (history; superseded by IMPLEMENTATION) | Background |
| `02_loop/sprint_plan_nov30.md` | Nov 30 sprint split and compute budget (history) | Background |
| **03_encoder/** (QAFE) | | |
| `03_encoder/qafe_spec.md` | Encoder idea, evidence, hypotheses, D1–D6 decisions | Phase 7 |
| `03_encoder/qafe_build_spec.md` | Encoder build spec: modules, interfaces, trainers, tests | Phase 7 |
| **04_prompts/** | | |
| `04_prompts/session_starters.md` | Copy-paste prompts for design sessions (execution, encoder, loop, safety) | Starting a chat session |
| `04_prompts/vision_encoder_session.md` | Full prompt for the encoder literature session | Encoder design session |
| **Other** | | |
| `literature/` | Earlier literature review + paper index (known errors; see reading brief §4) | Reference |
| `literature/model_cards/` | varloop and Inkling-2 cards | Backbone facts |
| `proposal/` | Proposal_v2 (superseded) + GRA research questions | History |
| `cerebrus/`, `OLD_CODE_REFERENCE.md` | Pre-reset CEREBRUS project | Only when porting old utilities |

Rules: one decision = one entry in `decisions.md`. Thresholds live in `configs/prereg.yaml`.
