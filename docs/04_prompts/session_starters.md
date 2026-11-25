# AdaLoop-VLM docs — index and session starters

Local root: `<CEREBRUS_ROOT>` = `/Users/vedaangchopra/all_data/complete_technical_work/all_projects_implemented/cerebrus`
Copies of the core docs are also in the claude.ai Project (CEREBRUS) under `claude/adaloop/`.

## Documents (read in this order)

| # | File | What it is | Use it for |
|---|---|---|---|
| 1 | `docs/01_research/vision_and_findings.md` | Vision, hypotheses, decision log, what we found / didn't find, prior-work map | Any new session — read first |
| 2 | `docs/02_loop/adaloop_build_spec.md` | Models, evaluation, agentic/safety tests, training recipes, reward, controller, gates, code interfaces | Building and running |
| 3 | `docs/02_loop/adaloop_execution_plan.md` | Phases A (measure) → B (build) → C (post-train), step IDs A1–C4, gates G1–G3 | Step-by-step plan |
| 4 | `docs/02_loop/sprint_plan_nov30.md` | Day-by-day to Nov 30, team split, data schemas, hardware, compute budget | Who does what, when |
| 5 | `docs/01_research/reading_brief.md` | Paper cards with page anchors, novelty verdict, objections, corrections | Literature evidence |
| 6 | `docs/04_prompts/vision_encoder_session.md` | Full prompt for the vision-encoder / noise-reduction session | Vision-encoder session |
| — | `docs/proposal/Proposal_v2.docx` | Original Nov 22 proposal (partly superseded — see vision doc §3) | History |
| — | `docs/literature/*` | Earlier review of the 77-PDF corpus (has known errors — reading brief §4) | Reference |
| — | `Papers/External_Fetched/*` | SmartVL, LoopLook, AdaLoop, LoopVL notes; 10 verified citations | External evidence |

## Session starters (paste into a new chat in the CEREBRUS Project, with `<CEREBRUS_ROOT>` attached)

Common preamble for every session:

> You are my research collaborator on AdaLoop-VLM. Be a critical peer: precise, direct, no padding, point out flaws. First read `docs/01_research/vision_and_findings.md`, then the files listed below. Treat `docs/literature/*` as a colleague's notes (the PDF wins). Anchor every claim to a file/section/page; never invent numbers. Then stop and ask me the first question on the agenda.

### A. Execution session (building, debugging, running)
> Read `docs/02_loop/adaloop_build_spec.md`, `docs/02_loop/sprint_plan_nov30.md`, `docs/02_loop/adaloop_execution_plan.md`, and the notebooks in `notebooks/`. My role: training track (controller, C1 LoRA, C2 adapter, C3 pilot). Agenda: (1) where we are against the sprint plan today; (2) the next blocking step and its Check; (3) code for that step — interfaces and scaffolds with TODOs, core logic left to me; (4) risks to the Friday launch.

### B. Vision-encoder session (noise reduction, global/local, text conditioning)
> Use the full prompt in `docs/04_prompts/vision_encoder_session.md`.

### C. Looping and controller session
> Read `docs/02_loop/adaloop_build_spec.md` §5–6, `docs/01_research/reading_brief.md` (LoopZip, StepPrune, SmartVL cards), `Papers/External_Fetched/LoopLook_*.md`, `AdaLoop_*.md`, `LoopVL_*.md`, `Unverified_citations.md` (Ouro, MoR, LoopFormer, RR-VLM rows), and the varloop model card. Agenda: (1) A3 feasibility results and the Gate G1 decision; (2) controller features and whether early readout is affordable; (3) look-again action design (restore vs re-encode crop, insert vs restart, KV consistency for decode); (4) per-token decode-time depth for January; (5) what would make the result generalise beyond this checkpoint.

### D. Safety session
> Read `docs/02_loop/adaloop_build_spec.md` §3, `docs/01_research/vision_and_findings.md` §4.2 (safety priors), the varloop card safety section, `Papers/External_Fetched/Unverified_citations.md` (Running Fewer Loops, SafeBridge, Token-Pruning-Induced Vulnerabilities, VPI-Bench), and corpus PDFs [062] [063] [064]. Agenda: (1) GUI-injection set construction and the conditional-ASR metric; (2) does early exit reduce refusal in a VLM; (3) does question-relevant pruning improve safety here; (4) when safety training (C4) is justified.

## Update rules
- After a session that changes a decision, add a row to the decision log in `docs/01_research/vision_and_findings.md` §3.
- Thresholds live in `configs/prereg.yaml`; change them only before the test split is run.
