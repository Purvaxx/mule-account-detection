# UI Architecture & Redesignability Contract

## Overview
This document defines the presentation layer architecture for the AML Investigation Dashboard.
The core philosophy is **Strict Backend Isolation**: the UI reads from the frozen ML detection pipeline and the persisted GraphSAGE checkpoint (`artifacts/models/experiment3_graphsage.pt`), but it **never modifies, retrains, or intercepts** the ML logic.

> **Current status.** The UI serves both the Reference Dataset and external workspaces end to end, including the **Run Analysis** flow that produces workspace predictions.

## Directory Structure
```
src/ui/
├── __init__.py
├── app.py             # HTTP server entry point. Routes the JSON API; compose_index() builds the page.
├── adapter.py         # Data/presentation boundary. Builds read-only view models from an AnalysisContext.
├── resolver.py        # Resolves a workspace_id (or the Reference Dataset) into an AnalysisContext.
├── context.py         # AnalysisContext dataclass + workspace error taxonomy.
├── workspaces.py      # Dataset Hub summary listing across all workspaces.
├── config.py          # Centralized UI configuration (ports, paths, UI parameters).
├── theme.py           # Design tokens: the only definition of the visual language.
├── styles.css         # Console stylesheet. Every value resolves through a theme.py token.
├── help_content.py    # Explanatory copy served to the UI (HELP_DICT).
├── components/
│   ├── presentation.js          # Status -> tone -> class, and value formatting. Pure.
│   ├── visuals.js               # The two canvas surfaces: network stage, score distribution.
│   ├── app.js                   # Alpine.js component: state, API calls, view transitions.
│   ├── app_bar.html             # Global shell: brand, dataset context, metrics, actions.
│   ├── view_hub.html            # Dataset Hub.
│   ├── view_upload.html         # Ingestion workflow.
│   ├── view_mapping.html        # Schema mapping + compatibility report.
│   ├── view_dashboard.html      # Summary, risk intelligence, network, ranked entities.
│   ├── investigation_dossier.html  # Selected-entity dossier.
│   ├── dataset_config.html      # Identity / currency / geography configuration panel.
│   ├── trust_banner.html        # Prediction confidence panel + signal list.
│   ├── entity_table.html        # Ranked entity listing fragment.
│   ├── relationship_table.html  # Counterparty relationships fragment.
│   └── help_modal.html          # Explanatory help dialog.
└── pages/
    └── index.html     # Document shell: head, Tailwind config, and <!-- INCLUDE: --> directives.
```

### Composition

`app.py`'s `compose_index()` reads `pages/index.html`, expands every
`<!-- INCLUDE: path -->` (repeatedly, so a fragment may include a fragment) and
substitutes the `<!-- THEME_VARIABLES -->` marker with the rendered token block.
It is a module-level function rather than handler code, so the *composed page*
— not one fragment of it — is what `tests/test_ui.py` asserts against.
`serve_index` is transport only.

Static assets and the composed page are served `Cache-Control: no-store`. The
page is built from the working tree on every request; a browser that
heuristically cached `app.js` would otherwise run a stale component against
freshly composed HTML, and the two would disagree about which state fields
exist.

## Responsibility Matrix

1. **`resolver.py`**: The single entry point to analysis state. `WorkspaceAnalysisResolver.get_analysis_context(workspace_id)` returns an `AnalysisContext` for either the Reference Dataset (`None` / `'default'`) or a named workspace. It caches the reference context and each workspace context. It also computes Layer 10 cycle evidence — from the *same* graph the predictions came from, never a second one — so that evidence lives and dies with the context's cache entry.
2. **`context.py`**: Defines the immutable `AnalysisContext` (`workspace_id`, predictions `DataFrame`, `MultiDiGraph`, `cycles`) and the error taxonomy — `WorkspaceNotFoundError`, `WorkspaceNotReadyError`, `InvalidWorkspaceError` — which `app.py` maps to HTTP 404 / 409 / 400.

### Cycle evidence on the context — **CURRENT**

`AnalysisContext.cycles` carries Layer 10 output (`src/detection/cycles.py`) for both analysis paths: the Reference Dataset and every external workspace. Because it is read off the graph the resolver already built, a workspace's identity strategy (`NAME` / `ACCOUNT_ID`) is respected with no special-casing — entity ids *are* the graph's nodes, so the two strategies naturally produce different entities and therefore different cycle identities.

It is **post-hoc investigator evidence, not model input**. It does not participate in `risk_score`, `predicted_class`, the 167-feature representation, compatibility status (`READY` / `DEGRADED` / `BLOCKED`), or the trust assessment. See `docs/architecture.md` §22.1.

The field is three-valued, and the three states must stay distinguishable:

| Value | Meaning |
|---|---|
| `None` | never attempted — a lightweight or stub context |
| `CycleAnalysis` | attempted and completed; an empty `cycles` tuple means the graph genuinely has no cycles |
| `CycleAnalysisUnavailable` | attempted and failed, carrying the reason |

Enumeration failure is recovered by `analyze_cycles_safely` over a narrow, enumerated error set, so a graph that defeats enumeration costs the analyst their cycle evidence and nothing else — never a prediction. Anything outside that set propagates, because it would be a defect rather than a property of the graph.

It is served read-only by `GET /api/cycles` and `GET /api/cycles/<cycle_id>` (see [API Surface](#api-surface)). Results are not persisted — they are derived from the graph on demand and live exactly as long as the context that carries them. There is no UI surface yet.
3. **`adapter.py`**: Ensures the UI only consumes structured detection results. Transforms an `AnalysisContext` into JSON view models. It takes a context — it does **not** call `run_inference()` itself.
4. **`workspaces.py`**: Builds the Dataset Hub summary — per workspace, its display name, original filename, manifest status, compatibility status, prediction status, record/column counts, the declared dataset configuration, and the persisted prediction confidence, sorted newest first.
5. **`theme.py`**: The only definition of the visual language — surfaces, type hierarchy, accents, risk states, graph encoding, and the form tokens (radii, elevation, spacing, motion). It reaches the page through exactly two adapters: `css_variables()` is injected into `pages/index.html` at compose time as `:root` custom properties, which the page's Tailwind config and `styles.css` both resolve against; and `/api/theme` serves `THEME`, from which `visuals.js` builds its Vis-Network options. No template, stylesheet or script may hardcode a colour — `tests/test_ui.py` enforces this per file. Form tokens are document-level only: `THEME`'s shape is part of the `/api/theme` contract and is asserted unchanged.
6. **`index.html` and `components/*.html`**: Layout composition using utility-first CSS (Tailwind) plus the semantic classes in `styles.css`. The page is a document shell; each view is a fragment.
7. **`app.js`**: The Alpine component — state, API calls, view transitions. `presentation.js` and `visuals.js` are spread into it: the first maps a status to a tone and a tone to classes (so a status cannot be painted two ways on two screens), the second owns the two surfaces drawn to a canvas.

## Premium Intelligence Dashboard

The dashboard is a single-page Alpine.js application composed server-side from `pages/index.html` plus fragment includes. It presents:

* **Workspace selector** — switches the active analysis context. The Reference Dataset is the default context; every data-bearing API call carries an optional `workspace_id` query parameter, and omitting it resolves to the Reference Dataset.
* **Dataset Hub** — upload surface and workspace inventory, driven by `GET /api/workspaces`. Shows manifest status, compatibility status, prediction status, and record/column counts per workspace. `can_predict` (eligible) and `has_predictions` (results exist) are separate fields.
* **Dataset configuration** — one card each for **Identity**, **Currency** and **Geography** on the Schema Mapping screen. Each renders the selectable options with what each requires, the saved verdict from `src/datasets/policy.py`, its stated reason, and the supporting evidence read from the uploaded file: observed currencies, geography values that resolve to no country, and concepts the mapping does not supply. Choices are local until **Confirm Configuration** posts them, so a displayed verdict always describes a saved configuration. The country picker offers only frozen-vocabulary countries.
* **Prediction confidence** — a workspace run carries a `trust.json` assessment (see `docs/workspaces.md`). The hub shows a confidence pill per workspace; the workspace dashboard shows a banner whose expandable list gives each signal as *observed / why it matters / rule*; the entity panel repeats a `LOW` / `LIMITED` caveat beside the risk score itself. The Reference Dataset has no such banner: it is the data the checkpoint was trained on.
* **Mapping review** — presents each source column with its proposed canonical concept, confidence, and the retained per-candidate evidence, so an operator can see *why* a column scored as it did. Operator overrides post back to the mapping endpoint.
* **Compatibility states** — surfaces `NEEDS_MAPPING`, `INCOMPATIBLE`, `READY`, and `FAILED`, together with per-field states (`AVAILABLE`, `DERIVABLE`, `NEEDS_REVIEW`, `MISSING`, `NOT_SUPPLIED`), each field's `requirement` class (`INFERENCE_REQUIRED` / `LABEL_ONLY` / `OPTIONAL_SIGNAL` / `OPTIONAL_METADATA` / `IDENTITY_ALTERNATE`), and their derivation rules. `NOT_SUPPLIED` fields are rendered in their own section, so the assumptions a run is made under are visible rather than merely not contradicted.
* **Readiness** — a `FULL` / `DEGRADED` / `BLOCKED` pill beside the validation status, with `readiness_reasons` listed one fact per line. The two answer different questions: the status says whether inference *may* run, readiness says what the model will actually *see* if it does. A `READY` dataset with no entity type and no resolvable geography reads as `DEGRADED`, and the "satisfies the contract" line no longer claims the dataset is fully compatible.
* **Run Analysis** — shown only when compatibility is `READY` (`can_predict`). Analysis state is displayed separately as `NOT RUN` / `ANALYZING` / `COMPLETE` / `FAILED`, so **READY never implies that predictions already exist**. **Open Dashboard** appears only once a run has completed; a `READY` workspace with no run shows a `NO_ANALYSIS` panel offering to start one. After a successful run the hub and the selected workspace dashboard are refreshed.
* **Entity investigation** — ranked risk table with model output alongside observed structural evidence.
* **Graph presentation** — Vis-Network ego-network explorer around a selected entity, bounded by `MAX_VISIBLE_NEIGHBORS` (default 15) and filterable by minimum transaction count and reciprocity. Visual encoding comes from `theme.py`.
* **Explanatory help** — `HELP_DICT` served via `/api/help`.

## API Surface

| Method | Route | Purpose |
|---|---|---|
| GET | `/api/theme` | Visual design tokens |
| GET | `/api/overview?workspace_id=` | Entity totals and ranked risk listing |
| GET | `/api/entity/<id>?workspace_id=` | Single-entity detail |
| GET | `/api/network/<id>?workspace_id=` | Ego network (`max_neighbors`, `min_tx`, `reciprocal`) |
| GET | `/api/cycles?workspace_id=&entity_id=&limit=&offset=` | Layer 10 cycle **index** — aggregates plus a page of summary rows |
| GET | `/api/cycles/<cycle_id>?workspace_id=` | Complete evidence for **one** cycle |
| GET | `/api/help` | Explanatory copy |
| GET | `/api/workspaces` | Dataset Hub summary |
| GET | `/api/workspaces/<id>/mapping` | Profile + mapping (generates the mapping on first request) |
| POST | `/api/workspaces/<id>/mapping` | Apply operator mapping overrides |
| GET | `/api/workspaces/<id>/config` | Current configuration, its assessment, selectable options, and the evidence |
| POST | `/api/workspaces/<id>/config` | Set identity / currency / geography policy; 400 on an unknown value, applied atomically |
| GET | `/api/workspaces/<id>/trust` | Prediction confidence assessment for the last run; 404 if none |
| POST | `/api/workspaces/<id>/validate` | Run canonicalization and compatibility validation |
| POST | `/api/workspaces/<id>/predict` | Run Analysis — delegates to `predict_workspace()`; 409 if not READY |
| POST | `/api/workspaces/upload` | Upload a `.csv` / `.xlsx` / `.xls` dataset |

There is deliberately **no route that triggers model training**. `/predict` runs *inference only*, against the frozen checkpoint, and delegates entirely to `predict_workspace()` — no inference logic is duplicated in the route layer.

### `GET /api/cycles` — **CURRENT**

Serves the Layer 10 cycle evidence already attached to the resolved `AnalysisContext`. It **reads**: the route performs no enumeration, and repeated requests reuse the cached context rather than recomputing the graph, the predictions or the analysis.

Addressing reuses the identifiers the existing investigation APIs already use — `workspace_id` selects the dataset exactly as on `/api/overview`, and `entity_id` is an *optional filter*, not a second identity scheme. Both ride in the query string because cycles are a collection over the whole context rather than a per-entity resource.

**This is investigator evidence, not model reasoning.** The response carries no `risk_score`, no `predicted_class`, no "cycle risk" number, no suspiciousness judgement, and no threshold this layer invented. It cannot alter model output; querying it is a pure read. A cycle is a structural fact — participation in one is not a determination of fraud.

**Two states, deliberately not interchangeable:**

| `state` | `cycles` | `metadata` | Meaning |
|---|---|---|---|
| `COMPLETE` | list (possibly `[]`) | present | Analysis ran. An empty list is the positive finding that there are no cycles — for the dataset, or for the queried entity. |
| `UNAVAILABLE` | `null` | `null` | No evidence exists. `unavailable.reason` is `CYCLE_ANALYSIS_FAILED` (attempted, did not complete) or `NOT_ATTEMPTED`. |

`cycles: null` versus `cycles: []` is the structural guarantee that "we could not analyse" is never read as "there are none". `unavailable` carries a stable `reason`, the exception *class name*, and fixed operator-facing text — never the exception message, never a traceback; the cause is logged server-side.

**Entity filtering selects, it never re-roots.** Querying `B` of `A → B → C → A` returns the whole `A → B → C → A` under its unchanged canonical `cycle_id`, not a path starting at `B`.

**Three counts, three meanings, never conflated:** `page.returned` is what this response carries, `summary.total_cycles` is the selection after any entity filter, and `metadata.counts.cycles_retained` is the whole analysis.

### Index and detail — **CURRENT**

The complete evidence for every cycle is far too large to be an initial page load: on the Reference Dataset it is **19.3 MB** across 5,589 cycles and takes ~500 ms just to serialize. A summary-only index is not enough on its own either — all 5,589 summary rows are still ~3.1 MB. So the surface is split in two, and the index is paged:

| | Payload | Serialize |
|---|---|---|
| Full evidence for every cycle (rejected) | 19.26 MB | 503 ms |
| Every summary row, unpaged (rejected) | ~3.1 MB | 66 ms |
| **`GET /api/cycles` default page (100 rows)** | **50.2 KB** | **17 ms** |
| **`GET /api/cycles/<id>` one cycle** | **2.8–4.8 KB** | **5–7 ms** |

`GET /api/cycles` returns `summary` (aggregates over the **whole selection**, computed before paging — `total_cycles`, `by_length`, shortest/longest, participating entity count, reciprocal-pair count), `page` (`limit`, `offset`, `returned`, `total`, `has_more`), and `cycles` (summary rows). A row carries what a list needs — identity, entity sequence, length, transaction count, amount evidence, balance, timestamps, temporal verdict — and sets `has_detail`.

**The reduction is stated, never silent.** `detail.route` names where the rest lives and `detail.omitted_fields` names exactly which fields a row does not carry (`legs`, `leg_amount_totals`, `legs_with_complete_amounts`, `amount_retention`, `amount_retention_basis`, `temporal_explanation`, `matched_rules`). `page.has_more` and `page.total` mean a client can always tell it has part of a set.

Paging is `limit` / `offset`, defaulting to `UIConfig.CYCLE_PAGE_SIZE` (100) and capped at `CYCLE_PAGE_MAX_SIZE` (500). These are **transport** bounds — they decide how many rows travel, never which cycles exist — and every response echoes the slice in force. An unparseable or out-of-range value is a 400; an `offset` past the end is a 200 with an empty page and the true `total`.

`GET /api/cycles/<cycle_id>` returns the complete Stage 1 evidence for one cycle, plus the `metadata` bounds it was found under, so a cycle fetched on its own is still qualifiable. Every field an index row shares with the detail response holds the identical value — nothing is rounded or re-derived for the summary.

**Cycle ids are resolved context-locally.** There is no global cycle registry: the id is looked up in the addressed context's own analysis and nowhere else, so an id minted by one workspace is simply absent in another and returns 404. Isolation is a property of the lookup rather than a check bolted on top of it.

An unknown `cycle_id` in a completed analysis is **404**. An unavailable analysis is **200 / `UNAVAILABLE` / `cycle: null`** — the server does not know whether that cycle exists and must not imply it does not.

`cycle.matched_rules` is served as a declared, **always-empty** list with a `matched_rules_note` explaining why. Stage 1 implemented Tier 1 (structural facts) and Tier 2 (derived properties) only; no Tier-3 named selection rule exists, and inventing one here would be a threshold with no evidence behind it. The empty seam makes that absence visible rather than silently missing.

**Status codes** follow the existing contract: unknown workspace 404, not-ready 409, invalid 400, unknown `entity_id` 404 (the same status and message `/api/entity/<id>` gives). An entity that exists but participates in no cycle is **200 / `COMPLETE` / `[]`** — a valid empty answer, not an error. Analysis being unavailable is likewise **200**, because the request was well formed; only client mistakes get 4xx.

`metadata` carries what a reader needs to reproduce and qualify the result: the effective `bounds`, the measured `temporal_precision` and its basis, `enumeration_complete` / `truncated` / `truncation_reason`, `elapsed_seconds`, and the graph and enumeration `counts`.

Serialization follows the convention in `src/datasets/profiler.py`: ISO 8601 for instants, native `int` / `float` / `bool` for numpy scalars, `null` for anything absent. A missing amount is `null` and is **never** rendered as `0`; `is_time_respecting` keeps its three values, where `null` means *not establishable* and never *false*.

## Redesigning the UI

Future prompts can completely overhaul the UI **without touching the ML pipeline**.

### 1. Visual/Theme Redesign
To change colors, typography, or graph appearance (e.g., "Make it a cyber-security visual style"):
- **Modify**: `src/ui/theme.py` — and only that. The document and graph adapters propagate it.

Templates use semantic `system-*` utility classes (`bg-system-surface`, `text-system-muted`, `border-system-critical/40`), which resolve to the tokens. Adding a token means adding it to `theme.py` and mapping it in the page's Tailwind config; it does not mean writing a hex value into a template. `tests/test_ui.py` fails on a `system-*` class the Tailwind config does not declare — an undeclared colour class is not an error at build or run time, it is simply an unstyled element.

Status colours are never chosen per screen. `presentation.js` maps a status to one of five **tones** (`success`, `warning`, `critical`, `accent`, `neutral`) and a tone to its classes, so workspace status, compatibility, readiness, analysis state, prediction confidence, a mapping decision and a profiler confidence band all resolve through one table.

The current direction is a premium financial-intelligence workstation: a true black canvas (`#080808`), deep plum (`#75226E`) used as *depth* rather than as fill — washes, glass and edges at restrained opacity — a pink primary accent (`#CF3688`) spent only on identity and the one primary action per surface, a calm analytical azure for data affordances and graph focus, near-white body copy, and **red (`#E25555`) spent only on risk and failure** so that seeing red always means something.

### 3D and motion

The console is 2D information architecture with 3D *moments*. Only the network stage goes three-dimensional: a `perspective` container holds parallax plum washes and a projected grid plane behind the Vis-Network canvas. Those layers are pointer-transparent, so hit-testing, dragging, zooming, node selection, search and filtering are exactly what Vis-Network does on a flat canvas — depth is presentation, and the projection and its filters are unchanged. Tables, mapping, configuration and every dense analytical surface stay flat and crisp.

Motion is short, eased out and never bouncing; `prefers-reduced-motion` collapses every animation and transition to 1ms.

### 2. Layout Changes
To move components (e.g., "Move the graph to the left", "Change cards to a table"):
- **Modify**: the view fragment in `src/ui/components/` that owns the surface. `src/ui/pages/index.html` is a document shell and normally does not change.
- Alpine.js variables (`overview`, `selectedEntity`) are decoupled from the layout, making HTML reorganization safe and fast.

### 3. Adding New Features
To display a new UI element:
- Add the required data parsing logic to `src/ui/adapter.py`.
- Bind it to the HTML template in `src/ui/pages/index.html`.

Adding a new **analysis source** should extend `resolver.py` so it still returns an `AnalysisContext`; the adapter and frontend then work unchanged.

## ML Isolation Contract
- **FROZEN BACKEND**: All files outside `src/ui/` (and `run_ui.py`) are strictly frozen.
- **NO TRAINING**: Normal UI execution must never trigger GraphSAGE training. It relies entirely on `experiment3_graphsage.pt`. `run_server()` resolves the reference context once at startup and exits if the checkpoint is missing.
- **NO ML LEAKAGE**: The UI must preserve the distinction between *Model Output*, *Observed Structural Features*, *Heuristic Structural Flags*, and *Prediction Confidence*. Confidence describes a run; it never re-scores, re-thresholds, reorders or suppresses a prediction, and a flag is never presented as a determination of fraud.
- **NO FEATURE CONSTRUCTION**: The UI never builds feature tensors. It consumes predictions already produced under the verified 167-feature contract.
- **READ-ONLY CONTEXT**: `AnalysisContext` is consumed, never mutated. Adapters return new dictionaries.
- **MODEL-FAMILY INDEPENDENCE (target, not yet delivered)**: the UI should not need to know whether a score came from GraphSAGE, gradient boosting, or a future edge-aware or temporal graph model. The prediction/evidence contract that would make that true is declared in `docs/architecture.md` §45.9. It is a declared seam, not current behaviour: the evidence columns the adapter reads today are the legacy 167-feature blocks by name, and `model_version` / `feature_version` / `confidence` reach the UI through the workspace manifest and the trust artifact rather than through the prediction itself. Closing that gap changes `src/detection/inference.py` and the adapter, and needs its own decision record. **Do not treat the current column names as the contract.** *Update 2026-09-11:* the contract now exists as `src/detection/contract.py` (`RiskAssessment`), and every workspace run persists its metadata as `assessment.json`; `LegacyGraphSAGEEngine.assess()` restates the frozen frame in contract terms without changing a value. The adapter has **not** been migrated to read it - that is the next UI step (`docs/system_architecture_decision.md` §6), and it needs no change to `inference.py`.

## Null-Safety of Template Bindings

Alpine evaluates every binding inside an `x-show` subtree even while that subtree
is hidden, and it evaluates `x-if` / `x-for` expressions eagerly. A binding that
dereferences state which is legitimately `null` therefore throws an uncaught
`TypeError` on an ordinary screen — the dashboard before an entity is selected,
the mapping screen before its configuration has loaded, any view whose state was
reset on exit.

Every state field `app.js` sets to `null` — `selectedEntity`,
`mappingActiveColumn`, `validationResult`, `configView`, `trust` — must be
dereferenced with optional chaining (`selectedEntity?.risk_score`) in templates.
Optional chaining short-circuits the whole member expression, so
`selectedEntity?.risk_score.toFixed(4)` is safe.

`tests/test_ui_workspace_integration.py` enforces this statically, per field and
per template, and separately checks that those fields are still nulled in
`app.js` so the expectation cannot silently go stale.

## Routing

`APIHandler.ROUTES` is a table of `(method, path segments, handler, error overrides)`. A `{name}` segment captures that segment and is passed to the handler. Matching is on the whole, exactly-sized segment list, so an extra segment does not match a shorter route.

The exception → HTTP status map is stated once (`WorkspaceNotFoundError` → 404, `WorkspaceNotReadyError` → 409, `InvalidWorkspaceError` → 400, anything else → 500). A route adds to it only where a status genuinely differs, and says why: `/predict` reports a non-`READY` workspace as **409** because that is a conflict with current state, not a malformed request.

Handlers are transport only. A handler returning a value sends it as JSON; raising `HttpError(status, message)` reports a specific outcome. Domain writes belong to `src/datasets` — the mapping-override route delegates to `WorkspaceManager.apply_mapping_overrides` rather than writing `mapping.json` itself.

## Known Defects Affecting the UI

* The upload extension allowlist is stated a second time in `app.js` as a client-side convenience check. The server, which imports `parsers.ALLOWED_EXTENSIONS`, is authoritative; the browser copy cannot import it.
* `POST /api/workspaces/upload` reports an ingestion refusal as HTTP 500 rather than 400; see `docs/workspaces.md` → *Known defects*.

## Launching
- **Headless Inference**: `python run_demo.py`
- **Interactive UI**: `python run_ui.py` → `http://127.0.0.1:8080`
