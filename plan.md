# Unified `code_search_snippet` Plan

## Goal

Keep the external tool name as `code_search_snippet`, but evolve its implementation from a single structural snippet retriever into a unified design-oriented retrieval system that combines:

- structured block retrieval from curated manifests
- semantic retrieval from the existing code RAG index
- a shared reranking layer that prefers design-relevant results over generic API templates

The target outcome is not "retrieve more code", but "retrieve code that materially improves operator design decisions".

## Why Change

Current behavior shows two complementary strengths and one clear gap:

- `code_search_snippet` is good at returning blocks with explicit `context_type`, `operator_family`, and `api_pattern` metadata.
- `code_rag` is better at semantic recall when the exact structural archetype is not present in the curated snippet corpus.
- Neither tool alone reliably solves sparse-design cases such as convolution- and pooling-style operators, where the system often falls back to generic `TPipe` / `DataCopy` examples.

The merge is therefore reasonable, but only if the new tool remains design-first. A naive concatenation of both result sets would likely amplify generic matches rather than improve design guidance.

## Design Principles

1. Preserve the public tool name: `code_search_snippet`.
2. Treat structured retrieval as the primary branch and semantic retrieval as a fallback or complement.
3. Rank results by design usefulness, not by generic API overlap.
4. Prefer diverse archetypes over near-duplicate template results.
5. Keep first-result latency low; semantic retrieval must not block the initial response path.
6. Make failures diagnosable by logging branch-level recall and rerank decisions.

## Scope

### In Scope

- unify `code_search_snippet` and `code_rag` behind one retriever interface
- extend searchable sources beyond the current curated manifest
- support both block-granularity and file-granularity retrieval
- add a shared ranking and fusion layer
- update evaluation to compare old and new retrieval quality on case-study operators

### Out of Scope

- changing the external tool name
- replacing the agent chooser prompt format in the first implementation step
- rebuilding all existing indexes in one shot before architecture stabilizes
- introducing heavyweight online rerank models in the first version

## Current Baseline

### Existing `code_search_snippet`

- structured block retrieval from a local manifest
- metadata + BM25 scoring, with optional dense scoring
- good when the corpus already contains the right archetype
- weak when the corpus lacks enough convolution / pooling / sliding-window examples

### Existing `code_rag`

- semantic retrieval over an embedding index
- file-level results with limited structural awareness
- useful for broader semantic recall
- weak at distinguishing design archetypes from generic implementation templates

## Target Architecture

Implement a unified retriever behind `code_search_snippet` with three internal layers.

### Layer 1: Intent Parsing

Parse each query into a structured retrieval intent:

- target context: `kernel_src`, `host_tiling_src`, `host_infer_shape`, etc.
- target family: `convolution`, `pooling`, `matrix`, `activation`, etc.
- target design pattern: `sliding_window`, `pointwise_conv`, `matmul_fused`, `reduce_window`, `tiling_block_dim`, etc.
- target APIs: `matmul_api`, `queue_api`, `datacopy_api`, `set_block_dim`, etc.
- retrieval mode: `design_archetype`, `api_usage`, `implementation_skeleton`, or `semantic_fallback`

This layer should extend the current query-intent logic rather than replace it.

### Layer 2: Multi-Branch Recall

Recall candidates from multiple internal branches.

#### Branch A: Structured Block Recall

- reuse the current manifest-based `code_search_snippet` path
- keep block-level recall as the highest-priority branch
- continue using `context_type`, `operator_family`, `api_pattern`, symbols, keywords, and BM25

#### Branch B: Semantic File Recall

- reuse the current `code_rag` embedding index
- return file-level or chunk-level semantic neighbors
- use only as supplement when structured recall is weak or ambiguous

#### Branch C: Expanded Structural Recall for `.h` / `.cpp`

- add lightweight extraction for host registration, tiling functions, infer-shape blocks, kernel classes, and entry points from `.h` / `.cpp`
- store them in the same or a sibling manifest format
- give the unified retriever more design-specific evidence outside `.asc`

### Layer 3: Fusion and Reranking

Fuse all recalled candidates into a single ranked list using design-aware signals.

Positive signals:

- exact `context_type` match
- exact or near `operator_family` match
- query-specific design pattern match
- key API match, especially non-generic APIs such as `matmul_api`
- symbol-level hits for archetype-specific classes or functions
- source diversity that adds complementary design evidence

Negative signals:

- generic-only matches dominated by `queue_api` / `datacopy_api`
- duplicate templates from multiple near-identical examples
- design-family mismatch such as `elementwise` results for a `convolution` request when better family matches exist
- file-level semantic matches that lack structural support

## Data Model Changes

Introduce a common retrieval unit for the unified tool.

Suggested fields:

- `retrieval_id`
- `source_scope` (`asc_devkit`, `workspace`, `generated`, `rag_index`)
- `granularity` (`block`, `chunk`, `file`)
- `path`
- `block_kind`
- `context_type`
- `symbol_name`
- `operator_family`
- `design_patterns`
- `api_pattern`
- `api_symbols`
- `text`
- `start_line`
- `end_line`
- `retrieval_branch`
- `branch_scores`
- `fusion_score`
- `why_matched`

This allows both old snippet blocks and RAG chunks to participate in one ranking pipeline.

## Query Routing Strategy

The new tool should not always invoke every branch equally.

### Preferred Routing

1. Run structured block recall first.
2. If structured recall returns strong family- and context-aligned results, return immediately or use semantic recall only for enrichment.
3. If structured recall is sparse, overly generic, or family-mismatched, run semantic recall and merge.
4. If both remain weak, explicitly label results as low-confidence fallback.

### Heuristics for Weak Structured Recall

- top results are all generic `elementwise` templates for a non-elementwise query
- top results repeat the same file archetype multiple times
- top results match only generic APIs and miss family-specific symbols
- query mentions a pattern not covered in the block corpus, such as `sliding window`, `pointwise conv`, `im2col`, or `pooling stride`

## Ranking Strategy

Move from branch-local ranking to unified design-aware ranking.

### Ranking Features

- family alignment score
- context alignment score
- design-pattern alignment score
- API specificity score
- symbol specificity score
- semantic similarity score
- source diversity bonus
- generic-template penalty
- duplicate-archetype penalty

### Important Constraint

`queue_api`, `datacopy_api`, and `vector_compute_api` must remain weak signals. They are necessary implementation features but poor design discriminators.

## Result Formatting

Keep the tool response name the same, but expand the returned metadata so the agent can reason about trustworthiness.

Each result should expose:

- `retrieval_branch`
- `granularity`
- `source_scope`
- `fusion_score`
- per-branch score summary
- `confidence`
- concise `why_matched`
- enough code text to guide design, not only implementation syntax

The formatted response should also avoid flooding the LLM with multiple near-identical add-custom or data-copy templates.

## Index and Corpus Workstreams

### Workstream A: Manifest Expansion

- extend current snippet ingestion to parse `.h` and `.cpp`
- identify structural regions such as host tiling, infer shape, infer dtype, op registration, kernel class, and kernel entry
- write them into a new or extended manifest

### Workstream B: RAG Reuse

- retain the existing embedding index format for semantic fallback
- optionally add chunk metadata so results can be mapped to structural regions when available

### Workstream C: Corpus Tagging

- enrich metadata with design-pattern tags: `pointwise_conv`, `sliding_window`, `reduce_window`, `matmul_fused`, `double_buffer`, `tile_over_spatial`, etc.
- add penalties or suppressed tags for overly generic templates

## Integration Plan

### Phase 0: Baseline Freeze

Goal: preserve current behavior and create a measurable starting point.

Tasks:

- snapshot current `code_search_snippet` and `code_rag` outputs on the three case-study ops
- define evaluation labels: `design-helpful`, `partially-helpful`, `generic`, `misleading`
- record latency and top-3 result quality

Deliverables:

- baseline evaluation report
- stable reproduction commands

### Phase 1: Shared Interface

Goal: unify retriever interfaces without changing ranking logic yet.

Tasks:

- define a shared internal candidate schema
- wrap current `code_rag` results into the same schema
- add `retrieval_branch` metadata to current `code_search_snippet`
- keep old output formatting working during transition

Deliverables:

- internal unified candidate object
- compatibility layer for existing nodes

### Phase 2: Multi-Branch Recall

Goal: allow `code_search_snippet` to recall from both structured and semantic branches.

Tasks:

- add a semantic fallback path inside the unified retriever
- add routing heuristics for when semantic fallback should run
- keep semantic retrieval non-blocking for first-result latency where possible

Deliverables:

- first working merged retriever
- branch-level logs for recall diagnostics

### Phase 3: `.h` / `.cpp` Structural Ingestion

Goal: reduce dependence on `.asc`-only examples.

Tasks:

- build parsers or lightweight regex extractors for `.h` / `.cpp`
- emit structural chunks for registration, tiling, infer-shape, kernel class, and entry sections
- tag each chunk with context and family metadata where possible

Deliverables:

- expanded manifest
- index build/update script changes

### Phase 4: Design-Aware Fusion Ranking

Goal: stop generic templates from dominating.

Tasks:

- implement family/context/design-pattern alignment features
- add generic-template penalties
- add duplicate suppression and diversity promotion
- tune weights against the case-study set

Deliverables:

- fused ranking implementation
- before/after retrieval comparison

### Phase 5: Agent Integration

Goal: make the agent benefit from the new tool rather than continue issuing overly generic queries.

Tasks:

- update chooser guidance so it requests design archetypes when appropriate
- ensure tool responses surface branch and confidence metadata
- optionally refine prompts to ask for one primary archetype plus one backup pattern

Deliverables:

- updated agent prompts or chooser logic
- new experiment outputs for case-study ops

### Phase 6: Evaluation and Rollout

Goal: verify the merge improves design usefulness without regressing speed.

Tasks:

- compare old `code_search_snippet`, old `code_rag`, and unified `code_search_snippet`
- evaluate relevance and design usefulness on the three case-study operators
- measure first-result latency and full run duration
- verify that gemm retrieval remains strong while conv and pooling improve

Deliverables:

- final evaluation report
- go/no-go recommendation for replacing standalone `code_rag`

## Implementation Files Likely to Change

Primary changes:

- `generator/agent/retrievers/code_search_snippet_retriever.py`
- `generator/agent/retrievers/code_retriever.py`
- `generator/agent/retrievers/__init__.py`
- `generator/agent/nodes/code_search_snippet.py`
- `generator/agent/agent_builder.py`
- `generator/agent/builtin_tools.py`
- code-search manifest builders and related scripts under `generator/scripts/`

Potential additions:

- shared retrieval schema module
- `.h/.cpp` structural ingestion helper
- unified evaluation script or notebook for retrieval quality

## Risks

### Risk 1: Generic semantic matches dominate

Mitigation:

- keep structured branch primary
- penalize generic templates in fusion ranking
- only run semantic fallback when structured evidence is weak

### Risk 2: Latency regresses

Mitigation:

- do not require semantic branch on every request
- preserve CPU-first, opt-in dense behavior for expensive scoring
- cache branch outputs and parsed manifests aggressively

### Risk 3: Output becomes harder for the LLM to use

Mitigation:

- keep top results concise
- expose branch/confidence metadata
- deduplicate near-identical templates

### Risk 4: `.h/.cpp` parsing is noisy

Mitigation:

- start with high-value structural regions only
- mark uncertain metadata conservatively
- prefer missing tags over wrong tags

## Acceptance Criteria

The unified `code_search_snippet` is successful if all of the following hold:

1. `gemm_add_relu` still retrieves `matmul_fused` / `MatmulLeakyKernel`-style examples in top results.
2. `conv_pointwise_2d` no longer defaults to `add_custom` and `KernelDataCopy` as the dominant design references.
3. `max_pooling_1d` retrieves at least one reduction- or pooling-structured archetype consistently, not just generic data-copy templates.
4. top results show more structural diversity and fewer near-duplicate templates.
5. first-result latency remains acceptable for interactive agent use.

## Suggested Milestones

### Milestone 1

Shared interface + dual-branch internal retriever with no rerank tuning.

### Milestone 2

Expanded `.h/.cpp` structural manifest and improved family tagging.

### Milestone 3

Design-aware fusion ranking validated on the three case-study operators.

### Milestone 4

Standalone `code_rag` deprecated from agent-facing tool selection, with unified `code_search_snippet` as the default code-design retrieval tool.

## Immediate Next Actions

1. Freeze a retrieval baseline on the existing three case-study outputs.
2. Define the shared candidate schema and branch interface.
3. Implement semantic fallback inside `code_search_snippet` without changing the external tool name.
4. Expand ingestion to `.h` and `.cpp` structural chunks.
5. Add fusion ranking and run side-by-side evaluation.