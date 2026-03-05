# Slow-Path Agent

The slow-path agent is CUCo's performance optimization engine. Starting from the fast-path baseline (generation 0), it runs an island-based evolutionary search with LLM-driven mutation to discover high-performance compute-communication kernels.

## Evolution Loop

**Module**: `cuco/core/runner.py`
**Class**: `EvolutionRunner`

```
For each generation:
  1. Select parent from population (fitness-weighted)
  2. Sample archive inspirations + top-k programs
  3. Assemble prompt (task msg + parent + inspirations + meta-recommendations)
  4. LLM generates mutation (diff / full rewrite / crossover)
  5. Apply patch to parent code
  6. Novelty check (reject near-duplicates)
  7. Submit for evaluation (build → run → score)
  8. Store result in database (including failures)
  9. Periodically: meta-summarization
  10. Periodically: island migration
```

### Two-Phase Scheduling

Evolution is typically split into two phases:

**Phase 1 — Explore** (first 40% of budget):
- 70% full rewrites, 15% diffs, 15% crossover
- Higher temperature (0.2, 0.5, 0.8)
- Goal: discover structurally diverse architectures (multi-stream, fused kernel, split put/wait, warp specialization)

**Phase 2 — Exploit** (remaining 60%):
- 60% full rewrites, 25% diffs, 15% crossover
- Lower temperature (0.0, 0.2, 0.5)
- Goal: refine the best architectures found during exploration

The phase split is controlled by `explore_fraction` in `run_evo.py`. The key insight: without initial diversity from the explore phase, the search converges slowly to a lower-quality local optimum.

## Mutation Forms

The LLM acts as a **variation operator** — not an open-ended code generator. Its output is structurally bounded by the EVOLVE-BLOCK markers.

### Diff Patches

Localized SEARCH/REPLACE edits within evolve blocks. The LLM proposes specific code changes:

```
<<<< SEARCH
gin.put(world, r, recvwin, offset, sendwin, offset, size, ncclGin_SignalInc(0));
==== REPLACE
gin.put(world, r, recvwin, offset, sendwin, offset, chunk_size, ncclGin_SignalInc(0));
gin.put(world, r, recvwin, offset + chunk_size, sendwin, offset + chunk_size,
        size - chunk_size, ncclGin_SignalInc(1));
>>>>
```

Best for: fine-grained parameter tuning, adding synchronization, reordering operations.

### Full Rewrites

Complete replacement of all code within EVOLVE-BLOCK markers. The LLM generates an entirely new implementation while preserving the frozen interface.

Best for: architectural changes (sequential → pipelined, single-kernel → multi-stream).

### Crossover

Synthesis from multiple archive programs. The LLM receives 2-3 high-performing candidates and combines their best aspects.

Best for: combining complementary strategies (e.g., one program's stream topology with another's synchronization pattern).

## Parent Selection

**Module**: `cuco/database/parents.py`

Three strategies are available:

### Power-Law Sampling

Default. Programs are ranked by fitness, and selection probability follows a power law distribution. The `exploitation_alpha` parameter controls selection pressure:
- `alpha = 0` → uniform random (maximum exploration)
- `alpha = 1` → strong bias toward top programs (maximum exploitation)

### Weighted Tree Sampling

Uses a sigmoid-based weighting over the fitness distribution. The `parent_selection_lambda` parameter controls sharpness — higher values concentrate selection on the best programs.

### Beam Search

Maintains `num_beams` best programs and only selects from this beam. Most exploitative strategy.

## Archive and Inspirations

**Module**: `cuco/database/inspirations.py`

Each mutation prompt includes context from other successful programs:

### Archive Inspirations

Drawn from a MAP-Elites diversity archive that maintains structurally distinct high-performing solutions. The archive ensures the LLM sees programs outside the current lineage, analogous to crossover across distant population members.

Configuration:
- `archive_size` — maximum archive capacity
- `num_archive_inspirations` — how many to include per prompt
- `elite_selection_ratio` — proportion of archive slots reserved for fitness elites

### Top-K Inspirations

The highest-scoring programs overall, regardless of structural diversity. Provides the LLM with clear performance targets.

Configuration:
- `num_top_k_inspirations` — how many to include per prompt

## Meta-Summarizer

**Module**: `cuco/core/summarizer.py`
**Class**: `MetaSummarizer`

Every `meta_rec_interval` generations, the meta-summarizer runs a three-step LLM pipeline:

1. **Summarize**: Digest the most recent batch of candidates — their scores, mutation types, architectural choices, and evaluation feedback.
2. **Update scratchpad**: Maintain a persistent global scratchpad tracking which strategies have been attempted, which succeeded, and which failed.
3. **Recommend**: Produce a ranked list of concrete optimization directions for the next generation.

These recommendations are injected into subsequent mutation prompts, creating a **closed-loop meta-learning** signal. Early recommendations may suggest exploring different fusion levels; later recommendations — informed by observing that multi-stream overlap consistently outperforms full fusion for a particular workload — redirect effort toward refining that strategy.

Output is stored in:
- `meta_memory.json` — persistent state (summaries, scratchpad, recommendations)
- `meta_N.txt` — human-readable snapshot at generation N

## Novelty Filtering

**Module**: `cuco/core/novelty_judge.py`
**Class**: `NoveltyJudge`

To prevent population collapse, candidates are checked for novelty before evaluation:

1. **Embedding similarity**: The candidate's code is embedded (via `EmbeddingClient`), and cosine similarity is computed against all existing database entries. If similarity exceeds `code_embed_sim_threshold` (default: 0.995), the candidate is rejected.

2. **LLM novelty assessment** (optional): An LLM judges whether the candidate introduces meaningful structural differences.

Rejected candidates are resampled up to `max_novelty_attempts` times.

## Island Model

**Module**: `cuco/database/islands.py`

The search uses multiple independent islands to maintain diversity:

### Assignment

Each program belongs to exactly one island. Islands can have different:
- Seed programs (`init_program_paths_per_island`)
- Task system messages (`task_sys_msg_per_island`)
- Communication APIs (`island_api_types`: e.g., island 0 = LSA, island 1 = GIN)

### Migration

Every `migration_interval` generations, top-performing programs are copied between islands:
- **Elitist migration**: Copy the best `migration_rate` fraction of each island to randomly selected targets
- **Directional migration**: Follow a configured `migration_graph` (e.g., LSA island → hybrid island ← GIN island)

Migration cross-pollinates successful patterns without collapsing diversity.

## Cascade Evaluation

Every candidate passes through a three-level cascade:

<table>
<tr><th>Level</th><th>Check</th><th>Cost</th><th>On Failure</th></tr>
<tr><td>L1</td><td>Compile (nvcc)</td><td>Seconds</td><td>Store with score 0, feed compiler errors to next mutation</td></tr>
<tr><td>L2</td><td>Run + verify (mpirun)</td><td>Seconds-minutes</td><td>Store with score 0, feed runtime errors</td></tr>
<tr><td>L3</td><td>Benchmark (best of N runs)</td><td>Seconds-minutes</td><td>Store with measured score</td></tr>
</table>

Failed candidates are retained in the database with their diagnostics. They serve as **negative examples** that inform future mutations — a form of explicit negative selection absent from classical evolutionary methods.

### LLM Feedback

At every cascade level, an LLM feedback agent receives the candidate code and evaluation outcome and generates a concise diagnostic. This feedback is stored with the candidate and injected when its lineage is later selected as a parent.

## Candidate Database

**Module**: `cuco/database/dbase.py`
**Class**: `ProgramDatabase`

All evaluated candidates — including failures — are persisted to an SQLite database. The database serves two roles:

1. **Candidate pool**: Source for parent selection, archive sampling, and inspiration retrieval across all islands.
2. **Knowledge base**: Backing store for the meta-summarizer, which queries historical results to distill cross-generation patterns.

### Embedding-Guided Retrieval

Each candidate's code embedding enables:
- **Novelty filtering**: Reject near-duplicates before evaluation
- **Nearest-neighbor lookup**: Surface structurally similar programs and their feedback
- **Clustering**: Group candidates by architectural similarity for visualization

### Schema

The `programs` table stores:

<table>
<tr><th>Column</th><th>Type</th><th>Description</th></tr>
<tr><td><code>id</code></td><td><code>TEXT</code></td><td>Unique identifier</td></tr>
<tr><td><code>code</code></td><td><code>TEXT</code></td><td>Full source code</td></tr>
<tr><td><code>generation</code></td><td><code>INTEGER</code></td><td>Generation number</td></tr>
<tr><td><code>island_idx</code></td><td><code>INTEGER</code></td><td>Island assignment</td></tr>
<tr><td><code>parent_id</code></td><td><code>TEXT</code></td><td>Parent program ID</td></tr>
<tr><td><code>combined_score</code></td><td><code>REAL</code></td><td>Fitness score</td></tr>
<tr><td><code>correct</code></td><td><code>INTEGER</code></td><td>0 or 1</td></tr>
<tr><td><code>public_metrics</code></td><td><code>TEXT</code></td><td>JSON timing data</td></tr>
<tr><td><code>text_feedback</code></td><td><code>TEXT</code></td><td>LLM feedback</td></tr>
<tr><td><code>embedding</code></td><td><code>BLOB</code></td><td>Code embedding vector</td></tr>
<tr><td><code>code_diff</code></td><td><code>TEXT</code></td><td>Mutation diff from parent</td></tr>
<tr><td><code>in_archive</code></td><td><code>INTEGER</code></td><td>Whether in MAP-Elites archive</td></tr>
</table>

## Configuration Summary

Key `EvolutionConfig` parameters for the slow-path agent:

<table>
<tr><th>Parameter</th><th>Default</th><th>Description</th></tr>
<tr><td><code>num_generations</code></td><td><code>10</code></td><td>Total generation budget</td></tr>
<tr><td><code>patch_types</code></td><td><code>["diff"]</code></td><td>Available mutation forms</td></tr>
<tr><td><code>patch_type_probs</code></td><td><code>[1.0]</code></td><td>Sampling probabilities</td></tr>
<tr><td><code>llm_models</code></td><td><code>["azure-gpt-4.1-mini"]</code></td><td>LLM models for mutation</td></tr>
<tr><td><code>llm_kwargs</code></td><td><code>{}</code></td><td>Temperature, max_tokens, etc.</td></tr>
<tr><td><code>meta_rec_interval</code></td><td><code>None</code></td><td>Generations between meta-summaries</td></tr>
<tr><td><code>max_novelty_attempts</code></td><td><code>3</code></td><td>Resamples before accepting a duplicate</td></tr>
<tr><td><code>code_embed_sim_threshold</code></td><td><code>1.0</code></td><td>Cosine similarity rejection threshold</td></tr>
<tr><td><code>use_text_feedback</code></td><td><code>False</code></td><td>Include LLM feedback in prompts</td></tr>
<tr><td><code>embedding_model</code></td><td><code>None</code></td><td>Model for code embeddings</td></tr>
</table>

Key `DatabaseConfig` parameters:

<table>
<tr><th>Parameter</th><th>Default</th><th>Description</th></tr>
<tr><td><code>num_islands</code></td><td><code>4</code></td><td>Number of independent islands</td></tr>
<tr><td><code>archive_size</code></td><td><code>100</code></td><td>MAP-Elites archive capacity</td></tr>
<tr><td><code>migration_interval</code></td><td><code>10</code></td><td>Generations between migrations</td></tr>
<tr><td><code>parent_selection_strategy</code></td><td><code>"power_law"</code></td><td>Selection algorithm</td></tr>
</table>

See [Configuration Reference](configuration.md) for the complete list.
