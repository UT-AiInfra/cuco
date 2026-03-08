# Configuration Reference

CUCo uses Python dataclasses for configuration. There are four main config objects: `EvolutionConfig`, `TransformConfig`, `DatabaseConfig`, and `JobConfig`. Configuration can be specified either in Python (via `run_evo.py`) or via Hydra YAML (via `cuco_launch`).

## EvolutionConfig

**Module**: `cuco/core/runner.py`

Controls the slow-path evolutionary search.

### Task and Language

<table>
  <tr>
    <th>Parameter</th>
    <th>Type</th>
    <th>Default</th>
    <th>Description</th>
  </tr>
  <tr>
    <td><code>task_sys_msg</code></td>
    <td><code>Optional[str]</code></td>
    <td><code>None</code></td>
    <td>System message describing the optimization task, constraints, API knowledge, and hardware context. This is the primary prompt customization point.</td>
  </tr>
  <tr>
    <td><code>language</code></td>
    <td><code>str</code></td>
    <td><code>"python"</code></td>
    <td>Source language. Set to <code>"cuda"</code> for CUDA kernels. Affects prompt templates.</td>
  </tr>
  <tr>
    <td><code>init_program_path</code></td>
    <td><code>Optional[str]</code></td>
    <td><code>None</code></td>
    <td>Path to the seed program file.</td>
  </tr>
  <tr>
    <td><code>results_dir</code></td>
    <td><code>Optional[str]</code></td>
    <td><code>None</code></td>
    <td>Directory for evolution artifacts (generations, database, meta files).</td>
  </tr>
</table>

### Mutation

<table>
  <tr>
    <th>Parameter</th>
    <th>Type</th>
    <th>Default</th>
    <th>Description</th>
  </tr>
  <tr>
    <td><code>patch_types</code></td>
    <td><code>List[str]</code></td>
    <td><code>["diff"]</code></td>
    <td>Available mutation forms: <code>"diff"</code>, <code>"full"</code>, <code>"cross"</code>.</td>
  </tr>
  <tr>
    <td><code>patch_type_probs</code></td>
    <td><code>List[float]</code></td>
    <td><code>[1.0]</code></td>
    <td>Sampling probabilities for each patch type. Must sum to 1.0.</td>
  </tr>
  <tr>
    <td><code>max_patch_resamples</code></td>
    <td><code>int</code></td>
    <td><code>3</code></td>
    <td>Times to retry patch generation if novelty check rejects it.</td>
  </tr>
  <tr>
    <td><code>max_patch_attempts</code></td>
    <td><code>int</code></td>
    <td><code>5</code></td>
    <td>Times to retry if patch application fails (malformed diff, etc.).</td>
  </tr>
</table>

### Evolution

<table>
  <tr>
    <th>Parameter</th>
    <th>Type</th>
    <th>Default</th>
    <th>Description</th>
  </tr>
  <tr>
    <td><code>num_generations</code></td>
    <td><code>int</code></td>
    <td><code>10</code></td>
    <td>Total generation budget for this run.</td>
  </tr>
  <tr>
    <td><code>max_parallel_jobs</code></td>
    <td><code>int</code></td>
    <td><code>2</code></td>
    <td>Maximum concurrent evaluation jobs.</td>
  </tr>
  <tr>
    <td><code>job_type</code></td>
    <td><code>str</code></td>
    <td><code>"local"</code></td>
    <td>Execution backend: <code>"local"</code>, <code>"slurm_docker"</code>, <code>"slurm_conda"</code>.</td>
  </tr>
</table>

### LLM

<table>
  <tr>
    <th>Parameter</th>
    <th>Type</th>
    <th>Default</th>
    <th>Description</th>
  </tr>
  <tr>
    <td><code>llm_models</code></td>
    <td><code>List[str]</code></td>
    <td><code>["azure-gpt-4.1-mini"]</code></td>
    <td>LLM models for mutation. Multiple models enable dynamic selection.</td>
  </tr>
  <tr>
    <td><code>llm_dynamic_selection</code></td>
    <td><code>Optional[str]</code></td>
    <td><code>None</code></td>
    <td>Model selection strategy: <code>None</code> (round-robin), <code>"ucb"</code> (bandit).</td>
  </tr>
  <tr>
    <td><code>llm_kwargs</code></td>
    <td><code>dict</code></td>
    <td><code>{}</code></td>
    <td>Extra kwargs: <code>temperatures</code> (list), <code>max_tokens</code> (int), <code>reasoning_effort</code> (str).</td>
  </tr>
</table>

### Meta-Summarizer

<table>
  <tr>
    <th>Parameter</th>
    <th>Type</th>
    <th>Default</th>
    <th>Description</th>
  </tr>
  <tr>
    <td><code>meta_rec_interval</code></td>
    <td><code>Optional[int]</code></td>
    <td><code>None</code></td>
    <td>Generations between meta-summaries. <code>None</code> disables meta-summarization.</td>
  </tr>
  <tr>
    <td><code>meta_llm_models</code></td>
    <td><code>Optional[List[str]]</code></td>
    <td><code>None</code></td>
    <td>LLM models for meta-summarization. Falls back to <code>llm_models</code>.</td>
  </tr>
  <tr>
    <td><code>meta_llm_kwargs</code></td>
    <td><code>dict</code></td>
    <td><code>{}</code></td>
    <td>LLM kwargs for meta-summarization.</td>
  </tr>
  <tr>
    <td><code>meta_max_recommendations</code></td>
    <td><code>int</code></td>
    <td><code>5</code></td>
    <td>Number of optimization recommendations per summary.</td>
  </tr>
</table>

### Novelty

<table>
  <tr>
    <th>Parameter</th>
    <th>Type</th>
    <th>Default</th>
    <th>Description</th>
  </tr>
  <tr>
    <td><code>max_novelty_attempts</code></td>
    <td><code>int</code></td>
    <td><code>3</code></td>
    <td>Resamples before accepting a near-duplicate.</td>
  </tr>
  <tr>
    <td><code>code_embed_sim_threshold</code></td>
    <td><code>float</code></td>
    <td><code>1.0</code></td>
    <td>Cosine similarity threshold for rejection. Lower = stricter (0.995 is typical).</td>
  </tr>
  <tr>
    <td><code>embedding_model</code></td>
    <td><code>Optional[str]</code></td>
    <td><code>None</code></td>
    <td>Embedding model name. <code>None</code> disables embedding-based novelty.</td>
  </tr>
  <tr>
    <td><code>novelty_llm_models</code></td>
    <td><code>Optional[List[str]]</code></td>
    <td><code>None</code></td>
    <td>LLM models for novelty assessment.</td>
  </tr>
  <tr>
    <td><code>use_text_feedback</code></td>
    <td><code>bool</code></td>
    <td><code>False</code></td>
    <td>Include LLM text feedback in mutation prompts.</td>
  </tr>
</table>

### Pre-Transform (Fast-Path)

<table>
  <tr>
    <th>Parameter</th>
    <th>Type</th>
    <th>Default</th>
    <th>Description</th>
  </tr>
  <tr>
    <td><code>pre_transform_enabled</code></td>
    <td><code>bool</code></td>
    <td><code>False</code></td>
    <td>Run fast-path before evolution.</td>
  </tr>
  <tr>
    <td><code>pre_transform_pipeline_steps</code></td>
    <td><code>List[str]</code></td>
    <td><code>["analyze", "host_to_device", "evolve_markers", "warmup"]</code></td>
    <td>Pipeline steps to run.</td>
  </tr>
  <tr>
    <td><code>pre_transform_two_stage</code></td>
    <td><code>bool</code></td>
    <td><code>True</code></td>
    <td>Split into infrastructure + replacement stages.</td>
  </tr>
  <tr>
    <td><code>pre_transform_max_iterations</code></td>
    <td><code>int</code></td>
    <td><code>20</code></td>
    <td>Max iterations (single-stage).</td>
  </tr>
  <tr>
    <td><code>pre_transform_stage_a_max_iterations</code></td>
    <td><code>int</code></td>
    <td><code>5</code></td>
    <td>Max iterations for infrastructure stage.</td>
  </tr>
  <tr>
    <td><code>pre_transform_stage_b_max_iterations</code></td>
    <td><code>int</code></td>
    <td><code>10</code></td>
    <td>Max iterations for replacement stage.</td>
  </tr>
  <tr>
    <td><code>pre_transform_rewrite_model</code></td>
    <td><code>Optional[str]</code></td>
    <td><code>None</code></td>
    <td>LLM for code generation. Falls back to first <code>llm_models</code> entry.</td>
  </tr>
  <tr>
    <td><code>pre_transform_judge_model</code></td>
    <td><code>str</code></td>
    <td><code>""</code></td>
    <td>LLM for judge feedback. Empty = same as rewriter.</td>
  </tr>
  <tr>
    <td><code>pre_transform_reference_code_path</code></td>
    <td><code>Optional[str]</code></td>
    <td><code>None</code></td>
    <td>Path to reference device-side example.</td>
  </tr>
  <tr>
    <td><code>pre_transform_nccl_api_docs</code></td>
    <td><code>str</code></td>
    <td><code>""</code></td>
    <td>NCCL API docs string for the rewriter.</td>
  </tr>
  <tr>
    <td><code>pre_transform_agent</code></td>
    <td><code>bool</code></td>
    <td><code>False</code></td>
    <td>Use Claude Code agent for transformation.</td>
  </tr>
  <tr>
    <td><code>pre_transform_agent_model</code></td>
    <td><code>str</code></td>
    <td><code>"opus"</code></td>
    <td>Claude model alias for agent mode.</td>
  </tr>
  <tr>
    <td><code>pre_transform_warmup_model</code></td>
    <td><code>Optional[str]</code></td>
    <td><code>None</code></td>
    <td>LLM for warmup injection.</td>
  </tr>
  <tr>
    <td><code>pre_transform_marker_model</code></td>
    <td><code>Optional[str]</code></td>
    <td><code>None</code></td>
    <td>LLM for evolve-block annotation.</td>
  </tr>
</table>

### Per-Island Customization

<table>
  <tr>
    <th>Parameter</th>
    <th>Type</th>
    <th>Default</th>
    <th>Description</th>
  </tr>
  <tr>
    <td><code>task_sys_msg_per_island</code></td>
    <td><code>Optional[Dict[int, str]]</code></td>
    <td><code>None</code></td>
    <td>Different task prompts per island.</td>
  </tr>
  <tr>
    <td><code>init_program_paths_per_island</code></td>
    <td><code>Optional[Dict[int, str]]</code></td>
    <td><code>None</code></td>
    <td>Different seed programs per island.</td>
  </tr>
  <tr>
    <td><code>reference_code_per_island</code></td>
    <td><code>Optional[Dict[int, str]]</code></td>
    <td><code>None</code></td>
    <td>Different reference code per island.</td>
  </tr>
</table>

### Dual Pre-Transform

<table>
  <tr>
    <th>Parameter</th>
    <th>Type</th>
    <th>Default</th>
    <th>Description</th>
  </tr>
  <tr>
    <td><code>pre_transform_dual</code></td>
    <td><code>bool</code></td>
    <td><code>False</code></td>
    <td>Run parallel LSA + GIN transformations.</td>
  </tr>
  <tr>
    <td><code>pre_transform_lsa_reference_code_path</code></td>
    <td><code>Optional[str]</code></td>
    <td><code>None</code></td>
    <td>Reference code for LSA transformation.</td>
  </tr>
  <tr>
    <td><code>pre_transform_lsa_nccl_api_docs</code></td>
    <td><code>str</code></td>
    <td><code>""</code></td>
    <td>NCCL docs for LSA transformation.</td>
  </tr>
  <tr>
    <td><code>pre_transform_lsa_island_idx</code></td>
    <td><code>int</code></td>
    <td><code>0</code></td>
    <td>Island index for LSA seed.</td>
  </tr>
  <tr>
    <td><code>pre_transform_gin_island_idx</code></td>
    <td><code>int</code></td>
    <td><code>1</code></td>
    <td>Island index for GIN seed.</td>
  </tr>
</table>

---

## TransformConfig

**Module**: `cuco/transform/transformer.py`

Controls the fast-path host-to-device transformation.

### LLM Settings

<table>
  <tr>
    <th>Parameter</th>
    <th>Type</th>
    <th>Default</th>
    <th>Description</th>
  </tr>
  <tr>
    <td><code>rewrite_model</code></td>
    <td><code>str</code></td>
    <td><code>"bedrock/us.anthropic.claude-sonnet-4-6"</code></td>
    <td>LLM for code generation.</td>
  </tr>
  <tr>
    <td><code>judge_model</code></td>
    <td><code>str</code></td>
    <td><code>""</code></td>
    <td>LLM for judge feedback. Empty = same as rewriter.</td>
  </tr>
  <tr>
    <td><code>rewrite_max_tokens</code></td>
    <td><code>int</code></td>
    <td><code>32768</code></td>
    <td>Max output tokens for rewrites.</td>
  </tr>
  <tr>
    <td><code>judge_max_tokens</code></td>
    <td><code>int</code></td>
    <td><code>2048</code></td>
    <td>Max output tokens for judge.</td>
  </tr>
  <tr>
    <td><code>rewrite_temperature</code></td>
    <td><code>float</code></td>
    <td><code>0.0</code></td>
    <td>Temperature for code generation.</td>
  </tr>
  <tr>
    <td><code>judge_temperature</code></td>
    <td><code>float</code></td>
    <td><code>0.0</code></td>
    <td>Temperature for judge.</td>
  </tr>
</table>

### Build Settings

<table>
  <tr>
    <th>Parameter</th>
    <th>Type</th>
    <th>Default</th>
    <th>Description</th>
  </tr>
  <tr>
    <td><code>nvcc_path</code></td>
    <td><code>str</code></td>
    <td><code>"/usr/local/cuda-13.1/bin/nvcc"</code></td>
    <td>Path to nvcc compiler.</td>
  </tr>
  <tr>
    <td><code>nccl_include</code></td>
    <td><code>str</code></td>
    <td>(NCCL include dir)</td>
    <td>Path to NCCL headers.</td>
  </tr>
  <tr>
    <td><code>nccl_static_lib</code></td>
    <td><code>str</code></td>
    <td>(NCCL static lib)</td>
    <td>Path to <code>libnccl_static.a</code>.</td>
  </tr>
  <tr>
    <td><code>cuda_lib64</code></td>
    <td><code>str</code></td>
    <td>(CUDA lib64 dir)</td>
    <td>Path to CUDA runtime libraries.</td>
  </tr>
  <tr>
    <td><code>mpi_include</code></td>
    <td><code>str</code></td>
    <td>(MPI include dir)</td>
    <td>Path to MPI headers.</td>
  </tr>
  <tr>
    <td><code>mpi_lib</code></td>
    <td><code>str</code></td>
    <td>(MPI lib dir)</td>
    <td>Path to MPI libraries.</td>
  </tr>
  <tr>
    <td><code>binary_name</code></td>
    <td><code>str</code></td>
    <td><code>"cuda_program"</code></td>
    <td>Output binary name.</td>
  </tr>
</table>

### Run Settings

<table>
  <tr>
    <th>Parameter</th>
    <th>Type</th>
    <th>Default</th>
    <th>Description</th>
  </tr>
  <tr>
    <td><code>num_mpi_ranks</code></td>
    <td><code>int</code></td>
    <td><code>4</code></td>
    <td>Number of MPI processes.</td>
  </tr>
  <tr>
    <td><code>run_timeout</code></td>
    <td><code>int</code></td>
    <td><code>120</code></td>
    <td>Timeout in seconds for each run.</td>
  </tr>
  <tr>
    <td><code>cuda_visible_devices</code></td>
    <td><code>str</code></td>
    <td><code>"0,1,2,3"</code></td>
    <td>GPU visibility mask.</td>
  </tr>
  <tr>
    <td><code>hostfile</code></td>
    <td><code>str</code></td>
    <td><code>""</code></td>
    <td>Path to MPI hostfile. Empty = local only.</td>
  </tr>
  <tr>
    <td><code>mpirun_extra_args</code></td>
    <td><code>tuple</code></td>
    <td><code>()</code></td>
    <td>Extra mpirun arguments (e.g., <code>("--map-by", "node")</code>).</td>
  </tr>
  <tr>
    <td><code>run_env_vars</code></td>
    <td><code>dict</code></td>
    <td><code>{}</code></td>
    <td>Extra environment variables passed via <code>-x</code>.</td>
  </tr>
</table>

### Loop Settings

<table>
  <tr>
    <th>Parameter</th>
    <th>Type</th>
    <th>Default</th>
    <th>Description</th>
  </tr>
  <tr>
    <td><code>max_iterations</code></td>
    <td><code>int</code></td>
    <td><code>5</code></td>
    <td>Max iterations (single-stage mode).</td>
  </tr>
  <tr>
    <td><code>verification_pass_str</code></td>
    <td><code>str</code></td>
    <td><code>"Verification: PASS"</code></td>
    <td>Expected output for correctness.</td>
  </tr>
  <tr>
    <td><code>api_type</code></td>
    <td><code>str</code></td>
    <td><code>"gin"</code></td>
    <td>Target API: <code>"gin"</code> or <code>"lsa"</code>.</td>
  </tr>
  <tr>
    <td><code>two_stage</code></td>
    <td><code>bool</code></td>
    <td><code>True</code></td>
    <td>Split into infrastructure + replacement.</td>
  </tr>
  <tr>
    <td><code>stage_a_max_iterations</code></td>
    <td><code>int</code></td>
    <td><code>5</code></td>
    <td>Max iterations for Stage A.</td>
  </tr>
  <tr>
    <td><code>stage_b_max_iterations</code></td>
    <td><code>int</code></td>
    <td><code>10</code></td>
    <td>Max iterations for Stage B.</td>
  </tr>
</table>

### Context

<table>
  <tr>
    <th>Parameter</th>
    <th>Type</th>
    <th>Default</th>
    <th>Description</th>
  </tr>
  <tr>
    <td><code>reference_code</code></td>
    <td><code>str</code></td>
    <td><code>""</code></td>
    <td>Working device-side example code.</td>
  </tr>
  <tr>
    <td><code>nccl_api_docs</code></td>
    <td><code>str</code></td>
    <td><code>""</code></td>
    <td>NCCL API documentation string.</td>
  </tr>
</table>

---

## DatabaseConfig

**Module**: `cuco/database/dbase.py`

Controls the candidate database, islands, and selection strategies.

### Core

<table>
  <tr>
    <th>Parameter</th>
    <th>Type</th>
    <th>Default</th>
    <th>Description</th>
  </tr>
  <tr>
    <td><code>db_path</code></td>
    <td><code>str</code></td>
    <td><code>"evolution_db.sqlite"</code></td>
    <td>SQLite database filename.</td>
  </tr>
  <tr>
    <td><code>num_islands</code></td>
    <td><code>int</code></td>
    <td><code>4</code></td>
    <td>Number of independent islands.</td>
  </tr>
  <tr>
    <td><code>archive_size</code></td>
    <td><code>int</code></td>
    <td><code>100</code></td>
    <td>MAP-Elites archive capacity.</td>
  </tr>
</table>

### Inspiration

<table>
  <tr>
    <th>Parameter</th>
    <th>Type</th>
    <th>Default</th>
    <th>Description</th>
  </tr>
  <tr>
    <td><code>elite_selection_ratio</code></td>
    <td><code>float</code></td>
    <td><code>0.3</code></td>
    <td>Proportion of archive reserved for fitness elites.</td>
  </tr>
  <tr>
    <td><code>num_archive_inspirations</code></td>
    <td><code>int</code></td>
    <td><code>5</code></td>
    <td>Archive programs per mutation prompt.</td>
  </tr>
  <tr>
    <td><code>num_top_k_inspirations</code></td>
    <td><code>int</code></td>
    <td><code>2</code></td>
    <td>Top-k programs per mutation prompt.</td>
  </tr>
</table>

### Islands and Migration

<table>
  <tr>
    <th>Parameter</th>
    <th>Type</th>
    <th>Default</th>
    <th>Description</th>
  </tr>
  <tr>
    <td><code>migration_interval</code></td>
    <td><code>int</code></td>
    <td><code>10</code></td>
    <td>Generations between migrations.</td>
  </tr>
  <tr>
    <td><code>migration_rate</code></td>
    <td><code>float</code></td>
    <td><code>0.1</code></td>
    <td>Fraction of island population to migrate.</td>
  </tr>
  <tr>
    <td><code>island_elitism</code></td>
    <td><code>bool</code></td>
    <td><code>True</code></td>
    <td>Keep best programs on their home islands.</td>
  </tr>
  <tr>
    <td><code>enforce_island_separation</code></td>
    <td><code>bool</code></td>
    <td><code>True</code></td>
    <td>Enforce full island separation for inspirations.</td>
  </tr>
  <tr>
    <td><code>island_api_types</code></td>
    <td><code>Optional[Dict[int, str]]</code></td>
    <td><code>None</code></td>
    <td>Per-island API types (e.g., <code>{0: "lsa", 1: "gin"}</code>).</td>
  </tr>
  <tr>
    <td><code>migration_graph</code></td>
    <td><code>Optional[Dict[int, List[int]]]</code></td>
    <td><code>None</code></td>
    <td>Directional migration (e.g., <code>{0: [2], 1: [2]}</code>).</td>
  </tr>
</table>

### Parent Selection

<table>
  <tr>
    <th>Parameter</th>
    <th>Type</th>
    <th>Default</th>
    <th>Description</th>
  </tr>
  <tr>
    <td><code>parent_selection_strategy</code></td>
    <td><code>str</code></td>
    <td><code>"power_law"</code></td>
    <td>Strategy: <code>"power_law"</code>, <code>"weighted"</code>, <code>"beam_search"</code>.</td>
  </tr>
  <tr>
    <td><code>exploitation_alpha</code></td>
    <td><code>float</code></td>
    <td><code>1.0</code></td>
    <td>Power-law exponent. 0 = uniform, 1 = strong bias.</td>
  </tr>
  <tr>
    <td><code>exploitation_ratio</code></td>
    <td><code>float</code></td>
    <td><code>0.2</code></td>
    <td>Probability of picking from archive vs. population.</td>
  </tr>
  <tr>
    <td><code>parent_selection_lambda</code></td>
    <td><code>float</code></td>
    <td><code>10.0</code></td>
    <td>Sigmoid sharpness for weighted selection.</td>
  </tr>
  <tr>
    <td><code>num_beams</code></td>
    <td><code>int</code></td>
    <td><code>5</code></td>
    <td>Beam width for beam search selection.</td>
  </tr>
</table>

### Embeddings

<table>
  <tr>
    <th>Parameter</th>
    <th>Type</th>
    <th>Default</th>
    <th>Description</th>
  </tr>
  <tr>
    <td><code>embedding_model</code></td>
    <td><code>str</code></td>
    <td><code>"text-embedding-3-small"</code></td>
    <td>Model for code embeddings.</td>
  </tr>
</table>

---

## JobConfig

**Module**: `cuco/launch/scheduler.py`

Controls how evaluation jobs are executed.

### Base Config

<table>
  <tr>
    <th>Parameter</th>
    <th>Type</th>
    <th>Default</th>
    <th>Description</th>
  </tr>
  <tr>
    <td><code>eval_program_path</code></td>
    <td><code>str</code></td>
    <td><code>""</code></td>
    <td>Path to <code>evaluate.py</code>.</td>
  </tr>
  <tr>
    <td><code>extra_cmd_args</code></td>
    <td><code>str</code></td>
    <td><code>""</code></td>
    <td>Extra CLI arguments for evaluate.py.</td>
  </tr>
</table>

### LocalJobConfig

For local execution via subprocess:

<table>
  <tr>
    <th>Parameter</th>
    <th>Type</th>
    <th>Default</th>
    <th>Description</th>
  </tr>
  <tr>
    <td><code>time</code></td>
    <td><code>str</code></td>
    <td><code>""</code></td>
    <td>Timeout (e.g., <code>"01:00:00"</code>). Empty = no timeout.</td>
  </tr>
  <tr>
    <td><code>conda_env</code></td>
    <td><code>str</code></td>
    <td><code>""</code></td>
    <td>Conda environment name. Empty = use current env.</td>
  </tr>
</table>

### SlurmDockerJobConfig

For Slurm execution with Docker:

<table>
  <tr>
    <th>Parameter</th>
    <th>Type</th>
    <th>Default</th>
    <th>Description</th>
  </tr>
  <tr>
    <td><code>image</code></td>
    <td><code>str</code></td>
    <td><code>""</code></td>
    <td>Docker image name.</td>
  </tr>
  <tr>
    <td><code>image_tar_path</code></td>
    <td><code>str</code></td>
    <td><code>""</code></td>
    <td>Path to Docker image tar (for offline nodes).</td>
  </tr>
  <tr>
    <td><code>docker_flags</code></td>
    <td><code>str</code></td>
    <td><code>""</code></td>
    <td>Extra docker run flags.</td>
  </tr>
  <tr>
    <td><code>partition</code></td>
    <td><code>str</code></td>
    <td><code>""</code></td>
    <td>Slurm partition.</td>
  </tr>
  <tr>
    <td><code>time</code></td>
    <td><code>str</code></td>
    <td><code>"01:00:00"</code></td>
    <td>Slurm time limit.</td>
  </tr>
  <tr>
    <td><code>cpus</code></td>
    <td><code>int</code></td>
    <td><code>4</code></td>
    <td>CPUs per task.</td>
  </tr>
  <tr>
    <td><code>gpus</code></td>
    <td><code>int</code></td>
    <td><code>1</code></td>
    <td>GPUs per task.</td>
  </tr>
  <tr>
    <td><code>mem</code></td>
    <td><code>str</code></td>
    <td><code>"32G"</code></td>
    <td>Memory per task.</td>
  </tr>
</table>

### SlurmCondaJobConfig

For Slurm execution with Conda:

<table>
  <tr>
    <th>Parameter</th>
    <th>Type</th>
    <th>Default</th>
    <th>Description</th>
  </tr>
  <tr>
    <td><code>conda_env</code></td>
    <td><code>str</code></td>
    <td><code>""</code></td>
    <td>Conda environment name.</td>
  </tr>
  <tr>
    <td><code>modules</code></td>
    <td><code>List[str]</code></td>
    <td><code>[]</code></td>
    <td>Environment modules to load.</td>
  </tr>
  <tr>
    <td><code>partition</code></td>
    <td><code>str</code></td>
    <td><code>""</code></td>
    <td>Slurm partition.</td>
  </tr>
  <tr>
    <td><code>time</code></td>
    <td><code>str</code></td>
    <td><code>"01:00:00"</code></td>
    <td>Slurm time limit.</td>
  </tr>
  <tr>
    <td><code>cpus</code></td>
    <td><code>int</code></td>
    <td><code>4</code></td>
    <td>CPUs per task.</td>
  </tr>
  <tr>
    <td><code>gpus</code></td>
    <td><code>int</code></td>
    <td><code>1</code></td>
    <td>GPUs per task.</td>
  </tr>
  <tr>
    <td><code>mem</code></td>
    <td><code>str</code></td>
    <td><code>"32G"</code></td>
    <td>Memory per task.</td>
  </tr>
</table>

---

## Environment Variables

CUCo reads credentials from a <code>.env</code> file in the repository root:

<table>
  <tr>
    <th>Variable</th>
    <th>Provider</th>
    <th>Description</th>
  </tr>
  <tr>
    <td><code>AWS_ACCESS_KEY_ID</code></td>
    <td>Bedrock</td>
    <td>AWS access key</td>
  </tr>
  <tr>
    <td><code>AWS_SECRET_ACCESS_KEY</code></td>
    <td>Bedrock</td>
    <td>AWS secret key</td>
  </tr>
  <tr>
    <td><code>AWS_REGION_NAME</code></td>
    <td>Bedrock</td>
    <td>AWS region (e.g., <code>us-east-1</code>)</td>
  </tr>
  <tr>
    <td><code>OPENAI_API_KEY</code></td>
    <td>OpenAI</td>
    <td>OpenAI API key</td>
  </tr>
  <tr>
    <td><code>AZURE_OPENAI_API_KEY</code></td>
    <td>Azure</td>
    <td>Azure OpenAI API key</td>
  </tr>
  <tr>
    <td><code>AZURE_API_VERSION</code></td>
    <td>Azure</td>
    <td>Azure API version</td>
  </tr>
  <tr>
    <td><code>AZURE_API_ENDPOINT</code></td>
    <td>Azure</td>
    <td>Azure endpoint URL</td>
  </tr>
  <tr>
    <td><code>DEEPSEEK_API_KEY</code></td>
    <td>DeepSeek</td>
    <td>DeepSeek API key</td>
  </tr>
  <tr>
    <td><code>GEMINI_API_KEY</code></td>
    <td>Gemini</td>
    <td>Google Gemini API key</td>
  </tr>
  <tr>
    <td><code>BEDROCK_BASE_URL</code></td>
    <td>Bedrock OpenAI</td>
    <td>Bedrock OpenAI-compatible base URL</td>
  </tr>
  <tr>
    <td><code>BEDROCK_API_KEY</code></td>
    <td>Bedrock OpenAI</td>
    <td>Bedrock API key for OpenAI-compatible endpoint</td>
  </tr>
</table>

See [LLM Backends](llm-backends.md) for provider-specific setup.

---

## Hydra Configuration

When using `cuco_launch` (the Bash entry point), configuration is loaded from YAML files in a `configs/` directory via Hydra:

```bash
cuco_launch database=my_db evolution=my_evo
```

This looks for `configs/database/my_db.yaml` and `configs/evolution/my_evo.yaml`. The YAML structure mirrors the dataclass fields.

When using `run_evo.py` directly (as in the workloads), configuration is assembled in Python — no YAML files needed. This is the recommended approach for workload-specific setups.
