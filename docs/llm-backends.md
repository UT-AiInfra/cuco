# LLM Backends

CUCo supports multiple LLM providers for code generation, mutation, judging, meta-summarization, and embeddings. This document covers setup for each provider.

## Provider Overview

<table>
  <tr>
    <th>Provider</th>
    <th>Model Name Format</th>
    <th>Client</th>
    <th>Use Case</th>
  </tr>
  <tr>
    <td>Anthropic (direct)</td>
    <td><code>claude-sonnet-4-6</code>, <code>claude-opus-4-6</code></td>
    <td><code>anthropic.Anthropic()</code></td>
    <td>Direct Anthropic API</td>
  </tr>
  <tr>
    <td>Anthropic (Bedrock)</td>
    <td><code>bedrock/us.anthropic.claude-opus-4-6-v1</code></td>
    <td><code>anthropic.AnthropicBedrock()</code></td>
    <td>AWS-managed Anthropic</td>
  </tr>
  <tr>
    <td>OpenAI</td>
    <td><code>gpt-4.1-mini</code>, <code>o3-mini</code></td>
    <td><code>openai.OpenAI()</code></td>
    <td>OpenAI API</td>
  </tr>
  <tr>
    <td>Azure OpenAI</td>
    <td><code>azure-gpt-4.1-mini</code></td>
    <td><code>openai.AzureOpenAI()</code></td>
    <td>Azure-managed OpenAI</td>
  </tr>
  <tr>
    <td>DeepSeek</td>
    <td><code>deepseek-chat</code>, <code>deepseek-reasoner</code></td>
    <td><code>openai.OpenAI(base_url=...)</code></td>
    <td>DeepSeek API</td>
  </tr>
  <tr>
    <td>Google Gemini</td>
    <td><code>gemini-2.0-flash</code>, <code>gemini-2.5-pro</code></td>
    <td><code>openai.OpenAI(base_url=...)</code></td>
    <td>Google AI API</td>
  </tr>
  <tr>
    <td>Claude CLI</td>
    <td><code>claude-cli/opus</code>, <code>claude-cli/sonnet</code></td>
    <td>subprocess</td>
    <td>Claude Code CLI</td>
  </tr>
</table>

## Anthropic (Direct API)

### Environment Variables

```bash
ANTHROPIC_API_KEY=sk-ant-...
```

### Available Models

```python
llm_models=["claude-sonnet-4-6"]
llm_models=["claude-opus-4-6"]
llm_models=["claude-haiku-4-5"]
```

## Anthropic via AWS Bedrock (recommended)

This is the default provider in the included examples.

### Environment Variables

```bash
AWS_ACCESS_KEY_ID=AKIA...
AWS_SECRET_ACCESS_KEY=...
AWS_REGION_NAME=us-east-1
```

### Available Models

Model names use the `bedrock/` prefix followed by the Bedrock model ID:

```python
# Sonnet 4.6
llm_models=["bedrock/us.anthropic.claude-sonnet-4-6"]

# Opus 4.6 (strongest, most expensive)
llm_models=["bedrock/us.anthropic.claude-opus-4-6-v1"]

# Sonnet 4.5
llm_models=["bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0"]

# Sonnet 4
llm_models=["bedrock/us.anthropic.claude-sonnet-4-20250514-v1:0"]

# Haiku 4.5 (fastest, cheapest)
llm_models=["bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0"]
```

### Pricing (per million tokens)

<table>
  <tr>
    <th>Model</th>
    <th>Input</th>
    <th>Output</th>
  </tr>
  <tr>
    <td>Claude Opus 4.6</td>
    <td>$5.00</td>
    <td>$25.00</td>
  </tr>
  <tr>
    <td>Claude Sonnet 4.6</td>
    <td>$3.00</td>
    <td>$15.00</td>
  </tr>
  <tr>
    <td>Claude Sonnet 4.5</td>
    <td>$3.00</td>
    <td>$15.00</td>
  </tr>
  <tr>
    <td>Claude Haiku 4.5</td>
    <td>$1.00</td>
    <td>$5.00</td>
  </tr>
</table>

## OpenAI

### Environment Variables

```bash
OPENAI_API_KEY=sk-...
```

### Available Models

```python
llm_models=["gpt-4.1-mini"]
llm_models=["gpt-4.1"]
llm_models=["o3-mini"]
llm_models=["o4-mini"]
```

## Azure OpenAI

### Environment Variables

```bash
AZURE_OPENAI_API_KEY=...
AZURE_API_VERSION=2024-02-15-preview
AZURE_API_ENDPOINT=https://your-resource.openai.azure.com/
```

### Model Names

Use the `azure-` prefix:

```python
llm_models=["azure-gpt-4.1-mini"]
llm_models=["azure-gpt-4.1"]
```

## DeepSeek

### Environment Variables

```bash
DEEPSEEK_API_KEY=...
```

### Available Models

```python
llm_models=["deepseek-chat"]
llm_models=["deepseek-reasoner"]
```

## Google Gemini

### Environment Variables

```bash
GEMINI_API_KEY=...
```

### Available Models

```python
llm_models=["gemini-2.0-flash"]
llm_models=["gemini-2.5-pro-preview-05-06"]
llm_models=["gemini-2.5-flash-preview-04-17"]
```

## Claude CLI

Uses the Claude Code CLI (`claude -p`) as a subprocess. No API key needed if Claude CLI is already authenticated.

```python
llm_models=["claude-cli/opus"]
llm_models=["claude-cli/sonnet"]
llm_models=["claude-cli/haiku"]
```

This is primarily used for the fast-path agent mode, where the LLM gets full file system autonomy.

## Reasoning Models

Some models support extended thinking / chain-of-thought reasoning. CUCo automatically enables this for known reasoning models:

<table>
  <tr>
    <th>Provider</th>
    <th>Reasoning Models</th>
  </tr>
  <tr>
    <td>Anthropic</td>
    <td><code>claude-3-7-sonnet-*</code>, <code>claude-sonnet-4-*</code>, <code>claude-opus-4-*</code></td>
  </tr>
  <tr>
    <td>OpenAI</td>
    <td><code>o3-mini</code>, <code>o4-mini</code></td>
  </tr>
  <tr>
    <td>DeepSeek</td>
    <td><code>deepseek-reasoner</code></td>
  </tr>
  <tr>
    <td>Gemini</td>
    <td><code>gemini-2.5-pro-*</code>, <code>gemini-2.5-flash-*</code></td>
  </tr>
</table>

For reasoning models, CUCo:
- Sets temperature to 1.0 (required by most reasoning APIs)
- Adds thinking/budget parameters (e.g., `thinking.budget_tokens` for Anthropic)
- Passes `reasoning_effort` if configured in `llm_kwargs`

## Embedding Models

Used for novelty filtering and similarity-based retrieval.

<table>
  <tr>
    <th>Provider</th>
    <th>Model</th>
    <th>Variable</th>
  </tr>
  <tr>
    <td>OpenAI</td>
    <td><code>text-embedding-3-small</code>, <code>text-embedding-3-large</code></td>
    <td><code>OPENAI_API_KEY</code></td>
  </tr>
  <tr>
    <td>Azure</td>
    <td><code>azure-text-embedding-3-small</code></td>
    <td><code>AZURE_OPENAI_API_KEY</code></td>
  </tr>
  <tr>
    <td>Gemini</td>
    <td><code>gemini-embedding-001</code></td>
    <td><code>GEMINI_API_KEY</code></td>
  </tr>
  <tr>
    <td>Bedrock</td>
    <td><code>bedrock-amazon.titan-embed-text-v1</code></td>
    <td><code>AWS_ACCESS_KEY_ID</code></td>
  </tr>
</table>

Configure via:

```python
evo_config = EvolutionConfig(
    embedding_model="bedrock-amazon.titan-embed-text-v1",
    ...
)
```

## Dynamic Model Selection

CUCo can automatically select between multiple models using a bandit algorithm:

```python
evo_config = EvolutionConfig(
    llm_models=[
        "bedrock/us.anthropic.claude-opus-4-6-v1",
        "bedrock/us.anthropic.claude-sonnet-4-6",
    ],
    llm_dynamic_selection="ucb",  # Asymmetric Upper Confidence Bound
    ...
)
```

The UCB bandit tracks which models produce higher-scoring candidates and allocates more queries to better-performing models over time.

Alternatively, use `None` (default) for round-robin selection across models.

## Cost Tracking

CUCo tracks API costs for all LLM calls. Each `QueryResult` includes `input_cost` and `output_cost` based on the pricing tables in `cuco/llm/models/pricing.py`. Cumulative costs are logged during evolution.

To add a new model, add its pricing entry to the appropriate dictionary in `pricing.py`:

```python
BEDROCK_MODELS = {
    "bedrock/your-new-model-id": {
        "input_price": X / M,
        "output_price": Y / M,
    },
    ...
}
```

## Choosing a Model

Recommendations for CUCo workloads:

<table>
  <tr>
    <th>Role</th>
    <th>Recommended</th>
    <th>Reasoning</th>
  </tr>
  <tr>
    <td>Mutation (slow-path)</td>
    <td>Opus 4.6 or Sonnet 4.6</td>
    <td>Complex code reasoning, large context</td>
  </tr>
  <tr>
    <td>Meta-summarization</td>
    <td>Opus 4.6</td>
    <td>Cross-generation pattern analysis</td>
  </tr>
  <tr>
    <td>Fast-path rewrite</td>
    <td>Sonnet 4.6</td>
    <td>Good balance of quality and cost</td>
  </tr>
  <tr>
    <td>Fast-path judge</td>
    <td>Same as rewriter</td>
    <td>Simpler task, lower token count</td>
  </tr>
  <tr>
    <td>Evaluation feedback</td>
    <td>Sonnet 4.6</td>
    <td>Quick factual analysis</td>
  </tr>
  <tr>
    <td>Embeddings</td>
    <td>Titan or text-embedding-3-small</td>
    <td>Cheap, fast</td>
  </tr>
</table>

For budget-conscious runs, Sonnet 4.6 works well for all roles. For maximum quality, use Opus 4.6 for mutation and meta-summarization.
