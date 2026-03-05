"""Claude Code CLI query integration.

Invokes the ``claude`` CLI in print mode (``-p``) for LLM queries.
This enables using Claude Code as the LLM backend for Cuco evolution.

Model names use the ``claude-cli/`` prefix:
    - ``claude-cli/opus``
    - ``claude-cli/sonnet``
    - ``claude-cli/haiku``
"""

from __future__ import annotations

import logging
import os
import subprocess
import time
from typing import Dict, List, Optional

from .result import QueryResult

logger = logging.getLogger(__name__)

CLAUDE_CLI_MODEL_MAP: Dict[str, str] = {
    "claude-cli/opus": "opus",
    "claude-cli/sonnet": "sonnet",
    "claude-cli/haiku": "haiku",
}


def query_claude_cli(
    client,  # unused — kept for interface compatibility with other query_* fns
    model_name: str,
    msg: str,
    system_msg: str,
    msg_history: List[Dict] = [],
    output_model=None,
    model_posteriors: Optional[Dict[str, float]] = None,
    **kwargs,
) -> QueryResult:
    """Query Claude via the Claude Code CLI (``claude -p``).

    The CLI is invoked in print-mode with tools disabled so it behaves as a
    pure text-in / text-out LLM — suitable for generating evolution patches,
    providing feedback, and other structured text tasks.

    Long prompts are passed via *stdin* to avoid shell argument-length limits.
    """
    model_alias = CLAUDE_CLI_MODEL_MAP.get(model_name, "opus")

    # Assemble conversation history into a single user message
    full_user_msg_parts: List[str] = []
    for h in msg_history:
        role = h.get("role", "")
        content = h.get("content", "")
        if role == "user":
            full_user_msg_parts.append(f"[Previous user message]\n{content}\n")
        elif role == "assistant":
            full_user_msg_parts.append(f"[Previous assistant response]\n{content}\n")
    full_user_msg_parts.append(msg)
    full_user_msg = "\n".join(full_user_msg_parts)

    cmd = [
        "claude",
        "-p",
        "--model", model_alias,
        "--output-format", "text",
        "--dangerously-skip-permissions",
        "--no-session-persistence",
        "--tools", "",
    ]

    if system_msg:
        cmd.extend(["--system-prompt", system_msg])

    env = os.environ.copy()
    for p in ["/usr/bin", "/usr/local/bin", "/usr/sbin", os.path.expanduser("~/.local/bin")]:
        if p not in env.get("PATH", ""):
            env["PATH"] = p + ":" + env.get("PATH", "")

    logger.info(f"Querying Claude CLI: model={model_alias}")

    start = time.perf_counter()
    try:
        result = subprocess.run(
            cmd,
            input=full_user_msg,
            capture_output=True,
            text=True,
            timeout=600,
            env=env,
        )

        duration = time.perf_counter() - start
        content = (result.stdout or "").strip()

        if result.returncode != 0:
            stderr = (result.stderr or "")[:500]
            logger.warning(f"Claude CLI exit {result.returncode}: {stderr}")
            if not content:
                content = f"Error: Claude CLI exit {result.returncode}. {stderr[:200]}"

        logger.info(f"Claude CLI response: {len(content)} chars in {duration:.1f}s")

    except subprocess.TimeoutExpired:
        logger.error("Claude CLI timed out (600s)")
        raise RuntimeError("Claude CLI query timed out after 600 seconds")
    except FileNotFoundError:
        raise RuntimeError(
            "claude CLI not found. Install with: npm install -g @anthropic-ai/claude-code"
        )

    input_tokens = (len(system_msg) + len(full_user_msg)) // 4
    output_tokens = len(content) // 4

    new_msg_history = list(msg_history) + [
        {"role": "user", "content": msg},
        {"role": "assistant", "content": content},
    ]

    return QueryResult(
        content=content,
        msg=msg,
        system_msg=system_msg,
        new_msg_history=new_msg_history,
        model_name=model_name,
        kwargs=kwargs,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost=0.0,
        input_cost=0.0,
        output_cost=0.0,
        thought="",
        model_posteriors=model_posteriors,
    )
