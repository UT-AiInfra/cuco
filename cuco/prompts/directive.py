"""Optimization directive schema, parsing, and validation.

The directive is a structured block the LLM must emit before generating code,
declaring its choices along the design space dimensions. It is stored in
Program.metadata["directive"] for downstream analysis.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

VALID_BACKENDS = {"GIN", "LSA", "Hybrid"}
VALID_SYNC_MECHANISMS = {"Barrier", "Signal", "SignalShadow", "Counter"}
VALID_ISSUERS = {"Thread", "Warp", "WarpSpan", "CTA"}
VALID_ORDERINGS = {"Relaxed", "Acquire", "Release", "AcqRel"}

DIRECTIVE_PROMPT = """
## Optimization Directive (REQUIRED)

Before writing any code, you MUST emit an optimization directive declaring your
design choices. Wrap it in <DIRECTIVE> tags using the exact format below:

<DIRECTIVE>
backend: <GIN | LSA | Hybrid>
sync_mechanism: <Barrier | Signal | SignalShadow | Counter>
placement: <free-text: e.g. deferred, tile-fused, tile-pipelined, stream-split>
sync_scope: <free-text: e.g. local, world, rail, hierarchical>
issuer: <Thread | Warp | WarpSpan | CTA>
granularity: <free-text: e.g. per-peer, per-tile, per-chunk>
ordering: <Relaxed | Acquire | Release | AcqRel>
contexts: <integer: number of GIN contexts, e.g. 1, 2, 4>
rationale: <1-2 sentences: why this configuration for this workload>
</DIRECTIVE>

Concrete dimensions (backend, sync_mechanism, issuer, ordering, contexts) must
use one of the listed values exactly. Intent dimensions (placement, sync_scope,
granularity) are free-text descriptions of your strategy.
"""


@dataclass
class OptimizationDirective:
    backend: str = ""
    sync_mechanism: str = ""
    placement: str = ""
    sync_scope: str = ""
    issuer: str = ""
    granularity: str = ""
    ordering: str = ""
    contexts: int = 1
    rationale: str = ""
    raw: str = ""
    parse_warnings: list = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d.pop("raw", None)
        return d

    @property
    def is_valid(self) -> bool:
        return bool(self.backend and self.sync_mechanism and self.issuer)


def parse_directive(llm_response: str) -> Optional[OptimizationDirective]:
    match = re.search(
        r"<DIRECTIVE>\s*(.*?)\s*</DIRECTIVE>", llm_response, re.DOTALL
    )
    if not match:
        return None

    raw_block = match.group(1).strip()
    directive = OptimizationDirective(raw=raw_block)
    warnings = []

    for line in raw_block.splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip().lower().replace(" ", "_")
        value = value.strip()

        if key == "backend":
            directive.backend = value
        elif key in ("sync_mechanism", "completion"):
            directive.sync_mechanism = value
        elif key == "placement":
            directive.placement = value
        elif key == "sync_scope":
            directive.sync_scope = value
        elif key == "issuer":
            directive.issuer = value
        elif key == "granularity":
            directive.granularity = value
        elif key == "ordering":
            directive.ordering = value
        elif key == "contexts":
            try:
                directive.contexts = int(value)
            except ValueError:
                directive.contexts = 1
                warnings.append(f"Could not parse contexts '{value}' as int, defaulting to 1")
        elif key == "rationale":
            directive.rationale = value

    directive.parse_warnings = warnings
    return directive


def validate_directive(directive: OptimizationDirective) -> list[str]:
    warnings = list(directive.parse_warnings)

    if directive.backend and directive.backend not in VALID_BACKENDS:
        warnings.append(
            f"Backend '{directive.backend}' not in {VALID_BACKENDS}"
        )
    if directive.sync_mechanism and directive.sync_mechanism not in VALID_SYNC_MECHANISMS:
        warnings.append(
            f"Sync mechanism '{directive.sync_mechanism}' not in {VALID_SYNC_MECHANISMS}"
        )
    if directive.issuer and directive.issuer not in VALID_ISSUERS:
        # Allow Tile<N> variants
        if not re.match(r"Tile<\d+>", directive.issuer):
            warnings.append(
                f"Issuer '{directive.issuer}' not in {VALID_ISSUERS} or Tile<N>"
            )
    if directive.ordering and directive.ordering not in VALID_ORDERINGS:
        warnings.append(
            f"Ordering '{directive.ordering}' not in {VALID_ORDERINGS}"
        )

    if directive.backend == "LSA" and directive.contexts > 1:
        warnings.append("LSA backend does not use GIN contexts; contexts > 1 is ignored")

    return warnings


def extract_directive_from_response(llm_response: str) -> Dict[str, Any]:
    directive = parse_directive(llm_response)
    if directive is None:
        logger.debug("No <DIRECTIVE> block found in LLM response")
        return {}

    warnings = validate_directive(directive)
    if warnings:
        for w in warnings:
            logger.warning(f"Directive validation: {w}")

    result = directive.to_dict()
    result["_valid"] = directive.is_valid
    result["_warnings"] = warnings
    return result
