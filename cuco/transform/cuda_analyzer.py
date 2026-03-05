"""Regex-based CUDA source analyzer for identifying communication patterns.

Extracts NCCL collective calls, memory allocations, kernel launches,
synchronization points, and structural information from CUDA source files.
Produces a structured AnalysisReport that can be fed to an LLM for
understanding and rewriting.

The communication graph layer (CommunicationNode / BufferRole) sits on top
of the flat extraction and connects each NCCL collective to its:
  - parsed semantic arguments (sendbuf, recvbuf, datatype, op, …)
  - buffer allocations
  - producer kernels (writes to sendbuf before the collective)
  - consumer kernels (reads from recvbuf after the collective)
  - GIN transformation hints
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# NCCL host-side collective function names (the targets for replacement)
# ---------------------------------------------------------------------------
NCCL_COLLECTIVES = {
    "ncclAllReduce",
    "ncclBroadcast",
    "ncclReduce",
    "ncclAllGather",
    "ncclReduceScatter",
    "ncclAllToAll",   # NCCL 2.19+
    "ncclAlltoAll",   # alternate casing used in some NCCL versions
    "ncclGather",
    "ncclScatter",
    "ncclSend",
    "ncclRecv",
    "ncclGroupStart",
    "ncclGroupEnd",
}

# NCCL host-side setup/teardown calls (not targets, but context)
NCCL_SETUP_CALLS = {
    "ncclCommInitRank",
    "ncclCommInitAll",
    "ncclGetUniqueId",
    "ncclCommDestroy",
    "ncclMemAlloc",
    "ncclMemFree",
    "ncclCommWindowRegister",
    "ncclCommWindowDeregister",
    "ncclDevCommCreate",
    "ncclDevCommDestroy",
}

# CUDA memory allocation patterns
CUDA_ALLOC_FUNCTIONS = {
    "cudaMalloc",
    "cudaMallocManaged",
    "cudaMallocHost",
    "ncclMemAlloc",
}

CUDA_FREE_FUNCTIONS = {
    "cudaFree",
    "cudaFreeHost",
    "ncclMemFree",
}

# CUDA synchronization
CUDA_SYNC_FUNCTIONS = {
    "cudaDeviceSynchronize",
    "cudaStreamSynchronize",
    "cudaEventSynchronize",
    "cudaEventRecord",
    "cudaStreamWaitEvent",
    "__syncthreads",
}

MPI_FUNCTIONS = {
    "MPI_Init",
    "MPI_Finalize",
    "MPI_Comm_rank",
    "MPI_Comm_size",
    "MPI_Bcast",
    "MPI_Barrier",
    "MPI_Allreduce",
    "MPI_Send",
    "MPI_Recv",
}


# ---------------------------------------------------------------------------
# NCCL collective function signatures — positional argument roles
# ---------------------------------------------------------------------------
# Each entry maps a collective name to its ordered parameter list.
# "sendbuf" / "recvbuf" are the buffer roles we trace through the graph.
NCCL_COLLECTIVE_SIGNATURES: Dict[str, List[str]] = {
    "ncclAllReduce":     ["sendbuf", "recvbuf", "count", "datatype", "op", "comm", "stream"],
    "ncclBroadcast":     ["sendbuf", "recvbuf", "count", "datatype", "root", "comm", "stream"],
    "ncclReduce":        ["sendbuf", "recvbuf", "count", "datatype", "op", "root", "comm", "stream"],
    "ncclAllGather":     ["sendbuf", "recvbuf", "sendcount", "datatype", "comm", "stream"],
    "ncclReduceScatter": ["sendbuf", "recvbuf", "recvcount", "datatype", "op", "comm", "stream"],
    "ncclAllToAll":      ["sendbuf", "recvbuf", "count", "datatype", "comm", "stream"],
    "ncclAlltoAll":      ["sendbuf", "recvbuf", "count", "datatype", "comm", "stream"],
    "ncclGather":        ["sendbuf", "recvbuf", "sendcount", "datatype", "root", "comm", "stream"],
    "ncclScatter":       ["sendbuf", "recvbuf", "recvcount", "datatype", "root", "comm", "stream"],
    "ncclSend":          ["sendbuf", "count", "datatype", "peer", "comm", "stream"],
    "ncclRecv":          ["recvbuf", "count", "datatype", "peer", "comm", "stream"],
}

_DEVICE_COMM_PATTERNS: Dict[str, str] = {
    "ncclAllReduce":     "AllReduce: send to all peers, reduce locally after all arrive",
    "ncclAllGather":     "AllGather: each rank writes its chunk at offset [rank * chunk_size] on every peer",
    "ncclReduceScatter": "ReduceScatter: chunk[i] → rank i; each rank reduces its received chunks",
    "ncclAllToAll":      "AlltoAll: rank r sends sendbuf[peer*chunk] → peer's recvbuf[r*chunk] + local self-copy",
    "ncclAlltoAll":      "AlltoAll: rank r sends sendbuf[peer*chunk] → peer's recvbuf[r*chunk] + local self-copy",
    "ncclBroadcast":     "Broadcast: root sends to all peers' recv buffers",
    "ncclReduce":        "Reduce: all ranks send to root, root reduces",
    "ncclGather":        "Gather: all ranks send chunk to root at offset [rank * chunk_size]",
    "ncclScatter":       "Scatter: root sends chunk[r] to each rank r",
    "ncclSend":          "P2P send to specific peer",
    "ncclRecv":          "P2P recv from specific peer",
}


# ---------------------------------------------------------------------------
# Data classes for extracted information
# ---------------------------------------------------------------------------


@dataclass
class FunctionCall:
    """A function call found in the source."""
    name: str
    line: int
    full_match: str          # The full text of the call (may span multiple lines)
    arguments_raw: str       # Raw argument text inside parentheses
    containing_function: Optional[str] = None
    inside_loop: Optional[str] = None  # e.g., "for (int i = 0; i < NUM_CHUNKS; i++)"
    loop_line: Optional[int] = None
    category: str = ""       # "nccl_collective", "nccl_setup", "cuda_alloc", etc.


@dataclass
class MemoryAllocation:
    """A memory allocation call."""
    allocator: str           # "cudaMalloc" or "ncclMemAlloc"
    buffer_name: str         # e.g., "d_C"
    size_expr: str           # e.g., "size_c"
    line: int
    containing_function: Optional[str] = None
    gin_compatible: bool = False  # True if ncclMemAlloc (compatible with device-side comm)


@dataclass
class KernelLaunch:
    """A CUDA kernel launch (<<<>>>)."""
    kernel_name: str
    line: int
    grid_expr: str
    block_expr: str
    shared_mem_expr: str
    stream_expr: str
    arguments_raw: str
    containing_function: Optional[str] = None
    inside_loop: Optional[str] = None
    loop_line: Optional[int] = None


@dataclass
class KernelDefinition:
    """A __global__ kernel definition."""
    name: str
    line_start: int
    line_end: int
    parameters_raw: str
    is_global: bool = True


@dataclass
class FunctionDefinition:
    """A function definition (non-kernel)."""
    name: str
    line_start: int
    line_end: int
    parameters_raw: str
    return_type: str = ""


@dataclass
class ForLoop:
    """A for loop found in the source."""
    line: int
    header: str              # e.g., "for (int i = 0; i < NUM_CHUNKS; i++)"
    containing_function: Optional[str] = None
    body_start: int = 0
    body_end: int = 0


# ---------------------------------------------------------------------------
# Communication graph data classes
# ---------------------------------------------------------------------------


@dataclass
class BufferRole:
    """A buffer's role (send or recv) in a communication operation."""
    role: str                      # "sendbuf" or "recvbuf"
    arg_expr: str                  # raw expression from the NCCL call, e.g. "d_quant_send"
    base_name: str                 # first identifier, e.g. "d_quant_send"
    alloc: Optional[MemoryAllocation] = None
    producers: List[str] = field(default_factory=list)   # "quantizeTokens<<<...>>> (line 109)"
    consumers: List[str] = field(default_factory=list)   # "expertCompute<<<...>>> (line 116)"


@dataclass
class CommunicationNode:
    """A single NCCL collective call with its full data-flow context.

    This is the primary unit of the communication graph.  Each node
    carries the parsed arguments, buffer provenance, and GIN hints.
    """
    collective: str                # "ncclAllToAll", "ncclAllReduce", ...
    line: int
    containing_function: Optional[str] = None
    inside_loop: Optional[str] = None
    loop_line: Optional[int] = None

    # Parsed semantic arguments
    sendbuf: Optional[BufferRole] = None
    recvbuf: Optional[BufferRole] = None
    count_expr: str = ""
    datatype: str = ""
    op: str = ""                   # "ncclSum" for reduce ops, "" otherwise
    root: str = ""                 # "0" for root-based ops, "" otherwise
    comm: str = ""
    stream: str = ""
    peer: str = ""                 # for ncclSend/ncclRecv
    in_place: bool = False         # sendbuf arg == recvbuf arg

    # Device-side communication transformation hints
    device_comm_pattern: str = ""
    buffers_needing_nccl_memalloc: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# AnalysisReport
# ---------------------------------------------------------------------------


@dataclass
class AnalysisReport:
    """Complete analysis of a CUDA source file."""
    source_path: str
    source_lines: List[str]
    num_lines: int

    # Extracted elements (flat inventory)
    nccl_collectives: List[FunctionCall] = field(default_factory=list)
    nccl_setup_calls: List[FunctionCall] = field(default_factory=list)
    memory_allocations: List[MemoryAllocation] = field(default_factory=list)
    kernel_launches: List[KernelLaunch] = field(default_factory=list)
    kernel_definitions: List[KernelDefinition] = field(default_factory=list)
    function_definitions: List[FunctionDefinition] = field(default_factory=list)
    sync_points: List[FunctionCall] = field(default_factory=list)
    mpi_calls: List[FunctionCall] = field(default_factory=list)
    for_loops: List[ForLoop] = field(default_factory=list)
    includes: List[Tuple[int, str]] = field(default_factory=list)  # (line, header)
    macros: List[Tuple[int, str, str]] = field(default_factory=list)  # (line, name, value)

    # Communication graph (rich, connected view)
    comm_nodes: List[CommunicationNode] = field(default_factory=list)

    def has_host_communication(self) -> bool:
        """True if any host-side NCCL collective calls remain."""
        return len(self.nccl_collectives) > 0

    # ------------------------------------------------------------------
    # Communication graph formatting
    # ------------------------------------------------------------------

    def format_comm_graph(self) -> str:
        """Produce the graph-oriented communication analysis for LLM consumption."""
        if not self.comm_nodes:
            return ""

        sections: List[str] = []
        sections.append("### Communication Graph")
        sections.append("")

        for idx, node in enumerate(self.comm_nodes, 1):
            sections.append(f"  Node {idx}: {node.collective} (line {node.line})")

            if node.containing_function:
                sections.append(f"  ├── In function: {node.containing_function}")
            if node.inside_loop:
                sections.append(f"  ├── In loop (line {node.loop_line}): {node.inside_loop}")
            sections.append(f"  ├── Stream: {node.stream or '(default)'}")
            sections.append(f"  ├── Communicator: {node.comm}")
            sections.append(f"  ├── Data type: {node.datatype}")
            sections.append(f"  ├── Count: {node.count_expr}")
            if node.op:
                sections.append(f"  ├── Reduction op: {node.op}")
            if node.root:
                sections.append(f"  ├── Root rank: {node.root}")
            if node.peer:
                sections.append(f"  ├── Peer rank: {node.peer}")
            sections.append(f"  ├── In-place: {'Yes' if node.in_place else 'No'}")
            sections.append("  │")

            # Send buffer
            if node.sendbuf:
                sb = node.sendbuf
                sections.append(f"  ├── Send buffer: {sb.base_name}")
                if sb.arg_expr != sb.base_name:
                    sections.append(f"  │   ├── Expression: {sb.arg_expr}")
                if sb.alloc:
                    compat = "device-comm compatible" if sb.alloc.gin_compatible else "needs ncclMemAlloc"
                    sections.append(
                        f"  │   ├── Allocated: {sb.alloc.allocator} "
                        f"(line {sb.alloc.line}, size: {sb.alloc.size_expr}) [{compat}]"
                    )
                else:
                    sections.append("  │   ├── Allocated: (not found — may be a derived pointer)")
                if sb.producers:
                    for p in sb.producers:
                        sections.append(f"  │   ├── Produced by: {p}")
                else:
                    sections.append("  │   ├── Produced by: (no kernel producer found — check host-side init)")
                if sb.consumers:
                    for c in sb.consumers:
                        sections.append(f"  │   └── Also consumed by: {c}")
                sections.append("  │")

            # Recv buffer
            if node.recvbuf:
                rb = node.recvbuf
                sections.append(f"  ├── Recv buffer: {rb.base_name}")
                if rb.arg_expr != rb.base_name:
                    sections.append(f"  │   ├── Expression: {rb.arg_expr}")
                if rb.alloc:
                    compat = "device-comm compatible" if rb.alloc.gin_compatible else "needs ncclMemAlloc"
                    sections.append(
                        f"  │   ├── Allocated: {rb.alloc.allocator} "
                        f"(line {rb.alloc.line}, size: {rb.alloc.size_expr}) [{compat}]"
                    )
                else:
                    sections.append("  │   ├── Allocated: (not found — may be a derived pointer)")
                if rb.consumers:
                    for c in rb.consumers:
                        sections.append(f"  │   ├── Consumed by: {c}")
                else:
                    sections.append("  │   ├── Consumed by: (no kernel consumer found — check host-side readback)")
                if rb.producers:
                    for p in rb.producers:
                        sections.append(f"  │   └── Also produced by: {p}")
                sections.append("  │")

            # Device-side communication transformation
            if node.device_comm_pattern:
                sections.append("  └── Device-side transformation:")
                sections.append(f"      ├── Pattern: {node.device_comm_pattern}")
                if node.buffers_needing_nccl_memalloc:
                    bufs = ", ".join(node.buffers_needing_nccl_memalloc)
                    sections.append(f"      └── Buffers needing ncclMemAlloc: {bufs}")
                else:
                    sections.append("      └── All buffers already compatible")
            sections.append("")

        # Execution order
        sections.append(self._format_execution_order())

        return "\n".join(sections)

    def _format_execution_order(self) -> str:
        """Build an execution-order timeline for each containing function."""
        func_events: Dict[str, List[Tuple[int, str, str]]] = {}

        for kl in self.kernel_launches:
            fn = kl.containing_function or "(global)"
            func_events.setdefault(fn, []).append(
                (kl.line, "compute", f"{kl.kernel_name}<<<...>>> (line {kl.line})")
            )
        for node in self.comm_nodes:
            fn = node.containing_function or "(global)"
            func_events.setdefault(fn, []).append(
                (node.line, "communicate", f"{node.collective} (line {node.line})")
            )

        if not func_events:
            return ""

        lines: List[str] = ["### Execution Order (per function)"]
        for fn, events in sorted(func_events.items()):
            events.sort(key=lambda e: e[0])
            labels = []
            for _, kind, desc in events:
                tag = "[compute]" if kind == "compute" else "[communicate]"
                labels.append(f"{desc} {tag}")
            lines.append(f"  {fn}: " + " → ".join(labels))
        lines.append("")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # LLM-facing text report
    # ------------------------------------------------------------------

    def format_for_llm(self) -> str:
        """Generate a focused analysis for LLM consumption.

        Outputs only the information an LLM needs to understand and
        transform the program's communication patterns:
          1. Communication dependency graph (collectives + buffers + producers/consumers)
          2. Execution order timeline (compute vs. communicate)
          3. Brief structural summary (counts only)
        """
        sections: List[str] = []
        sections.append(f"## CUDA Source Analysis: {self.source_path}")
        sections.append(f"Lines: {self.num_lines}\n")

        # -- Communication dependency graph (the core analysis) --
        comm_graph = self.format_comm_graph()
        if comm_graph:
            sections.append(comm_graph)
        elif self.nccl_collectives:
            sections.append("### Host-side NCCL collectives (no dependency graph built)")
            for call in self.nccl_collectives:
                loc = f" in {call.containing_function}" if call.containing_function else ""
                sections.append(f"  Line {call.line}: {call.name}({call.arguments_raw}){loc}")
            sections.append("")
        else:
            sections.append("No host-side NCCL collective calls found.\n")

        # -- Structural summary (compact, not a full dump) --
        parts: List[str] = []
        if self.kernel_definitions:
            names = [kd.name for kd in self.kernel_definitions]
            parts.append(f"kernels({len(names)}): {', '.join(names)}")
        if self.function_definitions:
            names = [fd.name for fd in self.function_definitions]
            parts.append(f"functions({len(names)}): {', '.join(names)}")
        if self.memory_allocations:
            gin_count = sum(1 for m in self.memory_allocations if m.gin_compatible)
            cuda_count = len(self.memory_allocations) - gin_count
            alloc_parts: List[str] = []
            if gin_count:
                alloc_parts.append(f"{gin_count} device-comm compatible")
            if cuda_count:
                alloc_parts.append(f"{cuda_count} cudaMalloc")
            parts.append(f"allocations: {', '.join(alloc_parts)}")
        if self.sync_points:
            host_sync = [s for s in self.sync_points if s.name != "__syncthreads"]
            if host_sync:
                parts.append(f"host syncs: {len(host_sync)}")
        if parts:
            sections.append("### Structure")
            for p in parts:
                sections.append(f"  {p}")
            sections.append("")

        return "\n".join(sections)


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------


class CUDAAnalyzer:
    """Regex-based analyzer for CUDA source files.

    Extracts communication patterns, memory allocations, kernel information,
    and structural context to produce an AnalysisReport.
    """

    def __init__(self, source_path: str | Path):
        self.source_path = Path(source_path)
        self.source = self.source_path.read_text(encoding="utf-8")
        self.lines = self.source.splitlines()
        self.num_lines = len(self.lines)

    def analyze(self) -> AnalysisReport:
        """Run full analysis and return the report."""
        report = AnalysisReport(
            source_path=str(self.source_path),
            source_lines=self.lines,
            num_lines=self.num_lines,
        )

        report.includes = self._find_includes()
        report.macros = self._find_macros()
        report.function_definitions = self._find_function_definitions()
        report.kernel_definitions = self._find_kernel_definitions()
        report.for_loops = self._find_for_loops()
        report.nccl_collectives = self._find_calls(NCCL_COLLECTIVES, "nccl_collective")
        report.nccl_setup_calls = self._find_calls(NCCL_SETUP_CALLS, "nccl_setup")
        report.memory_allocations = self._find_memory_allocations()
        report.kernel_launches = self._find_kernel_launches()
        report.sync_points = self._find_calls(CUDA_SYNC_FUNCTIONS, "sync")
        report.mpi_calls = self._find_calls(MPI_FUNCTIONS, "mpi")

        # Enrich with structural context
        self._annotate_containing_functions(report)
        self._annotate_loop_context(report)

        # Build the communication graph
        report.comm_nodes = self._build_comm_graph(report)

        return report

    # ===================================================================
    # Communication graph builder
    # ===================================================================

    def _build_comm_graph(self, report: AnalysisReport) -> List[CommunicationNode]:
        """Build CommunicationNode objects from the flat extraction data.

        For each NCCL collective call:
        1. Parse arguments per the known signature
        2. Resolve buffer names to allocations
        3. Find producer/consumer kernels
        4. Generate GIN transformation hints
        """
        aliases = self._resolve_pointer_aliases()
        memcpy_info = self._find_memcpy_memset_targets()
        nodes: List[CommunicationNode] = []

        for call in report.nccl_collectives:
            if call.name in ("ncclGroupStart", "ncclGroupEnd"):
                continue

            sig = NCCL_COLLECTIVE_SIGNATURES.get(call.name)
            if not sig:
                continue

            parsed = self._parse_collective_args(call.name, call.arguments_raw, sig)
            if not parsed:
                continue

            node = CommunicationNode(
                collective=call.name,
                line=call.line,
                containing_function=call.containing_function,
                inside_loop=call.inside_loop,
                loop_line=call.loop_line,
                count_expr=parsed.get("count", parsed.get("sendcount", parsed.get("recvcount", ""))),
                datatype=parsed.get("datatype", ""),
                op=parsed.get("op", ""),
                root=parsed.get("root", ""),
                comm=parsed.get("comm", ""),
                stream=parsed.get("stream", ""),
                peer=parsed.get("peer", ""),
            )

            # Build sendbuf role
            sendbuf_expr = parsed.get("sendbuf", "")
            if sendbuf_expr:
                base = _extract_base_buffer(sendbuf_expr)
                alloc = self._find_buffer_allocation(base, aliases, report)
                producers = self._find_producers(
                    base, call.line, call.containing_function,
                    aliases, memcpy_info, report)
                consumers = self._find_consumers(
                    base, call.line, call.containing_function,
                    aliases, memcpy_info, report)
                node.sendbuf = BufferRole(
                    role="sendbuf",
                    arg_expr=sendbuf_expr,
                    base_name=base,
                    alloc=alloc,
                    producers=producers,
                    consumers=consumers,
                )

            # Build recvbuf role
            recvbuf_expr = parsed.get("recvbuf", "")
            if recvbuf_expr:
                base = _extract_base_buffer(recvbuf_expr)
                alloc = self._find_buffer_allocation(base, aliases, report)
                producers = self._find_producers(
                    base, call.line, call.containing_function,
                    aliases, memcpy_info, report)
                consumers = self._find_consumers(
                    base, call.line, call.containing_function,
                    aliases, memcpy_info, report)
                node.recvbuf = BufferRole(
                    role="recvbuf",
                    arg_expr=recvbuf_expr,
                    base_name=base,
                    alloc=alloc,
                    producers=producers,
                    consumers=consumers,
                )

            # In-place detection
            if sendbuf_expr and recvbuf_expr:
                node.in_place = (sendbuf_expr.strip() == recvbuf_expr.strip())

            # Device-side communication hints
            node.device_comm_pattern = _DEVICE_COMM_PATTERNS.get(call.name, "")
            needs_alloc: List[str] = []
            for br in (node.sendbuf, node.recvbuf):
                if br and br.alloc and not br.alloc.gin_compatible:
                    if br.base_name not in needs_alloc:
                        needs_alloc.append(br.base_name)
            node.buffers_needing_nccl_memalloc = needs_alloc

            nodes.append(node)

        return nodes

    def _parse_collective_args(
        self, name: str, args_raw: str, sig: List[str],
    ) -> Optional[Dict[str, str]]:
        """Split *args_raw* and map positionally to *sig* roles."""
        clean = re.sub(r'//[^\n]*', '', args_raw)
        parts = _split_toplevel_commas(clean)
        if len(parts) < len(sig):
            return None
        return {role: " ".join(parts[i].split()).strip() for i, role in enumerate(sig)}

    # ------------------------------------------------------------------
    # Buffer resolution helpers
    # ------------------------------------------------------------------

    def _resolve_pointer_aliases(self) -> Dict[str, str]:
        """Find local pointer assignments like ``type* var = base + expr;``

        Returns ``{alias_var: base_buffer_name}``.
        """
        aliases: Dict[str, str] = {}
        pattern = re.compile(
            r'(?:const\s+)?\w[\w\s\*]*\*\s+(\w+)\s*=\s*(\w+)\s*(?:[+\-\[;]|$)',
            re.MULTILINE,
        )
        for m in pattern.finditer(self.source):
            alias = m.group(1)
            base = m.group(2)
            if base not in ("new", "nullptr", "NULL", "0"):
                aliases[alias] = base
        return aliases

    def _find_memcpy_memset_targets(self) -> List[Tuple[int, str, str]]:
        """Find cudaMemcpy/cudaMemset calls and extract (line, target_buf, kind).

        ``kind`` is "producer" for writes to device, "consumer" for reads from device.
        """
        results: List[Tuple[int, str, str]] = []

        # cudaMemcpy(dst, src, size, direction)
        for m in re.finditer(r'\bcudaMemcpy\s*\(', self.source):
            line = self.source[:m.start()].count("\n") + 1
            args = self._extract_balanced_parens(m.end())
            parts = _split_toplevel_commas(args)
            if len(parts) >= 4:
                direction = parts[3].strip()
                dst_base = _extract_base_buffer(parts[0])
                if "HostToDevice" in direction:
                    results.append((line, dst_base, "producer"))
                elif "DeviceToHost" in direction:
                    src_base = _extract_base_buffer(parts[1])
                    results.append((line, src_base, "consumer"))

        # cudaMemset(ptr, value, count)
        for m in re.finditer(r'\bcudaMemset\s*\(', self.source):
            line = self.source[:m.start()].count("\n") + 1
            args = self._extract_balanced_parens(m.end())
            parts = _split_toplevel_commas(args)
            if parts:
                dst_base = _extract_base_buffer(parts[0])
                results.append((line, dst_base, "producer"))

        return results

    def _find_buffer_allocation(
        self,
        buf_name: str,
        aliases: Dict[str, str],
        report: AnalysisReport,
    ) -> Optional[MemoryAllocation]:
        """Find the MemoryAllocation for *buf_name*, resolving aliases."""
        candidates = [buf_name]
        if buf_name in aliases:
            candidates.append(aliases[buf_name])
        for alias, base in aliases.items():
            if base == buf_name:
                candidates.append(alias)

        for ma in report.memory_allocations:
            if ma.buffer_name in candidates:
                return ma
        return None

    def _find_producers(
        self,
        buf_name: str,
        nccl_line: int,
        containing_fn: Optional[str],
        aliases: Dict[str, str],
        memcpy_info: List[Tuple[int, str, str]],
        report: AnalysisReport,
    ) -> List[str]:
        """Find kernel launches and host operations that write to *buf_name* before *nccl_line*."""
        names = _buffer_name_variants(buf_name, aliases)
        results: List[str] = []

        for kl in report.kernel_launches:
            if kl.line >= nccl_line:
                continue
            if containing_fn and kl.containing_function != containing_fn:
                continue
            kernel_args = _split_toplevel_commas(kl.arguments_raw)
            arg_bases = {_extract_base_buffer(a) for a in kernel_args}
            if names & arg_bases:
                results.append(f"{kl.kernel_name}<<<...>>> (line {kl.line})")

        for mc_line, mc_buf, mc_kind in memcpy_info:
            if mc_line >= nccl_line:
                continue
            if mc_kind == "producer" and mc_buf in names:
                results.append(f"cudaMemcpy/cudaMemset (line {mc_line})")

        return results

    def _find_consumers(
        self,
        buf_name: str,
        nccl_line: int,
        containing_fn: Optional[str],
        aliases: Dict[str, str],
        memcpy_info: List[Tuple[int, str, str]],
        report: AnalysisReport,
    ) -> List[str]:
        """Find kernel launches and host operations that read from *buf_name* after *nccl_line*."""
        names = _buffer_name_variants(buf_name, aliases)
        results: List[str] = []

        for kl in report.kernel_launches:
            if kl.line <= nccl_line:
                continue
            if containing_fn and kl.containing_function != containing_fn:
                continue
            kernel_args = _split_toplevel_commas(kl.arguments_raw)
            arg_bases = {_extract_base_buffer(a) for a in kernel_args}
            if names & arg_bases:
                results.append(f"{kl.kernel_name}<<<...>>> (line {kl.line})")

        for mc_line, mc_buf, mc_kind in memcpy_info:
            if mc_line <= nccl_line:
                continue
            if mc_kind == "consumer" and mc_buf in names:
                results.append(f"cudaMemcpy D2H (line {mc_line})")

        return results

    # ===================================================================
    # Flat extraction methods (unchanged from original)
    # ===================================================================

    def _find_includes(self) -> List[Tuple[int, str]]:
        includes = []
        for i, line in enumerate(self.lines, 1):
            m = re.match(r'\s*#\s*include\s+([<"][^>"]+[>"])', line)
            if m:
                includes.append((i, m.group(1)))
        return includes

    def _find_macros(self) -> List[Tuple[int, str, str]]:
        macros = []
        for i, line in enumerate(self.lines, 1):
            m = re.match(r'\s*#\s*define\s+(\w+)\s+(.*?)(?:\\)?$', line)
            if m:
                name, value = m.group(1), m.group(2).strip()
                if not value.endswith("\\"):
                    macros.append((i, name, value))
                else:
                    macros.append((i, name, "(multi-line macro)"))
        return macros

    def _find_function_definitions(self) -> List[FunctionDefinition]:
        """Find non-kernel function definitions."""
        defs = []
        pattern = re.compile(
            r'^(\w[\w\s\*&]*?)\s+(\w+)\s*\(([^)]*)\)\s*\{',
            re.MULTILINE,
        )
        for m in pattern.finditer(self.source):
            ret_type = m.group(1).strip()
            name = m.group(2)
            params = m.group(3).strip()
            line_start = self.source[:m.start()].count("\n") + 1

            prefix = self.source[max(0, m.start() - 100):m.start()]
            if "__global__" in prefix or "__global__" in ret_type:
                continue
            if name in ("if", "for", "while", "switch", "do", "else"):
                continue

            line_end = self._find_matching_brace(m.end() - 1)
            defs.append(FunctionDefinition(
                name=name,
                line_start=line_start,
                line_end=line_end,
                parameters_raw=params,
                return_type=ret_type,
            ))
        return defs

    def _find_kernel_definitions(self) -> List[KernelDefinition]:
        """Find __global__ kernel definitions."""
        defs = []
        pattern = re.compile(
            r'__global__\s+void\s+(\w+)\s*\(([^)]*)\)\s*\{',
            re.MULTILINE,
        )
        for m in pattern.finditer(self.source):
            name = m.group(1)
            params = m.group(2).strip()
            line_start = self.source[:m.start()].count("\n") + 1
            line_end = self._find_matching_brace(m.end() - 1)
            defs.append(KernelDefinition(
                name=name,
                line_start=line_start,
                line_end=line_end,
                parameters_raw=params,
            ))
        return defs

    def _find_for_loops(self) -> List[ForLoop]:
        """Find for-loop headers and their body ranges."""
        loops = []
        pattern = re.compile(r'(for\s*\([^)]*\))\s*\{', re.MULTILINE)
        for m in pattern.finditer(self.source):
            header = m.group(1).strip()
            line = self.source[:m.start()].count("\n") + 1
            brace_pos = m.end() - 1
            body_end_line = self._find_matching_brace(brace_pos)
            loops.append(ForLoop(
                line=line,
                header=header,
                body_start=line + 1,
                body_end=body_end_line,
            ))
        return loops

    def _find_calls(self, function_names: set, category: str) -> List[FunctionCall]:
        """Find function calls by name, extracting arguments."""
        calls = []
        names_pattern = "|".join(re.escape(n) for n in sorted(function_names))
        pattern = re.compile(
            rf'\b({names_pattern})\s*\(',
            re.MULTILINE,
        )
        for m in pattern.finditer(self.source):
            name = m.group(1)
            line = self.source[:m.start()].count("\n") + 1
            args_start = m.end()
            args_raw = self._extract_balanced_parens(args_start)
            full_match = self.source[m.start():args_start + len(args_raw) + 1].strip()
            calls.append(FunctionCall(
                name=name,
                line=line,
                full_match=full_match,
                arguments_raw=args_raw.strip(),
                category=category,
            ))
        return calls

    def _find_memory_allocations(self) -> List[MemoryAllocation]:
        """Find cudaMalloc / ncclMemAlloc calls and extract buffer info."""
        allocs = []
        for func_name in CUDA_ALLOC_FUNCTIONS:
            pattern = re.compile(
                rf'\b{re.escape(func_name)}\s*\(',
                re.MULTILINE,
            )
            for m in pattern.finditer(self.source):
                line = self.source[:m.start()].count("\n") + 1
                args_raw = self._extract_balanced_parens(m.end())
                buffer_name, size_expr = self._parse_alloc_args(func_name, args_raw)
                allocs.append(MemoryAllocation(
                    allocator=func_name,
                    buffer_name=buffer_name,
                    size_expr=size_expr,
                    line=line,
                    gin_compatible=(func_name == "ncclMemAlloc"),
                ))
        return allocs

    def _find_kernel_launches(self) -> List[KernelLaunch]:
        """Find CUDA kernel launches: name<<<grid, block, smem, stream>>>(args)."""
        launches = []
        pattern = re.compile(
            r'(\w+)\s*<<<\s*([^,>]+)\s*,\s*([^,>]+)\s*'
            r'(?:,\s*([^,>]+)\s*)?'
            r'(?:,\s*([^>]+)\s*)?>>>\s*\(',
            re.MULTILINE,
        )
        for m in pattern.finditer(self.source):
            kernel_name = m.group(1)
            grid = m.group(2).strip()
            block = m.group(3).strip()
            smem = (m.group(4) or "0").strip()
            stream = (m.group(5) or "0").strip()
            line = self.source[:m.start()].count("\n") + 1
            args_raw = self._extract_balanced_parens(m.end())

            launches.append(KernelLaunch(
                kernel_name=kernel_name,
                line=line,
                grid_expr=grid,
                block_expr=block,
                shared_mem_expr=smem,
                stream_expr=stream,
                arguments_raw=args_raw.strip(),
            ))
        return launches

    # --- Structural annotation ---

    def _annotate_containing_functions(self, report: AnalysisReport):
        """Annotate calls/allocs/launches with their containing function."""
        all_funcs = (
            [(fd.name, fd.line_start, fd.line_end) for fd in report.function_definitions]
            + [(kd.name, kd.line_start, kd.line_end) for kd in report.kernel_definitions]
        )

        def find_containing(line: int) -> Optional[str]:
            for name, start, end in all_funcs:
                if start <= line <= end:
                    return name
            return None

        for call in report.nccl_collectives + report.nccl_setup_calls + report.sync_points + report.mpi_calls:
            call.containing_function = find_containing(call.line)
        for ma in report.memory_allocations:
            ma.containing_function = find_containing(ma.line)
        for kl in report.kernel_launches:
            kl.containing_function = find_containing(kl.line)
        for fl in report.for_loops:
            fl.containing_function = find_containing(fl.line)

    def _annotate_loop_context(self, report: AnalysisReport):
        """Annotate calls/launches that are inside for loops."""
        def find_containing_loop(line: int) -> Optional[ForLoop]:
            for fl in report.for_loops:
                if fl.body_start <= line <= fl.body_end:
                    return fl
            return None

        for call in report.nccl_collectives + report.nccl_setup_calls + report.sync_points:
            loop = find_containing_loop(call.line)
            if loop:
                call.inside_loop = loop.header
                call.loop_line = loop.line
        for kl in report.kernel_launches:
            loop = find_containing_loop(kl.line)
            if loop:
                kl.inside_loop = loop.header
                kl.loop_line = loop.line

    # --- Low-level helpers ---

    def _find_matching_brace(self, open_brace_pos: int) -> int:
        """Find the line number of the closing brace matching the one at open_brace_pos."""
        depth = 1
        pos = open_brace_pos + 1
        while pos < len(self.source) and depth > 0:
            ch = self.source[pos]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
            pos += 1
        return self.source[:pos].count("\n") + 1

    def _extract_balanced_parens(self, start: int) -> str:
        """Extract text inside balanced parentheses starting after '('."""
        depth = 1
        pos = start
        while pos < len(self.source) and depth > 0:
            ch = self.source[pos]
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    return self.source[start:pos]
            pos += 1
        return self.source[start:pos]

    def _parse_alloc_args(self, func_name: str, args_raw: str) -> Tuple[str, str]:
        """Parse buffer name and size from allocation arguments."""
        parts = [p.strip() for p in args_raw.split(",")]
        if func_name == "ncclMemAlloc":
            buf = parts[0] if parts else "?"
            buf = re.sub(r'\(void\s*\*\*\)\s*&?', '', buf).strip("& ")
            size = parts[1] if len(parts) > 1 else "?"
            return buf, size
        else:
            buf = parts[0].strip("& ") if parts else "?"
            size = parts[1] if len(parts) > 1 else "?"
            return buf, size


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _split_toplevel_commas(raw: str) -> List[str]:
    """Split a string on commas, respecting nested parentheses and angle brackets."""
    parts: List[str] = []
    depth = 0
    current: List[str] = []
    for ch in raw:
        if ch in "(<[":
            depth += 1
            current.append(ch)
        elif ch in ")>]":
            depth -= 1
            current.append(ch)
        elif ch == "," and depth == 0:
            parts.append("".join(current).strip())
            current = []
        else:
            current.append(ch)
    trailing = "".join(current).strip()
    if trailing:
        parts.append(trailing)
    return parts


def _extract_base_buffer(expr: str) -> str:
    """Extract the base buffer variable name from an argument expression.

    Examples:
        "d_quant_send"                   → "d_quant_send"
        "d_C + (row_offset * N_DIM)"     → "d_C"
        "(void**)&d_C"                   → "d_C"
        "d_C_local + (local_row * N)"    → "d_C_local"
        "chunk_ptr"                      → "chunk_ptr"
    """
    cleaned = re.sub(r'\([^)]*\*[^)]*\)\s*', '', expr).strip()
    cleaned = cleaned.lstrip("&* ").strip()
    m = re.match(r'(\w+)', cleaned)
    return m.group(1) if m else expr.strip()


def _buffer_name_variants(buf_name: str, aliases: Dict[str, str]) -> set:
    """Return a set of names that refer to the same underlying buffer."""
    names = {buf_name}
    if buf_name in aliases:
        names.add(aliases[buf_name])
    for alias, base in aliases.items():
        if base == buf_name:
            names.add(alias)
    return names
