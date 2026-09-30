"""Program VM — closed-ISA register machine for evolved math algorithms.

Candidates are fixed-length straight-line programs over a closed opcode set.
A program is a [0,1]^D vector produced by the batched generator population;
`decode_params` maps it to index tensors + constants, and `run_programs`
executes a whole batch of programs on a whole batch of input cases in one
vectorized pass (torch, CPU or CUDA).

Safety model: there is NO code to scan. Programs cannot name anything outside
the op table / register file — no eval, no IO, no loops, no host objects.
Every op is a total function (protected div/mod/pow, clamped intermediates),
so every decoded program runs to completion in exactly S steps. This is the
"pure math" route: the ISA is the sandbox.

Honesty model ("real working math, not forced victory") lives in the domain:
held-out input cases, input-usage tracking, output-variance const-detection,
and bounded decoded constants (big targets must be COMPOSED, not embedded).

Register layout per program:
    [0 .. K-1]        task inputs  x0..xK-1
    [K .. K+M-1]      consts       c0..cM-1  (c0=0.0, c1=1.0 fixed)
    [K+M .. K+M+S-1]  step outputs r0..rS-1
Step s may read any register < K+M+s (visible prefix — every program is
well-formed by construction). Final output register "A" = regs[out_sel].

Param vector layout (D = 3S + M + 1):
    p[0*S : 1*S]      op select per step
    p[1*S : 2*S]      operand-a address per step
    p[2*S : 3*S]      operand-b address per step
    p[3*S : 3*S+M]    constant values (decoded, bounded)
    p[3*S+M]          output register select
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Op table — closed set, all total functions on floats.
# ---------------------------------------------------------------------------

_EPS = 1e-9
_CLAMP = 1e9

# (name, arity, fn, int_preserving)
def _pdiv(a, b):
    denom = torch.where(b.abs() > _EPS, b, torch.full_like(b, _EPS))
    return a / denom

def _pmod(a, b):
    return torch.fmod(a, torch.where(b.abs() > _EPS, b, torch.full_like(b, _EPS)))

def _ppow(a, b):
    return torch.sign(a) * torch.pow(a.abs() + _EPS, b.clamp(-4.0, 4.0))

OPS: list[tuple[str, int, Any, bool]] = [
    ("add",  2, lambda a, b: a + b,                        True),
    ("sub",  2, lambda a, b: a - b,                        True),
    ("mul",  2, lambda a, b: a * b,                        True),
    ("pdiv", 2, _pdiv,                                     False),
    ("pmod", 2, _pmod,                                     True),   # int-safe via fmod
    ("pow",  2, _ppow,                                     False),
    ("min",  2, torch.minimum,                             True),
    ("max",  2, torch.maximum,                             True),
    ("neg",  1, lambda a, b: -a,                           True),
    ("abs",  1, lambda a, b: a.abs(),                      True),
    ("floor",1, lambda a, b: torch.floor(a),               True),
    ("round",1, lambda a, b: torch.round(a),               True),
    ("sqrts",1, lambda a, b: torch.sqrt(a.abs() + _EPS),   False),
    ("sigm", 1, lambda a, b: torch.sigmoid(a),             False),
    ("expc", 1, lambda a, b: torch.exp(a.clamp(-8.0, 8.0)), False),
]
N_OPS = len(OPS)
OP_ARITY = torch.tensor([o[1] for o in OPS], dtype=torch.long)
INT_OPS = frozenset(i for i, o in enumerate(OPS) if o[3])

# Constant decode: bounded ints (and halves) so large targets must be composed
# from multiple steps — a program cannot embed a leaked answer constant.
CONST_LOW, CONST_HIGH = -64.0, 64.0


@dataclass
class ProgramSpec:
    """Shape of the program search space."""
    n_steps: int = 12        # S
    n_consts: int = 8        # M (first two are pinned to 0.0 / 1.0)
    max_inputs: int = 4      # K can vary per task; decode uses the task's K
    op_ids: tuple | None = None  # op whitelist (indices into OPS); None = all

    def output_dim(self) -> int:
        return 3 * self.n_steps + self.n_consts + 1


def decode_consts(p: torch.Tensor) -> torch.Tensor:
    """Map [0,1] const params to bounded values. First two slots are pinned."""
    v = p.detach().to(torch.float64).clamp(0.0, 1.0)
    c = torch.round(v * (CONST_HIGH - CONST_LOW) + CONST_LOW)
    if c.numel() >= 2:
        c = c.clone()
        c[0] = 0.0
        c[1] = 1.0
    return c


def decode_params(params: torch.Tensor, spec: ProgramSpec, n_inputs: int) -> dict:
    """Map a [0,1]^D vector to a program dict of integer/float tensors.

    Every decoded program is well-formed: operand addresses index only the
    visible prefix (inputs + consts + earlier step outputs).
    """
    p = params.detach().to(torch.float64).clamp(0.0, 1.0)
    S, M, K = spec.n_steps, spec.n_consts, n_inputs
    op_ids = list(spec.op_ids) if spec.op_ids else list(range(N_OPS))
    n_ops = len(op_ids)
    ops = torch.zeros(S, dtype=torch.long)
    a_addr = torch.zeros(S, dtype=torch.long)
    b_addr = torch.zeros(S, dtype=torch.long)
    for s in range(S):
        visible = K + M + s
        ops[s] = op_ids[min(int(p[s].item() * n_ops), n_ops - 1)]
        a_addr[s] = min(int(p[S + s].item() * visible), visible - 1)
        b_addr[s] = min(int(p[2 * S + s].item() * visible), visible - 1)
    consts = decode_consts(p[3 * S : 3 * S + M])
    out_sel = min(int(p[3 * S + M].item() * (K + M + S)), K + M + S - 1)
    return {
        "ops": ops, "a_addr": a_addr, "b_addr": b_addr,
        "consts": consts, "out_sel": torch.tensor(out_sel, dtype=torch.long),
        "n_inputs": K,
    }


def encode_program(prog: dict, spec: ProgramSpec) -> torch.Tensor:
    """Inverse-ish map back to [0,1]^D (context vector / surrogate input).

    Not a strict inverse of decode_params — mid-cell encoding.
    """
    S, M, K = spec.n_steps, spec.n_consts, int(prog["n_inputs"])
    p = torch.zeros(3 * S + M + 1, dtype=torch.float64)
    ops = torch.as_tensor(prog["ops"]).long()
    a = torch.as_tensor(prog["a_addr"]).long()
    b = torch.as_tensor(prog["b_addr"]).long()
    op_ids = list(spec.op_ids) if spec.op_ids else list(range(N_OPS))
    n_ops = len(op_ids)
    pos = {op: i for i, op in enumerate(op_ids)}
    for s in range(S):
        visible = K + M + s
        p[s] = (pos.get(ops[s].item(), 0) + 0.5) / n_ops
        p[S + s] = (min(a[s].item(), visible - 1) + 0.5) / visible
        p[2 * S + s] = (min(b[s].item(), visible - 1) + 0.5) / visible
    consts = torch.as_tensor(prog["consts"]).to(torch.float64)
    cnorm = (consts.clamp(CONST_LOW, CONST_HIGH) - CONST_LOW) / (CONST_HIGH - CONST_LOW)
    p[3 * S : 3 * S + M] = cnorm[:M]
    out = int(torch.as_tensor(prog["out_sel"]).item())
    p[3 * S + M] = (min(out, K + M + S - 1) + 0.5) / (K + M + S)
    return p.float()


def program_from_lists(ops, a_addr, b_addr, consts, out_sel, n_inputs) -> dict:
    """Build a program dict from explicit lists (oracle programs, seeds, tests)."""
    return {
        "ops": torch.as_tensor(ops, dtype=torch.long),
        "a_addr": torch.as_tensor(a_addr, dtype=torch.long),
        "b_addr": torch.as_tensor(b_addr, dtype=torch.long),
        "consts": torch.as_tensor(consts, dtype=torch.float64),
        "out_sel": torch.as_tensor(out_sel, dtype=torch.long),
        "n_inputs": int(n_inputs),
    }


def stack_programs(progs: list[dict], device=None) -> dict:
    """Stack per-candidate program dicts into batched tensors [B, ...]."""
    dev = device
    return {
        "ops": torch.stack([p["ops"] for p in progs]).to(dev),          # [B,S]
        "a_addr": torch.stack([p["a_addr"] for p in progs]).to(dev),    # [B,S]
        "b_addr": torch.stack([p["b_addr"] for p in progs]).to(dev),    # [B,S]
        "consts": torch.stack([p["consts"] for p in progs]).to(dev),    # [B,M]
        "out_sel": torch.stack([p["out_sel"] for p in progs]).view(-1).to(dev),  # [B]
        "n_inputs": int(progs[0]["n_inputs"]),
    }


def run_programs(prog: dict, inputs: torch.Tensor, clamp: float = _CLAMP) -> dict:
    """Execute B programs × T input cases in one vectorized pass.

    Args:
        prog: stacked program dict (from stack_programs; B=1 for a single one)
        inputs: [B, T, K] or [T, K] (broadcast across programs)

    Returns dict with:
        out:      [B, T]   value of register A per case
        used_in:  [B, K]   bool — which inputs any step actually reads
        regs:     [B, T, K+M+S] full register trace (for debugging)
    """
    if inputs.dim() == 2:
        inputs = inputs.unsqueeze(0)
    B = prog["ops"].shape[0]
    if inputs.shape[0] == 1 and B > 1:
        inputs = inputs.expand(B, -1, -1)
    T = inputs.shape[1]
    K_task = inputs.shape[2]
    # Register layout is fixed at decode time (n_inputs = layout K). Tasks with
    # fewer inputs leave dead input slots reading 0 — keeps program semantics
    # independent of the task's arity.
    K = int(prog.get("n_inputs", K_task))
    dev = prog["ops"].device
    dt = torch.float64
    S = prog["ops"].shape[1]
    M = prog["consts"].shape[1]
    R = K + M + S

    inputs = inputs.to(dev).to(dt)
    regs = torch.zeros(B, T, R, device=dev, dtype=dt)
    regs[:, :, :min(K_task, K)] = inputs[..., :min(K_task, K)]
    regs[:, :, K:K + M] = prog["consts"].to(dt).unsqueeze(1)

    arity = OP_ARITY.to(dev)                      # [N_OPS]
    used_in = torch.zeros(B, K_task, dtype=torch.bool, device=dev)
    in_arange = torch.arange(K_task, device=dev).view(1, K_task)

    op_fns = [o[2] for o in OPS]
    for s in range(S):
        visible = K + M + s
        aidx = prog["a_addr"][:, s].clamp(0, visible - 1)   # [B]
        bidx = prog["b_addr"][:, s].clamp(0, visible - 1)   # [B]
        a = regs.gather(2, aidx.view(B, 1, 1).expand(B, T, 1)).squeeze(2)
        b = regs.gather(2, bidx.view(B, 1, 1).expand(B, T, 1)).squeeze(2)

        # Evaluate all ops, select by opcode — the whole population for this
        # step is a handful of small tensor ops.
        outs = torch.stack([fn(a, b) for fn in op_fns], dim=-1)  # [B,T,N_OPS]
        sel = prog["ops"][:, s].view(B, 1, 1).expand(B, T, 1)
        r = outs.gather(2, sel).squeeze(2)
        r = torch.nan_to_num(r, nan=0.0, posinf=clamp, neginf=-clamp).clamp(-clamp, clamp)
        regs[:, :, K + M + s] = r

        # Input-usage tracking: mark reads of provided input registers.
        # Binary ops read both operands; unary ops read only 'a'.
        binary = arity[prog["ops"][:, s]] == 2                    # [B]
        used_in |= (aidx.view(B, 1) == in_arange)
        used_in |= (bidx.view(B, 1) == in_arange) & binary.view(B, 1)

    out_idx = prog["out_sel"].clamp(0, R - 1)
    out = regs.gather(2, out_idx.view(B, 1, 1).expand(B, T, 1)).squeeze(2)
    return {"out": out, "used_in": used_in, "regs": regs}


def dep_steps(prog: dict, K: int, M: int) -> tuple[set[int], set[int]]:
    """Backward dependency cone of the output register.

    Returns (step_indices, input_addrs_read). Only ops whose results actually
    reach A count — dead-end steps cannot fake input usage.
    """
    S = prog["ops"].numel() if torch.is_tensor(prog["ops"]) else len(prog["ops"])
    ops = torch.as_tensor(prog["ops"]).tolist()
    a = torch.as_tensor(prog["a_addr"]).tolist()
    b = torch.as_tensor(prog["b_addr"]).tolist()
    out = int(torch.as_tensor(prog["out_sel"]).item())
    needed: set[int] = set()
    used_ins: set[int] = set()
    stack = [out]
    while stack:
        r = stack.pop()
        if r < K:                      # input register
            used_ins.add(r)
            continue
        s = r - (K + M)
        if r < K + M or s in needed or s >= S:   # const or already-seen step
            continue
        needed.add(s)
        ar = (a[s], b[s]) if OP_ARITY[ops[s]].item() == 2 else (a[s],)
        stack.extend(ar)
    return needed, used_ins


def steps_used(prog: dict, K: int, M: int) -> int:
    """Count how many steps feed the output (backward dependency trace)."""
    return len(dep_steps(prog, K, M)[0])


def render(prog: dict, K: int | None = None) -> str:
    """Readable rendering: 'A = r2' preceded by step assignments."""
    M = prog["consts"].numel() if torch.is_tensor(prog["consts"]) else len(prog["consts"])
    K = K if K is not None else int(prog.get("n_inputs", 0))
    consts = torch.as_tensor(prog["consts"]).tolist()
    ops = torch.as_tensor(prog["ops"]).tolist()
    a = torch.as_tensor(prog["a_addr"]).tolist()
    b = torch.as_tensor(prog["b_addr"]).tolist()
    out = int(torch.as_tensor(prog["out_sel"]).item())

    def name(i):
        if i < K:
            return f"x{i}"
        if i < K + M:
            v = consts[i - K]
            return f"{v:g}"
        return f"r{i - K - M}"

    lines = []
    for s in range(S := len(ops)):
        op = OPS[ops[s]]
        if op[1] == 1:
            rhs = f"{op[0]}({name(a[s])})"
        else:
            rhs = f"({name(a[s])} {op[0]} {name(b[s])})"
        lines.append(f"r{s} = {rhs}")
    lines.append(f"A = {name(out)}")
    return "; ".join(lines)


def config_dict(prog: dict, raw_params: torch.Tensor | None = None) -> dict:
    """JSON-serializable config dict for a decoded program."""
    cfg = {
        "ops": torch.as_tensor(prog["ops"]).tolist(),
        "a_addr": torch.as_tensor(prog["a_addr"]).tolist(),
        "b_addr": torch.as_tensor(prog["b_addr"]).tolist(),
        "consts": [float(c) for c in torch.as_tensor(prog["consts"]).tolist()],
        "out_sel": int(torch.as_tensor(prog["out_sel"]).item()),
        "n_inputs": int(prog["n_inputs"]),
    }
    if raw_params is not None:
        cfg["_p"] = [float(v) for v in raw_params.detach().cpu().flatten().tolist()]
    return cfg


def prog_from_config(config: dict) -> dict:
    """Reconstruct a program dict from a JSON-safe config dict."""
    return program_from_lists(
        config["ops"], config["a_addr"], config["b_addr"],
        config["consts"], config["out_sel"], config.get("n_inputs", 0),
    )
