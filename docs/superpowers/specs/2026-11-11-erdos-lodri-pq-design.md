# Erdős–Lódri UltraGraph `p/q` E2E — Design

**Date:** 2026-11-11
**Status:** approved for execution (operator directed SDD + cavecrew)
**Target workspace:** `wukong-zeno` (5 crates, frozen ABI)

## Intent

Prove, on real code in this workspace, the two claims the Erdős–Lódri
architecture rests on:

1. **Zero-float determinism** — exact rational `p/q` Cayley rotations preserve
   L2 energy and invariant structure bit-for-bit, while an `f32`/`f64` baseline
   drifts.
2. **Graph compression without losing cross-module bridges** — an Erdős-style
   percolation / regularity / shadow-cut pass over a 4-partition graph yields
   exact conservation (`Σ(L₃x) = 0`), exact percolation ratio, and an
   inspectable shadow edge set, then folds through kompress-super-ultra.

## Non-goals

- No new third-party dependencies.
- No changes to `pivot-abi` (frozen), no changes to sibling crate internals.
- No physical-quantum claim; "quantum" means quantum-inspired unitary linear
  algebra over ℚ / 𝔽ₚ on ordinary silicon.
- No LAN, Ollama, entheai, or dogfeed work in this plan. Those are later
  programs (`erdos-lodri-pq` is lane A only).
- No `f64` anywhere in library crates. Floats are allowed **only** in the
  benchmark example, which is a separate cargo target.

## Constraints (from `AGENTS.md`)

- `#![no_std]`, allocation-free, dependency-free library crates.
- `#![deny(clippy::float_arithmetic)]` at every crate root.
- `pivot-abi` checked arithmetic returns `Option`; never panics on overflow.
- Wire little-endian; `KompressFrame32` = 32 bytes; `CentroidPod64` = 128 bytes.
- Examples are the only `std` entry points.

## Approaches considered

**A. Reuse frozen ABI traits (chosen).** Both examples consume crates through
the existing `TernarySimdPort` / `KompressPort` / `LaplacianOraclePort` /
`QulbCapsulePort` traits and `pivot-abi` value types, exactly like
`examples/omni_zeno_smoke.rs`. No crate edits; new helpers (Cayley gate,
Mersenne projection, Molloy–Reed ratio, CSR builder) live in the examples.

**B. Add Cayley/`𝔽ₚ` primitives to `mem8-wave`.** Rejected: `pivot-abi` is
frozen and sibling crates own narrow lanes; this would widen the ABI for a
demonstration.

**C. Edit `omni_zeno_smoke.rs` in place.** Rejected: that example is the
canonical digest test (`0x5F745302BDDBB7DE`); mutating it risks the workspace
contract. The new work lands as independent examples.

## Design

Two independent example targets under `examples/`:

### `examples/cayley_bench.rs`
- Exact rational Cayley gate `c = (s²−r²)/(s²+r²)`, `σ = 2rs/(s²+r²)` over
  `RationalPQ`, applied to a 16-lane wave state.
- Per-step exact L2 energy equality assertion over ℚ.
- `K = 10_000` step modular run: each fraction projected via `p·q^(P−2) mod P`
  with `P = 2^31−1` (local `mod_pow`), energy invariant checked in 𝔽ₚ so
  denominators never grow.
- `f64` baseline over the same gate sequence; drift reported as integer parts
  per million (no float used in pass/fail).
- FNV-1a digest of the final rational + modular state; nonzero exit on any
  exact invariant failure.

### `examples/erdos_lodri_e2e.rs`
- 16 vertices in four meso partitions `V_1..V_4` (4 each), K₄ intra-cluster
  mesh, pivot-halo bridges `(5,13) (6,13) (9,1) (10,14)`, shadow cut edges
  `(2,7) (3,12) (7,8)`.
- Build CSR (`offsets`, `targets`, `weights`) in the example.
- Stage 1: ternary micro contraction via `felix_simd::contract64_trit_i16`.
- Stage 2: Molloy–Reed `κ = Σk² / Σk` as exact `RationalPQ`; bipartite density
  of `V_2 × V_4` via `regularity_oracle::bipartite_density`.
- Stage 3: `apply_ternary_laplacian`; assert `Σ out == 0` exactly. Then the
  same Cayley gates as the benchmark (shared logic duplicated deliberately:
  examples must not import one another).
- Stage 4: `fold_super_ultra` → `unfold_super_ultra` fixed-point assertion,
  `UltraBackyardLoop::qulb_step` termination, seal/activate/consume capsule.
- Print six deterministic lines (micro dot, κ, density + shadow count, Hodge
  sum, energy equality + `P`, fold frame). FNV-1a digest printed last; assert
  `0` drift before any print.

## Review focus

- Repeated Cayley composition: denominator growth → `checked_mul` overflow must
  surface as a clear assertion, never a wrap or panic.
- `P = 2^31−1` projection with a denominator divisible by `P`: must be
  rejected/handled, not silently inverted.
- CSR orientation: `apply_ternary_laplacian` is one-sided (`u → v`), while the
  shadow graph is undirected; edges must be added in both directions where the
  invariant requires it, or conservation must be asserted on the oriented form.
- Shadow edges must remain visible in output after compression (the point of
  the exercise); assert the count.
- Example-only float use must not leak a `float_arithmetic` lint into library
  crates (`cargo clippy --workspace --all-targets -- -D warnings`).

## Left to the builder

Exact helper placement inside the two example files, local variable names, and
the exact formatting of the six output lines beyond the labels above.
