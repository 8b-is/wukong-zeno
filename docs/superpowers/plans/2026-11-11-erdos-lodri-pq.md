# Erdős–Lódri UltraGraph `p/q` E2E Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add two self-checking examples that prove exact `p/q` Cayley invariance and end-to-end Erdős percolation / shadow-cut / kompress folding on the frozen `wukong-zeno` ABI.

**Architecture:** Two independent `std` example targets consume the five library crates through their public types and traits only. All new maths (Cayley gate, Mersenne projection, Molloy–Reed ratio, CSR assembly) lives inside the examples; no crate source is edited.

**Tech Stack:** Rust 2021, no third-party dependencies, `pivot-abi` / `felix-simd` / `mem8-wave` / `regularity-oracle` / `phoenix-capsule`.

**Spec:** `docs/superpowers/specs/2026-11-11-erdos-lodri-pq-design.md`

## Global Constraints

- Library crates: `#![no_std]`, allocation-free, no dependencies, no floats. Do not edit them.
- `pivot-abi` is frozen. Use `RationalPQ::from_ratio(num: i128, den: u128) -> Option<Self>` and the `checked_*` arithmetic; overflow is an error, never a panic or wrap.
- `KompressFrame32` is exactly 32 bytes; `CentroidPod64` exactly 128 bytes. Do not redefine them.
- Examples are the only `std` entry points. `f64` is permitted **only** in `examples/cayley_bench.rs`.
- Verification gate: `cargo test --workspace --doc`, `cargo clippy --workspace --all-targets -- -D warnings`, then `cargo run --release --example <name>`.
- Commit after each task. Never touch `examples/omni_zeno_smoke.rs` or its digest `0x5F745302BDDBB7DE`.

## Interfaces (frozen, use verbatim)

```rust
// pivot-abi
RationalPQ::from_ratio(num: i128, den: u128) -> Option<RationalPQ>
RationalPQ::ZERO, RationalPQ::ONE
RationalPQ::{checked_add, checked_sub, checked_mul, checked_div}(self, other) -> Option<RationalPQ>
TritPlane64 { pub pos: u64, pub neg: u64 }          // invariant pos & neg == 0
TritPlane64::set(&mut self, lane: usize, trit: Trit) -> Result<(), AbiError>
CentroidPod64(pub [i16; 64]); CentroidPod64::{set, max_abs, l1_abs_sum}

// felix-simd
contract64_trit_i16(weights: TritPlane64, act: &CentroidPod64) -> i32

// mem8-wave
fold_super_ultra(pod: &CentroidPod64, threshold: RationalPQ, epoch: u32, budget: u16) -> KompressFrame32
unfold_super_ultra(frame: &KompressFrame32) -> Result<CentroidPod64, FrameError>

// regularity-oracle
apply_ternary_laplacian(x: &[RationalPQ], offsets: &[u32], targets: &[u32],
                        weights: &[RationalPQ], out: &mut [RationalPQ]) -> Result<(), OracleError>
bipartite_density(edges: usize, left: usize, right: usize) -> Result<RationalPQ, OracleError>

// phoenix-capsule
Capsule::seal(payload, authority: RationalPQ, budget: u16) -> Capsule<T, Sealed>
Capsule::activate(self) -> Capsule<T, Active>
Capsule::attenuate(self, decay: RationalPQ) -> Result<Capsule<T, Attenuated>, AttenuateError<T>>
Capsule::consume(self) -> T
UltraBackyardLoop::new(state: TritPlane64, authority: RationalPQ, budget: u16)
UltraBackyardLoop::qulb_step<K: TernarySimdPort>(&mut self, weights: TritPlane64, decay: RationalPQ)
    -> Result<QulbStep, QulbError>

// wukong-zeno root
fnv1a64(bytes: &[u8]) -> u64
```

---

### Task 1: Exact Cayley vs float baseline benchmark

**Files:**
- Create: `examples/cayley_bench.rs`
- Test: the example is its own check — it `assert!`s every exact invariant and exits nonzero on failure.

**Interfaces:**
- Consumes: `pivot_abi::RationalPQ`, `wukong_zeno::fnv1a64`.
- Produces: nothing importable; prints deterministic lines consumed by the verification step.

- [ ] **Step 1: Write the example**

```rust
// examples/cayley_bench.rs — std + f64 allowed here only.
use pivot_abi::RationalPQ;
use wukong_zeno::fnv1a64;

const P: u64 = 2_147_483_647; // 2^31 - 1

fn cayley(r: i128, s: i128) -> (RationalPQ, RationalPQ) {
    let den = (s * s + r * r) as u128;
    (
        RationalPQ::from_ratio(s * s - r * r, den).expect("c"),
        RationalPQ::from_ratio(2 * r * s, den).expect("sigma"),
    )
}

fn gate(psi: &mut [RationalPQ; 16], u: usize, v: usize, r: i128, s: i128) {
    let (c, sigma) = cayley(r, s);
    let pu = psi[u];
    let pv = psi[v];
    psi[u] = c.checked_mul(pu).unwrap().checked_sub(sigma.checked_mul(pv).unwrap()).unwrap();
    psi[v] = sigma.checked_mul(pu).unwrap().checked_add(c.checked_mul(pv).unwrap()).unwrap();
}

fn energy(psi: &[RationalPQ; 16]) -> RationalPQ {
    psi.iter().fold(RationalPQ::ZERO, |acc, &x| {
        acc.checked_add(x.checked_mul(x).unwrap()).unwrap()
    })
}

fn mod_pow(mut b: u64, mut e: u64, m: u64) -> u64 { /* square-and-multiply, u128 intermediates */ todo!() }

fn to_mod_p(x: RationalPQ) -> u64 { /* p/q -> p * q^(P-2) mod P via mod_pow */ todo!() }

fn energy_mod_p(psi: &[RationalPQ; 16]) -> u64 {
    psi.iter().fold(0u64, |acc, &x| {
        let m = to_mod_p(x);
        (acc + (m as u128 * m as u128 % P as u128) as u64) % P
    })
}

fn main() {
    // 1. single-step exact energy equality
    // 2. K = 10_000 modular steps, energy invariant in F_P every step
    // 3. f64 baseline drift reported as integer ppm
    // 4. digest of exact + modular final state
    // assert! exact invariants; print lines; exit(1) on failure
}
```

Fill the four `todo!()` bodies and `main` so that:
- `gate` is applied to a seeded 16-lane state (`RationalPQ::from_ratio(raw, 8)`).
- Exact phase: copy the state, apply one gate, assert `energy(after) == energy(before)`.
- Modular phase: apply gates `(r,s) = (1,2), (2,3), (1,3), (3,4)` in a `for step in 0..10_000` loop; before each step record `energy_mod_p`, after each step assert equality. If a denominator is `0 mod P`, panic with a message naming the step (the spec forbids silent inversion).
- Float phase: same four gates with `f64` `c = (s*s - r*r) as f64 / den as f64`, `sigma = (2*r*s) as f64 / den as f64`; compute the same four rotations from an `f64` copy of the seed; report `drift_ppm = (|e0 - e1| * 1_000_000.0 / e0) as u64`.
- Print:
  `cayley exact  : energy preserved over 10000 steps, P=2147483647`
  `cayley float  : drift_ppm=<n>`
  `digest        : 0x<16 hex>`
- `std::process::exit(1)` if any exact assertion would fail (use `assert!` — non-zero exit is fine).

- [ ] **Step 2: Run it**

Run: `cargo run --release --example cayley_bench`
Expected: three lines as above, exit 0, `drift_ppm` prints a nonzero value for the float baseline.

- [ ] **Step 3: Lint**

Run: `cargo clippy --workspace --all-targets -- -D warnings`
Expected: clean (the float use is confined to the example target).

- [ ] **Step 4: Commit**

```bash
git add examples/cayley_bench.rs
git commit -m "feat(examples): exact p/q Cayley vs f64 drift benchmark"
```

---

### Task 2: Erdős–Lódri E2E graph example

**Files:**
- Create: `examples/erdos_lodri_e2e.rs`

**Interfaces:**
- Consumes: every signature in the Interfaces block, plus duplicate local `cayley`/`gate` helpers (examples never import one another).
- Produces: nothing importable.

- [ ] **Step 1: Write the example**

Build the graph described in the spec:
- vertices `0..16`, partitions `[0,1,2,3]`, `[4,5,6,7]`, `[8,9,10,11]`, `[12,13,14,15]`
- K₄ intra-partition edges, weight `3/4`
- pivot-halo edges `(5,13) (6,13) (9,1) (10,14)`, weight `1/1`
- shadow edges `(2,7) (3,12) (7,8)`, weight `1/3`
- CSR assembly must be one-sided `u -> v` per `apply_ternary_laplacian`; add both directions for undirected edges so conservation is asserted on the oriented form.

Then:
1. Micro: `TritPlane64` from the spec's 16-trit mask, `CentroidPod64` from the 16 raw values, `contract64_trit_i16`, print `micro`.
2. Percolation: `κ = Σk²/Σk` via `RationalPQ::from_ratio`; assert `κ > 2`; `bipartite_density(shadow_edges_between_V2_V4, 4, 4)`; count shadow edges.
3. Hodge: `apply_ternary_laplacian` on the seeded wave state; assert the sum is exactly `RationalPQ::ZERO`.
4. Unitary: four Cayley gates across pivot + shadow edges; assert exact energy equality before/after the chain.
5. Fold: `fold_super_ultra(&pod, threshold = 3/4, epoch = 1, budget = 14)`; unfold; assert refold fixed point; `UltraBackyardLoop` step until `fixed_point`, assert it terminates within `budget`; `Capsule::seal(pod, ONE, 14).activate().attenuate(1/2)?.consume()`.
6. Digest: `fnv1a64` over the printed numeric fields (little-endian bytes); print `digest : 0x<16 hex>`.

Print exactly these labels before the digest:
```
erdos-lodri e2e
micro         : <i32>
kappa         : <num>/<den> (supercritical=<bool>)
density V2xV4 : <num>/<den> shadow_edges=<n>
hodge sum     : 0/1
energy        : t0=<num>/<den> t1=<num>/<den> equal=true
fold          : active=<n> scale=<num>/<den> crc=0x<02X> fixed_point=true
digest        : 0x<16 hex>
```

- [ ] **Step 2: Run it**

Run: `cargo run --release --example erdos_lodri_e2e`
Expected: the seven lines above, exit 0. `hodge sum` must be `0/1`, `equal=true`, `fixed_point=true`.

- [ ] **Step 3: Confirm the canonical smoke still passes unchanged**

Run: `cargo run --release --example omni_zeno_smoke`
Expected: `digest : 0x5F745302BDDBB7DE` (unchanged).

- [ ] **Step 4: Full gate**

Run: `cargo test --workspace --doc && cargo clippy --workspace --all-targets -- -D warnings`
Expected: pass, clean.

- [ ] **Step 5: Commit**

```bash
git add examples/erdos_lodri_e2e.rs
git commit -m "feat(examples): Erdős–Lódri p/q E2E shadow-cut pipeline"
```

---

## Review Focus

Each spec risk gets its check in the owning task:

| Risk | Check |
|---|---|
| Denominator overflow on repeated Cayley | Task 1 modular loop asserts per step; Task 2 chain asserts once |
| Denominator `≡ 0 mod P` | Task 1 panics with step index; no silent inversion |
| CSR orientation vs undirected shadow graph | Task 2 adds both directions and asserts `Σ = 0` |
| Shadow edges vanish after compression | Task 2 prints and asserts `shadow_edges=3` |
| Float leaks into library crates | Task 1/2 clippy gate with `--all-targets -D warnings` |
