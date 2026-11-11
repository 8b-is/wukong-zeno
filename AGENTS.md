# wukong-zeno v0.2.0 — workspace contract

OS-agnostic, dependency-free `#![no_std]` workspace. Five crates, one frozen ABI,
zero circular dependencies.

## Crates

| Crate | Owner lane | Depends on |
|---|---|---|
| `crates/pivot-abi` | frozen central mediator | nothing |
| `crates/felix-simd` | `V_3` kernels | `pivot-abi` |
| `crates/mem8-wave` | `V_1` + kompress-super-ultra | `pivot-abi` |
| `crates/regularity-oracle` | `V_2` Laplacian oracle | `pivot-abi` |
| `crates/phoenix-capsule` | `V_4` qULB capsule | `pivot-abi` |

Siblings never import each other. The root package `wukong-zeno` exists only to
host the `examples/*.rs` targets (`omni_zeno_smoke`, `cayley_bench`,
`erdos_lodri_e2e`, `fp31_bench`) and may use `std`.

## Hard invariants

- No `std`/`alloc`/`libc`/OS clocks/env in library crates. `#![no_std]` at every
  library root. Unit tests may add `#[cfg(test)] extern crate std;`.
- No IEEE-754 floats. Every crate root carries
  `#![deny(clippy::float_arithmetic)]`. All ratios are `RationalPQ` /
  `CompactPQ`; all markers are `Trit`.
- `pivot-abi`, `mem8-wave`, `regularity-oracle`, `phoenix-capsule` are
  `#![forbid(unsafe_code)]`. Only `felix-simd/src/arch/` may contain `unsafe`,
  each block with a `// SAFETY:` comment and a parity test against
  `felix-simd::scalar`.
- Wire bytes are explicit little-endian. SIMD selection is compile-time `cfg`
  only (no runtime `is_x86_feature_detected!`, no `getauxval`).
- Capability state is linear: `Sealed -> Active -> Attenuated -> Consumed`,
  never `Copy`/`Clone`, double-consume must fail to compile (`compile_fail`
  doctest in `phoenix-capsule`).

## ABI contract (`pivot-abi`)

- `RationalPQ { num: i64, den: NonZeroU64 }`, `CompactPQ { num: i32, den: NonZeroU32 }`.
  Constructors reduce with gcd; comparison/equality uses i128 cross-multiplication.
  Checked arithmetic returns `Option`, never panics on overflow or division by zero.
- `Trit { Neg = -1, Zero = 0, Pos = 1 }`, `TritPlane64 { pos: u64, neg: u64 }`
  with invariant `pos & neg == 0`.
- `PackedTrit5(u8)`: 5 trits per byte, `3^5 = 243`; values 243..=255 are invalid.
- `CentroidPod64([i16; 64])`: 128 bytes, align 64.
- `KompressFrame32`: exactly 32 bytes / align 32:
  `scale_num i32 | scale_den NonZeroU32 | epoch_tag u32 | insert4_hdr [u8;4] |
  qulb_budget u16 | active_lanes u8 | packed_trits [u8;13]`.
  `insert4_hdr = [b'K', b'S', b'U', crc8]`.
- Port traits: `TernarySimdPort` (static kernels), `KompressPort`,
  `LaplacianOraclePort`, `QulbCapsulePort`.

## kompress-super-ultra fold contract (`mem8-wave`)

`fold_super_ultra(&CentroidPod64, threshold: RationalPQ, epoch: u32, budget: u16)
-> KompressFrame32` runs exactly five stages:

1. `sphere_pop` — zero lanes with `|v| < threshold * max_abs`; count nonzero
   survivors. Zero lanes are never active, so a zero pod has `active_lanes == 0`
   and `threshold <= 0` keeps every nonzero lane. `active_lanes` counts threshold
   survivors, not decoded nonzero trits: a survivor below `max_abs / 3` may
   still quantize to `Trit::Zero`.
2. `cross_laminate` — reversible even/odd interleave (`new[2k]=old[k]`,
   `new[2k+1]=old[32+k]`).
3. `die_mold_1` — ternary quantize against `max_abs / 3`; `scale` is the exact
   reduced L1 mean magnitude of active (nonzero surviving) lanes (`CompactPQ`).
4. `insert_4` — CRC-8 poly `0x07` over the 32 frame bytes with the CRC byte
   zeroed; write `[b'K', b'S', b'U', crc8]`.
5. `die_mold_2` — pack 64 trits + 1 zero pad trit into `[u8; 13]` (`PackedTrit5`).

`unfold_super_ultra` validates magic, CRC, pad trit, packed digits, active lane
count, then reconstructs `trit * scale` with deterministic integer rounding and
applies the inverse laminate. Because rounding is integer, the *second* fold
generation is a fixed point (`fold(unfold(fold(unfold(fold(x))))) == fold(unfold(fold(x)))`);
use that as the round-trip test. Any single-byte tamper must yield
`FrameError::CrcMismatch` (or a stricter structural error).

## Laplacian contract (`regularity-oracle`)

CSR arrays `offsets: &[u32]` (`n + 1`), `targets: &[u32]` (`m`), `weights:
&[RationalPQ]` (`m`). Each edge `e` in `u`'s range is oriented `u -> v`:

```
flux = w_e * (x[u] - x[v]); out[u] += flux; out[v] -= flux
```

Exact conservation: `sum(out) == 0` with zero drift. All arithmetic checked;
overflow is `OracleError`, never a panic.

## qULB contract (`phoenix-capsule`)

`UltraBackyardLoop` holds `TritPlane64`, `RationalPQ` authority, `u16` budget.
Each step contracts with the kernel port, applies the monotone ternary mask
(`pos &= weights.pos`, `neg &= weights.neg`), strictly decrements the budget and
attenuates authority by `decay` (`0 < decay < 1`). It terminates on a ternary
fixed point or budget/authority exhaustion. Authority must be strictly positive;
zero or negative authority is `QulbError::Exhausted`.

`Capsule::attenuate` is fallible without breaking linearity: a failed
attenuation returns `AttenuateError<T>` carrying the original `Active` capsule,
so the payload is never silently dropped.

## Verification gate

```sh
cargo test --workspace --doc
cargo clippy --workspace --all-targets -- -D warnings
cargo check --workspace --lib
cargo run --release --example omni_zeno_smoke
```

Optional cross-target checks when installed:

```sh
cargo check -p pivot-abi -p felix-simd -p mem8-wave -p regularity-oracle -p phoenix-capsule --target wasm32-unknown-unknown
cargo check -p pivot-abi -p felix-simd -p mem8-wave -p regularity-oracle -p phoenix-capsule --target aarch64-unknown-none
```

Lane rule: edit only your crate directory, run only `cargo test -p <crate>`
during development, never add third-party dependencies.

## Version control

This repository is a colocated Jujutsu repo. Use `jj` for status, commits, and
pushes (`jj status`, `jj commit`, `jj git push`); the underlying git repo stays
intact. Remotes: `origin` = GitHub `8b-is/wukong-zeno`, `codeberg` =
Codeberg `8b-is/wukong-zeno`.
