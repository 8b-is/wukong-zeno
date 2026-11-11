# wukong-zeno

OS-agnostic, `no_std`, dependency-free Rust workspace for exact `p/q`
balanced-ternary computation:

- `pivot-abi` - frozen central mediator ABI (rationals, trits, pods, frames, ports).
- `felix-simd` - NEON / SSE2 kernels with safe scalar parity.
- `mem8-wave` - kompress-super-ultra 128 B -> 32 B fold with CRC-8 frames.
- `regularity-oracle` - exact rational ternary Laplacian and partitioning.
- `phoenix-capsule` - qULB typestate capability capsules.

The root `wukong-zeno` package contains only allocation-free helpers plus the
`omni_zeno_smoke` example, which is the single place where `std` is allowed.

## Contracts

- Library crates are `#![no_std]`, allocation-free, and have no third-party
  dependencies.
- No floating point: `#![deny(clippy::float_arithmetic)]` in every crate.
- `unsafe` is confined to `felix-simd/src/arch/`; every other crate is
  `#![forbid(unsafe_code)]`.
- All wire values are little-endian; `KompressFrame32` is exactly 32 bytes and
  `CentroidPod64` is exactly 128 bytes.
- Arithmetic is checked and deterministic; there are no clocks, environment
  reads, or OS calls in the libraries.

## Layout

```text
wukong-zeno/
  crates/pivot-abi/          frozen ABI and port traits
  crates/felix-simd/         scalar + aarch64 NEON + x86_64 SSE2 kernels
  crates/mem8-wave/          fold/unfold, CRC-8, packed trits
  crates/regularity-oracle/  Laplacian, degree partitions, density, pivots
  crates/phoenix-capsule/    Sealed -> Active -> Attenuated -> Consumed
  examples/omni_zeno_smoke.rs
```

## Build and verify

```sh
cargo test --workspace
cargo test --workspace --doc
cargo clippy --workspace --all-targets -- -D warnings
cargo run --release --example omni_zeno_smoke

# Cross-target compilation (no std available on the bare-metal target)
cargo check --workspace --lib --target wasm32-unknown-unknown
cargo check --workspace --lib --target x86_64-apple-darwin
cargo check --workspace --lib --target aarch64-unknown-none
```

Expected smoke output:

```text
wukong-zeno v0.2.0 - omni zeno smoke
kernel parity : 1000/1000 vectors, 0 mismatches
kompress      : 128B -> 32B -> 128B, active=57 scale=163774/19 crc=0xC2
               refold=fixed-point tamper=CRC-mismatch unfolded_l1=370660
oracle        : Delta_H3=0 partitions=[0:2 1:1 2:2 3:1 4:1 5:1 6:1 7:1] pivots=2 density=5/8
qulb          : fixed_point=true authority=1/3 budget=14 energy=32 capsule=0x8B15
digest        : 0x5F745302BDDBB7DE
```

The digest is an FNV-1a 64-bit fingerprint of the frame, Laplacian result, and
qULB state; it must stay stable across hosts and architectures.

## License

MIT OR Apache-2.0.
