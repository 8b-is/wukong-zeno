#!/usr/bin/env bash
# wukong-zeno verification gate: exact p/q examples + library contracts.
set -euo pipefail
cd "$(dirname "$0")/.."

cargo test --workspace --doc
cargo check --workspace --lib
cargo clippy --workspace --all-targets -- -D warnings
cargo run --release --example omni_zeno_smoke
cargo run --release --example cayley_bench
cargo run --release --example erdos_lodri_e2e
cargo run --release --example fp31_bench
