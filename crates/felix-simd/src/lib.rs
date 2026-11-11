#![no_std]
#![deny(clippy::float_arithmetic)]
#![deny(unsafe_op_in_unsafe_fn)]
#![deny(unsafe_code)]

//! `V_3`: balanced-ternary and integer SIMD kernels with safe scalar parity.

#[cfg(test)]
extern crate std;

pub mod arch;
pub mod scalar;

pub use arch::{contract64_trit_i16, dot64_i16, dot64_trit_plane, louds_rank1, splitmix64_x4};

use pivot_abi::{CentroidPod64, TernarySimdPort, TritPlane64};

/// Marker for the portable, fully safe scalar kernels.
#[derive(Clone, Copy, Debug, Default)]
pub struct ScalarKernels;

impl TernarySimdPort for ScalarKernels {
    fn dot64_trit_plane(a: TritPlane64, b: TritPlane64) -> i32 {
        scalar::dot64_trit_plane(a, b)
    }

    fn contract64_trit_i16(weights: TritPlane64, act: &CentroidPod64) -> i32 {
        scalar::contract64_trit_i16(weights, act)
    }

    fn dot64_i16(a: &CentroidPod64, b: &CentroidPod64) -> i32 {
        scalar::dot64_i16(a, b)
    }

    fn louds_rank1(bits: &[u64; 2], idx: usize) -> u32 {
        scalar::louds_rank1(bits, idx)
    }

    fn splitmix64_x4(state: &mut [u64; 4]) -> [u64; 4] {
        scalar::splitmix64_x4(state)
    }
}

/// Marker for the compile-time-dispatched architecture kernels.
#[derive(Clone, Copy, Debug, Default)]
pub struct ArchKernels;

impl TernarySimdPort for ArchKernels {
    fn dot64_trit_plane(a: TritPlane64, b: TritPlane64) -> i32 {
        arch::dot64_trit_plane(a, b)
    }

    fn contract64_trit_i16(weights: TritPlane64, act: &CentroidPod64) -> i32 {
        arch::contract64_trit_i16(weights, act)
    }

    fn dot64_i16(a: &CentroidPod64, b: &CentroidPod64) -> i32 {
        arch::dot64_i16(a, b)
    }

    fn louds_rank1(bits: &[u64; 2], idx: usize) -> u32 {
        arch::louds_rank1(bits, idx)
    }

    fn splitmix64_x4(state: &mut [u64; 4]) -> [u64; 4] {
        arch::splitmix64_x4(state)
    }
}

#[cfg(test)]
mod tests;
