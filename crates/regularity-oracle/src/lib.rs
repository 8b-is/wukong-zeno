#![no_std]
#![forbid(unsafe_code)]
#![deny(clippy::float_arithmetic)]

//! Exact-rational ternary Laplacian oracle (`L_3 = B W B^T`).
//!
//! The oracle consumes sparse directed graphs in compressed sparse row (CSR)
//! form: `offsets` has `vertices + 1` entries, `targets[e]` names the
//! destination of edge `e`, and `weights[e]` is its exact rational weight.
//! Every edge `e` in source vertex `u`'s CSR range is oriented `u -> v` and
//! contributes
//!
//! ```text
//! flux = w_e * (x[u] - x[v]); out[u] += flux; out[v] -= flux
//! ```
//!
//! so the accumulated vector conserves `sum(out) == 0` exactly. All rational
//! arithmetic is checked; overflow is reported as [`OracleError::Overflow`]
//! and the crate never panics. It also provides deterministic degree
//! partitioning, exact bipartite density, and pivot-vertex selection. The crate
//! is `no_std`, allocation-free, float-free, and contains no `unsafe`.

use pivot_abi::{LaplacianOraclePort, OracleError, RationalPQ};

#[cfg(test)]
extern crate std;

/// Stateless exact-rational ternary Laplacian.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct TernaryLaplacian;

impl LaplacianOraclePort for TernaryLaplacian {
    /// Apply `L_3 = B W B^T` to `x`, accumulating into a caller-zeroed `out`.
    ///
    /// See [`apply_ternary_laplacian`] for the validation rules and the
    /// orientation convention.
    fn apply_ternary_laplacian(
        x: &[RationalPQ],
        offsets: &[u32],
        targets: &[u32],
        weights: &[RationalPQ],
        out: &mut [RationalPQ],
    ) -> Result<(), OracleError> {
        crate::apply_ternary_laplacian(x, offsets, targets, weights, out)
    }
}

/// Apply `L_3 = B W B^T` to `x`, accumulating into a caller-zeroed `out`.
///
/// Vertices are `0..x.len()`. `offsets` must contain exactly `x.len() + 1`
/// monotone entries with `offsets[0] == 0` and `offsets[last] == targets.len()`,
/// `out.len()` must equal `x.len()`, `targets.len()` must equal
/// `weights.len()`, and every target must be a valid vertex index.
///
/// Each edge `e` in source `u`'s range `offsets[u]..offsets[u + 1]` targets
/// `v = targets[e]` and contributes `flux = weights[e] * (x[u] - x[v])` to
/// `out[u]` and `-flux` to `out[v]`. This orientation is one-sided: a directed
/// edge `u -> v` touches only `u` and `v`, and only in that direction.
///
/// All rational operations are checked. On an arithmetic overflow the function
/// returns [`OracleError::Overflow`] without panicking; `out` may then hold a
/// partially accumulated prefix. On success the exact conservation identity
/// `sum(out) == RationalPQ::ZERO` holds.
pub fn apply_ternary_laplacian(
    x: &[RationalPQ],
    offsets: &[u32],
    targets: &[u32],
    weights: &[RationalPQ],
    out: &mut [RationalPQ],
) -> Result<(), OracleError> {
    let vertices = x.len();
    if out.len() != vertices {
        return Err(OracleError::LengthMismatch);
    }
    if offsets.len() != vertices + 1 {
        return Err(OracleError::InvalidOffsets);
    }
    if targets.len() != weights.len() {
        return Err(OracleError::LengthMismatch);
    }
    validate_csr(offsets, targets.len())?;
    for &target in targets {
        if target as usize >= vertices {
            return Err(OracleError::TargetOutOfRange);
        }
    }
    for (source, window) in offsets.windows(2).enumerate() {
        let start = window[0] as usize;
        let end = window[1] as usize;
        for edge in start..end {
            let target = targets[edge] as usize;
            let difference = x[source]
                .checked_sub(x[target])
                .ok_or(OracleError::Overflow)?;
            let flux = weights[edge]
                .checked_mul(difference)
                .ok_or(OracleError::Overflow)?;
            out[source] = out[source].checked_add(flux).ok_or(OracleError::Overflow)?;
            out[target] = out[target].checked_sub(flux).ok_or(OracleError::Overflow)?;
        }
    }
    Ok(())
}

/// Validate strict CSR structure against an expected edge count.
fn validate_csr(offsets: &[u32], edge_count: usize) -> Result<(), OracleError> {
    let Some(&first) = offsets.first() else {
        return Err(OracleError::InvalidOffsets);
    };
    if first != 0 {
        return Err(OracleError::InvalidCsr);
    }
    let mut previous = first;
    for &offset in offsets.iter().skip(1) {
        if offset < previous {
            return Err(OracleError::InvalidCsr);
        }
        previous = offset;
    }
    match offsets.last() {
        Some(&last) if last as usize == edge_count => Ok(()),
        _ => Err(OracleError::InvalidCsr),
    }
}

/// Deterministically bucket vertices by normalized CSR degree.
///
/// `offsets` has `vertices + 1` entries and `out.len()` must equal `vertices`.
/// Bucket `out[u]` is `min(parts - 1, degree(u) * parts / (max_degree + 1))`,
/// evaluated with integer arithmetic, where `degree(u) = offsets[u + 1] -
/// offsets[u]` and `max_degree` is the maximum degree in the graph.
///
/// An empty graph (`offsets == [0]`) succeeds without writing. A zero `parts`
/// yields [`OracleError::EmptyPartition`]; inverted CSR ranges yield
/// [`OracleError::InvalidCsr`].
pub fn partition_by_degree(offsets: &[u32], out: &mut [u8], parts: u8) -> Result<(), OracleError> {
    let Some(vertices) = offsets.len().checked_sub(1) else {
        return Err(OracleError::InvalidOffsets);
    };
    if out.len() != vertices {
        return Err(OracleError::LengthMismatch);
    }
    if parts == 0 {
        return Err(OracleError::EmptyPartition);
    }
    if vertices == 0 {
        return Ok(());
    }
    let mut previous = offsets[0];
    for &offset in offsets.iter().skip(1) {
        if offset < previous {
            return Err(OracleError::InvalidCsr);
        }
        previous = offset;
    }
    let mut max_degree = 0u64;
    for window in offsets.windows(2) {
        let degree = u64::from(window[1] - window[0]);
        if degree > max_degree {
            max_degree = degree;
        }
    }
    let scale = max_degree + 1;
    let part_count = u64::from(parts);
    let last_bucket = u64::from(parts - 1);
    for (vertex, window) in offsets.windows(2).enumerate() {
        let degree = u64::from(window[1] - window[0]);
        let bucket = (degree * part_count / scale).min(last_bucket);
        out[vertex] = bucket as u8;
    }
    Ok(())
}

/// Exact bipartite edge density `edges / (left * right)`.
///
/// The product `left * right` is formed in `u128`; the result is reduced by
/// [`RationalPQ::from_ratio`]. A zero side count yields
/// [`OracleError::EmptyPartition`]. A product overflow, a numerator that does
/// not fit the ABI, or any other conversion failure yields
/// [`OracleError::Overflow`].
pub fn bipartite_density(
    edges: usize,
    left: usize,
    right: usize,
) -> Result<RationalPQ, OracleError> {
    if left == 0 || right == 0 {
        return Err(OracleError::EmptyPartition);
    }
    let product = (left as u128)
        .checked_mul(right as u128)
        .ok_or(OracleError::Overflow)?;
    let numerator = edges as u128;
    if numerator > i128::MAX as u128 {
        return Err(OracleError::Overflow);
    }
    RationalPQ::from_ratio(numerator as i128, product).ok_or(OracleError::Overflow)
}

/// Write the indices of vertices whose degree is at least the ceiling of the
/// mean degree, in ascending order.
///
/// Returns the number of indices written. If `out` is shorter than the number
/// of qualifying vertices, only the first `out.len()` indices are written and
/// that smaller count is returned. Empty, monotone-violating, or malformed
/// offset arrays yield `0` without writing. The function allocates nothing.
pub fn pivot_indices(offsets: &[u32], out: &mut [u32]) -> usize {
    if offsets.len() < 2 {
        return 0;
    }
    let vertices = offsets.len() - 1;
    let mut previous = offsets[0];
    for &offset in offsets.iter().skip(1) {
        if offset < previous {
            return 0;
        }
        previous = offset;
    }
    let total = u64::from(offsets[vertices] - offsets[0]);
    let vertex_count = vertices as u64;
    if vertex_count == 0 {
        return 0;
    }
    let threshold = total.div_ceil(vertex_count);
    let mut written = 0usize;
    for (vertex, window) in offsets.windows(2).enumerate() {
        let degree = u64::from(window[1] - window[0]);
        if degree >= threshold {
            let Ok(index) = u32::try_from(vertex) else {
                break;
            };
            if written >= out.len() {
                break;
            }
            out[written] = index;
            written += 1;
        }
    }
    written
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::vec;
    use std::vec::Vec;

    fn rat(num: i64, den: u64) -> RationalPQ {
        RationalPQ::new(num, den).expect("valid rational")
    }

    fn zeros(count: usize) -> Vec<RationalPQ> {
        vec![RationalPQ::ZERO; count]
    }

    fn sum(values: &[RationalPQ]) -> RationalPQ {
        let mut total = RationalPQ::ZERO;
        for &value in values {
            total = total.checked_add(value).expect("conservation sum");
        }
        total
    }

    #[test]
    fn conserves_on_directed_triangle() {
        let offsets = vec![0u32, 1, 2, 3];
        let targets = vec![1u32, 2, 0];
        let weights = vec![rat(1, 2), rat(2, 3), rat(3, 2)];
        let samples = [
            vec![rat(1, 1), rat(0, 1), rat(0, 1)],
            vec![rat(0, 1), rat(5, 1), rat(-2, 1)],
            vec![rat(7, 3), rat(-1, 2), rat(11, 6)],
            vec![rat(-9, 4), rat(3, 8), rat(0, 1)],
        ];
        for x in samples {
            let mut out = zeros(x.len());
            apply_ternary_laplacian(&x, &offsets, &targets, &weights, &mut out).unwrap();
            assert_eq!(sum(&out), RationalPQ::ZERO);
        }
    }

    #[test]
    fn conserves_on_four_partition_graph() {
        let offsets = vec![0u32, 2, 4, 6, 8];
        let targets = vec![1u32, 2, 2, 3, 0, 3, 0, 1];
        let weights = vec![
            rat(1, 2),
            rat(2, 3),
            rat(3, 2),
            rat(5, 4),
            rat(1, 2),
            rat(2, 3),
            rat(3, 2),
            rat(5, 4),
        ];
        let samples = [
            vec![rat(1, 1), rat(2, 1), rat(3, 1), rat(4, 1)],
            vec![rat(0, 1), rat(0, 1), rat(0, 1), rat(0, 1)],
            vec![rat(-5, 2), rat(7, 3), rat(0, 1), rat(9, 5)],
            vec![rat(1, 7), rat(-2, 9), rat(13, 11), rat(-4, 13)],
        ];
        for x in samples {
            let mut out = zeros(x.len());
            apply_ternary_laplacian(&x, &offsets, &targets, &weights, &mut out).unwrap();
            assert_eq!(sum(&out), RationalPQ::ZERO);
        }
    }

    #[test]
    fn path_action_matches_hand_computation() {
        let offsets = vec![0u32, 1, 3, 4];
        let targets = vec![1u32, 0, 2, 1];
        let weight = rat(5, 4);
        let weights = vec![weight, weight, weight, weight];
        let x = vec![rat(2, 1), rat(0, 1), rat(0, 1)];
        let mut out = zeros(3);
        apply_ternary_laplacian(&x, &offsets, &targets, &weights, &mut out).unwrap();
        assert_eq!(out, vec![rat(5, 1), rat(-5, 1), RationalPQ::ZERO]);
    }

    #[test]
    fn directed_edge_is_one_sided() {
        let offsets = vec![0u32, 1, 1, 1];
        let targets = vec![1u32];
        let weights = vec![rat(2, 1)];
        let x = vec![rat(3, 1), rat(1, 1), rat(0, 1)];
        let mut out = zeros(3);
        apply_ternary_laplacian(&x, &offsets, &targets, &weights, &mut out).unwrap();
        assert_eq!(out, vec![rat(4, 1), rat(-4, 1), RationalPQ::ZERO]);

        let reversed_offsets = vec![0u32, 0, 1, 1];
        let reversed_targets = vec![0u32];
        let mut reversed = zeros(3);
        apply_ternary_laplacian(
            &x,
            &reversed_offsets,
            &reversed_targets,
            &weights,
            &mut reversed,
        )
        .unwrap();
        assert_eq!(reversed, vec![rat(4, 1), rat(-4, 1), RationalPQ::ZERO]);
    }

    #[test]
    fn rejects_invalid_offsets_length() {
        let x = vec![rat(1, 1), rat(2, 1), rat(3, 1)];
        let offsets = vec![0u32, 1, 2];
        let targets = vec![1u32, 2];
        let weights = vec![rat(1, 1), rat(1, 1)];
        let mut out = zeros(3);
        assert_eq!(
            apply_ternary_laplacian(&x, &offsets, &targets, &weights, &mut out),
            Err(OracleError::InvalidOffsets)
        );
    }

    #[test]
    fn rejects_length_mismatches() {
        let x = vec![rat(1, 1), rat(2, 1), rat(3, 1)];
        let offsets = vec![0u32, 1, 1, 1];
        let targets = vec![1u32];
        let weights: Vec<RationalPQ> = Vec::new();
        let mut out = zeros(3);
        assert_eq!(
            apply_ternary_laplacian(&x, &offsets, &targets, &weights, &mut out),
            Err(OracleError::LengthMismatch)
        );

        let weights = vec![rat(1, 1)];
        let mut short_out = zeros(2);
        assert_eq!(
            apply_ternary_laplacian(&x, &offsets, &targets, &weights, &mut short_out),
            Err(OracleError::LengthMismatch)
        );
    }

    #[test]
    fn rejects_out_of_range_target() {
        let x = vec![rat(1, 1), rat(2, 1), rat(3, 1)];
        let offsets = vec![0u32, 1, 1, 1];
        let targets = vec![3u32];
        let weights = vec![rat(1, 1)];
        let mut out = zeros(3);
        assert_eq!(
            apply_ternary_laplacian(&x, &offsets, &targets, &weights, &mut out),
            Err(OracleError::TargetOutOfRange)
        );
    }

    #[test]
    fn rejects_inverted_and_non_canonical_csr() {
        let x = vec![rat(1, 1), rat(2, 1), rat(3, 1)];
        let weights = vec![rat(1, 1)];
        let targets = vec![1u32];
        let mut out = zeros(3);
        let inverted = vec![0u32, 2, 1, 1];
        assert_eq!(
            apply_ternary_laplacian(&x, &inverted, &targets, &weights, &mut out),
            Err(OracleError::InvalidCsr)
        );
        let trailing = vec![0u32, 1, 1, 2];
        assert_eq!(
            apply_ternary_laplacian(&x, &trailing, &targets, &weights, &mut out),
            Err(OracleError::InvalidCsr)
        );
        let base = vec![1u32, 1, 1, 1];
        assert_eq!(
            apply_ternary_laplacian(&x, &base, &targets, &weights, &mut out),
            Err(OracleError::InvalidCsr)
        );
    }

    #[test]
    fn overflow_in_multiplication_is_reported() {
        let huge = rat(i64::MAX, 1);
        let x = vec![huge, RationalPQ::ZERO];
        let offsets = vec![0u32, 1, 1];
        let targets = vec![1u32];
        let weights = vec![huge];
        let mut out = zeros(2);
        assert_eq!(
            apply_ternary_laplacian(&x, &offsets, &targets, &weights, &mut out),
            Err(OracleError::Overflow)
        );
    }

    #[test]
    fn overflow_in_accumulation_is_reported() {
        let huge = rat(i64::MAX, 1);
        let x = vec![RationalPQ::ONE, RationalPQ::ZERO];
        let offsets = vec![0u32, 2, 2];
        let targets = vec![1u32, 1];
        let weights = vec![huge, huge];
        let mut out = zeros(2);
        assert_eq!(
            apply_ternary_laplacian(&x, &offsets, &targets, &weights, &mut out),
            Err(OracleError::Overflow)
        );
    }

    #[test]
    fn trait_port_matches_free_function() {
        let x = vec![rat(1, 1), rat(0, 1), rat(-3, 2)];
        let offsets = vec![0u32, 1, 2, 3];
        let targets = vec![1u32, 2, 0];
        let weights = vec![rat(1, 2), rat(2, 3), rat(3, 2)];
        let mut direct = zeros(3);
        let mut via_port = zeros(3);
        apply_ternary_laplacian(&x, &offsets, &targets, &weights, &mut direct).unwrap();
        TernaryLaplacian::apply_ternary_laplacian(&x, &offsets, &targets, &weights, &mut via_port)
            .unwrap();
        assert_eq!(direct, via_port);
    }

    #[test]
    fn degree_partition_uses_exact_buckets() {
        let offsets = vec![0u32, 2, 3, 3, 5];
        let mut out = vec![0u8; 4];
        partition_by_degree(&offsets, &mut out, 3).unwrap();
        assert_eq!(out, vec![2u8, 1, 0, 2]);

        partition_by_degree(&offsets, &mut out, 2).unwrap();
        assert_eq!(out, vec![1u8, 0, 0, 1]);

        partition_by_degree(&offsets, &mut out, 1).unwrap();
        assert_eq!(out, vec![0u8, 0, 0, 0]);
    }

    #[test]
    fn degree_partition_is_deterministic() {
        let offsets = vec![0u32, 1, 3, 6, 6, 7];
        let mut first = vec![0u8; 5];
        let mut second = vec![0u8; 5];
        partition_by_degree(&offsets, &mut first, 4).unwrap();
        partition_by_degree(&offsets, &mut second, 4).unwrap();
        assert_eq!(first, second);
        let mut expected = Vec::new();
        let max_degree = 3u64;
        for window in offsets.windows(2) {
            let degree = u64::from(window[1] - window[0]);
            expected.push((degree * 4 / (max_degree + 1)).min(3) as u8);
        }
        assert_eq!(first, expected);
    }

    #[test]
    fn degree_partition_rejects_bad_inputs() {
        let offsets = vec![0u32, 2, 2, 3];
        let mut out = vec![0u8; 3];
        assert_eq!(
            partition_by_degree(&offsets, &mut out, 0),
            Err(OracleError::EmptyPartition)
        );
        let mut short = vec![0u8; 2];
        assert_eq!(
            partition_by_degree(&offsets, &mut short, 2),
            Err(OracleError::LengthMismatch)
        );
        let inverted = vec![0u32, 2, 1, 3];
        assert_eq!(
            partition_by_degree(&inverted, &mut out, 2),
            Err(OracleError::InvalidCsr)
        );
        assert_eq!(
            partition_by_degree(&[], &mut [], 2),
            Err(OracleError::InvalidOffsets)
        );
    }

    #[test]
    fn degree_partition_accepts_empty_graph() {
        let mut empty: [u8; 0] = [];
        assert_eq!(partition_by_degree(&[0u32], &mut empty, 3), Ok(()));
    }

    #[test]
    fn bipartite_density_reduces_exactly() {
        let density = bipartite_density(3, 4, 6).unwrap();
        assert_eq!(density.num(), 1);
        assert_eq!(density.den(), 8);
        assert_eq!(density, rat(1, 8));

        assert_eq!(bipartite_density(0, 4, 6).unwrap(), RationalPQ::ZERO);
        assert_eq!(bipartite_density(24, 4, 6).unwrap(), RationalPQ::ONE);
        assert_eq!(bipartite_density(5, 1, 2).unwrap(), rat(5, 2));
    }

    #[test]
    fn bipartite_density_rejects_empty_sides() {
        assert_eq!(bipartite_density(3, 0, 6), Err(OracleError::EmptyPartition));
        assert_eq!(bipartite_density(3, 4, 0), Err(OracleError::EmptyPartition));
    }

    #[cfg(target_pointer_width = "64")]
    #[test]
    fn bipartite_density_reports_unrepresentable_numerator() {
        assert_eq!(
            bipartite_density(usize::MAX, 1, 1),
            Err(OracleError::Overflow)
        );
    }

    #[test]
    fn pivot_indices_selects_at_or_above_mean_ceiling() {
        let offsets = vec![0u32, 2, 3, 3];
        let mut out = vec![0u32; 3];
        assert_eq!(pivot_indices(&offsets, &mut out), 2);
        assert_eq!(out, vec![0u32, 1, 0]);
    }

    #[test]
    fn pivot_indices_handles_fractional_mean_ceiling() {
        let offsets = vec![0u32, 1, 1, 1, 2];
        let mut out = vec![0u32; 4];
        assert_eq!(pivot_indices(&offsets, &mut out), 2);
        assert_eq!(out, vec![0u32, 3, 0, 0]);
    }

    #[test]
    fn pivot_indices_stops_when_output_is_small() {
        let offsets = vec![0u32, 2, 3, 3];
        let mut small = vec![0u32; 1];
        assert_eq!(pivot_indices(&offsets, &mut small), 1);
        assert_eq!(small, vec![0u32]);

        let mut empty: [u32; 0] = [];
        assert_eq!(pivot_indices(&offsets, &mut empty), 0);
    }

    #[test]
    fn pivot_indices_handles_degenerate_offsets() {
        let mut out = vec![0u32; 2];
        assert_eq!(pivot_indices(&[], &mut out), 0);
        assert_eq!(pivot_indices(&[0u32], &mut out), 0);
        assert_eq!(pivot_indices(&[0u32, 2, 1], &mut out), 0);
        assert_eq!(pivot_indices(&[0u32, 0, 0], &mut out), 2);
        assert_eq!(out, vec![0u32, 1]);
    }
}
