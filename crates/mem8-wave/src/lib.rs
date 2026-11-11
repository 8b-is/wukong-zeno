#![no_std]
#![forbid(unsafe_code)]
#![deny(clippy::float_arithmetic)]

//! Deterministic `kompress-super-ultra` folding for the `wukong-zeno`
//! workspace.
//!
//! [`fold_super_ultra`] runs five integer-only stages over a 128-byte
//! [`CentroidPod64`] and produces a 32-byte [`KompressFrame32`]:
//!
//! 1. `sphere_pop` — zero lanes below `threshold * max_abs` and count the
//!    nonzero survivors. Zero lanes are never active, so a zero pod has no
//!    active lanes and `threshold <= 0` keeps every nonzero lane.
//! 2. `cross_laminate` — reversible even/odd interleave.
//! 3. `die_mold_1` — balanced-ternary quantization against `max_abs / 3` plus
//!    the exact reduced L1 mean magnitude of the active (nonzero surviving)
//!    lanes.
//! 4. `insert_4` — little-endian serialization, CRC-8 seal, `KSU` magic.
//! 5. `die_mold_2` — pack 64 trits plus one zero pad trit into 13 bytes.
//!
//! No `std`, no allocation, no floating point, and no `unsafe`.

#[cfg(test)]
extern crate std;

use pivot_abi::{
    CentroidPod64, CompactPQ, FrameError, KompressFrame32, KompressPort, PackedTrit5, RationalPQ,
    Trit, CRC8_POLY, INSERT4_MAGIC,
};

/// Offset of the CRC byte inside the 32-byte wire layout (`insert4_hdr[3]`).
const CRC_OFFSET: usize = 15;

/// CRC-8 with polynomial `0x07`, initial value `0`, no reflection, and no
/// final XOR.
///
/// This is the seal used by the `insert_4` stage. The CRC is always computed
/// over the 32 frame bytes with the CRC byte itself (offset 15) held at zero.
#[must_use]
pub fn crc8(bytes: &[u8]) -> u8 {
    let mut crc = 0u8;
    for &byte in bytes {
        crc ^= byte;
        let mut bit = 0u8;
        while bit < 8 {
            crc = if crc & 0x80 != 0 {
                (crc << 1) ^ CRC8_POLY
            } else {
                crc << 1
            };
            bit += 1;
        }
    }
    crc
}

/// Serialize the frame fields into their explicit little-endian wire layout.
///
/// `scale_num` 0..4, `scale_den` 4..8, `epoch_tag` 8..12, `insert4_hdr`
/// 12..16, `qulb_budget` 16..18, `active_lanes` 18, `packed_trits` 19..32.
fn serialize_frame(frame: &KompressFrame32) -> [u8; 32] {
    let mut bytes = [0u8; 32];
    bytes[0..4].copy_from_slice(&frame.scale_num.to_le_bytes());
    bytes[4..8].copy_from_slice(&frame.scale_den.get().to_le_bytes());
    bytes[8..12].copy_from_slice(&frame.epoch_tag.to_le_bytes());
    bytes[12..16].copy_from_slice(&frame.insert4_hdr);
    bytes[16..18].copy_from_slice(&frame.qulb_budget.to_le_bytes());
    bytes[18] = frame.active_lanes;
    bytes[19..32].copy_from_slice(&frame.packed_trits);
    bytes
}

/// CRC of the typed frame fields, with the CRC byte held at zero.
fn frame_crc(frame: &KompressFrame32) -> u8 {
    let mut bytes = serialize_frame(frame);
    bytes[CRC_OFFSET] = 0;
    crc8(&bytes)
}

/// Stage 1: zero lanes with `|v| < threshold * max_abs`, counting nonzero
/// survivors.
///
/// The comparison is exact and happens in `i128`:
/// `|v| * threshold.den < max_abs * threshold.num`. Zero lanes are never
/// active: they are already zero and do not participate in the scale mean.
/// Consequently a zero pod has `active == 0`, and `threshold <= 0` keeps
/// every nonzero lane. Counting only nonzero lanes is what makes the second
/// fold generation an exact fixed point for every threshold.
fn sphere_pop(pod: &CentroidPod64, threshold: RationalPQ) -> (CentroidPod64, u8) {
    let max_abs = pod.max_abs();
    let mut cleaned = *pod;
    if max_abs == 0 {
        return (cleaned, 0);
    }
    let den = threshold.den.get() as i128;
    let bound = i128::from(max_abs) * i128::from(threshold.num);
    let mut active = 0u8;
    for lane in &mut cleaned.0 {
        if *lane == 0 {
            continue;
        }
        let value = i64::from(*lane);
        let magnitude = i128::from(if value < 0 { -value } else { value });
        if magnitude * den < bound {
            *lane = 0;
        } else {
            active += 1;
        }
    }
    (cleaned, active)
}

/// Stage 2: reversible even/odd interleave, `new[2k] = old[k]`,
/// `new[2k+1] = old[32+k]`.
fn cross_laminate(pod: &CentroidPod64) -> [i16; 64] {
    let mut interleaved = [0i16; 64];
    let mut k = 0usize;
    while k < 32 {
        interleaved[2 * k] = pod.0[k];
        interleaved[2 * k + 1] = pod.0[32 + k];
        k += 1;
    }
    interleaved
}

/// Inverse of [`cross_laminate`]: `old[k] = interleaved[2k]`,
/// `old[32+k] = interleaved[2k+1]`.
fn inverse_cross_laminate(interleaved: &[i16; 64]) -> [i16; 64] {
    let mut pod = [0i16; 64];
    let mut k = 0usize;
    while k < 32 {
        pod[k] = interleaved[2 * k];
        pod[32 + k] = interleaved[2 * k + 1];
        k += 1;
    }
    pod
}

/// Stage 3 quantization against `max_abs / 3`, exactly as integer comparisons:
/// `v*3 > max_abs` is positive, `v*3 < -max_abs` is negative, else zero.
fn quantize_lane(value: i16, max_abs: i64) -> Trit {
    let scaled = i64::from(value) * 3;
    if scaled > max_abs {
        Trit::Pos
    } else if scaled < -max_abs {
        Trit::Neg
    } else {
        Trit::Zero
    }
}

/// Exact reduced L1 mean magnitude over the lanes that survived `sphere_pop`.
fn survivor_scale(pod: &CentroidPod64, active_lanes: u8) -> CompactPQ {
    if active_lanes == 0 {
        return CompactPQ::ZERO;
    }
    let Ok(num) = i32::try_from(pod.l1_abs_sum()) else {
        return CompactPQ::ZERO;
    };
    CompactPQ::new(num, u32::from(active_lanes)).unwrap_or(CompactPQ::ZERO)
}

/// Stage 5: pack 64 trits plus one zero pad trit into 13 base-3 bytes.
fn pack_trits(trits: &[Trit; 64]) -> [u8; 13] {
    let mut packed = [0u8; 13];
    let mut group = 0usize;
    while group < 12 {
        let base = group * 5;
        packed[group] = PackedTrit5::pack5([
            trits[base],
            trits[base + 1],
            trits[base + 2],
            trits[base + 3],
            trits[base + 4],
        ])
        .0;
        group += 1;
    }
    packed[12] = PackedTrit5::pack5([trits[60], trits[61], trits[62], trits[63], Trit::Zero]).0;
    packed
}

/// Unpack all 65 trits (64 lanes plus pad) from the 13 payload bytes.
fn unpack_trits(packed: &[u8; 13]) -> [Trit; 65] {
    let mut trits = [Trit::Zero; 65];
    let mut group = 0usize;
    while group < 13 {
        let digits = PackedTrit5(packed[group]).unpack5();
        let base = group * 5;
        let mut digit = 0usize;
        while digit < 5 {
            trits[base + digit] = digits[digit];
            digit += 1;
        }
        group += 1;
    }
    trits
}

/// Reconstruct one lane as `round(trit * scale_num / scale_den)`.
///
/// Rounding is half away from zero, done in `i128`, and the result saturates
/// to the `i16` range.
fn scale_lane(trit: Trit, scale_num: i128, scale_den: i128) -> i16 {
    let numerator = i128::from(trit.to_i8()) * scale_num;
    let magnitude = if numerator < 0 { -numerator } else { numerator };
    let rounded = (magnitude + scale_den / 2) / scale_den;
    let signed = if numerator < 0 { -rounded } else { rounded };
    if signed > i128::from(i16::MAX) {
        i16::MAX
    } else if signed < i128::from(i16::MIN) {
        i16::MIN
    } else {
        signed as i16
    }
}

/// Fold a centroid pod into a 32-byte kompress frame.
///
/// The five stages run in contract order and are fully deterministic: no
/// clocks, no environment, no allocation, and no floating point.
#[must_use]
pub fn fold_super_ultra(
    pod: &CentroidPod64,
    threshold: RationalPQ,
    epoch: u32,
    budget: u16,
) -> KompressFrame32 {
    let max_abs = pod.max_abs();
    let (cleaned, active_lanes) = sphere_pop(pod, threshold);
    let interleaved = cross_laminate(&cleaned);

    let mut trits = [Trit::Zero; 64];
    for (trit, value) in trits.iter_mut().zip(interleaved.iter()) {
        *trit = quantize_lane(*value, max_abs);
    }

    let scale = survivor_scale(&cleaned, active_lanes);
    let mut frame = KompressFrame32 {
        scale_num: scale.num,
        scale_den: scale.den,
        epoch_tag: epoch,
        insert4_hdr: [INSERT4_MAGIC[0], INSERT4_MAGIC[1], INSERT4_MAGIC[2], 0],
        qulb_budget: budget,
        active_lanes,
        packed_trits: pack_trits(&trits),
    };
    frame.insert4_hdr[3] = frame_crc(&frame);
    frame
}

/// Validate and unfold a kompress frame back into an approximate pod.
///
/// Validation order is magic, CRC, active-lane count, packed digits, pad
/// trit, then scale. Reconstruction uses deterministic half-away-from-zero
/// integer rounding and the inverse laminate.
///
/// # Errors
///
/// Returns [`FrameError`] for a bad magic (`BadMagic`), a CRC mismatch
/// (`CrcMismatch`), more than 64 active lanes (`InvalidActiveLanes`), a
/// non-canonical packed byte (`InvalidPackedTrit`), a nonzero pad trit
/// (`NonZeroPad`), or a negative scale numerator (`InvalidScale`).
pub fn unfold_super_ultra(frame: &KompressFrame32) -> Result<CentroidPod64, FrameError> {
    if frame.insert4_hdr[0] != INSERT4_MAGIC[0]
        || frame.insert4_hdr[1] != INSERT4_MAGIC[1]
        || frame.insert4_hdr[2] != INSERT4_MAGIC[2]
    {
        return Err(FrameError::BadMagic);
    }

    let expected = frame.insert4_hdr[3];
    let actual = frame_crc(frame);
    if expected != actual {
        return Err(FrameError::CrcMismatch { expected, actual });
    }

    if frame.active_lanes > 64 {
        return Err(FrameError::InvalidActiveLanes {
            active: frame.active_lanes,
        });
    }

    for &byte in &frame.packed_trits {
        if byte > PackedTrit5::MAX {
            return Err(FrameError::InvalidPackedTrit { byte });
        }
    }

    let trits = unpack_trits(&frame.packed_trits);
    if trits[64] != Trit::Zero {
        return Err(FrameError::NonZeroPad);
    }

    if frame.scale_num < 0 {
        return Err(FrameError::InvalidScale);
    }

    let scale_num = i128::from(frame.scale_num);
    let scale_den = i128::from(frame.scale_den.get());
    let mut interleaved = [0i16; 64];
    for (lane, trit) in interleaved.iter_mut().zip(trits.iter()) {
        *lane = scale_lane(*trit, scale_num, scale_den);
    }
    Ok(CentroidPod64(inverse_cross_laminate(&interleaved)))
}

/// Zero-sized [`KompressPort`] implementation for the super-ultra fold.
pub struct SuperUltraFold;

impl KompressPort for SuperUltraFold {
    fn fold(
        pod: &CentroidPod64,
        threshold: RationalPQ,
        epoch: u32,
        budget: u16,
    ) -> KompressFrame32 {
        fold_super_ultra(pod, threshold, epoch, budget)
    }

    fn unfold(frame: &KompressFrame32) -> Result<CentroidPod64, FrameError> {
        unfold_super_ultra(frame)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPOCH: u32 = 0x5EED_1234;
    const BUDGET: u16 = 0xABCD;

    fn threshold_half() -> RationalPQ {
        RationalPQ::new(1, 2).unwrap()
    }

    fn ramp_pod(step: i16) -> CentroidPod64 {
        let mut values = [0i16; 64];
        let mut index = 0usize;
        while index < 64 {
            values[index] = (index as i16 - 32) * step;
            index += 1;
        }
        CentroidPod64(values)
    }

    fn alternating_pod() -> CentroidPod64 {
        let mut values = [0i16; 64];
        let mut index = 0usize;
        while index < 64 {
            values[index] = if index % 2 == 0 { 12_000 } else { -9_000 };
            index += 1;
        }
        CentroidPod64(values)
    }

    fn extremes_pod() -> CentroidPod64 {
        let mut values = [0i16; 64];
        values[0] = i16::MIN;
        values[1] = i16::MAX;
        values[17] = 20_000;
        values[42] = -20_000;
        CentroidPod64(values)
    }

    fn sparse_pod() -> CentroidPod64 {
        let mut values = [0i16; 64];
        values[0] = 30_000;
        values[5] = -30_000;
        values[33] = 100;
        CentroidPod64(values)
    }

    fn sample_pods() -> [CentroidPod64; 5] {
        [
            ramp_pod(1000),
            alternating_pod(),
            extremes_pod(),
            CentroidPod64::ZERO,
            sparse_pod(),
        ]
    }

    fn reseal(frame: &mut KompressFrame32) {
        frame.insert4_hdr[3] = frame_crc(frame);
    }

    fn manual_frame(scale_num: i32, scale_den: u32, trits: &[Trit; 64]) -> KompressFrame32 {
        let mut frame = KompressFrame32 {
            scale_num,
            scale_den: core::num::NonZeroU32::new(scale_den).unwrap(),
            epoch_tag: EPOCH,
            insert4_hdr: [INSERT4_MAGIC[0], INSERT4_MAGIC[1], INSERT4_MAGIC[2], 0],
            qulb_budget: BUDGET,
            active_lanes: 64,
            packed_trits: pack_trits(trits),
        };
        frame.insert4_hdr[3] = frame_crc(&frame);
        frame
    }

    #[test]
    fn crc8_known_vector() {
        assert_eq!(crc8(b"123456789"), 0xF4);
        assert_eq!(crc8(&[]), 0x00);
    }

    #[test]
    fn sphere_pop_counts_exact_survivors() {
        let mut values = [0i16; 64];
        values[0] = 100;
        values[1] = -100;
        values[2] = 49;
        values[3] = -50;
        values[4] = 50;
        values[5] = -49;
        let pod = CentroidPod64(values);
        let (cleaned, active) = sphere_pop(&pod, threshold_half());
        assert_eq!(active, 4);
        assert_eq!(cleaned.0[0], 100);
        assert_eq!(cleaned.0[1], -100);
        assert_eq!(cleaned.0[2], 0);
        assert_eq!(cleaned.0[3], -50);
        assert_eq!(cleaned.0[4], 50);
        assert_eq!(cleaned.0[5], 0);
    }

    #[test]
    fn sphere_pop_zero_pod_has_no_active_lanes() {
        let (cleaned, active) = sphere_pop(&CentroidPod64::ZERO, threshold_half());
        assert_eq!(active, 0);
        assert_eq!(cleaned, CentroidPod64::ZERO);
    }

    #[test]
    fn die_mold_thresholds_are_inclusive_at_zero() {
        assert_eq!(quantize_lane(10_000, 30_000), Trit::Zero);
        assert_eq!(quantize_lane(-10_000, 30_000), Trit::Zero);
        assert_eq!(quantize_lane(10_001, 30_000), Trit::Pos);
        assert_eq!(quantize_lane(-10_001, 30_000), Trit::Neg);
    }

    #[test]
    fn ramp_fold_matches_known_scale() {
        let frame = fold_super_ultra(&ramp_pod(1000), threshold_half(), EPOCH, BUDGET);
        assert_eq!(frame.active_lanes, 33);
        assert_eq!(frame.scale_num, 784_000);
        assert_eq!(frame.scale_den.get(), 33);
    }

    #[test]
    fn fold_unfold_cycle_is_crc_valid() {
        let threshold = threshold_half();
        for pod in sample_pods() {
            let frame = fold_super_ultra(&pod, threshold, EPOCH, BUDGET);
            assert_eq!(frame.epoch_tag, EPOCH);
            assert_eq!(frame.qulb_budget, BUDGET);
            assert_eq!(&frame.insert4_hdr[..3], &INSERT4_MAGIC[..]);
            assert_eq!(frame.insert4_hdr[3], frame_crc(&frame));
            assert!(frame.active_lanes <= 64);
            assert!(unfold_super_ultra(&frame).is_ok());
        }
    }

    #[test]
    fn tampered_scale_num_fails_crc() {
        let frame = fold_super_ultra(&ramp_pod(1000), threshold_half(), EPOCH, BUDGET);
        let mut tampered = frame;
        tampered.scale_num = tampered.scale_num.wrapping_add(1);
        match unfold_super_ultra(&tampered) {
            Err(FrameError::CrcMismatch { expected, actual }) => {
                assert_eq!(expected, frame.insert4_hdr[3]);
                assert_ne!(actual, expected);
            }
            other => panic!("expected CrcMismatch, got {other:?}"),
        }
    }

    #[test]
    fn tampered_magic_is_rejected() {
        let frame = fold_super_ultra(&ramp_pod(1000), threshold_half(), EPOCH, BUDGET);
        let mut tampered = frame;
        tampered.insert4_hdr[0] = b'X';
        assert_eq!(unfold_super_ultra(&tampered), Err(FrameError::BadMagic));
    }

    #[test]
    fn invalid_packed_digit_is_rejected() {
        let mut frame = fold_super_ultra(&ramp_pod(1000), threshold_half(), EPOCH, BUDGET);
        frame.packed_trits[0] = 243;
        reseal(&mut frame);
        assert_eq!(
            unfold_super_ultra(&frame),
            Err(FrameError::InvalidPackedTrit { byte: 243 })
        );
    }

    #[test]
    fn nonzero_pad_is_rejected() {
        let mut frame = fold_super_ultra(&ramp_pod(1000), threshold_half(), EPOCH, BUDGET);
        let digits = PackedTrit5(frame.packed_trits[12]).unpack5();
        frame.packed_trits[12] =
            PackedTrit5::pack5([digits[0], digits[1], digits[2], digits[3], Trit::Pos]).0;
        reseal(&mut frame);
        assert_eq!(unfold_super_ultra(&frame), Err(FrameError::NonZeroPad));
    }

    #[test]
    fn invalid_active_lanes_is_rejected() {
        let mut frame = fold_super_ultra(&ramp_pod(1000), threshold_half(), EPOCH, BUDGET);
        frame.active_lanes = 65;
        reseal(&mut frame);
        assert_eq!(
            unfold_super_ultra(&frame),
            Err(FrameError::InvalidActiveLanes { active: 65 })
        );
    }

    #[test]
    fn negative_scale_is_rejected() {
        let mut frame = fold_super_ultra(&ramp_pod(1000), threshold_half(), EPOCH, BUDGET);
        frame.scale_num = -1;
        reseal(&mut frame);
        assert_eq!(unfold_super_ultra(&frame), Err(FrameError::InvalidScale));
    }

    #[test]
    fn unfold_rounds_half_away_from_zero() {
        let mut trits = [Trit::Zero; 64];
        trits[0] = Trit::Pos;
        trits[1] = Trit::Neg;

        let pod = unfold_super_ultra(&manual_frame(1, 2, &trits)).unwrap();
        assert_eq!(pod.0[0], 1);
        assert_eq!(pod.0[32], -1);

        let pod = unfold_super_ultra(&manual_frame(3, 2, &trits)).unwrap();
        assert_eq!(pod.0[0], 2);
        assert_eq!(pod.0[32], -2);
    }

    #[test]
    fn unfold_saturates_to_i16() {
        let mut trits = [Trit::Zero; 64];
        trits[0] = Trit::Pos;
        trits[1] = Trit::Neg;
        let pod = unfold_super_ultra(&manual_frame(i32::MAX, 1, &trits)).unwrap();
        assert_eq!(pod.0[0], i16::MAX);
        assert_eq!(pod.0[32], i16::MIN);
    }

    #[test]
    fn second_fold_generation_is_a_fixed_point() {
        let thresholds = [
            threshold_half(),
            RationalPQ::ZERO,
            RationalPQ::new(-1, 2).unwrap(),
            RationalPQ::new(1, 1000).unwrap(),
            RationalPQ::ONE,
            RationalPQ::new(3, 2).unwrap(),
        ];
        for threshold in thresholds {
            for pod in sample_pods() {
                let f1 = fold_super_ultra(&pod, threshold, EPOCH, BUDGET);
                let p1 = unfold_super_ultra(&f1).expect("first unfold should succeed");
                let f2 = fold_super_ultra(&p1, threshold, EPOCH, BUDGET);
                let p2 = unfold_super_ultra(&f2).expect("second unfold should succeed");
                let f3 = fold_super_ultra(&p2, threshold, EPOCH, BUDGET);
                assert_eq!(f3, f2, "threshold {:?} pod {:?}", threshold, pod.0);
            }
        }
    }

    #[test]
    fn port_impl_matches_free_functions() {
        let pod = ramp_pod(1000);
        let frame = SuperUltraFold::fold(&pod, threshold_half(), EPOCH, BUDGET);
        assert_eq!(
            frame,
            fold_super_ultra(&pod, threshold_half(), EPOCH, BUDGET)
        );
        assert_eq!(SuperUltraFold::unfold(&frame), unfold_super_ultra(&frame));
    }
}
