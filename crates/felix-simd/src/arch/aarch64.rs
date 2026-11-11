#![allow(unsafe_code)]
use core::arch::aarch64::{
    vaddq_s32, vaddvq_s32, vaddvq_s64, vaddvq_u8, vandq_u16, vcgtq_u16, vcntq_u8, vdupq_n_s32,
    vdupq_n_s64, vdupq_n_u16, vget_high_s16, vget_low_s16, vld1q_s16, vld1q_u16, vld1q_u64,
    vmull_s16, vpadalq_s32, vreinterpretq_s16_u16, vreinterpretq_u8_u64, vshrq_n_u16, vsubq_s16,
};
use pivot_abi::{CentroidPod64, TritPlane64};

use crate::scalar;

const LANE_BITS: [u16; 8] = [1, 2, 4, 8, 16, 32, 64, 128];

fn saturate_i64(value: i64) -> i32 {
    if value > i32::MAX as i64 {
        i32::MAX
    } else if value < i32::MIN as i64 {
        i32::MIN
    } else {
        value as i32
    }
}

pub fn dot64_trit_plane(a: TritPlane64, b: TritPlane64) -> i32 {
    let agree = (a.pos & b.pos) | (a.neg & b.neg);
    let disagree = (a.pos & b.neg) | (a.neg & b.pos);
    let agree_words = [agree, 0u64];
    let disagree_words = [disagree, 0u64];
    // SAFETY: both arrays hold exactly two initialized `u64` lanes, which is
    // the full extent read by `vld1q_u64`; the reinterpretations are the same
    // 128 bits and the byte-popcount sum cannot exceed 128 on `u8`.
    unsafe {
        let agree_bytes = vreinterpretq_u8_u64(vld1q_u64(agree_words.as_ptr()));
        let disagree_bytes = vreinterpretq_u8_u64(vld1q_u64(disagree_words.as_ptr()));
        let agree_count = vaddvq_u8(vcntq_u8(agree_bytes)) as i32;
        let disagree_count = vaddvq_u8(vcntq_u8(disagree_bytes)) as i32;
        agree_count - disagree_count
    }
}

pub fn contract64_trit_i16(weights: TritPlane64, act: &CentroidPod64) -> i32 {
    // SAFETY: `LANE_BITS` has eight initialized `u16` lanes and every
    // `vld1q_s16` reads lanes `[chunk * 8, chunk * 8 + 8)` with `chunk < 8`,
    // staying inside the 64-lane `[i16; 64]` payload. The plain integer
    // accumulators cannot overflow `i32` because the bound is 64 * 32768.
    unsafe {
        let lane_bits = vld1q_u16(LANE_BITS.as_ptr());
        let zero = vdupq_n_u16(0);
        let mut acc = vdupq_n_s32(0);
        let mut chunk = 0;
        while chunk < 8 {
            let shift = chunk * 8;
            let pos_hits = vcgtq_u16(
                vandq_u16(vdupq_n_u16((weights.pos >> shift) as u16), lane_bits),
                zero,
            );
            let neg_hits = vcgtq_u16(
                vandq_u16(vdupq_n_u16((weights.neg >> shift) as u16), lane_bits),
                zero,
            );
            let mask = vsubq_s16(
                vreinterpretq_s16_u16(vshrq_n_u16::<15>(pos_hits)),
                vreinterpretq_s16_u16(vshrq_n_u16::<15>(neg_hits)),
            );
            let activations = vld1q_s16(act.0.as_ptr().add(shift));
            acc = vaddq_s32(
                acc,
                vmull_s16(vget_low_s16(activations), vget_low_s16(mask)),
            );
            acc = vaddq_s32(
                acc,
                vmull_s16(vget_high_s16(activations), vget_high_s16(mask)),
            );
            chunk += 1;
        }
        vaddvq_s32(acc)
    }
}

pub fn dot64_i16(a: &CentroidPod64, b: &CentroidPod64) -> i32 {
    // SAFETY: every `vld1q_s16` reads eight initialized lanes at offsets
    // `0, 8, ..., 56` of the two 64-lane `[i16; 64]` payloads, so all loads
    // stay in bounds. `vmull_s16` widens each product exactly into `i32` and
    // `vpadalq_s32` accumulates pairwise into `i64`; the worst-case magnitude
    // is 64 * 2^30, far below the `i64` limit.
    unsafe {
        let mut acc = vdupq_n_s64(0);
        let mut index = 0;
        while index < 64 {
            let left = vld1q_s16(a.0.as_ptr().add(index));
            let right = vld1q_s16(b.0.as_ptr().add(index));
            acc = vpadalq_s32(acc, vmull_s16(vget_low_s16(left), vget_low_s16(right)));
            acc = vpadalq_s32(acc, vmull_s16(vget_high_s16(left), vget_high_s16(right)));
            index += 8;
        }
        saturate_i64(vaddvq_s64(acc))
    }
}

pub fn louds_rank1(bits: &[u64; 2], idx: usize) -> u32 {
    scalar::louds_rank1(bits, idx)
}

pub fn splitmix64_x4(state: &mut [u64; 4]) -> [u64; 4] {
    scalar::splitmix64_x4(state)
}
