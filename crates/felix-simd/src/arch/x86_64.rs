#![allow(unsafe_code)]
use core::arch::x86_64::{
    __m128i, _mm_add_epi32, _mm_add_epi64, _mm_and_si128, _mm_cmpgt_epi16, _mm_loadu_si128,
    _mm_madd_epi16, _mm_set1_epi16, _mm_set1_epi32, _mm_setr_epi16, _mm_setzero_si128,
    _mm_srai_epi16, _mm_srai_epi32, _mm_storeu_si128, _mm_unpackhi_epi16, _mm_unpackhi_epi32,
    _mm_unpacklo_epi16, _mm_unpacklo_epi32,
};
use pivot_abi::{CentroidPod64, TritPlane64};

use crate::scalar;

fn saturate_i64(value: i64) -> i32 {
    if value > i32::MAX as i64 {
        i32::MAX
    } else if value < i32::MIN as i64 {
        i32::MIN
    } else {
        value as i32
    }
}

fn horizontal_sum_i32(value: __m128i) -> i32 {
    let mut lanes = [0i32; 4];
    // SAFETY: `lanes` owns exactly 16 bytes, the size written by
    // `_mm_storeu_si128`, and the store is unaligned so no alignment is
    // required.
    unsafe {
        _mm_storeu_si128(lanes.as_mut_ptr().cast(), value);
    }
    lanes[0] + lanes[1] + lanes[2] + lanes[3]
}

pub fn dot64_trit_plane(a: TritPlane64, b: TritPlane64) -> i32 {
    scalar::dot64_trit_plane(a, b)
}

pub fn contract64_trit_i16(weights: TritPlane64, act: &CentroidPod64) -> i32 {
    // SAFETY: every `_mm_loadu_si128` reads 16 bytes at lane offsets
    // `0, 8, ..., 56`, fully inside the 128-byte `[i16; 64]` payload. The
    // masks are 0xFFFF/0x0000 i16 lanes, widening uses the arithmetic sign
    // lane, and the accumulator bound is 64 * 32768, far inside `i32`.
    unsafe {
        let lane_bits = _mm_setr_epi16(1, 2, 4, 8, 16, 32, 64, 128);
        let zero = _mm_setzero_si128();
        let mut positive = zero;
        let mut negative = zero;
        let mut chunk = 0;
        while chunk < 8 {
            let shift = chunk * 8;
            let pos_hits = _mm_cmpgt_epi16(
                _mm_and_si128(
                    _mm_set1_epi16((weights.pos >> shift) as u8 as i16),
                    lane_bits,
                ),
                zero,
            );
            let neg_hits = _mm_cmpgt_epi16(
                _mm_and_si128(
                    _mm_set1_epi16((weights.neg >> shift) as u8 as i16),
                    lane_bits,
                ),
                zero,
            );
            let activations = _mm_loadu_si128(act.0.as_ptr().add(shift).cast());
            let pos_values = _mm_and_si128(activations, pos_hits);
            let neg_values = _mm_and_si128(activations, neg_hits);
            let pos_sign = _mm_srai_epi16::<15>(pos_values);
            let neg_sign = _mm_srai_epi16::<15>(neg_values);
            positive = _mm_add_epi32(positive, _mm_unpacklo_epi16(pos_values, pos_sign));
            positive = _mm_add_epi32(positive, _mm_unpackhi_epi16(pos_values, pos_sign));
            negative = _mm_add_epi32(negative, _mm_unpacklo_epi16(neg_values, neg_sign));
            negative = _mm_add_epi32(negative, _mm_unpackhi_epi16(neg_values, neg_sign));
            chunk += 1;
        }
        horizontal_sum_i32(positive) - horizontal_sum_i32(negative)
    }
}

pub fn dot64_i16(a: &CentroidPod64, b: &CentroidPod64) -> i32 {
    // SAFETY: every `_mm_loadu_si128` reads 16 bytes at lane offsets
    // `0, 8, ..., 56`, fully inside the two 128-byte `[i16; 64]` payloads.
    // Masking one operand lane to zero before `_mm_madd_epi16` keeps every
    // adjacent pair sum to a single exact product (at most 2^30), and each
    // product is sign-extended into `i64` before accumulation, so the final
    // bound 64 * 2^30 never overflows.
    unsafe {
        let even_mask = _mm_set1_epi32(0x0000_FFFFu32 as i32);
        let odd_mask = _mm_set1_epi32(0xFFFF_0000u32 as i32);
        let mut even_acc = _mm_setzero_si128();
        let mut odd_acc = _mm_setzero_si128();
        let mut index = 0;
        while index < 64 {
            let left = _mm_loadu_si128(a.0.as_ptr().add(index).cast());
            let right = _mm_loadu_si128(b.0.as_ptr().add(index).cast());
            let left_odd = _mm_and_si128(left, odd_mask);
            let right_even = _mm_and_si128(right, even_mask);
            let even_products = _mm_madd_epi16(left, right_even);
            let odd_products = _mm_madd_epi16(left_odd, right);
            let even_sign = _mm_srai_epi32::<31>(even_products);
            let odd_sign = _mm_srai_epi32::<31>(odd_products);
            even_acc = _mm_add_epi64(even_acc, _mm_unpacklo_epi32(even_products, even_sign));
            even_acc = _mm_add_epi64(even_acc, _mm_unpackhi_epi32(even_products, even_sign));
            odd_acc = _mm_add_epi64(odd_acc, _mm_unpacklo_epi32(odd_products, odd_sign));
            odd_acc = _mm_add_epi64(odd_acc, _mm_unpackhi_epi32(odd_products, odd_sign));
            index += 8;
        }
        let total = _mm_add_epi64(even_acc, odd_acc);
        let mut lanes = [0i64; 2];
        _mm_storeu_si128(lanes.as_mut_ptr().cast(), total);
        saturate_i64(lanes[0] + lanes[1])
    }
}

pub fn louds_rank1(bits: &[u64; 2], idx: usize) -> u32 {
    scalar::louds_rank1(bits, idx)
}

pub fn splitmix64_x4(state: &mut [u64; 4]) -> [u64; 4] {
    scalar::splitmix64_x4(state)
}
