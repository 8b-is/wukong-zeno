use pivot_abi::{CentroidPod64, TritPlane64};

pub fn dot64_trit_plane(a: TritPlane64, b: TritPlane64) -> i32 {
    let agree = ((a.pos & b.pos) | (a.neg & b.neg)).count_ones() as i32;
    let disagree = ((a.pos & b.neg) | (a.neg & b.pos)).count_ones() as i32;
    agree - disagree
}

pub fn contract64_trit_i16(weights: TritPlane64, act: &CentroidPod64) -> i32 {
    let mut acc = 0i32;
    let mut lane = 0;
    while lane < 64 {
        let bit = 1u64 << lane;
        let positive = weights.pos & bit != 0;
        let negative = weights.neg & bit != 0;
        // A lane with both bits set is outside the `TritPlane64` invariant.
        // Treat it as neutral so the scalar, NEON, and SSE2 paths agree even
        // on malformed planes instead of silently choosing opposite signs.
        if positive && !negative {
            acc += act.0[lane] as i32;
        } else if negative && !positive {
            acc -= act.0[lane] as i32;
        }
        lane += 1;
    }
    acc
}

pub fn dot64_i16(a: &CentroidPod64, b: &CentroidPod64) -> i32 {
    let mut acc = 0i64;
    let mut lane = 0;
    while lane < 64 {
        acc += a.0[lane] as i64 * b.0[lane] as i64;
        lane += 1;
    }
    if acc > i32::MAX as i64 {
        i32::MAX
    } else if acc < i32::MIN as i64 {
        i32::MIN
    } else {
        acc as i32
    }
}

pub fn louds_rank1(bits: &[u64; 2], idx: usize) -> u32 {
    let clamped = idx.min(128);
    if clamped == 0 {
        return 0;
    }
    if clamped < 64 {
        return (bits[0] & ((1u64 << clamped) - 1)).count_ones();
    }
    if clamped == 64 {
        return bits[0].count_ones();
    }
    let upper = clamped - 64;
    let mut rank = bits[0].count_ones();
    if upper == 64 {
        rank += bits[1].count_ones();
    } else {
        rank += (bits[1] & ((1u64 << upper) - 1)).count_ones();
    }
    rank
}

pub fn splitmix64_x4(state: &mut [u64; 4]) -> [u64; 4] {
    const GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;
    const MIX_A: u64 = 0xBF58_476D_1CE4_E5B9;
    const MIX_B: u64 = 0x94D0_49BB_1331_11EB;

    let mut out = [0u64; 4];
    let mut lane = 0;
    while lane < 4 {
        let mut z = state[lane].wrapping_add(GAMMA);
        state[lane] = z;
        z = (z ^ (z >> 30)).wrapping_mul(MIX_A);
        z = (z ^ (z >> 27)).wrapping_mul(MIX_B);
        out[lane] = z ^ (z >> 31);
        lane += 1;
    }
    out
}
