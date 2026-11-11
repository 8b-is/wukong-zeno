use crate::{arch, scalar, ArchKernels, ScalarKernels};
use pivot_abi::{CentroidPod64, TernarySimdPort, TritPlane64};

const SEED: [u64; 4] = [
    0x0123_4567_89AB_CDEF,
    0xFEDC_BA98_7654_3210,
    0x0F0F_0F0F_0F0F_0F0F,
    0xFFFF_0000_FFFF_0000,
];

fn next_word(state: &mut [u64; 4]) -> u64 {
    scalar::splitmix64_x4(state)[0]
}

fn next_plane(state: &mut [u64; 4]) -> TritPlane64 {
    let pos = next_word(state);
    let neg = next_word(state) & !pos;
    TritPlane64 { pos, neg }
}

fn next_pod(state: &mut [u64; 4]) -> CentroidPod64 {
    let mut lanes = [0i16; 64];
    let mut index = 0;
    while index < 64 {
        let words = scalar::splitmix64_x4(state);
        let mut lane = 0;
        while lane < 4 {
            let word = words[lane];
            lanes[index + lane * 2] = word as u16 as i16;
            lanes[index + lane * 2 + 1] = (word >> 16) as u16 as i16;
            lane += 1;
        }
        index += 8;
    }
    CentroidPod64(lanes)
}

fn filled(value: i16) -> CentroidPod64 {
    CentroidPod64([value; 64])
}

#[test]
fn thousand_deterministic_vectors_match_scalar() {
    let mut state = SEED;
    let mut iteration = 0;
    while iteration < 1000 {
        let left_plane = next_plane(&mut state);
        let right_plane = next_plane(&mut state);
        let left_pod = next_pod(&mut state);
        let right_pod = next_pod(&mut state);

        assert_eq!(
            arch::dot64_trit_plane(left_plane, right_plane),
            scalar::dot64_trit_plane(left_plane, right_plane)
        );
        assert_eq!(
            arch::contract64_trit_i16(left_plane, &left_pod),
            scalar::contract64_trit_i16(left_plane, &left_pod)
        );
        assert_eq!(
            arch::dot64_i16(&left_pod, &right_pod),
            scalar::dot64_i16(&left_pod, &right_pod)
        );

        iteration += 1;
    }
}

#[test]
fn extreme_inputs_match_scalar_and_saturate() {
    let zeros = CentroidPod64::ZERO;
    let mins = filled(i16::MIN);
    let maxs = filled(i16::MAX);
    let mut alternating_lanes = [0i16; 64];
    let mut inverse_lanes = [0i16; 64];
    let mut index = 0;
    while index < 64 {
        if (index & 1) == 0 {
            alternating_lanes[index] = i16::MIN;
            inverse_lanes[index] = i16::MAX;
        } else {
            alternating_lanes[index] = i16::MAX;
            inverse_lanes[index] = i16::MIN;
        }
        index += 1;
    }
    let alternating = CentroidPod64(alternating_lanes);
    let inverse = CentroidPod64(inverse_lanes);

    let pods = [zeros, mins, maxs, alternating, inverse];
    let mut left = 0;
    while left < pods.len() {
        let mut right = 0;
        while right < pods.len() {
            assert_eq!(
                arch::dot64_i16(&pods[left], &pods[right]),
                scalar::dot64_i16(&pods[left], &pods[right])
            );
            right += 1;
        }
        left += 1;
    }

    assert_eq!(scalar::dot64_i16(&zeros, &maxs), 0);
    assert_eq!(arch::dot64_i16(&zeros, &maxs), 0);
    assert_eq!(scalar::dot64_i16(&mins, &mins), i32::MAX);
    assert_eq!(arch::dot64_i16(&mins, &mins), i32::MAX);
    assert_eq!(scalar::dot64_i16(&maxs, &maxs), i32::MAX);
    assert_eq!(arch::dot64_i16(&maxs, &maxs), i32::MAX);
    assert_eq!(scalar::dot64_i16(&mins, &maxs), i32::MIN);
    assert_eq!(arch::dot64_i16(&mins, &maxs), i32::MIN);
    assert_eq!(scalar::dot64_i16(&alternating, &inverse), i32::MIN);
    assert_eq!(arch::dot64_i16(&alternating, &inverse), i32::MIN);

    let all_pos = TritPlane64 {
        pos: u64::MAX,
        neg: 0,
    };
    let all_neg = TritPlane64 {
        pos: 0,
        neg: u64::MAX,
    };
    let mixed = TritPlane64 {
        pos: 0xAAAA_AAAA_AAAA_AAAA,
        neg: 0x5555_5555_5555_5555,
    };

    let planes = [all_pos, all_neg, mixed, TritPlane64::ZERO];
    let mut left = 0;
    while left < planes.len() {
        let mut right = 0;
        while right < planes.len() {
            assert_eq!(
                arch::dot64_trit_plane(planes[left], planes[right]),
                scalar::dot64_trit_plane(planes[left], planes[right])
            );
            right += 1;
        }
        assert_eq!(
            arch::contract64_trit_i16(planes[left], &mins),
            scalar::contract64_trit_i16(planes[left], &mins)
        );
        assert_eq!(
            arch::contract64_trit_i16(planes[left], &maxs),
            scalar::contract64_trit_i16(planes[left], &maxs)
        );
        left += 1;
    }

    assert_eq!(scalar::dot64_trit_plane(all_pos, all_pos), 64);
    assert_eq!(arch::dot64_trit_plane(all_pos, all_pos), 64);
    assert_eq!(scalar::dot64_trit_plane(all_pos, all_neg), -64);
    assert_eq!(arch::dot64_trit_plane(all_pos, all_neg), -64);
    assert_eq!(scalar::dot64_trit_plane(all_neg, all_neg), 64);
    assert_eq!(arch::dot64_trit_plane(all_neg, all_neg), 64);
    assert_eq!(scalar::contract64_trit_i16(all_pos, &mins), -2_097_152);
    assert_eq!(arch::contract64_trit_i16(all_pos, &mins), -2_097_152);
    assert_eq!(scalar::contract64_trit_i16(all_neg, &mins), 2_097_152);
    assert_eq!(arch::contract64_trit_i16(all_neg, &mins), 2_097_152);
    assert_eq!(scalar::contract64_trit_i16(all_pos, &maxs), 2_097_088);
    assert_eq!(arch::contract64_trit_i16(all_pos, &maxs), 2_097_088);
}

#[test]
fn overlapping_planes_are_neutral_on_every_path() {
    let all_overlap = TritPlane64 {
        pos: u64::MAX,
        neg: u64::MAX,
    };
    let mixed_overlap = TritPlane64 {
        pos: 0xAAAA_AAAA_AAAA_AAAA,
        neg: 0xAAAA_AAAA_AAAA_AAAA,
    };
    let all_pos = TritPlane64 {
        pos: u64::MAX,
        neg: 0,
    };
    let pod = filled(i16::MAX);

    for plane in [all_overlap, mixed_overlap] {
        assert_eq!(scalar::contract64_trit_i16(plane, &pod), 0);
        assert_eq!(arch::contract64_trit_i16(plane, &pod), 0);
        assert_eq!(scalar::dot64_trit_plane(plane, all_pos), 0);
        assert_eq!(arch::dot64_trit_plane(plane, all_pos), 0);
        assert_eq!(scalar::dot64_trit_plane(all_pos, plane), 0);
        assert_eq!(arch::dot64_trit_plane(all_pos, plane), 0);
    }

    let partially_overlapping = TritPlane64 {
        pos: 0xFFFF_FFFF_FFFF_FFFF,
        neg: 0x0000_0000_0000_00FF,
    };
    assert_eq!(
        scalar::contract64_trit_i16(partially_overlapping, &pod),
        arch::contract64_trit_i16(partially_overlapping, &pod)
    );
    assert_eq!(
        scalar::dot64_trit_plane(partially_overlapping, all_pos),
        arch::dot64_trit_plane(partially_overlapping, all_pos)
    );
}

#[test]
fn louds_rank_boundaries_match_scalar() {
    let bits = [u64::MAX, 1];
    let indices = [0usize, 1, 63, 64, 65, 127, 128];
    let expected = [0u32, 1, 63, 64, 65, 65, 65];
    let mut index = 0;
    while index < indices.len() {
        assert_eq!(scalar::louds_rank1(&bits, indices[index]), expected[index]);
        assert_eq!(arch::louds_rank1(&bits, indices[index]), expected[index]);
        index += 1;
    }

    assert_eq!(scalar::louds_rank1(&bits, 200), 65);
    assert_eq!(arch::louds_rank1(&bits, 200), 65);

    let pattern = [0xAAAA_AAAA_AAAA_AAAA, 0x8000_0000_0000_0001];
    let mut index = 0;
    while index <= 130 {
        assert_eq!(
            arch::louds_rank1(&pattern, index),
            scalar::louds_rank1(&pattern, index)
        );
        index += 1;
    }
}

#[test]
fn splitmix_matches_known_vector_and_is_deterministic() {
    let mut state = [0u64; 4];
    let produced = scalar::splitmix64_x4(&mut state);
    assert_eq!(produced, [0xE220_A839_7B1D_CDAF; 4]);
    assert_eq!(state, [0x9E37_79B9_7F4A_7C15; 4]);

    let mut repeat_state = [0u64; 4];
    assert_eq!(scalar::splitmix64_x4(&mut repeat_state), produced);

    let mut arch_state = [0u64; 4];
    assert_eq!(arch::splitmix64_x4(&mut arch_state), produced);
    assert_eq!(arch_state, state);
}

#[test]
fn marker_structs_expose_the_five_kernels() {
    let pod = CentroidPod64::ZERO;
    let plane = TritPlane64::ZERO;
    let bits = [0u64; 2];

    assert_eq!(ScalarKernels::dot64_trit_plane(plane, plane), 0);
    assert_eq!(ArchKernels::dot64_trit_plane(plane, plane), 0);
    assert_eq!(ScalarKernels::contract64_trit_i16(plane, &pod), 0);
    assert_eq!(ArchKernels::contract64_trit_i16(plane, &pod), 0);
    assert_eq!(ScalarKernels::dot64_i16(&pod, &pod), 0);
    assert_eq!(ArchKernels::dot64_i16(&pod, &pod), 0);
    assert_eq!(ScalarKernels::louds_rank1(&bits, 0), 0);
    assert_eq!(ArchKernels::louds_rank1(&bits, 0), 0);

    let mut scalar_state = [1u64, 2, 3, 4];
    let mut arch_state = scalar_state;
    assert_eq!(
        ScalarKernels::splitmix64_x4(&mut scalar_state),
        ArchKernels::splitmix64_x4(&mut arch_state)
    );
}
