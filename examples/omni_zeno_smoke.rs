//! End-to-end smoke run for the `wukong-zeno` v0.2.0 workspace.
//!
//! This is the only `std` entry point in the workspace. It exercises every
//! crate exclusively through the frozen `pivot-abi` traits and prints a compact
//! deterministic summary.

use felix_simd::{ArchKernels, ScalarKernels};
use mem8_wave::{crc8, SuperUltraFold};
use phoenix_capsule::{Capsule, UltraBackyardLoop};
use pivot_abi::{
    CentroidPod64, FrameError, KompressFrame32, KompressPort, LaplacianOraclePort, QulbCapsulePort,
    RationalPQ, TernarySimdPort, TritPlane64,
};
use regularity_oracle::{bipartite_density, partition_by_degree, pivot_indices, TernaryLaplacian};

const VECTORS: u32 = 1_000;

fn main() {
    println!("wukong-zeno v0.2.0 - omni zeno smoke");

    let (mismatches, checked) = kernel_parity();
    assert_eq!(mismatches, 0, "arch/scalar kernels diverged");
    println!("kernel parity : {checked}/{VECTORS} vectors, {mismatches} mismatches");

    let (frame, second, third, unfolded, tamper) = kompress_cycle();
    assert_eq!(
        second, third,
        "second fold generation must be a fixed point"
    );
    assert!(matches!(tamper, Err(FrameError::CrcMismatch { .. })));
    println!(
        "kompress      : 128B -> 32B -> 128B, active={} scale={}/{} crc=0x{:02X}",
        frame.active_lanes,
        frame.scale_num,
        frame.scale_den.get(),
        frame.insert4_hdr[3]
    );
    println!(
        "               refold=fixed-point tamper=CRC-mismatch unfolded_l1={}",
        unfolded.l1_abs_sum()
    );

    let (out, delta, partitions, pivots, density) = oracle_cycle();
    assert_eq!(delta, RationalPQ::ZERO, "Laplacian conservation failed");
    print!("oracle        : Delta_H3=0 partitions=[");
    for (index, bucket) in partitions.iter().enumerate() {
        if index > 0 {
            print!(" ");
        }
        print!("{index}:{bucket}");
    }
    println!("] pivots={pivots} density={density}");

    let (step, capsule_value) = qulb_cycle();
    assert!(step.fixed_point, "qULB did not converge");
    assert!(step.budget < 16);
    assert!(step.authority < RationalPQ::new(3, 4).unwrap());
    println!(
        "qulb          : fixed_point=true authority={} budget={} energy={} capsule=0x{capsule_value:04X}",
        step.authority, step.budget, step.energy
    );

    let digest = state_digest(&frame, &out, &step);
    println!("digest        : 0x{digest:016X}");
    println!("status        : OK (no floats, no std in crates, no allocation in crates)");
}

fn kernel_parity() -> (u32, u32) {
    let mut state = [0x0123_4567_89AB_CDEFu64, 0x0FED_CBA9_8765_4321, 7, 11];
    let mut mismatches = 0u32;
    let mut checked = 0u32;

    while checked < VECTORS {
        let a = fill_pod(&mut state);
        let b = fill_pod(&mut state);
        let plane_a = plane_from_pod(&a);
        let plane_b = plane_from_pod(&b);

        let arch_dot = <ArchKernels as TernarySimdPort>::dot64_i16(&a, &b);
        let scalar_dot = <ScalarKernels as TernarySimdPort>::dot64_i16(&a, &b);
        let arch_contract = <ArchKernels as TernarySimdPort>::contract64_trit_i16(plane_a, &b);
        let scalar_contract = <ScalarKernels as TernarySimdPort>::contract64_trit_i16(plane_a, &b);
        let arch_trit = <ArchKernels as TernarySimdPort>::dot64_trit_plane(plane_a, plane_b);
        let scalar_trit = <ScalarKernels as TernarySimdPort>::dot64_trit_plane(plane_a, plane_b);

        if arch_dot != scalar_dot || arch_contract != scalar_contract || arch_trit != scalar_trit {
            mismatches += 1;
        }
        checked += 1;
    }

    let extremes = CentroidPod64([i16::MIN; 64]);
    let maximum = CentroidPod64([i16::MAX; 64]);
    if <ArchKernels as TernarySimdPort>::dot64_i16(&extremes, &maximum)
        != <ScalarKernels as TernarySimdPort>::dot64_i16(&extremes, &maximum)
    {
        mismatches += 1;
    }

    let mut ranks_ok = true;
    for idx in [0usize, 1, 63, 64, 65, 127, 128] {
        ranks_ok &= <ArchKernels as TernarySimdPort>::louds_rank1(
            &[0xFFFF_FFFF_FFFF_FFFF, 0x0F0F_0F0F_0F0F_0F0F],
            idx,
        ) == <ScalarKernels as TernarySimdPort>::louds_rank1(
            &[0xFFFF_FFFF_FFFF_FFFF, 0x0F0F_0F0F_0F0F_0F0F],
            idx,
        );
    }
    if !ranks_ok {
        mismatches += 1;
    }

    (mismatches, checked)
}

fn fill_pod(state: &mut [u64; 4]) -> CentroidPod64 {
    let mut lanes = [0i16; 64];
    let mut cursor = 0usize;
    while cursor < 64 {
        let words = <ScalarKernels as TernarySimdPort>::splitmix64_x4(state);
        let mut lane = 0usize;
        while lane < 8 && cursor < 64 {
            let word = words[lane % 4];
            lanes[cursor] = ((word >> ((lane % 4) * 16)) as u16) as i16;
            cursor += 1;
            lane += 1;
        }
    }
    CentroidPod64(lanes)
}

fn plane_from_pod(pod: &CentroidPod64) -> TritPlane64 {
    let mut pos = 0u64;
    let mut neg = 0u64;
    let mut lane = 0usize;
    while lane < 64 {
        match pod.0[lane].rem_euclid(3) {
            1 => pos |= 1u64 << lane,
            2 => neg |= 1u64 << lane,
            _ => {}
        }
        lane += 1;
    }
    TritPlane64::new(pos, neg).unwrap()
}

fn kompress_cycle() -> (
    KompressFrame32,
    KompressFrame32,
    KompressFrame32,
    CentroidPod64,
    Result<CentroidPod64, FrameError>,
) {
    let mut lanes = [0i16; 64];
    let mut index = 0usize;
    while index < 64 {
        lanes[index] = (((index as i32 * 977) % 30_001) - 15_000) as i16;
        index += 1;
    }
    let pod = CentroidPod64(lanes);
    let threshold = RationalPQ::new(1, 8).unwrap();

    let first = <SuperUltraFold as KompressPort>::fold(&pod, threshold, 42, 64);
    let unfolded = <SuperUltraFold as KompressPort>::unfold(&first).unwrap();
    let second = <SuperUltraFold as KompressPort>::fold(&unfolded, threshold, 42, 64);
    let unfolded_second = <SuperUltraFold as KompressPort>::unfold(&second).unwrap();
    let third = <SuperUltraFold as KompressPort>::fold(&unfolded_second, threshold, 42, 64);

    let mut tampered = first;
    tampered.scale_num = tampered.scale_num.wrapping_add(1);
    let tamper = <SuperUltraFold as KompressPort>::unfold(&tampered);

    (first, second, third, unfolded, tamper)
}

fn oracle_cycle() -> ([RationalPQ; 8], RationalPQ, [u8; 8], usize, RationalPQ) {
    let offsets = [0u32, 2, 3, 5, 6, 7, 8, 9, 10];
    let targets = [1u32, 4, 2, 3, 6, 0, 5, 6, 7, 4];
    let weights = [
        RationalPQ::new(1, 2).unwrap(),
        RationalPQ::new(2, 5).unwrap(),
        RationalPQ::new(2, 3).unwrap(),
        RationalPQ::new(3, 2).unwrap(),
        RationalPQ::new(3, 7).unwrap(),
        RationalPQ::new(5, 4).unwrap(),
        RationalPQ::new(1, 3).unwrap(),
        RationalPQ::new(4, 5).unwrap(),
        RationalPQ::new(7, 6).unwrap(),
        RationalPQ::new(1, 7).unwrap(),
    ];
    let x = [
        RationalPQ::from_integer(1),
        RationalPQ::from_integer(2),
        RationalPQ::from_integer(3),
        RationalPQ::from_integer(4),
        RationalPQ::from_integer(5),
        RationalPQ::from_integer(6),
        RationalPQ::from_integer(7),
        RationalPQ::from_integer(8),
    ];
    let mut out = [RationalPQ::ZERO; 8];
    <TernaryLaplacian as LaplacianOraclePort>::apply_ternary_laplacian(
        &x, &offsets, &targets, &weights, &mut out,
    )
    .unwrap();

    let mut delta = RationalPQ::ZERO;
    for value in &out {
        delta = delta.checked_add(*value).unwrap();
    }

    let mut partitions = [0u8; 8];
    partition_by_degree(&offsets, &mut partitions, 4).unwrap();
    let mut pivot_buffer = [0u32; 8];
    let pivots = pivot_indices(&offsets, &mut pivot_buffer);
    let density = bipartite_density(10, 4, 4).unwrap();

    (out, delta, partitions, pivots, density)
}

fn qulb_cycle() -> (phoenix_capsule::QulbStep, u16) {
    let state = TritPlane64::new(u64::MAX, 0).unwrap();
    let weights = TritPlane64::new(0x0F0F_0F0F_0F0F_0F0F, 0).unwrap();
    let authority = RationalPQ::new(3, 4).unwrap();
    let decay = RationalPQ::new(2, 3).unwrap();

    let mut ultra = UltraBackyardLoop::new(state, authority, 16);
    let initial = <UltraBackyardLoop as QulbCapsulePort>::authority(&ultra);
    assert_eq!(initial, authority);

    let step = ultra
        .run_to_fixed_point::<ArchKernels>(weights, decay, 16)
        .unwrap();

    let capsule = Capsule::seal(0x8B15u16, authority, 4)
        .activate()
        .attenuate(decay)
        .unwrap()
        .consume();

    (step, capsule)
}

fn state_digest(
    frame: &KompressFrame32,
    out: &[RationalPQ; 8],
    step: &phoenix_capsule::QulbStep,
) -> u64 {
    let mut buffer = [0u8; 256];
    let mut cursor = 0usize;

    put_u64(&mut buffer, &mut cursor, u64::from(frame.scale_num as u32));
    put_u64(&mut buffer, &mut cursor, u64::from(frame.scale_den.get()));
    put_u64(&mut buffer, &mut cursor, u64::from(frame.epoch_tag));
    put_u64(&mut buffer, &mut cursor, u64::from(frame.insert4_hdr[3]));
    put_u64(&mut buffer, &mut cursor, u64::from(frame.qulb_budget));
    put_u64(&mut buffer, &mut cursor, u64::from(frame.active_lanes));
    put_u64(&mut buffer, &mut cursor, crc8(&frame.insert4_hdr) as u64);

    for value in out {
        put_u64(&mut buffer, &mut cursor, value.num as u64);
        put_u64(&mut buffer, &mut cursor, value.den.get());
    }

    put_u64(&mut buffer, &mut cursor, step.state.pos);
    put_u64(&mut buffer, &mut cursor, step.state.neg);
    put_u64(&mut buffer, &mut cursor, step.authority.num as u64);
    put_u64(&mut buffer, &mut cursor, step.authority.den.get());
    put_u64(&mut buffer, &mut cursor, u64::from(step.budget));
    put_u64(&mut buffer, &mut cursor, u64::from(step.fixed_point));

    wukong_zeno::fnv1a64(&buffer[..cursor])
}

fn put_u64(buffer: &mut [u8; 256], cursor: &mut usize, value: u64) {
    let bytes = value.to_le_bytes();
    buffer[*cursor..*cursor + 8].copy_from_slice(&bytes);
    *cursor += 8;
}
