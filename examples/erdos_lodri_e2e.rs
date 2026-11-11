//! Erdős–Lódri `p/q` end-to-end pipeline over the frozen `wukong-zeno` ABI.
//!
//! This example is intentionally self-contained: it duplicates the local
//! `cayley`/`gate` helpers instead of importing the sibling
//! `examples/cayley_bench.rs` (examples never import one another). It builds the
//! four-partition graph, runs the exact ternary Laplacian, an exact rational
//! Cayley chain, the kompress-super-ultra fold, and the qULB capability loop.
//!
//! No floating point is used anywhere in this file.

use felix_simd::ScalarKernels;
use mem8_wave::{fold_super_ultra, unfold_super_ultra};
use phoenix_capsule::{Capsule, UltraBackyardLoop};
use pivot_abi::{CentroidPod64, RationalPQ, Trit, TritPlane64};
use regularity_oracle::{apply_ternary_laplacian, bipartite_density};
use wukong_zeno::fnv1a64;

/// Number of vertices in the Erdős–Lódri graph.
const VERTICES: usize = 16;

/// Raw lane values backing both the micro contraction and the wave state.
///
/// The spec fixes the graph and the pipeline, but leaves the 16-trit mask and
/// the 16 raw pod values to the builder. These values are deterministic,
/// mostly coprime to the exact denominators, and keep every later rational
/// well inside the ABI range.
const RAW_LANES: [i16; 16] = [
    8, -7, 6, -5, 12, -11, 10, -9, 4, -3, 2, -1, 14, -13, 16, -15,
];

/// Shadow cut edges `(u, v)`, each of weight `1/3`.
///
/// Shadow membership is decided by this explicit list, never by weight
/// equality, so another edge that happens to carry `1/3` is not miscounted.
const SHADOW_CUT: [(u32, u32); 3] = [(2, 7), (3, 12), (7, 8)];

/// One undirected weighted edge, added in both directions when the CSR is built.
#[derive(Clone, Copy)]
struct Edge {
    u: u32,
    v: u32,
    weight: RationalPQ,
}

/// One-sided `u -> v` compressed sparse row form of the graph.
struct Csr {
    offsets: Vec<u32>,
    targets: Vec<u32>,
    weights: Vec<RationalPQ>,
}

fn main() {
    // ---------------------------------------------------------------- graph
    let edges = undirected_edges();
    assert_eq!(edges.len(), 31, "K4 + halo + shadow edge count");
    let csr = build_csr(&edges);
    assert_eq!(csr.targets.len(), 62, "both directions must be present");

    // ---------------------------------------------------------------- micro
    // 16-trit mask: lane `l` is Pos when `l % 3 == 0`, Neg when `l % 3 == 1`,
    // and Zero otherwise. The spec leaves this mask to the builder.
    let mut pod = CentroidPod64::ZERO;
    for (lane, &value) in RAW_LANES.iter().enumerate() {
        pod.set(lane, value).expect("pod lane in range");
    }
    let mut mask = TritPlane64::ZERO;
    for lane in 0..RAW_LANES.len() {
        let trit = match lane % 3 {
            0 => Trit::Pos,
            1 => Trit::Neg,
            _ => Trit::Zero,
        };
        mask.set(lane, trit).expect("mask lane in range");
    }
    let micro = felix_simd::contract64_trit_i16(mask, &pod);

    // ---------------------------------------------------------- percolation
    let mut degree = [0u32; VERTICES];
    for edge in &edges {
        degree[edge.u as usize] += 1;
        degree[edge.v as usize] += 1;
    }
    let mut degree_sum = 0u64;
    let mut degree_sq_sum = 0u128;
    for &k in &degree {
        let k = u64::from(k);
        degree_sum += k;
        degree_sq_sum += u128::from(k) * u128::from(k);
    }
    assert_eq!(degree_sum, 62, "handshake: 2 * 31 undirected edges");
    let kappa = RationalPQ::from_ratio(degree_sq_sum as i128, u128::from(degree_sum))
        .expect("Molloy-Reed ratio fits the ABI");
    let supercritical = kappa > RationalPQ::from_integer(2);
    assert!(supercritical, "Molloy-Reed kappa must exceed 2");

    let shadow_edges = edges
        .iter()
        .filter(|edge| SHADOW_CUT.contains(&(edge.u, edge.v)))
        .count();
    assert_eq!(shadow_edges, 3, "shadow edges must remain visible");

    // Broad V2 x V4 cut: every edge with one endpoint in V2 = [4..7] and one in
    // V4 = [12..15], whatever its weight. This includes the `1/1` halo bridges
    // `(5, 13)` and `(6, 13)`, so the cut has 2 of 16 possible crossings.
    let in_v2 = |vertex: u32| (4..=7).contains(&vertex);
    let in_v4 = |vertex: u32| (12..=15).contains(&vertex);
    let cut_edges = edges
        .iter()
        .filter(|edge| {
            (in_v2(edge.u) && in_v4(edge.v)) || (in_v4(edge.u) && in_v2(edge.v))
        })
        .count();
    assert_eq!(cut_edges, 2, "V2 x V4 cut is the two halo bridges");
    let density =
        bipartite_density(cut_edges, 4, 4).expect("bipartite density over V2 x V4");

    // ---------------------------------------------------------------- hodge
    let mut wave = [RationalPQ::ZERO; VERTICES];
    for (lane, &raw) in RAW_LANES.iter().enumerate() {
        wave[lane] = RationalPQ::from_ratio(i128::from(raw), 8).expect("seeded wave lane");
    }
    let mut hodge_out = [RationalPQ::ZERO; VERTICES];
    apply_ternary_laplacian(
        &wave,
        &csr.offsets,
        &csr.targets,
        &csr.weights,
        &mut hodge_out,
    )
    .expect("Laplacian on the oriented CSR");
    let mut hodge = RationalPQ::ZERO;
    for value in hodge_out {
        hodge = hodge.checked_add(value).expect("Hodge sum");
    }
    assert_eq!(hodge, RationalPQ::ZERO, "Hodge conservation must be exact");

    // --------------------------------------------------------------- unitary
    let t0 = energy(&wave);
    // Four gates across pivot-halo and shadow edges, disjoint pairs.
    let chain = [
        (5usize, 13usize, 1i128, 2i128),
        (2, 7, 2, 3),
        (3, 12, 1, 3),
        (10, 14, 3, 4),
    ];
    for (index, &(u, v, r, s)) in chain.iter().enumerate() {
        gate(&mut wave, u, v, r, s, index);
    }
    let t1 = energy(&wave);
    assert_eq!(t0, t1, "exact rational Cayley chain must preserve energy");
    assert!(t0 > RationalPQ::ZERO, "seed energy must be positive");

    // ------------------------------------------------------------------ fold
    let threshold = RationalPQ::new(3, 4).expect("threshold");
    let frame = fold_super_ultra(&pod, threshold, 1, 14);
    let unfolded = unfold_super_ultra(&frame).expect("first unfold");
    let second = fold_super_ultra(&unfolded, threshold, 1, 14);
    let unfolded_second = unfold_super_ultra(&second).expect("second unfold");
    let third = fold_super_ultra(&unfolded_second, threshold, 1, 14);
    // The printed fixed-point flag comes from the fold/unfold refold assertion,
    // not from the qULB loop: the second fold generation is the fixed point.
    let fixed_point = second == third;
    assert!(
        fixed_point,
        "second fold generation must be an exact fixed point"
    );

    let decay = RationalPQ::new(1, 2).expect("decay");
    let mut ultra = UltraBackyardLoop::new(mask, RationalPQ::ONE, 14);
    let mut steps = 0u16;
    let mut loop_fixed = false;
    while steps < 14 {
        let report = ultra
            .qulb_step::<ScalarKernels>(mask, decay)
            .expect("qULB step");
        steps += 1;
        if report.fixed_point {
            loop_fixed = true;
            break;
        }
    }
    assert!(loop_fixed, "qULB loop must terminate within its budget");
    assert!(steps <= 14, "qULB loop must respect the budget");

    let consumed = Capsule::seal(pod, RationalPQ::ONE, 14)
        .activate()
        .attenuate(decay)
        .unwrap_or_else(|_| panic!("capsule attenuation failed"))
        .consume();
    assert_eq!(consumed, pod, "capsule must return the sealed pod");

    // --------------------------------------------------------------- digest
    let mut buffer = Vec::with_capacity(192);
    buffer.extend_from_slice(&micro.to_le_bytes());
    push_rational(&mut buffer, kappa);
    push_rational(&mut buffer, density);
    push_u64(&mut buffer, u64::try_from(shadow_edges).expect("shadow count"));
    push_rational(&mut buffer, hodge);
    push_rational(&mut buffer, t0);
    push_rational(&mut buffer, t1);
    push_u64(&mut buffer, u64::from(frame.active_lanes));
    push_u64(
        &mut buffer,
        u64::try_from(frame.scale_num).expect("scale numerator is non-negative"),
    );
    push_u64(&mut buffer, u64::from(frame.scale_den.get()));
    push_u64(&mut buffer, u64::from(frame.insert4_hdr[3]));
    push_u64(&mut buffer, u64::from(fixed_point));
    let digest = fnv1a64(&buffer);

    // ---------------------------------------------------------------- output
    println!("erdos-lodri e2e");
    println!("micro         : {micro}");
    println!(
        "kappa         : {}/{} (supercritical={supercritical})",
        kappa.num, kappa.den
    );
    println!(
        "density V2xV4 : {}/{} shadow_edges={shadow_edges}",
        density.num, density.den
    );
    println!("hodge sum     : {}/{}", hodge.num, hodge.den);
    println!(
        "energy        : t0={}/{} t1={}/{} equal=true",
        t0.num, t0.den, t1.num, t1.den
    );
    println!(
        "fold          : active={} scale={}/{} crc=0x{:02X} fixed_point={fixed_point}",
        frame.active_lanes,
        frame.scale_num,
        frame.scale_den.get(),
        frame.insert4_hdr[3]
    );
    println!("digest        : 0x{digest:016X}");
}

/// Build the four-partition graph: K4 intra mesh, pivot halo, shadow cut.
fn undirected_edges() -> Vec<Edge> {
    let mut edges = Vec::with_capacity(31);

    let intra = RationalPQ::new(3, 4).expect("intra weight");
    for base in [0u32, 4, 8, 12] {
        for left in 0..4u32 {
            for right in (left + 1)..4u32 {
                edges.push(Edge {
                    u: base + left,
                    v: base + right,
                    weight: intra,
                });
            }
        }
    }

    let halo = RationalPQ::new(1, 1).expect("halo weight");
    for (u, v) in [(5u32, 13u32), (6, 13), (9, 1), (10, 14)] {
        edges.push(Edge {
            u,
            v,
            weight: halo,
        });
    }

    let shadow = RationalPQ::new(1, 3).expect("shadow weight");
    for (u, v) in SHADOW_CUT {
        edges.push(Edge {
            u,
            v,
            weight: shadow,
        });
    }

    edges
}

/// Assemble a one-sided `u -> v` CSR, adding both directions per undirected edge.
fn build_csr(edges: &[Edge]) -> Csr {
    let mut offsets = Vec::with_capacity(VERTICES + 1);
    let mut targets = Vec::with_capacity(edges.len() * 2);
    let mut weights = Vec::with_capacity(edges.len() * 2);

    offsets.push(0u32);
    for source in 0..VERTICES as u32 {
        for edge in edges {
            if edge.u == source {
                targets.push(edge.v);
                weights.push(edge.weight);
            }
            if edge.v == source {
                targets.push(edge.u);
                weights.push(edge.weight);
            }
        }
        offsets.push(u32::try_from(targets.len()).expect("CSR offset fits u32"));
    }

    Csr {
        offsets,
        targets,
        weights,
    }
}

/// Local duplicate of the exact Cayley rotor coefficients used by the benchmark.
fn cayley(r: i128, s: i128) -> (RationalPQ, RationalPQ) {
    let denominator = (s * s + r * r) as u128;
    let cosine = RationalPQ::from_ratio(s * s - r * r, denominator).expect("cayley cosine");
    let sine = RationalPQ::from_ratio(2 * r * s, denominator).expect("cayley sine");
    (cosine, sine)
}

/// Apply one exact rotation to lanes `u` and `v`; overflow names the gate.
fn gate(psi: &mut [RationalPQ; VERTICES], u: usize, v: usize, r: i128, s: i128, index: usize) {
    let (cosine, sine) = cayley(r, s);
    let pu = psi[u];
    let pv = psi[v];
    let cosine_u = cosine
        .checked_mul(pu)
        .unwrap_or_else(|| panic!("gate {index}: c * psi[u] overflow"));
    let sine_v = sine
        .checked_mul(pv)
        .unwrap_or_else(|| panic!("gate {index}: sigma * psi[v] overflow"));
    let sine_u = sine
        .checked_mul(pu)
        .unwrap_or_else(|| panic!("gate {index}: sigma * psi[u] overflow"));
    let cosine_v = cosine
        .checked_mul(pv)
        .unwrap_or_else(|| panic!("gate {index}: c * psi[v] overflow"));
    psi[u] = cosine_u
        .checked_sub(sine_v)
        .unwrap_or_else(|| panic!("gate {index}: rotation subtract overflow"));
    psi[v] = sine_u
        .checked_add(cosine_v)
        .unwrap_or_else(|| panic!("gate {index}: rotation add overflow"));
}

/// Exact L2 energy over the rational wave state.
fn energy(psi: &[RationalPQ; VERTICES]) -> RationalPQ {
    let mut total = RationalPQ::ZERO;
    for &value in psi {
        let square = value
            .checked_mul(value)
            .unwrap_or_else(|| panic!("energy square overflow"));
        total = total
            .checked_add(square)
            .unwrap_or_else(|| panic!("energy accumulation overflow"));
    }
    total
}

fn push_u64(buffer: &mut Vec<u8>, value: u64) {
    buffer.extend_from_slice(&value.to_le_bytes());
}

fn push_rational(buffer: &mut Vec<u8>, value: RationalPQ) {
    buffer.extend_from_slice(&value.num.to_le_bytes());
    buffer.extend_from_slice(&value.den.get().to_le_bytes());
}
