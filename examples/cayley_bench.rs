//! Exact `p/q` Cayley rotation benchmark against an `f64` baseline.
//!
//! This example is its own check: every exact invariant is asserted and a
//! failed assertion exits nonzero. Floats are confined to this target and are
//! only used for the reported drift; they never participate in pass/fail.

use pivot_abi::RationalPQ;
use wukong_zeno::fnv1a64;

const P: u64 = 2_147_483_647; // 2^31 - 1, prime
const LANES: usize = 16;
const STEPS: usize = 10_000;
const GATES: [(i128, i128); 4] = [(1, 2), (2, 3), (1, 3), (3, 4)];

fn cayley(r: i128, s: i128) -> (RationalPQ, RationalPQ) {
    let den = (s * s + r * r) as u128;
    (
        RationalPQ::from_ratio(s * s - r * r, den).expect("c"),
        RationalPQ::from_ratio(2 * r * s, den).expect("sigma"),
    )
}

fn gate(psi: &mut [RationalPQ; LANES], u: usize, v: usize, r: i128, s: i128) {
    let (c, sigma) = cayley(r, s);
    let pu = psi[u];
    let pv = psi[v];
    psi[u] = c
        .checked_mul(pu)
        .unwrap()
        .checked_sub(sigma.checked_mul(pv).unwrap())
        .unwrap();
    psi[v] = sigma
        .checked_mul(pu)
        .unwrap()
        .checked_add(c.checked_mul(pv).unwrap())
        .unwrap();
}

fn energy(psi: &[RationalPQ; LANES]) -> RationalPQ {
    psi.iter().fold(RationalPQ::ZERO, |acc, &x| {
        acc.checked_add(x.checked_mul(x).unwrap()).unwrap()
    })
}

fn mod_pow(b: u64, mut e: u64, m: u64) -> u64 {
    let m = m as u128;
    let mut result = 1u128 % m;
    let mut base = b as u128 % m;
    while e != 0 {
        if e & 1 == 1 {
            result = (result * base) % m;
        }
        base = (base * base) % m;
        e >>= 1;
    }
    result as u64
}

/// Project `p/q` into `F_P` as `p * q^(P-2) mod P`.
///
/// A denominator divisible by `P` is never silently inverted; it panics and
/// names the step that produced it.
fn to_mod_p(x: RationalPQ, step: usize) -> u64 {
    let den = x.den.get();
    if den % P == 0 {
        panic!("denominator divisible by P at step {step}: {den}");
    }
    let num = x.num.rem_euclid(P as i64) as u64;
    ((num as u128 * mod_pow(den % P, P - 2, P) as u128) % P as u128) as u64
}

fn energy_mod(psi: &[u64; LANES]) -> u64 {
    let mut acc = 0u128;
    for &x in psi {
        acc = (acc + (x as u128 * x as u128) % P as u128) % P as u128;
    }
    acc as u64
}

fn seed_state() -> [RationalPQ; LANES] {
    let mut psi = [RationalPQ::ZERO; LANES];
    for (lane, slot) in psi.iter_mut().enumerate() {
        let raw = (lane as i128) * 3 - 10;
        *slot = RationalPQ::from_ratio(raw, 8).expect("seed lane");
    }
    psi
}

fn main() {
    let seed = seed_state();

    // 1. Exact single-step energy equality.
    let exact_before = energy(&seed);
    let mut exact_after = seed;
    gate(&mut exact_after, 0, 1, GATES[0].0, GATES[0].1);
    let exact_after_energy = energy(&exact_after);
    assert_eq!(
        exact_before, exact_after_energy,
        "exact energy changed after one Cayley gate"
    );

    // 2. Ten thousand modular steps in F_P; denominators never grow.
    let mut modular = [0u64; LANES];
    for (lane, slot) in modular.iter_mut().enumerate() {
        *slot = to_mod_p(seed[lane], 0);
    }
    for step in 0..STEPS {
        let before = energy_mod(&modular);
        let (r, s) = GATES[step % GATES.len()];
        let (c, sigma) = cayley(r, s);
        let c_mod = to_mod_p(c, step);
        let sigma_mod = to_mod_p(sigma, step);

        let u = step % LANES;
        let v = (step + 1) % LANES;
        let pu = modular[u] as u128;
        let pv = modular[v] as u128;
        let p = P as u128;
        let new_u = ((c_mod as u128 * pu) % p + p - (sigma_mod as u128 * pv) % p) % p;
        let new_v = ((sigma_mod as u128 * pu) % p + (c_mod as u128 * pv) % p) % p;
        modular[u] = new_u as u64;
        modular[v] = new_v as u64;

        let after = energy_mod(&modular);
        assert_eq!(before, after, "modular energy drifted at step {step}");
    }

    // 3. Float baseline over the same gate sequence; report drift only.
    let mut float_state = [0.0f64; LANES];
    for (lane, slot) in float_state.iter_mut().enumerate() {
        *slot = seed[lane].num as f64 / seed[lane].den.get() as f64;
    }
    let float_initial = float_energy(&float_state);
    for step in 0..STEPS {
        let (r, s) = GATES[step % GATES.len()];
        let den = (s * s + r * r) as f64;
        let c = (s * s - r * r) as f64 / den;
        let sigma = (2 * r * s) as f64 / den;

        let u = step % LANES;
        let v = (step + 1) % LANES;
        let pu = float_state[u];
        let pv = float_state[v];
        float_state[u] = c * pu - sigma * pv;
        float_state[v] = sigma * pu + c * pv;
    }
    let float_final = float_energy(&float_state);
    // Report the measured drift honestly: truncated integer ppm plus the
    // relative value in scientific notation. Pass/fail never reads it.
    let relative_drift = (float_initial - float_final).abs() / float_initial;
    let drift_ppm = (relative_drift * 1_000_000.0) as u64;

    // 4. Digest of the exact final rational state plus the modular final state.
    let mut bytes = Vec::with_capacity(LANES * 24);
    for value in &exact_after {
        bytes.extend_from_slice(&value.num.to_le_bytes());
        bytes.extend_from_slice(&value.den.get().to_le_bytes());
    }
    for &lane in &modular {
        bytes.extend_from_slice(&lane.to_le_bytes());
    }
    let digest = fnv1a64(&bytes);

    println!("cayley exact  : energy preserved (1 gate, exact Q), P=2147483647");
    println!("cayley modular: energy preserved over {STEPS} steps in F_P");
    println!("cayley float  : drift_ppm={drift_ppm} rel={relative_drift:.2e}");
    println!("digest        : 0x{digest:016X}");
}

fn float_energy(psi: &[f64; LANES]) -> f64 {
    psi.iter().map(|x| x * x).sum()
}
