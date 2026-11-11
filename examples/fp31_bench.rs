//! Branchless Montgomery-free 𝔽ₚ₃₁ Cayley round-trip benchmark.
//!
//! The exact backend keeps every lane in the prime field `P = 2^31 - 1` and
//! multiplies with the double Mersenne fold, so the hot loop contains no `%`
//! and no division. Two million gate applications forward and back must return
//! the seed bit-for-bit and preserve the field L2 energy exactly.
//!
//! The `f32`/`f64` backends run the identical schedule. Their drift is only
//! reported; it is never asserted to be nonzero or zero.
//!
//! `Fp31` is intentionally local to this example: no library crate is touched.

use std::time::Instant;

use pivot_abi::RationalPQ;
use wukong_zeno::fnv1a64;

/// Number of lanes in the wave state.
const LANES: usize = 16;

/// Forward rounds. Each round applies the four gates once.
const ROUNDS: usize = 250_000;

/// Gate applications counted by `ns_per_gate`: `ROUNDS * 4` forward plus the
/// same number of inverse applications.
const TOTAL_APPLICATIONS: u128 = (ROUNDS as u128) * 8;

/// Lane pairs touched by the four non-commuting gates.
const PAIRS: [(usize, usize); 4] = [(5, 13), (9, 1), (2, 7), (3, 12)];

/// Cayley parameters `(r, s)` for the four gates.
const GATE_PARAMS: [(i32, i32); 4] = [(1, 2), (2, 3), (1, 3), (3, 4)];

/// Reverse order: undo the last forward gate first.
const REVERSE_ORDER: [usize; 4] = [3, 2, 1, 0];

/// Element of the prime field `P = 2^31 - 1`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct Fp31(u32);

#[allow(clippy::should_implement_trait)]
impl Fp31 {
    /// The Mersenne prime `0x7FFF_FFFF`.
    const P: u32 = 0x7FFF_FFFF;

    /// Additive identity.
    const ZERO: Self = Self(0);

    /// Multiplicative identity.
    const ONE: Self = Self(1);

    /// Branchless double Mersenne fold: no `%`, no division.
    const fn mul(self, rhs: Self) -> Self {
        let prod = (self.0 as u64) * (rhs.0 as u64);
        let p = Self::P as u64;
        let s1 = (prod & p) + (prod >> 31);
        let s2 = ((s1 & p) + (s1 >> 31)) as u32;
        Self(if s2 >= Self::P { s2 - Self::P } else { s2 })
    }

    /// Modular addition; both operands are already reduced.
    const fn add(self, rhs: Self) -> Self {
        let sum = self.0 + rhs.0;
        Self(if sum >= Self::P { sum - Self::P } else { sum })
    }

    /// Modular subtraction; both operands are already reduced.
    const fn sub(self, rhs: Self) -> Self {
        Self(if self.0 >= rhs.0 {
            self.0 - rhs.0
        } else {
            self.0 + Self::P - rhs.0
        })
    }

    /// Square-and-multiply exponentiation.
    fn pow(self, mut exponent: u32) -> Self {
        let mut base = self;
        let mut result = Self::ONE;
        while exponent > 0 {
            if exponent & 1 == 1 {
                result = result.mul(base);
            }
            base = base.mul(base);
            exponent >>= 1;
        }
        result
    }

    /// Fermat inverse `a^(P-2)`; maps zero to zero (never used for a zero
    /// denominator because projection rejects those first).
    fn inv(self) -> Self {
        self.pow(Self::P - 2)
    }

    /// Reduce a signed integer into the field.
    fn from_i32(value: i32) -> Self {
        Self(value.rem_euclid(Self::P as i32) as u32)
    }
}

/// Project `p/q` into `Fp31` as `p * q^(P-2) mod P`.
///
/// A denominator divisible by `P` is never silently inverted: the panic names
/// the lane that produced it.
fn to_fp31(value: RationalPQ, lane: usize) -> Fp31 {
    let modulus = u64::from(Fp31::P);
    let den = value.den.get();
    if den % modulus == 0 {
        panic!("fp31 seed lane {lane}: denominator {den} is divisible by P");
    }
    let num_mod = Fp31(value.num.rem_euclid(i64::from(Fp31::P)) as u32);
    let den_mod = Fp31(u32::try_from(den % modulus).expect("denominator residue fits in u32"));
    num_mod.mul(den_mod.inv())
}

/// Cayley rotor `U = [[c, -sigma], [sigma, c]]` over `Fp31`.
#[derive(Clone, Copy)]
struct CayleyGateFp31 {
    c: Fp31,
    sigma: Fp31,
}

impl CayleyGateFp31 {
    /// Build `c = (s^2 - r^2)/(s^2 + r^2)` and `sigma = 2rs/(s^2 + r^2)`.
    fn from_rs(r: i32, s: i32) -> Self {
        let r = i64::from(r);
        let s = i64::from(s);
        let rr = r * r;
        let ss = s * s;
        let denominator = Fp31::from_i32(
            i32::try_from(ss + rr).expect("Cayley denominator fits in i32"),
        );
        debug_assert!(
            denominator != Fp31::ZERO,
            "Cayley denominator must be nonzero"
        );
        let inverse = denominator.inv();
        let cosine = Fp31::from_i32(
            i32::try_from(ss - rr).expect("Cayley cosine numerator fits in i32"),
        )
        .mul(inverse);
        let sine = Fp31::from_i32(
            i32::try_from(2 * r * s).expect("Cayley sine numerator fits in i32"),
        )
        .mul(inverse);
        Self {
            c: cosine,
            sigma: sine,
        }
    }

    /// Forward rotation `U`: `psi[u] = c*pu - sigma*pv`, `psi[v] = sigma*pu + c*pv`.
    fn apply_forward(&self, psi: &mut [Fp31; LANES], u: usize, v: usize) {
        let pu = psi[u];
        let pv = psi[v];
        psi[u] = self.c.mul(pu).sub(self.sigma.mul(pv));
        psi[v] = self.sigma.mul(pu).add(self.c.mul(pv));
    }

    /// Transposed rotation `Uᵀ`: `psi[u] = c*pu + sigma*pv`, `psi[v] = c*pv - sigma*pu`.
    fn apply_inverse(&self, psi: &mut [Fp31; LANES], u: usize, v: usize) {
        let pu = psi[u];
        let pv = psi[v];
        psi[u] = self.c.mul(pu).add(self.sigma.mul(pv));
        psi[v] = self.c.mul(pv).sub(self.sigma.mul(pu));
    }
}

/// 𝔽ₚ L2 energy `sum psi[i]^2`.
fn energy(psi: &[Fp31; LANES]) -> Fp31 {
    let mut total = Fp31::ZERO;
    for &value in psi.iter() {
        total = total.add(value.mul(value));
    }
    total
}

/// Deterministic seed rationals `(3*lane - 10)/8`.
fn seed_rationals() -> [RationalPQ; LANES] {
    let mut seeds = [RationalPQ::ZERO; LANES];
    for (lane, slot) in seeds.iter_mut().enumerate() {
        let raw = (lane as i128) * 3 - 10;
        *slot = RationalPQ::from_ratio(raw, 8).expect("seed lane fits the ABI");
    }
    seeds
}

/// Forward `f32` rotation; coefficients are precomputed outside the timer.
fn apply_f32(psi: &mut [f32; LANES], u: usize, v: usize, c: f32, sigma: f32) {
    let pu = psi[u];
    let pv = psi[v];
    psi[u] = c * pu - sigma * pv;
    psi[v] = sigma * pu + c * pv;
}

/// Transposed `f32` rotation.
fn apply_inverse_f32(psi: &mut [f32; LANES], u: usize, v: usize, c: f32, sigma: f32) {
    let pu = psi[u];
    let pv = psi[v];
    psi[u] = c * pu + sigma * pv;
    psi[v] = c * pv - sigma * pu;
}

/// Forward `f64` rotation; coefficients are precomputed outside the timer.
fn apply_f64(psi: &mut [f64; LANES], u: usize, v: usize, c: f64, sigma: f64) {
    let pu = psi[u];
    let pv = psi[v];
    psi[u] = c * pu - sigma * pv;
    psi[v] = sigma * pu + c * pv;
}

/// Transposed `f64` rotation.
fn apply_inverse_f64(psi: &mut [f64; LANES], u: usize, v: usize, c: f64, sigma: f64) {
    let pu = psi[u];
    let pv = psi[v];
    psi[u] = c * pu + sigma * pv;
    psi[v] = c * pv - sigma * pu;
}

fn energy_f32(psi: &[f32; LANES]) -> f32 {
    psi.iter().map(|value| value * value).sum()
}

fn energy_f64(psi: &[f64; LANES]) -> f64 {
    psi.iter().map(|value| value * value).sum()
}

fn nanos_per_gate(forward: std::time::Duration, reverse: std::time::Duration) -> u64 {
    let total = forward.as_nanos() + reverse.as_nanos();
    u64::try_from(total / TOTAL_APPLICATIONS).unwrap_or(u64::MAX)
}

fn main() {
    let seed_q = seed_rationals();
    let seed_state = {
        let mut state = [Fp31::ZERO; LANES];
        for (lane, slot) in state.iter_mut().enumerate() {
            // Projection happens once, before any timer starts.
            *slot = to_fp31(seed_q[lane], lane);
        }
        state
    };

    // The four gates are built once, also outside the timed region.
    let gates = [
        CayleyGateFp31::from_rs(GATE_PARAMS[0].0, GATE_PARAMS[0].1),
        CayleyGateFp31::from_rs(GATE_PARAMS[1].0, GATE_PARAMS[1].1),
        CayleyGateFp31::from_rs(GATE_PARAMS[2].0, GATE_PARAMS[2].1),
        CayleyGateFp31::from_rs(GATE_PARAMS[3].0, GATE_PARAMS[3].1),
    ];

    let initial_energy = energy(&seed_state);
    let mut psi = seed_state;

    let forward_start = Instant::now();
    for _ in 0..ROUNDS {
        for (gate, &(u, v)) in gates.iter().zip(PAIRS.iter()) {
            gate.apply_forward(&mut psi, u, v);
        }
    }
    let forward_elapsed = forward_start.elapsed();
    let midpoint_energy = energy(&psi);

    let reverse_start = Instant::now();
    for _ in 0..ROUNDS {
        for &index in REVERSE_ORDER.iter() {
            gates[index].apply_inverse(&mut psi, PAIRS[index].0, PAIRS[index].1);
        }
    }
    let reverse_elapsed = reverse_start.elapsed();

    let final_energy = energy(&psi);
    assert_eq!(
        midpoint_energy, initial_energy,
        "fp31 midpoint L2 energy drifted"
    );
    assert_eq!(
        final_energy, initial_energy,
        "fp31 final L2 energy drifted"
    );

    let mut reversal_errors = 0usize;
    for (value, seed) in psi.iter().zip(seed_state.iter()) {
        if value != seed {
            reversal_errors += 1;
        }
    }
    assert_eq!(
        reversal_errors, 0,
        "fp31 reverse pass must restore every lane bit-for-bit"
    );

    let energy_equal = midpoint_energy == initial_energy && final_energy == initial_energy;
    let ns_fp31 = nanos_per_gate(forward_elapsed, reverse_elapsed);

    // ------------------------------------------------------------ f32 baseline
    let gates32 = GATE_PARAMS.map(|(r, s)| {
        let rr = (r * r) as f32;
        let ss = (s * s) as f32;
        let denominator = rr + ss;
        ((ss - rr) / denominator, (2 * r * s) as f32 / denominator)
    });
    let mut psi32 = [0.0f32; LANES];
    for (lane, slot) in psi32.iter_mut().enumerate() {
        *slot = seed_q[lane].num as f32 / seed_q[lane].den.get() as f32;
    }
    let seed32 = psi32;
    let initial32 = energy_f32(&psi32);
    let forward_start32 = Instant::now();
    for _ in 0..ROUNDS {
        for (index, &(u, v)) in PAIRS.iter().enumerate() {
            let (c, sigma) = gates32[index];
            apply_f32(&mut psi32, u, v, c, sigma);
        }
    }
    let forward_elapsed32 = forward_start32.elapsed();
    let midpoint32 = energy_f32(&psi32);
    let reverse_start32 = Instant::now();
    for _ in 0..ROUNDS {
        for &index in REVERSE_ORDER.iter() {
            let (c, sigma) = gates32[index];
            apply_inverse_f32(&mut psi32, PAIRS[index].0, PAIRS[index].1, c, sigma);
        }
    }
    let reverse_elapsed32 = reverse_start32.elapsed();
    let drift32 = (midpoint32 - initial32).abs() / initial32;
    let mut max_delta32 = 0.0f32;
    for (lane, &value) in psi32.iter().enumerate() {
        let delta = (value - seed32[lane]).abs();
        if delta > max_delta32 {
            max_delta32 = delta;
        }
    }
    let ns_f32 = (forward_elapsed32.as_nanos() + reverse_elapsed32.as_nanos()) as f64
        / TOTAL_APPLICATIONS as f64;

    // ------------------------------------------------------------ f64 baseline
    let gates64 = GATE_PARAMS.map(|(r, s)| {
        let rr = (r * r) as f64;
        let ss = (s * s) as f64;
        let denominator = rr + ss;
        ((ss - rr) / denominator, (2 * r * s) as f64 / denominator)
    });
    let mut psi64 = [0.0f64; LANES];
    for (lane, slot) in psi64.iter_mut().enumerate() {
        *slot = seed_q[lane].num as f64 / seed_q[lane].den.get() as f64;
    }
    let seed64 = psi64;
    let initial64 = energy_f64(&psi64);
    let forward_start64 = Instant::now();
    for _ in 0..ROUNDS {
        for (index, &(u, v)) in PAIRS.iter().enumerate() {
            let (c, sigma) = gates64[index];
            apply_f64(&mut psi64, u, v, c, sigma);
        }
    }
    let forward_elapsed64 = forward_start64.elapsed();
    let midpoint64 = energy_f64(&psi64);
    let reverse_start64 = Instant::now();
    for _ in 0..ROUNDS {
        for &index in REVERSE_ORDER.iter() {
            let (c, sigma) = gates64[index];
            apply_inverse_f64(&mut psi64, PAIRS[index].0, PAIRS[index].1, c, sigma);
        }
    }
    let reverse_elapsed64 = reverse_start64.elapsed();
    let drift64 = (midpoint64 - initial64).abs() / initial64;
    let mut max_delta64 = 0.0f64;
    for (lane, &value) in psi64.iter().enumerate() {
        let delta = (value - seed64[lane]).abs();
        if delta > max_delta64 {
            max_delta64 = delta;
        }
    }
    let ns_f64 = (forward_elapsed64.as_nanos() + reverse_elapsed64.as_nanos()) as f64
        / TOTAL_APPLICATIONS as f64;

    // ---------------------------------------------------------------- digest
    // Fingerprint the restored state plus the three exact energy samples.
    let mut digest_bytes = Vec::with_capacity(LANES * 4 + 12);
    for &lane in psi.iter() {
        digest_bytes.extend_from_slice(&lane.0.to_le_bytes());
    }
    digest_bytes.extend_from_slice(&initial_energy.0.to_le_bytes());
    digest_bytes.extend_from_slice(&midpoint_energy.0.to_le_bytes());
    digest_bytes.extend_from_slice(&final_energy.0.to_le_bytes());
    let digest = fnv1a64(&digest_bytes);

    // ---------------------------------------------------------------- output
    println!(
        "fp31          : gates=2000000 ns_per_gate={ns_fp31} energy_equal={energy_equal} reversal_errors={reversal_errors}"
    );
    println!("fp31 digest   : 0x{digest:016X}");
    println!(
        "f32           : ns_per_gate={ns_f32:.3} energy_drift={drift32:.3e} max_delta={max_delta32:.3e}"
    );
    println!(
        "f64           : ns_per_gate={ns_f64:.3} energy_drift={drift64:.3e} max_delta={max_delta64:.3e}"
    );
}
