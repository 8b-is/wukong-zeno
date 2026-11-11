//! Exact-arithmetic FP31 gate benchmark with float drift comparison.

use pivot_abi::RationalPQ;
use std::time::Instant;
use wukong_zeno::fnv1a64;

const ROUNDS: usize = 250_000;
const GATE_APPLICATIONS: u128 = 2_000_000;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Fp31(u32);

#[allow(clippy::should_implement_trait)]
impl Fp31 {
    const P: u32 = 0x7FFF_FFFF;
    const ZERO: Self = Self(0);
    const ONE: Self = Self(1);

    fn add(self, rhs: Self) -> Self {
        let sum = self.0 + rhs.0;
        Self(if sum >= Self::P { sum - Self::P } else { sum })
    }

    fn sub(self, rhs: Self) -> Self {
        Self(if self.0 >= rhs.0 {
            self.0 - rhs.0
        } else {
            Self::P - (rhs.0 - self.0)
        })
    }

    fn mul(self, rhs: Self) -> Self {
        let prod = (self.0 as u64) * (rhs.0 as u64);
        let p = Self::P as u64;
        let s1 = (prod & p) + (prod >> 31);
        let s2 = ((s1 & p) + (s1 >> 31)) as u32;
        Self(if s2 >= Self::P { s2 - Self::P } else { s2 })
    }

    fn pow(self, mut e: u32) -> Self {
        let mut base = self;
        let mut acc = Self::ONE;
        while e != 0 {
            if e & 1 != 0 {
                acc = acc.mul(base);
            }
            base = base.mul(base);
            e >>= 1;
        }
        acc
    }

    fn inv(self) -> Self {
        self.pow(Self::P - 2)
    }

    fn from_i32(x: i32) -> Self {
        Self(x.rem_euclid(Self::P as i32) as u32)
    }

    fn from_rational(q: RationalPQ) -> Self {
        let n = Fp31::from_i32((q.num.rem_euclid(Fp31::P as i64)) as i32);
        let d = Fp31::from_i32((q.den.get() % Fp31::P as u64) as i32);
        assert!(d.0 != 0);
        n.mul(d.inv())
    }
}

struct Gate {
    c: Fp31,
    sigma: Fp31,
}

fn gate(r: i32, s: i32) -> Gate {
    let r = Fp31::from_i32(r);
    let s = Fp31::from_i32(s);
    let r2 = r.mul(r);
    let s2 = s.mul(s);
    let den = s2.add(r2);
    assert!(den.0 != 0);
    let inv = den.inv();
    Gate {
        c: s2.sub(r2).mul(inv),
        sigma: Fp31::from_i32(2).mul(r.mul(s)).mul(inv),
    }
}

impl Gate {
    fn apply(&self, psi: &mut [Fp31; 16], u: usize, v: usize) {
        let x = psi[u];
        let y = psi[v];
        psi[u] = self.c.mul(x).sub(self.sigma.mul(y));
        psi[v] = self.sigma.mul(x).add(self.c.mul(y));
    }

    fn apply_inv(&self, psi: &mut [Fp31; 16], u: usize, v: usize) {
        let x = psi[u];
        let y = psi[v];
        psi[u] = self.c.mul(x).add(self.sigma.mul(y));
        psi[v] = self.c.mul(y).sub(self.sigma.mul(x));
    }
}

fn energy(psi: &[Fp31; 16]) -> Fp31 {
    psi.iter()
        .fold(Fp31::ZERO, |acc, &x| acc.add(x.mul(x)))
}

fn gate_f32(r: f32, s: f32) -> (f32, f32) {
    let r2 = r * r;
    let s2 = s * s;
    let den = s2 + r2;
    ((s2 - r2) / den, (2.0 * r * s) / den)
}

fn gate_f64(r: f64, s: f64) -> (f64, f64) {
    let r2 = r * r;
    let s2 = s * s;
    let den = s2 + r2;
    ((s2 - r2) / den, (2.0 * r * s) / den)
}

fn apply_f32(g: (f32, f32), psi: &mut [f32; 16], u: usize, v: usize, inverse: bool) {
    let (c, sigma) = g;
    let x = psi[u];
    let y = psi[v];
    if inverse {
        psi[u] = c * x + sigma * y;
        psi[v] = c * y - sigma * x;
    } else {
        psi[u] = c * x - sigma * y;
        psi[v] = sigma * x + c * y;
    }
}

fn apply_f64(g: (f64, f64), psi: &mut [f64; 16], u: usize, v: usize, inverse: bool) {
    let (c, sigma) = g;
    let x = psi[u];
    let y = psi[v];
    if inverse {
        psi[u] = c * x + sigma * y;
        psi[v] = c * y - sigma * x;
    } else {
        psi[u] = c * x - sigma * y;
        psi[v] = sigma * x + c * y;
    }
}

fn energy_f32(psi: &[f32; 16]) -> f32 {
    psi.iter().fold(0.0_f32, |acc, &x| acc + x * x)
}

fn energy_f64(psi: &[f64; 16]) -> f64 {
    psi.iter().fold(0.0_f64, |acc, &x| acc + x * x)
}

fn main() {
    let raw: [i32; 16] = [
        12, -7, 5, 19, -11, 23, 8, -4, 15, -9, 6, 14, -17, 21, 3, -8,
    ];
    let seed: [Fp31; 16] = core::array::from_fn(|i| {
        Fp31::from_rational(RationalPQ::from_ratio(raw[i] as i128, 8).unwrap())
    });

    let gate_params: [(i32, i32); 4] = [(5, 13), (9, 1), (2, 7), (3, 12)];
    let pairs: [(usize, usize); 4] = [(1, 2), (2, 3), (1, 3), (3, 4)];
    let gates: [Gate; 4] =
        core::array::from_fn(|i| gate(gate_params[i].0, gate_params[i].1));

    let seed = std::hint::black_box(seed);
    let gates = std::hint::black_box(gates);

    let mut psi = seed;

    let start = Instant::now();
    for _ in 0..ROUNDS {
        for (g, &(u, v)) in gates.iter().zip(pairs.iter()) {
            g.apply(&mut psi, u, v);
        }
    }
    let mid_energy = energy(&psi);
    let _ = std::hint::black_box(mid_energy);
    for _ in 0..ROUNDS {
        for (g, &(u, v)) in gates.iter().zip(pairs.iter()).rev() {
            g.apply_inv(&mut psi, u, v);
        }
    }
    let elapsed = start.elapsed();
    let _ = std::hint::black_box(&psi);

    let energy_equal = energy(&psi) == energy(&seed);
    let reversal_errors = psi
        .iter()
        .zip(seed.iter())
        .filter(|&(a, b)| a != b)
        .count();
    assert!(energy_equal, "energy not preserved");
    assert_eq!(reversal_errors, 0, "reversal mismatches");
    assert_eq!(psi, seed, "reversal did not restore seed");

    let mut digest_bytes = Vec::with_capacity(16 * 4 + 16 * 16);
    for lane in &psi {
        digest_bytes.extend_from_slice(&lane.0.to_le_bytes());
    }
    for &value in &raw {
        let q = RationalPQ::from_ratio(value as i128, 8).unwrap();
        digest_bytes.extend_from_slice(&q.num.to_le_bytes());
        digest_bytes.extend_from_slice(&q.den.get().to_le_bytes());
    }
    let digest = fnv1a64(&digest_bytes);

    let ns_per_gate = elapsed.as_nanos() / GATE_APPLICATIONS;

    let seed_f32: [f32; 16] = core::array::from_fn(|i| raw[i] as f32 / 8.0);
    let seed_f64: [f64; 16] = core::array::from_fn(|i| raw[i] as f64 / 8.0);
    let gates_f32: [(f32, f32); 4] = core::array::from_fn(|i| {
        gate_f32(gate_params[i].0 as f32, gate_params[i].1 as f32)
    });
    let gates_f64: [(f64, f64); 4] = core::array::from_fn(|i| {
        gate_f64(gate_params[i].0 as f64, gate_params[i].1 as f64)
    });

    let seed_f32 = std::hint::black_box(seed_f32);
    let seed_f64 = std::hint::black_box(seed_f64);
    let gates_f32 = std::hint::black_box(gates_f32);
    let gates_f64 = std::hint::black_box(gates_f64);

    let mut psi_f32 = seed_f32;
    let start_f32 = Instant::now();
    for _ in 0..ROUNDS {
        for (g, &(u, v)) in gates_f32.iter().zip(pairs.iter()) {
            apply_f32(*g, &mut psi_f32, u, v, false);
        }
    }
    let e1_f32 = energy_f32(&psi_f32);
    for _ in 0..ROUNDS {
        for (g, &(u, v)) in gates_f32.iter().zip(pairs.iter()).rev() {
            apply_f32(*g, &mut psi_f32, u, v, true);
        }
    }
    let elapsed_f32 = start_f32.elapsed();
    let _ = std::hint::black_box(&psi_f32);
    let e0_f32 = energy_f32(&seed_f32);
    let energy_drift_f32 = (e1_f32 - e0_f32).abs();
    let max_delta_f32 = psi_f32
        .iter()
        .zip(seed_f32.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    let ns_per_gate_f32 = elapsed_f32.as_nanos() / GATE_APPLICATIONS;

    let mut psi_f64 = seed_f64;
    let start_f64 = Instant::now();
    for _ in 0..ROUNDS {
        for (g, &(u, v)) in gates_f64.iter().zip(pairs.iter()) {
            apply_f64(*g, &mut psi_f64, u, v, false);
        }
    }
    let e1_f64 = energy_f64(&psi_f64);
    for _ in 0..ROUNDS {
        for (g, &(u, v)) in gates_f64.iter().zip(pairs.iter()).rev() {
            apply_f64(*g, &mut psi_f64, u, v, true);
        }
    }
    let elapsed_f64 = start_f64.elapsed();
    let _ = std::hint::black_box(&psi_f64);
    let e0_f64 = energy_f64(&seed_f64);
    let energy_drift_f64 = (e1_f64 - e0_f64).abs();
    let max_delta_f64 = psi_f64
        .iter()
        .zip(seed_f64.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    let ns_per_gate_f64 = elapsed_f64.as_nanos() / GATE_APPLICATIONS;

    println!(
        "fp31          : gates=2000000 ns_per_gate={} energy_equal={} reversal_errors={}",
        ns_per_gate, energy_equal, reversal_errors
    );
    println!("fp31 digest   : 0x{:016X}", digest);
    println!(
        "f32           : ns_per_gate={} energy_drift={:e} max_delta={:e}",
        ns_per_gate_f32, energy_drift_f32, max_delta_f32
    );
    println!(
        "f64           : ns_per_gate={} energy_drift={:e} max_delta={:e}",
        ns_per_gate_f64, energy_drift_f64, max_delta_f64
    );
}
