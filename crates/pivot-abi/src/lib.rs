#![no_std]
#![forbid(unsafe_code)]
#![deny(clippy::float_arithmetic)]

//! Frozen central mediator ABI for the `wukong-zeno` workspace.
//!
//! This crate contains only data pods, exact arithmetic, and boundary traits.
//! It has no dependencies, no allocation, no `std`, and no `unsafe`. The four
//! founder crates depend on this ABI and never on each other.

use core::cmp::Ordering;
use core::fmt;
use core::num::{NonZeroU32, NonZeroU64};

/// Greatest common divisor for unsigned 128-bit integers.
const fn gcd_u128(mut a: u128, mut b: u128) -> u128 {
    while b != 0 {
        let remainder = a % b;
        a = b;
        b = remainder;
    }
    a
}

/// Exact reduced rational `p/q` with a positive denominator.
///
/// The fields are public to keep the ABI a plain data pod, but construct values
/// through [`RationalPQ::new`] or [`RationalPQ::from_ratio`] so the fraction is
/// reduced. Equality and ordering use `i128` cross-multiplication, so even a
/// manually built unreduced value compares correctly.
#[derive(Clone, Copy, Debug)]
pub struct RationalPQ {
    /// Reduced numerator.
    pub num: i64,
    /// Reduced, strictly positive denominator.
    pub den: NonZeroU64,
}

impl RationalPQ {
    /// Zero, canonically `0/1`.
    pub const ZERO: Self = Self {
        num: 0,
        den: NonZeroU64::MIN,
    };

    /// One, canonically `1/1`.
    pub const ONE: Self = Self {
        num: 1,
        den: NonZeroU64::MIN,
    };

    /// Build a reduced rational, returning `None` for a zero denominator.
    #[must_use]
    pub const fn new(num: i64, den: u64) -> Option<Self> {
        if den == 0 {
            return None;
        }
        if num == 0 {
            return Some(Self::ZERO);
        }
        let divisor = gcd_u128(num.unsigned_abs() as u128, den as u128);
        let reduced_num = (num as i128 / divisor as i128) as i64;
        let reduced_den = (den as u128 / divisor) as u64;
        match NonZeroU64::new(reduced_den) {
            Some(den) => Some(Self {
                num: reduced_num,
                den,
            }),
            None => None,
        }
    }

    /// Build a reduced rational from an integer numerator.
    #[must_use]
    pub const fn from_integer(value: i64) -> Self {
        Self {
            num: value,
            den: NonZeroU64::MIN,
        }
    }

    /// Build a reduced rational from wide intermediates.
    #[must_use]
    pub fn from_ratio(num: i128, den: u128) -> Option<Self> {
        if den == 0 {
            return None;
        }
        if num == 0 {
            return Some(Self::ZERO);
        }
        let divisor = gcd_u128(num.unsigned_abs(), den);
        if divisor == 0 {
            return None;
        }
        let reduced_num = num / divisor as i128;
        let reduced_den = den / divisor;
        if reduced_num < i64::MIN as i128 || reduced_num > i64::MAX as i128 {
            return None;
        }
        if reduced_den > u64::MAX as u128 {
            return None;
        }
        let den = NonZeroU64::new(reduced_den as u64)?;
        Some(Self {
            num: reduced_num as i64,
            den,
        })
    }

    /// Numerator.
    #[must_use]
    pub const fn num(self) -> i64 {
        self.num
    }

    /// Denominator (always positive).
    #[must_use]
    pub const fn den(self) -> u64 {
        self.den.get()
    }

    /// True when the value is exactly zero.
    #[must_use]
    pub const fn is_zero(self) -> bool {
        self.num == 0
    }

    /// True when the value is strictly positive.
    #[must_use]
    pub const fn is_positive(self) -> bool {
        self.num > 0
    }

    /// True when the value is strictly negative.
    #[must_use]
    pub const fn is_negative(self) -> bool {
        self.num < 0
    }

    /// Absolute value, or `None` for `i64::MIN`.
    #[must_use]
    pub const fn checked_abs(self) -> Option<Self> {
        match self.num.checked_abs() {
            Some(num) => Some(Self { num, den: self.den }),
            None => None,
        }
    }

    /// Negation, or `None` for `i64::MIN`.
    #[must_use]
    pub const fn checked_neg(self) -> Option<Self> {
        match self.num.checked_neg() {
            Some(num) => Some(Self { num, den: self.den }),
            None => None,
        }
    }

    /// Exact addition with overflow detection.
    #[must_use]
    pub fn checked_add(self, other: Self) -> Option<Self> {
        let left = (self.num as i128).checked_mul(other.den.get() as i128)?;
        let right = (other.num as i128).checked_mul(self.den.get() as i128)?;
        let num = left.checked_add(right)?;
        let den = (self.den.get() as u128).checked_mul(other.den.get() as u128)?;
        Self::from_ratio(num, den)
    }

    /// Exact subtraction with overflow detection.
    #[must_use]
    pub fn checked_sub(self, other: Self) -> Option<Self> {
        self.checked_add(other.checked_neg()?)
    }

    /// Exact multiplication with overflow detection.
    #[must_use]
    pub fn checked_mul(self, other: Self) -> Option<Self> {
        let num = (self.num as i128).checked_mul(other.num as i128)?;
        let den = (self.den.get() as u128).checked_mul(other.den.get() as u128)?;
        Self::from_ratio(num, den)
    }

    /// Exact division with overflow detection; dividing by zero yields `None`.
    #[must_use]
    pub fn checked_div(self, other: Self) -> Option<Self> {
        if other.num == 0 {
            return None;
        }
        let mut num = (self.num as i128).checked_mul(other.den.get() as i128)?;
        let den = (self.den.get() as u128).checked_mul(other.num.unsigned_abs() as u128)?;
        if other.num < 0 {
            num = num.checked_neg()?;
        }
        Self::from_ratio(num, den)
    }

    /// Total comparison using `i128` cross-multiplication.
    #[must_use]
    pub const fn cmp_rational(&self, other: &Self) -> Ordering {
        let left = self.num as i128 * other.den.get() as i128;
        let right = other.num as i128 * self.den.get() as i128;
        if left < right {
            Ordering::Less
        } else if left > right {
            Ordering::Greater
        } else {
            Ordering::Equal
        }
    }
}

impl PartialEq for RationalPQ {
    fn eq(&self, other: &Self) -> bool {
        self.cmp_rational(other) == Ordering::Equal
    }
}

impl Eq for RationalPQ {}

impl PartialOrd for RationalPQ {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(core::cmp::Ord::cmp(self, other))
    }
}

impl Ord for RationalPQ {
    fn cmp(&self, other: &Self) -> Ordering {
        self.cmp_rational(other)
    }
}

impl fmt::Display for RationalPQ {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.den.get() == 1 {
            write!(formatter, "{}", self.num)
        } else {
            write!(formatter, "{}/{}", self.num, self.den.get())
        }
    }
}

/// Compact exact rational used inside a 32-byte kompress frame.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CompactPQ {
    /// Reduced numerator.
    pub num: i32,
    /// Reduced, strictly positive denominator.
    pub den: NonZeroU32,
}

impl CompactPQ {
    /// Zero, canonically `0/1`.
    pub const ZERO: Self = Self {
        num: 0,
        den: NonZeroU32::MIN,
    };

    /// One, canonically `1/1`.
    pub const ONE: Self = Self {
        num: 1,
        den: NonZeroU32::MIN,
    };

    /// Build a reduced compact rational, returning `None` for a zero denominator.
    #[must_use]
    pub const fn new(num: i32, den: u32) -> Option<Self> {
        if den == 0 {
            return None;
        }
        if num == 0 {
            return Some(Self::ZERO);
        }
        let divisor = gcd_u128(num.unsigned_abs() as u128, den as u128);
        let reduced_num = (num as i128 / divisor as i128) as i32;
        let reduced_den = (den as u128 / divisor) as u32;
        match NonZeroU32::new(reduced_den) {
            Some(den) => Some(Self {
                num: reduced_num,
                den,
            }),
            None => None,
        }
    }

    /// Numerator.
    #[must_use]
    pub const fn num(self) -> i32 {
        self.num
    }

    /// Denominator (always positive).
    #[must_use]
    pub const fn den(self) -> u32 {
        self.den.get()
    }

    /// Widen to the canonical workspace rational.
    #[must_use]
    pub const fn to_rational(self) -> RationalPQ {
        match NonZeroU64::new(self.den.get() as u64) {
            Some(den) => RationalPQ {
                num: self.num as i64,
                den,
            },
            None => RationalPQ::ZERO,
        }
    }

    /// Narrow an already-reduced rational when it fits the compact layout.
    #[must_use]
    pub fn from_rational(value: RationalPQ) -> Option<Self> {
        if value.num < i32::MIN as i64 || value.num > i32::MAX as i64 {
            return None;
        }
        if value.den.get() > u32::MAX as u64 {
            return None;
        }
        let den = NonZeroU32::new(value.den.get() as u32)?;
        Some(Self {
            num: value.num as i32,
            den,
        })
    }

    /// Total comparison using `i128` cross-multiplication.
    #[must_use]
    pub const fn cmp_rational(&self, other: &Self) -> Ordering {
        let left = self.num as i128 * other.den.get() as i128;
        let right = other.num as i128 * self.den.get() as i128;
        if left < right {
            Ordering::Less
        } else if left > right {
            Ordering::Greater
        } else {
            Ordering::Equal
        }
    }
}

impl PartialOrd for CompactPQ {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(core::cmp::Ord::cmp(self, other))
    }
}

impl Ord for CompactPQ {
    fn cmp(&self, other: &Self) -> Ordering {
        self.cmp_rational(other)
    }
}

/// Balanced ternary digit.
#[repr(i8)]
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub enum Trit {
    /// Negative trit, `-1`.
    Neg = -1,
    /// Neutral trit, `0`.
    Zero = 0,
    /// Positive trit, `+1`.
    Pos = 1,
}

impl Trit {
    /// Balanced negation.
    #[must_use]
    pub const fn trit_neg(self) -> Self {
        match self {
            Self::Neg => Self::Pos,
            Self::Zero => Self::Zero,
            Self::Pos => Self::Neg,
        }
    }

    /// Balanced ternary product (never overflows).
    #[must_use]
    pub const fn trit_mul(self, other: Self) -> Self {
        let product = self as i8 * other as i8;
        match product {
            -1 => Self::Neg,
            1 => Self::Pos,
            _ => Self::Zero,
        }
    }

    /// Agreement consensus: equal trits survive, disagreement becomes zero.
    #[must_use]
    pub const fn trit_consensus(self, other: Self) -> Self {
        if self as i8 == other as i8 {
            self
        } else {
            Self::Zero
        }
    }

    /// Convert a signed byte in `-1..=1`.
    #[must_use]
    pub const fn from_i8(value: i8) -> Option<Self> {
        match value {
            -1 => Some(Self::Neg),
            0 => Some(Self::Zero),
            1 => Some(Self::Pos),
            _ => None,
        }
    }

    /// Signed byte representation.
    #[must_use]
    pub const fn to_i8(self) -> i8 {
        self as i8
    }
}

/// Errors from bit-plane access.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AbiError {
    /// Lane index was `>= 64`.
    LaneOutOfRange,
    /// The plane violates `pos & neg == 0`.
    NonCanonicalPlane,
}

/// 64-lane balanced-ternary bit plane with invariant `pos & neg == 0`.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct TritPlane64 {
    /// Lanes holding `+1`.
    pub pos: u64,
    /// Lanes holding `-1`.
    pub neg: u64,
}

impl TritPlane64 {
    /// Empty plane.
    pub const ZERO: Self = Self { pos: 0, neg: 0 };

    /// Build a plane only when the two masks are disjoint.
    #[must_use]
    pub const fn new(pos: u64, neg: u64) -> Option<Self> {
        if pos & neg == 0 {
            Some(Self { pos, neg })
        } else {
            None
        }
    }

    /// True when `pos & neg == 0`.
    #[must_use]
    pub const fn is_canonical(self) -> bool {
        self.pos & self.neg == 0
    }

    /// Write one lane.
    pub fn set(&mut self, lane: usize, trit: Trit) -> Result<(), AbiError> {
        if lane >= 64 {
            return Err(AbiError::LaneOutOfRange);
        }
        let bit = 1u64 << lane;
        self.pos &= !bit;
        self.neg &= !bit;
        match trit {
            Trit::Pos => self.pos |= bit,
            Trit::Neg => self.neg |= bit,
            Trit::Zero => {}
        }
        Ok(())
    }

    /// Read one lane.
    pub const fn get(&self, lane: usize) -> Result<Trit, AbiError> {
        if lane >= 64 {
            return Err(AbiError::LaneOutOfRange);
        }
        let bit = 1u64 << lane;
        if self.pos & bit != 0 {
            Ok(Trit::Pos)
        } else if self.neg & bit != 0 {
            Ok(Trit::Neg)
        } else {
            Ok(Trit::Zero)
        }
    }

    /// Number of `+1` lanes.
    #[must_use]
    pub const fn count_pos(self) -> u32 {
        self.pos.count_ones()
    }

    /// Number of `-1` lanes.
    #[must_use]
    pub const fn count_neg(self) -> u32 {
        self.neg.count_ones()
    }

    /// Number of non-zero lanes.
    #[must_use]
    pub const fn count_nonzero(self) -> u32 {
        self.pos.count_ones() + self.neg.count_ones()
    }
}

/// Five balanced-ternary trits packed into one byte (`3^5 = 243`).
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct PackedTrit5(pub u8);

impl PackedTrit5 {
    /// Canonical all-zero byte: digits `[0, 0, 0, 0, 0]` encode `1+3+9+27+81`.
    pub const ZERO: Self = Self::pack5([Trit::Zero; 5]);

    /// Largest canonical payload byte.
    pub const MAX: u8 = 242;

    /// Pack exactly five trits, least significant first.
    #[must_use]
    pub const fn pack5(digits: [Trit; 5]) -> Self {
        let mut value: u16 = 0;
        let mut scale: u16 = 1;
        let mut index = 0;
        while index < 5 {
            value += trit_digit(digits[index]) as u16 * scale;
            scale *= 3;
            index += 1;
        }
        Self(value as u8)
    }

    /// Unpack five trits, least significant first.
    #[must_use]
    pub const fn unpack5(self) -> [Trit; 5] {
        let mut digits = [Trit::Zero; 5];
        let mut value = self.0 as u16;
        let mut index = 0;
        while index < 5 {
            digits[index] = match value % 3 {
                0 => Trit::Neg,
                1 => Trit::Zero,
                _ => Trit::Pos,
            };
            value /= 3;
            index += 1;
        }
        digits
    }

    /// True when the byte is a valid base-3 payload (`0..=242`).
    #[must_use]
    pub const fn is_canonical(self) -> bool {
        self.0 <= Self::MAX
    }
}

const fn trit_digit(trit: Trit) -> u8 {
    match trit {
        Trit::Neg => 0,
        Trit::Zero => 1,
        Trit::Pos => 2,
    }
}

/// 64 signed 16-bit centroid lanes, one full cache line pair.
#[repr(C, align(64))]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CentroidPod64(pub [i16; 64]);

impl CentroidPod64 {
    /// All-zero pod.
    pub const ZERO: Self = Self([0; 64]);

    /// Read one lane.
    #[must_use]
    pub const fn get(&self, lane: usize) -> Option<i16> {
        if lane < 64 {
            Some(self.0[lane])
        } else {
            None
        }
    }

    /// Write one lane.
    pub fn set(&mut self, lane: usize, value: i16) -> Result<(), AbiError> {
        if lane >= 64 {
            return Err(AbiError::LaneOutOfRange);
        }
        self.0[lane] = value;
        Ok(())
    }

    /// Greatest absolute lane value, as a non-negative `i64`.
    #[must_use]
    pub fn max_abs(&self) -> i64 {
        let mut maximum = 0i64;
        let mut index = 0;
        while index < 64 {
            let value = self.0[index] as i64;
            let absolute = if value < 0 { -value } else { value };
            if absolute > maximum {
                maximum = absolute;
            }
            index += 1;
        }
        maximum
    }

    /// Sum of absolute lane values.
    #[must_use]
    pub fn l1_abs_sum(&self) -> i64 {
        let mut sum = 0i64;
        let mut index = 0;
        while index < 64 {
            let value = self.0[index] as i64;
            sum += if value < 0 { -value } else { value };
            index += 1;
        }
        sum
    }
}

/// Canonical magic written into the first three bytes of `insert4_hdr`.
pub const INSERT4_MAGIC: [u8; 3] = *b"KSU";

/// CRC-8 polynomial (`x^8 + x^2 + x + 1`) used by the insert-4 stage.
pub const CRC8_POLY: u8 = 0x07;

/// Exactly 32 bytes: the kompress-super-ultra folded wire frame.
#[repr(C, align(32))]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct KompressFrame32 {
    /// Sign-preserving numerator of the L1 mean-magnitude scale.
    pub scale_num: i32,
    /// Denominator of the L1 mean-magnitude scale.
    pub scale_den: NonZeroU32,
    /// Monotonic producer epoch tag.
    pub epoch_tag: u32,
    /// `[b'K', b'S', b'U', crc8]`.
    pub insert4_hdr: [u8; 4],
    /// Remaining qULB budget carried with the frame.
    pub qulb_budget: u16,
    /// Number of lanes that survived `sphere_pop` (`0..=64`).
    pub active_lanes: u8,
    /// 65 trits (64 lanes plus one zero pad) at five trits per byte.
    pub packed_trits: [u8; 13],
}

impl KompressFrame32 {
    /// Empty frame for the given epoch and budget.
    #[must_use]
    pub const fn empty(epoch_tag: u32, qulb_budget: u16) -> Self {
        Self {
            scale_num: 0,
            scale_den: NonZeroU32::MIN,
            epoch_tag,
            insert4_hdr: [0, 0, 0, 0],
            qulb_budget,
            active_lanes: 0,
            packed_trits: [PackedTrit5::ZERO.0; 13],
        }
    }
}

/// Errors produced while unpacking a kompress frame.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FrameError {
    /// `insert4_hdr[0..3]` was not `KSU`.
    BadMagic,
    /// CRC-8 did not match the stored byte.
    CrcMismatch {
        /// CRC stored in the frame.
        expected: u8,
        /// CRC recomputed over the payload.
        actual: u8,
    },
    /// A packed byte was greater than 242.
    InvalidPackedTrit {
        /// Offending payload byte.
        byte: u8,
    },
    /// The 65th (pad) trit was not zero.
    NonZeroPad,
    /// `active_lanes` exceeded 64.
    InvalidActiveLanes {
        /// Stored count.
        active: u8,
    },
    /// Scale numerator was negative or otherwise unusable.
    InvalidScale,
}

/// Errors produced by the sparse Laplacian oracle.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OracleError {
    /// `offsets` was not exactly `vertices + 1` entries.
    InvalidOffsets,
    /// `targets` and `weights` had different lengths.
    LengthMismatch,
    /// A CSR range was inverted or out of bounds.
    InvalidCsr,
    /// A target vertex index was out of bounds.
    TargetOutOfRange,
    /// Checked rational arithmetic overflowed.
    Overflow,
    /// A partition count of zero was requested.
    EmptyPartition,
}

/// Errors produced by the qULB capability loop.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum QulbError {
    /// Decay factor was not inside `(0, 1)`.
    InvalidDecay,
    /// Budget or authority was already exhausted.
    Exhausted,
    /// Checked rational arithmetic overflowed.
    Overflow,
}

/// Static balanced-ternary SIMD kernel boundary.
pub trait TernarySimdPort {
    /// Zero-multiply balanced ternary dot product over two bit planes.
    fn dot64_trit_plane(a: TritPlane64, b: TritPlane64) -> i32;

    /// Contract ternary weights against signed activations without multiplication.
    fn contract64_trit_i16(weights: TritPlane64, act: &CentroidPod64) -> i32;

    /// Widening 64-lane signed inner product, saturating at the `i32` bounds.
    fn dot64_i16(a: &CentroidPod64, b: &CentroidPod64) -> i32;

    /// Rank-1 query over a 128-bit LOUDS bit pair; `idx` is in `0..=128`.
    fn louds_rank1(bits: &[u64; 2], idx: usize) -> u32;

    /// Four deterministic SplitMix64 lanes, advanced in place.
    fn splitmix64_x4(state: &mut [u64; 4]) -> [u64; 4];
}

/// `kompress-super-ultra` fold/unfold boundary.
pub trait KompressPort {
    /// Fold a 128-byte pod into a 32-byte frame.
    fn fold(pod: &CentroidPod64, threshold: RationalPQ, epoch: u32, budget: u16)
        -> KompressFrame32;

    /// Reconstruct an approximate pod after structural and CRC validation.
    fn unfold(frame: &KompressFrame32) -> Result<CentroidPod64, FrameError>;
}

/// Exact-rational ternary Laplacian boundary.
pub trait LaplacianOraclePort {
    /// Apply `L_3 = B W B^T` to `x`, accumulating into a caller-zeroed `out`.
    ///
    /// `offsets` is CSR source offsets (`vertices + 1`), `targets[e]` is the
    /// destination of directed edge `e`, and `weights[e]` is its exact weight.
    fn apply_ternary_laplacian(
        x: &[RationalPQ],
        offsets: &[u32],
        targets: &[u32],
        weights: &[RationalPQ],
        out: &mut [RationalPQ],
    ) -> Result<(), OracleError>;
}

/// Linear capability capsule boundary used by the qULB loop.
pub trait QulbCapsulePort {
    /// Current authority ceiling.
    fn authority(&self) -> RationalPQ;
    /// Remaining deterministic step budget.
    fn budget(&self) -> u16;
    /// Multiply authority by `decay` and return the new ceiling.
    fn attenuate(&mut self, decay: RationalPQ) -> Result<RationalPQ, QulbError>;
}

const _: () = {
    assert!(core::mem::size_of::<CentroidPod64>() == 128);
    assert!(core::mem::align_of::<CentroidPod64>() == 64);
    assert!(core::mem::size_of::<KompressFrame32>() == 32);
    assert!(core::mem::align_of::<KompressFrame32>() == 32);
    assert!(core::mem::size_of::<PackedTrit5>() == 1);
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rational_reduces_and_orders() {
        let half = RationalPQ::new(2, 4).unwrap();
        assert_eq!(half.num, 1);
        assert_eq!(half.den.get(), 2);
        let third = RationalPQ::new(1, 3).unwrap();
        assert!(half > third);
        assert_eq!(
            half.checked_add(third).unwrap(),
            RationalPQ::new(5, 6).unwrap()
        );
        assert_eq!(
            half.checked_mul(third).unwrap(),
            RationalPQ::new(1, 6).unwrap()
        );
        assert_eq!(
            half.checked_div(third).unwrap(),
            RationalPQ::new(3, 2).unwrap()
        );
        assert!(RationalPQ::new(1, 0).is_none());
        assert!(half.checked_div(RationalPQ::ZERO).is_none());
    }

    #[test]
    fn rational_handles_negative_denominators_as_positive_storage() {
        let value = RationalPQ::new(-6, 4).unwrap();
        assert_eq!(value.num, -3);
        assert_eq!(value.den.get(), 2);
        assert_eq!(value.checked_neg().unwrap(), RationalPQ::new(3, 2).unwrap());
    }

    #[test]
    fn compact_round_trip() {
        let value = RationalPQ::new(9, 8).unwrap();
        let compact = CompactPQ::from_rational(value).unwrap();
        assert_eq!(compact.to_rational(), value);
        assert_eq!(compact, CompactPQ::new(18, 16).unwrap());
    }

    #[test]
    fn trit_algebra() {
        assert_eq!(Trit::Neg.trit_neg(), Trit::Pos);
        assert_eq!(Trit::Neg.trit_mul(Trit::Neg), Trit::Pos);
        assert_eq!(Trit::Zero.trit_mul(Trit::Neg), Trit::Zero);
        assert_eq!(Trit::Pos.trit_consensus(Trit::Pos), Trit::Pos);
        assert_eq!(Trit::Pos.trit_consensus(Trit::Neg), Trit::Zero);
        assert_eq!(Trit::from_i8(2), None);
    }

    #[test]
    fn trit_plane_access() {
        let mut plane = TritPlane64::new(0, 0).unwrap();
        plane.set(0, Trit::Pos).unwrap();
        plane.set(63, Trit::Neg).unwrap();
        assert_eq!(plane.get(0).unwrap(), Trit::Pos);
        assert_eq!(plane.get(63).unwrap(), Trit::Neg);
        assert_eq!(plane.get(1).unwrap(), Trit::Zero);
        assert!(plane.get(64).is_err());
        assert!(plane.set(64, Trit::Zero).is_err());
        assert!(plane.is_canonical());
        assert!(TritPlane64::new(1, 1).is_none());
    }

    #[test]
    fn packed_trit5_is_bijective_for_canonical_bytes() {
        for value in 0u8..=PackedTrit5::MAX {
            let packed = PackedTrit5(value);
            assert!(packed.is_canonical());
            assert_eq!(PackedTrit5::pack5(packed.unpack5()), packed);
        }
        assert!(!PackedTrit5(243).is_canonical());
        assert!(!PackedTrit5(255).is_canonical());
    }

    #[test]
    fn pod_metrics() {
        let mut pod = CentroidPod64::ZERO;
        pod.set(0, -32768).unwrap();
        pod.set(1, 7).unwrap();
        assert_eq!(pod.max_abs(), 32768);
        assert_eq!(pod.l1_abs_sum(), 32775);
        assert!(pod.set(64, 0).is_err());
    }

    #[test]
    fn frame_is_32_bytes() {
        assert_eq!(core::mem::size_of::<KompressFrame32>(), 32);
        assert_eq!(core::mem::align_of::<KompressFrame32>(), 32);
        let frame = KompressFrame32::empty(7, 9);
        assert_eq!(frame.epoch_tag, 7);
        assert_eq!(frame.qulb_budget, 9);
    }
}
