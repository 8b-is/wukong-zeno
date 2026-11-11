#![no_std]
#![forbid(unsafe_code)]
#![deny(clippy::float_arithmetic)]

//! Linear capability capsules and the deterministic `qULB` loop for the
//! `wukong-zeno` workspace.
//!
//! [`Capsule`] models authority as a linear value: the typestate only exposes
//! the transitions `Sealed -> Active -> Attenuated`, and both `consume`
//! methods take the capsule by value, so a consumed capsule cannot be used
//! again. The capsule is deliberately not `Copy`, not `Clone`, and not
//! `Default`.
//!
//! [`UltraBackyardLoop`] is the runtime counterpart. It carries a 64-lane
//! balanced-ternary state plane, an exact strictly-positive [`RationalPQ`]
//! authority ceiling, and a deterministic `u16` step budget. Every step contracts the plane with
//! a monotone ternary mask, reports ternary kernel telemetry, strictly
//! decrements the budget, and attenuates authority by a decay factor in
//! `(0, 1)`.
//!
//! The crate is `no_std`, allocation-free, float-free, and contains no
//! `unsafe`.

#[cfg(test)]
extern crate std;

use core::fmt;
use core::marker::PhantomData;

pub use pivot_abi::{QulbCapsulePort, QulbError, RationalPQ, TernarySimdPort, TritPlane64};

/// Typestate marker: authority has been minted but the capsule is not active.
#[derive(Debug)]
pub struct Sealed;

/// Typestate marker: the capsule may attenuate or be consumed directly.
#[derive(Debug)]
pub struct Active;

/// Typestate marker: the capsule's authority has been attenuated once.
#[derive(Debug)]
pub struct Attenuated;

/// Typestate marker: the capsule has been consumed.
///
/// Consumption returns the payload directly, so no `Capsule<T, Consumed>`
/// value is ever constructed; the marker documents the terminal state of the
/// linear lifecycle.
#[derive(Debug)]
pub struct Consumed;

/// A linear capability capsule with a compile-time lifecycle.
///
/// The value moves through [`Sealed`], [`Active`], and [`Attenuated`]. Both
/// `consume` implementations take `self` by value, which makes double
/// consumption a compile-time error. The capsule implements neither `Copy` nor
/// `Clone` nor `Default`.
pub struct Capsule<T, S> {
    value: T,
    authority: RationalPQ,
    budget: u16,
    state: PhantomData<S>,
}

/// Error returned when an attenuation step fails.
///
/// The original [`Active`] capsule is returned unchanged, including its
/// payload, so a failed attenuation never silently drops a linear value.
pub struct AttenuateError<T> {
    error: QulbError,
    capsule: Capsule<T, Active>,
}

impl<T> AttenuateError<T> {
    /// The underlying qULB error.
    #[must_use]
    pub fn error(&self) -> QulbError {
        self.error
    }

    /// Borrow the capsule that failed to attenuate.
    #[must_use]
    pub fn capsule(&self) -> &Capsule<T, Active> {
        &self.capsule
    }

    /// Recover the unchanged capsule and its payload.
    #[must_use]
    pub fn into_capsule(self) -> Capsule<T, Active> {
        self.capsule
    }
}

impl<T> fmt::Debug for AttenuateError<T> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("AttenuateError")
            .field("error", &self.error)
            .finish_non_exhaustive()
    }
}

impl<T> fmt::Display for AttenuateError<T> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "attenuation failed ({:?}); the active capsule was preserved",
            self.error
        )
    }
}

impl<T, S> Capsule<T, S> {
    /// Current authority ceiling.
    #[must_use]
    pub fn authority(&self) -> RationalPQ {
        self.authority
    }

    /// Remaining deterministic step budget.
    #[must_use]
    pub fn budget(&self) -> u16 {
        self.budget
    }

    fn into_value(self) -> T {
        self.value
    }
}

impl<T> Capsule<T, Sealed> {
    /// Seal a payload with an authority ceiling and a step budget.
    #[must_use]
    pub fn seal(value: T, authority: RationalPQ, budget: u16) -> Self {
        Self {
            value,
            authority,
            budget,
            state: PhantomData,
        }
    }

    /// Move the capsule into the active state.
    #[must_use]
    pub fn activate(self) -> Capsule<T, Active> {
        Capsule {
            value: self.value,
            authority: self.authority,
            budget: self.budget,
            state: PhantomData,
        }
    }
}

impl<T> Capsule<T, Active> {
    /// Attenuate authority by `decay` and spend exactly one budget unit.
    ///
    /// On failure the original active capsule, including its payload, is
    /// returned inside [`AttenuateError`]; linearity is never broken by an
    /// error path.
    ///
    /// # Errors
    ///
    /// Returns [`QulbError::InvalidDecay`] unless `0 < decay < 1`,
    /// [`QulbError::Exhausted`] when authority is not strictly positive or the
    /// budget is already zero, and [`QulbError::Overflow`] when checked
    /// rational multiplication fails.
    pub fn attenuate(self, decay: RationalPQ) -> Result<Capsule<T, Attenuated>, AttenuateError<T>> {
        if let Err(error) = validate_decay(decay) {
            return Err(AttenuateError {
                error,
                capsule: self,
            });
        }
        if let Err(error) = ensure_capacity(self.authority, self.budget) {
            return Err(AttenuateError {
                error,
                capsule: self,
            });
        }
        match self.authority.checked_mul(decay) {
            Some(authority) => Ok(Capsule {
                value: self.value,
                authority,
                budget: self.budget - 1,
                state: PhantomData,
            }),
            None => Err(AttenuateError {
                error: QulbError::Overflow,
                capsule: self,
            }),
        }
    }

    /// Consume the capsule and return its payload.
    ///
    /// ```compile_fail
    /// use phoenix_capsule::Capsule;
    /// let capsule = Capsule::seal(7u8, pivot_abi::RationalPQ::ONE, 4).activate();
    /// let _first = capsule.consume();
    /// let _second = capsule.consume();
    /// ```
    #[must_use]
    pub fn consume(self) -> T {
        self.into_value()
    }
}

impl<T> Capsule<T, Attenuated> {
    /// Consume the capsule and return its payload.
    #[must_use]
    pub fn consume(self) -> T {
        self.into_value()
    }
}

/// Deterministic telemetry returned by one qULB step.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct QulbStep {
    /// Ternary state after the step, or the current pre-mask state for the
    /// zero-step snapshot sentinel.
    pub state: TritPlane64,
    /// Kernel dot-product telemetry for the pre-mask state and the weights.
    pub energy: i32,
    /// Authority ceiling after attenuation.
    pub authority: RationalPQ,
    /// Remaining budget after the step.
    pub budget: u16,
    /// True when the monotone mask left the state unchanged.
    pub fixed_point: bool,
}

/// Mutable `ultrabackyardconcepts` qULB loop state.
#[derive(Debug, Eq, PartialEq)]
pub struct UltraBackyardLoop {
    state: TritPlane64,
    authority: RationalPQ,
    budget: u16,
}

impl UltraBackyardLoop {
    /// Build a loop from a ternary state, authority ceiling, and step budget.
    #[must_use]
    pub fn new(state: TritPlane64, authority: RationalPQ, budget: u16) -> Self {
        Self {
            state,
            authority,
            budget,
        }
    }

    /// Current ternary state plane.
    #[must_use]
    pub fn state(&self) -> TritPlane64 {
        self.state
    }

    /// Current authority ceiling.
    #[must_use]
    pub fn authority(&self) -> RationalPQ {
        self.authority
    }

    /// Remaining deterministic step budget.
    #[must_use]
    pub fn budget(&self) -> u16 {
        self.budget
    }

    /// Advance the loop by one monotone contraction step.
    ///
    /// The step applies `pos &= weights.pos` and `neg &= weights.neg`, so the
    /// state is canonical whenever the inputs are canonical, and the number of
    /// non-zero lanes can only shrink.
    ///
    /// # Errors
    ///
    /// Returns [`QulbError::InvalidDecay`] unless `0 < decay < 1`,
    /// [`QulbError::Exhausted`] when authority is not strictly positive or the
    /// budget is already zero, and
    /// [`QulbError::Overflow`] when checked rational multiplication fails. On
    /// an error the loop state, authority, and budget are left unchanged.
    pub fn qulb_step<K: TernarySimdPort>(
        &mut self,
        weights: TritPlane64,
        decay: RationalPQ,
    ) -> Result<QulbStep, QulbError> {
        validate_decay(decay)?;
        ensure_capacity(self.authority, self.budget)?;

        let energy = K::dot64_trit_plane(self.state, weights);
        let next = TritPlane64 {
            pos: self.state.pos & weights.pos,
            neg: self.state.neg & weights.neg,
        };
        let fixed_point = next == self.state;

        let authority = self
            .authority
            .checked_mul(decay)
            .ok_or(QulbError::Overflow)?;

        self.state = next;
        self.authority = authority;
        self.budget -= 1;

        Ok(QulbStep {
            state: next,
            energy,
            authority,
            budget: self.budget,
            fixed_point,
        })
    }

    /// Run steps until the ternary state reaches a fixed point or the budget
    /// or authority is exhausted.
    ///
    /// Returns the first fixed-point step, the last completed step when the
    /// loop exhausts its capacity, or the last completed step when `max_steps`
    /// is reached first. A `max_steps` of zero performs no step and returns a
    /// read-only snapshot of the current state.
    ///
    /// # Errors
    ///
    /// Returns [`QulbError::InvalidDecay`] or [`QulbError::Overflow`] for a bad
    /// step, and [`QulbError::Exhausted`] when the loop is already exhausted
    /// before the first step.
    pub fn run_to_fixed_point<K: TernarySimdPort>(
        &mut self,
        weights: TritPlane64,
        decay: RationalPQ,
        max_steps: u16,
    ) -> Result<QulbStep, QulbError> {
        if max_steps == 0 {
            return Ok(self.snapshot::<K>(weights));
        }

        let mut last = None;
        let mut elapsed = 0u16;
        while elapsed < max_steps {
            match self.qulb_step::<K>(weights, decay) {
                Ok(report) => {
                    let done = report.fixed_point;
                    last = Some(report);
                    if done {
                        break;
                    }
                }
                Err(QulbError::Exhausted) => {
                    return match last {
                        Some(report) => Ok(report),
                        None => Err(QulbError::Exhausted),
                    };
                }
                Err(error) => return Err(error),
            }
            elapsed += 1;
        }

        match last {
            Some(report) => Ok(report),
            None => Err(QulbError::Exhausted),
        }
    }

    fn snapshot<K: TernarySimdPort>(&self, weights: TritPlane64) -> QulbStep {
        let next = TritPlane64 {
            pos: self.state.pos & weights.pos,
            neg: self.state.neg & weights.neg,
        };
        QulbStep {
            state: self.state,
            energy: K::dot64_trit_plane(self.state, weights),
            authority: self.authority,
            budget: self.budget,
            fixed_point: next == self.state,
        }
    }
}

impl QulbCapsulePort for UltraBackyardLoop {
    fn authority(&self) -> RationalPQ {
        self.authority
    }

    fn budget(&self) -> u16 {
        self.budget
    }

    fn attenuate(&mut self, decay: RationalPQ) -> Result<RationalPQ, QulbError> {
        validate_decay(decay)?;
        ensure_capacity(self.authority, self.budget)?;
        let authority = self
            .authority
            .checked_mul(decay)
            .ok_or(QulbError::Overflow)?;
        self.authority = authority;
        Ok(authority)
    }
}

/// Decay must be strictly inside the open interval `(0, 1)`.
fn validate_decay(decay: RationalPQ) -> Result<(), QulbError> {
    if decay <= RationalPQ::ZERO || decay >= RationalPQ::ONE {
        return Err(QulbError::InvalidDecay);
    }
    Ok(())
}

/// Authority must be strictly positive and budget must be non-zero.
fn ensure_capacity(authority: RationalPQ, budget: u16) -> Result<(), QulbError> {
    if authority <= RationalPQ::ZERO || budget == 0 {
        return Err(QulbError::Exhausted);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use pivot_abi::CentroidPod64;

    struct TestKernels;

    impl TernarySimdPort for TestKernels {
        fn dot64_trit_plane(a: TritPlane64, b: TritPlane64) -> i32 {
            let agree = ((a.pos & b.pos) | (a.neg & b.neg)).count_ones() as i32;
            let disagree = ((a.pos & b.neg) | (a.neg & b.pos)).count_ones() as i32;
            agree - disagree
        }

        fn contract64_trit_i16(_weights: TritPlane64, _act: &CentroidPod64) -> i32 {
            0
        }

        fn dot64_i16(_a: &CentroidPod64, _b: &CentroidPod64) -> i32 {
            0
        }

        fn louds_rank1(_bits: &[u64; 2], _idx: usize) -> u32 {
            0
        }

        fn splitmix64_x4(_state: &mut [u64; 4]) -> [u64; 4] {
            [0; 4]
        }
    }

    fn rat(num: i64, den: u64) -> RationalPQ {
        RationalPQ::new(num, den).expect("valid rational")
    }

    fn half() -> RationalPQ {
        rat(1, 2)
    }

    fn sample_state() -> TritPlane64 {
        TritPlane64::new(0b0000_0111, 0b0001_1000).expect("canonical state")
    }

    fn sample_weights() -> TritPlane64 {
        TritPlane64::new(0b0010_0110, 0b0100_1000).expect("canonical weights")
    }

    #[test]
    fn markers_are_zero_sized() {
        assert_eq!(core::mem::size_of::<Sealed>(), 0);
        assert_eq!(core::mem::size_of::<Active>(), 0);
        assert_eq!(core::mem::size_of::<Attenuated>(), 0);
        assert_eq!(core::mem::size_of::<Consumed>(), 0);
    }

    #[test]
    fn capsule_attenuates_exactly_and_decrements_budget() {
        let capsule = Capsule::seal(7u32, rat(3, 4), 4).activate();
        assert_eq!(capsule.authority(), rat(3, 4));
        assert_eq!(capsule.budget(), 4);

        let capsule = capsule.attenuate(rat(2, 3)).expect("valid decay");
        assert_eq!(capsule.authority(), rat(1, 2));
        assert_eq!(capsule.budget(), 3);
        assert_eq!(capsule.consume(), 7);
    }

    #[test]
    fn active_capsule_consumes_directly() {
        let capsule = Capsule::seal(7u8, RationalPQ::ONE, 4).activate();
        assert_eq!(capsule.consume(), 7);
    }

    #[test]
    fn capsule_consumes_a_non_copy_payload() {
        struct Payload {
            id: u32,
        }

        let capsule = Capsule::seal(Payload { id: 9 }, RationalPQ::ONE, 4).activate();
        assert_eq!(capsule.consume().id, 9);
    }

    fn expect_attenuation_error<T>(
        result: Result<Capsule<T, Attenuated>, AttenuateError<T>>,
    ) -> (QulbError, Capsule<T, Active>) {
        match result {
            Ok(_) => panic!("attenuation unexpectedly succeeded"),
            Err(failure) => (failure.error(), failure.into_capsule()),
        }
    }

    fn expect_attenuation<T>(
        result: Result<Capsule<T, Attenuated>, AttenuateError<T>>,
    ) -> Capsule<T, Attenuated> {
        match result {
            Ok(capsule) => capsule,
            Err(failure) => panic!("attenuation failed: {:?}", failure.error()),
        }
    }

    #[test]
    fn capsule_rejects_invalid_decay_and_preserves_the_payload() {
        for decay in [RationalPQ::ZERO, RationalPQ::ONE, rat(3, 2), rat(-1, 2)] {
            let (error, capsule) = expect_attenuation_error(
                Capsule::seal(9u8, RationalPQ::ONE, 4)
                    .activate()
                    .attenuate(decay),
            );
            assert_eq!(error, QulbError::InvalidDecay);
            assert_eq!(capsule.authority(), RationalPQ::ONE);
            assert_eq!(capsule.budget(), 4);
            assert_eq!(capsule.consume(), 9);
        }
    }

    #[test]
    fn capsule_error_recovery_keeps_a_non_copy_payload() {
        struct Payload {
            id: u32,
        }

        let capsule = Capsule::seal(Payload { id: 11 }, RationalPQ::ONE, 4).activate();
        let (error, capsule) = expect_attenuation_error(capsule.attenuate(RationalPQ::ONE));
        assert_eq!(error, QulbError::InvalidDecay);
        assert_eq!(capsule.consume().id, 11);
    }

    #[test]
    fn capsule_zero_or_negative_authority_is_exhausted() {
        for authority in [RationalPQ::ZERO, rat(-1, 2)] {
            let (error, capsule) = expect_attenuation_error(
                Capsule::seal(3u8, authority, 4)
                    .activate()
                    .attenuate(half()),
            );
            assert_eq!(error, QulbError::Exhausted);
            assert_eq!(capsule.authority(), authority);
            assert_eq!(capsule.consume(), 3);
        }
    }

    #[test]
    fn capsule_zero_budget_is_exhausted() {
        let (error, capsule) = expect_attenuation_error(
            Capsule::seal(5u8, RationalPQ::ONE, 0)
                .activate()
                .attenuate(half()),
        );
        assert_eq!(error, QulbError::Exhausted);
        assert_eq!(capsule.budget(), 0);
        assert_eq!(capsule.consume(), 5);
    }

    #[test]
    fn capsule_reports_overflow_and_preserves_the_capsule() {
        let authority = rat(i64::MAX, 1);
        let (error, capsule) = expect_attenuation_error(
            Capsule::seal(13u8, authority, 4)
                .activate()
                .attenuate(rat(3, 4)),
        );
        assert_eq!(error, QulbError::Overflow);
        assert_eq!(capsule.authority(), authority);
        assert_eq!(capsule.budget(), 4);
        assert_eq!(capsule.consume(), 13);
    }

    #[test]
    fn attenuation_is_strictly_monotone_for_positive_authority() {
        for decay in [rat(1, 2), rat(2, 3), rat(9, 10)] {
            let before = rat(7, 1);
            let capsule = Capsule::seal((), before, 4).activate();
            let after = expect_attenuation(capsule.attenuate(decay)).authority();
            assert!(after < before);
            assert!(after > RationalPQ::ZERO);
        }
    }

    #[test]
    fn qulb_step_reports_exact_energy_and_next_state() {
        let mut capsule = UltraBackyardLoop::new(sample_state(), RationalPQ::ONE, 4);
        let step = capsule
            .qulb_step::<TestKernels>(sample_weights(), half())
            .expect("valid step");

        assert_eq!(step.energy, 3);
        assert_eq!(
            step.state,
            TritPlane64::new(0b0000_0110, 0b0000_1000).expect("canonical next state")
        );
        assert!(!step.fixed_point);
        assert_eq!(step.authority, half());
        assert_eq!(step.budget, 3);
        assert_eq!(capsule.state(), step.state);
        assert_eq!(capsule.authority(), half());
        assert_eq!(capsule.budget(), 3);
    }

    #[test]
    fn qulb_step_detects_fixed_point() {
        let state = sample_state();
        let mut capsule = UltraBackyardLoop::new(state, RationalPQ::ONE, 4);
        let step = capsule
            .qulb_step::<TestKernels>(state, half())
            .expect("valid step");

        assert!(step.fixed_point);
        assert_eq!(step.state, state);
        assert_eq!(step.energy, 5);
        assert_eq!(step.budget, 3);
        assert_eq!(step.authority, half());
    }

    #[test]
    fn monotone_contraction_terminates_with_reduced_authority_and_budget() {
        let state = TritPlane64::new(u64::MAX, 0).expect("canonical state");
        let weights = TritPlane64::new(0x0F0F_0F0F_0F0F_0F0F, 0).expect("canonical weights");
        let mut capsule = UltraBackyardLoop::new(state, RationalPQ::ONE, 64);

        let step = capsule
            .run_to_fixed_point::<TestKernels>(weights, half(), 64)
            .expect("contraction converges");

        assert!(step.fixed_point);
        assert_eq!(step.state, weights);
        assert!(step.budget < 64);
        assert!(step.authority < RationalPQ::ONE);
        assert_eq!(capsule.state(), weights);
    }

    #[test]
    fn run_to_fixed_point_with_zero_steps_is_a_snapshot() {
        let state = sample_state();
        let mut capsule = UltraBackyardLoop::new(state, rat(3, 4), 2);
        let step = capsule
            .run_to_fixed_point::<TestKernels>(sample_weights(), half(), 0)
            .expect("snapshot");

        assert_eq!(step.state, state);
        assert_eq!(step.energy, 3);
        assert_eq!(step.authority, rat(3, 4));
        assert_eq!(step.budget, 2);
        assert!(!step.fixed_point);
    }

    #[test]
    fn qulb_step_rejects_invalid_decay() {
        for decay in [RationalPQ::ZERO, RationalPQ::ONE, rat(3, 2)] {
            let mut capsule = UltraBackyardLoop::new(sample_state(), RationalPQ::ONE, 4);
            assert_eq!(
                capsule
                    .qulb_step::<TestKernels>(sample_weights(), decay)
                    .err(),
                Some(QulbError::InvalidDecay)
            );
        }
    }

    #[test]
    fn qulb_step_reports_budget_exhaustion() {
        let mut capsule = UltraBackyardLoop::new(sample_state(), RationalPQ::ONE, 1);
        capsule
            .qulb_step::<TestKernels>(sample_weights(), half())
            .expect("first step fits the budget");
        assert_eq!(capsule.budget(), 0);
        assert_eq!(
            capsule
                .qulb_step::<TestKernels>(sample_weights(), half())
                .err(),
            Some(QulbError::Exhausted)
        );
    }

    #[test]
    fn qulb_step_reports_zero_authority_exhaustion() {
        let mut capsule = UltraBackyardLoop::new(sample_state(), RationalPQ::ZERO, 4);
        assert_eq!(
            capsule
                .qulb_step::<TestKernels>(sample_weights(), half())
                .err(),
            Some(QulbError::Exhausted)
        );
    }

    #[test]
    fn qulb_step_rejects_negative_authority_without_mutation() {
        let state = sample_state();
        let mut capsule = UltraBackyardLoop::new(state, rat(-1, 2), 4);
        assert_eq!(
            capsule
                .qulb_step::<TestKernels>(sample_weights(), half())
                .err(),
            Some(QulbError::Exhausted)
        );
        assert_eq!(capsule.state(), state);
        assert_eq!(capsule.authority(), rat(-1, 2));
        assert_eq!(capsule.budget(), 4);

        assert_eq!(
            QulbCapsulePort::attenuate(&mut capsule, half()).err(),
            Some(QulbError::Exhausted)
        );
        assert_eq!(capsule.authority(), rat(-1, 2));
    }

    #[test]
    fn run_to_fixed_point_reports_initial_exhaustion() {
        let mut capsule = UltraBackyardLoop::new(sample_state(), RationalPQ::ONE, 0);
        assert_eq!(
            capsule
                .run_to_fixed_point::<TestKernels>(sample_weights(), half(), 8)
                .err(),
            Some(QulbError::Exhausted)
        );
    }

    #[test]
    fn run_to_fixed_point_returns_last_step_on_exhaustion() {
        let mut capsule = UltraBackyardLoop::new(sample_state(), RationalPQ::ONE, 1);
        let step = capsule
            .run_to_fixed_point::<TestKernels>(sample_weights(), half(), 8)
            .expect("one step completes before exhaustion");

        assert!(!step.fixed_point);
        assert_eq!(step.budget, 0);
        assert_eq!(step.authority, half());
        assert_eq!(capsule.budget(), 0);
    }

    #[test]
    fn qulb_step_reports_overflow_without_mutating_state() {
        let state = sample_state();
        let mut capsule = UltraBackyardLoop::new(state, rat(i64::MAX, 1), 4);
        assert_eq!(
            capsule
                .qulb_step::<TestKernels>(sample_weights(), rat(3, 4))
                .err(),
            Some(QulbError::Overflow)
        );
        assert_eq!(capsule.state(), state);
        assert_eq!(capsule.authority(), rat(i64::MAX, 1));
        assert_eq!(capsule.budget(), 4);
    }

    #[test]
    fn qulb_steps_are_deterministic() {
        let mut first = UltraBackyardLoop::new(sample_state(), rat(9, 10), 8);
        let mut second = UltraBackyardLoop::new(sample_state(), rat(9, 10), 8);

        for _ in 0..4 {
            let left = first
                .qulb_step::<TestKernels>(sample_weights(), half())
                .expect("first loop step");
            let right = second
                .qulb_step::<TestKernels>(sample_weights(), half())
                .expect("second loop step");
            assert_eq!(left, right);
        }

        assert_eq!(first, second);
    }

    #[test]
    fn qulb_capsule_port_matches_inherent_accessors() {
        let mut capsule = UltraBackyardLoop::new(sample_state(), rat(3, 4), 4);
        assert_eq!(QulbCapsulePort::authority(&capsule), capsule.authority());
        assert_eq!(QulbCapsulePort::budget(&capsule), capsule.budget());

        let attenuated = QulbCapsulePort::attenuate(&mut capsule, rat(2, 3)).expect("valid decay");
        assert_eq!(attenuated, rat(1, 2));
        assert_eq!(capsule.authority(), rat(1, 2));
        assert_eq!(capsule.budget(), 4);
    }

    #[test]
    fn qulb_capsule_port_rejects_invalid_decay_and_exhaustion() {
        let mut capsule = UltraBackyardLoop::new(sample_state(), RationalPQ::ONE, 4);
        assert_eq!(
            QulbCapsulePort::attenuate(&mut capsule, RationalPQ::ONE).err(),
            Some(QulbError::InvalidDecay)
        );

        let mut zero_authority = UltraBackyardLoop::new(sample_state(), RationalPQ::ZERO, 4);
        assert_eq!(
            QulbCapsulePort::attenuate(&mut zero_authority, half()).err(),
            Some(QulbError::Exhausted)
        );

        let mut zero_budget = UltraBackyardLoop::new(sample_state(), RationalPQ::ONE, 0);
        assert_eq!(
            QulbCapsulePort::attenuate(&mut zero_budget, half()).err(),
            Some(QulbError::Exhausted)
        );
    }
}
