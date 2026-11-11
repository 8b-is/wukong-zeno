#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
pub mod aarch64;

#[cfg(target_arch = "x86_64")]
pub mod x86_64;

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
pub use self::aarch64::{
    contract64_trit_i16, dot64_i16, dot64_trit_plane, louds_rank1, splitmix64_x4,
};

#[cfg(target_arch = "x86_64")]
pub use self::x86_64::{
    contract64_trit_i16, dot64_i16, dot64_trit_plane, louds_rank1, splitmix64_x4,
};

#[cfg(not(any(
    all(target_arch = "aarch64", target_feature = "neon"),
    target_arch = "x86_64"
)))]
pub use crate::scalar::{
    contract64_trit_i16, dot64_i16, dot64_trit_plane, louds_rank1, splitmix64_x4,
};
