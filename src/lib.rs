#![no_std]
#![forbid(unsafe_code)]
#![deny(clippy::float_arithmetic)]

//! Host-agnostic smoke-harness package for the `wukong-zeno` workspace.
//!
//! The library body only supplies deterministic, allocation-free helpers used
//! by the `examples/*.rs` targets, which are the only places where `std` is
//! permitted.

/// FNV-1a 64-bit digest used for the smoke-run state fingerprint.
#[must_use]
pub fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    let mut index = 0;
    while index < bytes.len() {
        hash ^= bytes[index] as u64;
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        index += 1;
    }
    hash
}

#[cfg(test)]
mod tests {
    #[test]
    fn fnv1a_known_vector() {
        assert_eq!(super::fnv1a64(b""), 0xcbf2_9ce4_8422_2325);
        assert_eq!(super::fnv1a64(b"a"), 0xaf63_dc4c_8601_ec8c);
    }
}
