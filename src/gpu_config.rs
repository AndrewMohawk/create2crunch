pub(crate) const WORK_SIZE: u64 = 0x100000000; // Optimized for H100's memory
pub(crate) const WORK_FACTOR: u128 = (WORK_SIZE as u128) / 1_000_000;

pub(crate) fn get_optimal_work_group_size() -> usize {
    1024 // Optimal for H100 (matches warp size * 32)
}