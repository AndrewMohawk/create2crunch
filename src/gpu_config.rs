pub(crate) const WORK_SIZE: u32 = 0x4000000; // Significantly reduced batch size
pub(crate) const WORK_FACTOR: u128 = (WORK_SIZE as u128) / 1_000_000;

pub(crate) fn get_optimal_work_group_size() -> usize {
    64 // Further reduced for better occupancy
}

pub(crate) fn get_local_mem_size() -> usize {
    16 * 1024 // 16KB local memory per workgroup
}