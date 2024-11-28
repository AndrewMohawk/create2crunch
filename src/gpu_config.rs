pub(crate) const WORK_SIZE: u32 = 0x10000000; // Further reduced for memory constraints
pub(crate) const WORK_FACTOR: u128 = (WORK_SIZE as u128) / 1_000_000;

pub(crate) fn get_optimal_work_group_size() -> usize {
    128 // Reduced for better memory utilization
}

pub(crate) fn get_local_mem_size() -> usize {
    32 * 1024 // 32KB local memory per workgroup for H100
}