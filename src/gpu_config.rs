pub(crate) const WORK_SIZE: u32 = 0x40000000; // Adjusted for better H100 occupancy
pub(crate) const WORK_FACTOR: u128 = (WORK_SIZE as u128) / 1_000_000;

pub(crate) fn get_optimal_work_group_size() -> usize {
    256 // Better balance between occupancy and register pressure
}