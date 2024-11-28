// pub(crate) const COMPUTE_UNITS: u32 = 132; // H100 specific
// pub(crate) const WAVES_PER_EU: u32 = 32;
pub(crate) fn get_optimal_work_group_size() -> usize {
    256 // Optimal for H100
}