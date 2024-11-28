pub(crate) const PATTERN_BYTES: [[u8; 4]; 12] = [
    [0x00, 0x00, 0x00, 0x44], // 9+ leading zeros then 4
    [0x00, 0x00, 0x00, 0x04], // Alternative leading zero pattern
    [0x00, 0x00, 0x04, 0x44], // 8 leading zeros then 4s
    [0x00, 0x00, 0x44, 0x44], // 7 leading zeros then 4s
    [0x00, 0x04, 0x44, 0x44], // 6 leading zeros then 4s
    [0x00, 0x44, 0x44, 0x44], // 5 leading zeros then 4s
    [0x04, 0x44, 0x44, 0x44], // 4 leading zeros then 4s
    [0x44, 0x44, 0x44, 0x44], // All 4s pattern
    [0x00, 0x00, 0x00, 0x40], // High leading zeros variant
    [0x00, 0x00, 0x40, 0x44], // Mixed high zeros and 4s
    [0x00, 0x40, 0x44, 0x44], // Another mixed pattern
    [0x40, 0x44, 0x44, 0x44], // Heavy on 4s with leading bits
];

pub(crate) fn get_optimal_pattern_sequence() -> Vec<[u8; 4]> {
    let mut patterns = Vec::with_capacity(PATTERN_BYTES.len() * 2);
    
    // Add base patterns
    patterns.extend_from_slice(&PATTERN_BYTES);
    
    // Add variations with shifted bits
    for pattern in PATTERN_BYTES.iter() {
        let mut shifted = [0u8; 4];
        for i in 0..4 {
            shifted[i] = pattern[i].rotate_left(4);
        }
        patterns.push(shifted);
    }
    
    patterns
}