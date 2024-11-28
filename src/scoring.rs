// scoring.rs
pub(crate) fn score_address_hex(addr: &str) -> Option<u32> {
    let addr = addr.trim_start_matches("0x");
    let mut score = 0;
    let mut first_non_zero_pos = None;
    
    // Count leading zeros and find first non-zero
    for (i, c) in addr.chars().enumerate() {
        if c == '0' {
            if first_non_zero_pos.is_none() {
                score += 10; // 10 points per leading zero
            }
        } else {
            if first_non_zero_pos.is_none() {
                first_non_zero_pos = Some(i);
                if c != '4' {
                    return None; // Must start with 4 after zeros
                }
            }
        }
    }

    if first_non_zero_pos.is_none() {
        return None;
    }

    // Count sequences of 4s
    let mut consecutive_fours = 0;
    let mut max_consecutive_fours = 0;
    
    for c in addr.chars() {
        if c == '4' {
            consecutive_fours += 1;
            max_consecutive_fours = max_consecutive_fours.max(consecutive_fours);
        } else {
            consecutive_fours = 0;
        }
    }

    // Scoring for sequences
    if max_consecutive_fours >= 4 {
        score += 40; // Base score for four 4s
        score += 20; // Bonus for clean sequence
        
        if max_consecutive_fours >= 8 {
            score += 20; // Extra bonus for longer sequence
        }
    }

    // Count total 4s
    score += addr.chars().filter(|&c| c == '4').count() as u32;

    // Bonus for specific positions
    if addr.len() >= 4 {
        if addr.ends_with("4444") {
            score += 20;
        }
        if addr.starts_with("0000000000004444") {
            score += 30; // Bonus for ideal pattern
        }
    }

    Some(score)
}
