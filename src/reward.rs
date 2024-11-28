use rustc_hash::FxHashMap;

pub struct Reward {
    reward: FxHashMap<usize, &'static str>,
}

impl Reward {
    pub fn new() -> Self {
        let mut reward = FxHashMap::default();
        
        for leading_zeros in 8..20 {  // Start at 8 leading zeros minimum
            for total_fours in 4..40 {  // Must have at least 4 fours
                let mut score = 0;
                score += leading_zeros * 10;
                
                if total_fours >= 4 {
                    score += 40;
                    score += 20;
                    
                    if total_fours >= 8 {
                        score += 20;
                    }
                }
                
                score += total_fours;
                
                if score > 174 {  // Only store potentially winning scores
                    let key = leading_zeros * 20 + total_fours;
                    let score_str = score.to_string();
                    let static_str: &'static str = Box::leak(score_str.into_boxed_str());
                    reward.insert(key, static_str);
                }
            }
        }
        
        // Special high-value patterns
        reward.insert(200, "175");  // Only store if better than current best
        reward.insert(180, "176");
        
        Reward { reward }
    }

    #[inline]
    pub fn get(&self, value: &usize) -> Option<&'static str> {
        self.reward.get(value).copied()
    }

    #[inline]
    pub fn is_competitive(&self, score: &str) -> bool {
        score.parse::<u32>().map_or(false, |num| num > 174)
    }
}