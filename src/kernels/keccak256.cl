// Previous license and comments remain the same...

// Add scoring macros for efficient address checking
#define is_four(x) ((x) == 0x4)
#define is_zero(x) ((x) == 0x0)

#define check_nibble(byte, high) (high ? ((byte) >> 4) : ((byte) & 0xF))

#define score_address(d) ({ \
  int score = 0; \
  bool found_first = false; \
  int consecutive_fours = 0; \
  int last_four_count = 0; \
  bool has_four_sequence = false; \
  \
  /* Check first 20 bytes (40 nibbles) */ \
  for(int i = 0; i < 20; i++) { \
    uchar high = check_nibble(d[i], 1); \
    uchar low = check_nibble(d[i], 0); \
    \
    /* Process high nibble */ \
    if(!found_first) { \
      if(is_zero(high)) { \
        score += 10; \
      } else { \
        found_first = true; \
        if(!is_four(high)) return 0; \
      } \
    } \
    \
    /* Count fours and check sequences */ \
    if(is_four(high)) { \
      score += 1; \
      consecutive_fours++; \
      if(consecutive_fours == 4 && !has_four_sequence) { \
        score += 40; \
        has_four_sequence = true; \
        /* Check next nibble after sequence */ \
        if(i < 19 && !is_four(low)) { \
          score += 20; \
        } \
      } \
    } else { \
      consecutive_fours = 0; \
    } \
    \
    /* Process low nibble */ \
    if(!found_first) { \
      if(is_zero(low)) { \
        score += 10; \
      } else { \
        found_first = true; \
        if(!is_four(low)) return 0; \
      } \
    } \
    \
    if(is_four(low)) { \
      score += 1; \
      consecutive_fours++; \
      if(consecutive_fours == 4 && !has_four_sequence) { \
        score += 40; \
        has_four_sequence = true; \
        /* Check next byte's high nibble */ \
        if(i < 19 && !is_four(check_nibble(d[i+1], 1))) { \
          score += 20; \
        } \
      } \
    } else { \
      consecutive_fours = 0; \
    } \
    \
    /* Track last 4 nibbles */ \
    if(i >= 18) { \
      if(is_four(high)) last_four_count++; \
      if(is_four(low)) last_four_count++; \
    } \
  } \
  \
  /* Add points for last four being all 4s */ \
  if(last_four_count == 4) { \
    score += 20; \
  } \
  \
  score; \
})

__kernel void hashMessage(
  __constant uchar const *d_message,
  __constant uint const *d_nonce,
  __global volatile ulong *restrict solutions
) {
  // Previous buffer setup remains the same...

  // Apply keccakf
  keccakf(spongeBuffer);

  // Score the address and check if it beats current best
  int score = score_address(digest);
  if (score > 174) {
    solutions[0] = nonce.uint64_t;
  } else if (
    hasLeading(digest) 
#if TOTAL_ZEROES <= 20
    || hasTotal(digest)
#endif
  ) {
    // Keep existing zero-based checks as fallback
    solutions[0] = nonce.uint64_t;
  }
}
