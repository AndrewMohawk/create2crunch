/*
   Copyright 2018 Lip Wee Yeo Amano

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/

/**
* Based on the following, with small tweaks and optimizations:
*
* https://github.com/lwYeo/SoliditySHA3Miner/blob/master/SoliditySHA3Miner/
*   Miner/Kernels/OpenCL/sha3KingKernel.cl
*
* Originally modified for openCL processing by lwYeo
*
* Original implementor: David Leon Gil
*
* License: CC0, attribution kindly requested. Blame taken too, but not
* liability.
*/

/******** Keccak-f[1600] (for finding efficient Ethereum addresses) ********/

#define OPENCL_PLATFORM_UNKNOWN 0
#define OPENCL_PLATFORM_AMD   2

#ifndef PLATFORM
# define PLATFORM       OPENCL_PLATFORM_UNKNOWN
#endif

#if PLATFORM == OPENCL_PLATFORM_AMD
# pragma OPENCL EXTENSION   cl_amd_media_ops : enable
#endif

typedef union _nonce_t
{
  ulong   uint64_t;
  uint    uint32_t[2];
  uchar   uint8_t[8];
} nonce_t;

#if PLATFORM == OPENCL_PLATFORM_AMD
static inline ulong rol(const ulong x, const uint s)
{
  uint2 output;
  uint2 x2 = as_uint2(x);

  output = (s > 32u) ? amd_bitalign((x2).yx, (x2).xy, 64u - s) : amd_bitalign((x2).xy, (x2).yx, 32u - s);
  return as_ulong(output);
}
#else
#define rol(x, s) (((x) << s) | ((x) >> (64u - s)))
#endif

#define rol1(x) rol(x, 1u)

#define theta_(m, n, o) \
t = b[m] ^ rol1(b[n]); \
a[o + 0] ^= t; \
a[o + 5] ^= t; \
a[o + 10] ^= t; \
a[o + 15] ^= t; \
a[o + 20] ^= t; \

#define theta() \
b[0] = a[0] ^ a[5] ^ a[10] ^ a[15] ^ a[20]; \
b[1] = a[1] ^ a[6] ^ a[11] ^ a[16] ^ a[21]; \
b[2] = a[2] ^ a[7] ^ a[12] ^ a[17] ^ a[22]; \
b[3] = a[3] ^ a[8] ^ a[13] ^ a[18] ^ a[23]; \
b[4] = a[4] ^ a[9] ^ a[14] ^ a[19] ^ a[24]; \
theta_(4, 1, 0); \
theta_(0, 2, 1); \
theta_(1, 3, 2); \
theta_(2, 4, 3); \
theta_(3, 0, 4);

#define rhoPi_(m, n) t = b[0]; b[0] = a[m]; a[m] = rol(t, n); \

#define rhoPi() t = a[1]; b[0] = a[10]; a[10] = rol1(t); \
rhoPi_(7, 3); \
rhoPi_(11, 6); \
rhoPi_(17, 10); \
rhoPi_(18, 15); \
rhoPi_(3, 21); \
rhoPi_(5, 28); \
rhoPi_(16, 36); \
rhoPi_(8, 45); \
rhoPi_(21, 55); \
rhoPi_(24, 2); \
rhoPi_(4, 14); \
rhoPi_(15, 27); \
rhoPi_(23, 41); \
rhoPi_(19, 56); \
rhoPi_(13, 8); \
rhoPi_(12, 25); \
rhoPi_(2, 43); \
rhoPi_(20, 62); \
rhoPi_(14, 18); \
rhoPi_(22, 39); \
rhoPi_(9, 61); \
rhoPi_(6, 20); \
rhoPi_(1, 44);

#define chi_(n) \
b[0] = a[n + 0]; \
b[1] = a[n + 1]; \
b[2] = a[n + 2]; \
b[3] = a[n + 3]; \
b[4] = a[n + 4]; \
a[n + 0] = b[0] ^ ((~b[1]) & b[2]); \
a[n + 1] = b[1] ^ ((~b[2]) & b[3]); \
a[n + 2] = b[2] ^ ((~b[3]) & b[4]); \
a[n + 3] = b[3] ^ ((~b[4]) & b[0]); \
a[n + 4] = b[4] ^ ((~b[0]) & b[1]);

#define chi() chi_(0); chi_(5); chi_(10); chi_(15); chi_(20);

#define iota(x) a[0] ^= x;

#define iteration(x) theta(); rhoPi(); chi(); iota(x);

__attribute__((reqd_work_group_size(256, 1, 1)))
static inline void keccakf(ulong *a)
{
  ulong b[5] __attribute__((aligned(32)));
  ulong t;

  iteration(0x0000000000000001); // iteration 1
  iteration(0x0000000000008082); // iteration 2
  iteration(0x800000000000808a); // iteration 3
  iteration(0x8000000080008000); // iteration 4
  iteration(0x000000000000808b); // iteration 5
  iteration(0x0000000080000001); // iteration 6
  iteration(0x8000000080008081); // iteration 7
  iteration(0x8000000000008009); // iteration 8
  iteration(0x000000000000008a); // iteration 9
  iteration(0x0000000000000088); // iteration 10
  iteration(0x0000000080008009); // iteration 11
  iteration(0x000000008000000a); // iteration 12
  iteration(0x000000008000808b); // iteration 13
  iteration(0x800000000000008b); // iteration 14
  iteration(0x8000000000008089); // iteration 15
  iteration(0x8000000000008003); // iteration 16
  iteration(0x8000000000008002); // iteration 17
  iteration(0x8000000000000080); // iteration 18
  iteration(0x000000000000800a); // iteration 19
  iteration(0x800000008000000a); // iteration 20
  iteration(0x8000000080008081); // iteration 21
  iteration(0x8000000000008080); // iteration 22
  iteration(0x0000000080000001); // iteration 23

  // iteration 24 (partial)
#define o ((uint *)(a))
  // Theta (partial)
  b[0] = a[0] ^ a[5] ^ a[10] ^ a[15] ^ a[20];
  b[1] = a[1] ^ a[6] ^ a[11] ^ a[16] ^ a[21];
  b[2] = a[2] ^ a[7] ^ a[12] ^ a[17] ^ a[22];
  b[3] = a[3] ^ a[8] ^ a[13] ^ a[18] ^ a[23];
  b[4] = a[4] ^ a[9] ^ a[14] ^ a[19] ^ a[24];

  a[0] ^= b[4] ^ rol1(b[1]);
  a[6] ^= b[0] ^ rol1(b[2]);
  a[12] ^= b[1] ^ rol1(b[3]);
  a[18] ^= b[2] ^ rol1(b[4]);
  a[24] ^= b[3] ^ rol1(b[0]);

  // Rho Pi (partial)
  o[3] = (o[13] >> 20) | (o[12] << 12);
  a[2] = rol(a[12], 43);
  a[3] = rol(a[18], 21);
  a[4] = rol(a[24], 14);

  // Chi (partial)
  o[3] ^= ((~o[5]) & o[7]);
  o[4] ^= ((~o[6]) & o[8]);
  o[5] ^= ((~o[7]) & o[9]);
  o[6] ^= ((~o[8]) & o[0]);
  o[7] ^= ((~o[9]) & o[1]);
#undef o
}

#define hasTotal(d) ( \
  (!(d[0])) + (!(d[1])) + (!(d[2])) + (!(d[3])) + \
  (!(d[4])) + (!(d[5])) + (!(d[6])) + (!(d[7])) + \
  (!(d[8])) + (!(d[9])) + (!(d[10])) + (!(d[11])) + \
  (!(d[12])) + (!(d[13])) + (!(d[14])) + (!(d[15])) + \
  (!(d[16])) + (!(d[17])) + (!(d[18])) + (!(d[19])) \
>= TOTAL_ZEROES)

#if LEADING_ZEROES == 8
#define hasLeading(d) (!(((uint*)d)[0]) && !(((uint*)d)[1]))
#elif LEADING_ZEROES == 7
#define hasLeading(d) (!(((uint*)d)[0]) && !(((uint*)d)[1] & 0x00ffffffu))
#elif LEADING_ZEROES == 6
#define hasLeading(d) (!(((uint*)d)[0]) && !(((uint*)d)[1] & 0x0000ffffu))
#elif LEADING_ZEROES == 5
#define hasLeading(d) (!(((uint*)d)[0]) && !(((uint*)d)[1] & 0x000000ffu))
#elif LEADING_ZEROES == 4
#define hasLeading(d) (!(((uint*)d)[0]))
#elif LEADING_ZEROES == 3
#define hasLeading(d) (!(((uint*)d)[0] & 0x00ffffffu))
#elif LEADING_ZEROES == 2
#define hasLeading(d) (!(((uint*)d)[0] & 0x0000ffffu))
#elif LEADING_ZEROES == 1
#define hasLeading(d) (!(((uint*)d)[0] & 0x000000ffu))
#else
static inline bool hasLeading(uchar const *d)
{
#pragma unroll
  for (uint i = 0; i < LEADING_ZEROES; ++i) {
    if (d[i] != 0) return false;
  }
  return true;
}
#endif

__kernel void hashMessage(
  __constant uchar const *d_message,
  __constant uint const *d_nonce,
  __global volatile ulong *restrict solutions
) {
  __local ulong shared_sponge[64][25];
  const uint gid = get_global_id(0);
  const uint lid = get_local_id(0);
  
  ulong *spongeBuffer = &shared_sponge[lid][0];
#define sponge ((uchar *)spongeBuffer)
#define digest (sponge + 12)

  // Process multiple hashes per thread
  #pragma unroll 4
  for(uint batch = 0; batch < 4; batch++) {
    uint current_nonce = gid * 4 + batch;
    
    // Initialize sponge state
    #pragma unroll 8 
    for(int i = 0; i < 200; i += 8) {
      vstore8(0UL, 0, &sponge[i]);
    }

    // Set control character
    sponge[0] = 0xffu;

    // Copy factory and caller address
    #pragma unroll
    for(int i = 1; i < 41; i++) {
      sponge[i] = *(&S_1 + (i-1));
    }

    // Copy message pattern
    #pragma unroll
    for(int i = 0; i < 4; i++) {
      sponge[41 + i] = d_message[i];
    }

    // Set nonce
    nonce_t nonce;
    nonce.uint32_t[0] = current_nonce;
    nonce.uint32_t[1] = d_nonce[0];
    
    #pragma unroll
    for(int i = 0; i < 8; i++) {
      sponge[45 + i] = nonce.uint8_t[i];
    }

    // Copy init code hash
    #pragma unroll
    for(int i = 53; i < 85; i++) {
      sponge[i] = *(&S_53 + (i-53));
    }

    // Padding
    sponge[85] = 0x01u;
    sponge[135] = 0x80u;

    // Process hash
    keccakf(spongeBuffer);

    // Check result
    if(check_score(digest, 20)) {
      atomic_min(&solutions[0], nonce.uint64_t);
      return;
    }
  }
}
