// MIT License
// 
// Copyright (c) 2024 Ben Buhrow
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// 
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// MIT License
// 
// Copyright (c) 2024 Pierre
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// 
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <stdint.h>
#include <immintrin.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>	//for gettimeofday using gcc
#include "gmp.h"

typedef __uint128_t uint128_t;

typedef struct
{
	uint64_t data[2][8];
} vec_u104_t;

#define and64 _mm512_and_epi64
#define store64 _mm512_store_epi64
#define storeu64 _mm512_storeu_epi64
#define mstoreu64 _mm512_mask_storeu_epi64
#define storeu512 _mm512_storeu_si512
#define add64 _mm512_add_epi64
#define sub64 _mm512_sub_epi64
#define set64 _mm512_set1_epi64
#define srli64 _mm512_srli_epi64
#define load64 _mm512_load_epi64
#define loadu64 _mm512_loadu_epi64
#define loadu512 _mm512_loadu_si512
#define castpd _mm512_castsi512_pd
#define castepu _mm512_castpd_si512

/* ============= Begin routines borrowed or adapted from Perig ==============
See https://github.com/Boutoukoat/Euler-Sprp-fast-primality-checks
and https://www.mersenneforum.org/node/22163/page10
*/


static uint128_t my_random(void)
{
	// based on linear congruential generator, period = 2^128
	static uint128_t seed = ((uint128_t)0x123456789ull << 92) + ((uint128_t)0xabcdef << 36) + 0x987654321ull;
	seed = seed * 137 + 13;
	// shuffle
	uint128_t x = seed ^ (seed >> 17) ^ (seed << 13);
	return x;
}

// count trailing zeroes in binary representation 
static __inline 
uint64_t my_ctz52(uint64_t n)
{
#if (INLINE_ASM && defined(__x86_64__))
#if defined(__BMI1__)
	uint64_t t;
	asm(" tzcntq %1, %0\n": "=r"(t) : "r"(n) : "flags");
	return t;
#else
	if (n)
		return __builtin_ctzll(n);
	return 52;
#endif
#else
#if defined(__GNUC__)
	if (n)
		return __builtin_ctzll(n);
	return 52;
#else
	if (n == 0)
		return 52;
	uint64_t r = 0;
	if ((n & 0xFFFFFFFFull) == 0)
		r += 32, n >>= 32;
	if ((n & 0xFFFFull) == 0)
		r += 16, n >>= 16;
	if ((n & 0xFFull) == 0)
		r += 8, n >>= 8;
	if ((n & 0xFull) == 0)
		r += 4, n >>= 4;
	if ((n & 0x3ull) == 0)
		r += 2, n >>= 2;
	if ((n & 0x1ull) == 0)
		r += 1;
	return r;
#endif
#endif
}

// count trailing zeroes in binary representation 
static __inline 
uint64_t my_ctz104(uint64_t n_lo, uint64_t n_hi)
{
	if (n_lo) {
		return my_ctz52(n_lo);
	}
	return 52 + my_ctz52(n_hi);
}

// count leading zeroes in binary representation
static __inline 
uint64_t my_clz52(uint64_t n)
{
#if (INLINE_ASM && defined(__x86_64__))
#ifdef __BMI1__
	uint64_t t;
	asm(" lzcntq %1, %0\n": "=r"(t) : "r"(n) : "flags");
	return t;
#else
	if (n)
		return __builtin_clzll(n);
	return 52;
#endif
#else
#if defined(__GNUC__)
	if (n)
		return __builtin_clzll(n);
	return 52;
#else
	if (n == 0)
		return 52;
	uint64_t r = 0;
	if ((n & (0xFFFFFFFFull << 32)) == 0)
		r += 32, n <<= 32;
	if ((n & (0xFFFFull << 48)) == 0)
		r += 16, n <<= 16;
	if ((n & (0xFFull << 56)) == 0)
		r += 8, n <<= 8;
	if ((n & (0xFull << 60)) == 0)
		r += 4, n <<= 4;
	if ((n & (0x3ull << 62)) == 0)
		r += 2, n <<= 2;
	if ((n & (0x1ull << 63)) == 0)
		r += 1;
	return r;
#endif
#endif
}

// count leading zeroes in binary representation 
static __inline 
uint64_t my_clz104(uint64_t n_lo, uint64_t n_hi)
{
	if (n_hi) {
		return my_clz52(n_hi);
	}
	return 52 + my_clz52(n_lo);
}

static inline uint64_t my_rdtsc(void)
{
#if defined(__x86_64__)
	// supported by GCC and Clang for x86 platform
	return _rdtsc();
#elif INLINE_ASM && defined(__aarch64__)
	// should be a 64 bits wallclock counter
	// document for old/recent architecture and/or BMC chipsets mention it
	// could be a 56 bit counter.
	uint64_t val;

	asm volatile ("mrs %0, cntvct_el0":"=r" (val));

	// I am not sure what the clock unit is, it depends on pre-scaler setup
	// A multiplication by 32 might be needed on my platform 
	return val * 32;	// aarch64 emulation on x86_64 ?
	return ((val / 3) * 25) << 4;	// maybe for ARM M1 ?
	return val;
#else
#error "todo : unsupported _rdtsc implementation\n"
	return 0;
#endif
}

/* ============= End routines borrowed or adapted from Perig ============== */

static __m512i lo52mask;

#ifdef IFMA

#define MICRO_ECM_FORCE_INLINE __inline

MICRO_ECM_FORCE_INLINE static __m512i mul52hi(__m512i b, __m512i c)
{
	return _mm512_madd52hi_epu64(_mm512_set1_epi64(0), c, b);
}

MICRO_ECM_FORCE_INLINE static __m512i mul52lo(__m512i b, __m512i c)
{
	return _mm512_madd52lo_epu64(_mm512_set1_epi64(0), c, b);
}

MICRO_ECM_FORCE_INLINE static void mul52lohi(__m512i b, __m512i c, __m512i* l, __m512i* h)
{
	*l = _mm512_madd52lo_epu64(_mm512_set1_epi64(0), c, b);
	*h = _mm512_madd52hi_epu64(_mm512_set1_epi64(0), c, b);
	return;
}

#else

static __m512d dbias;
static __m512i vbias1;
static __m512i vbias2;
static __m512i vbias3;

__inline static __m512i mul52lo(__m512i b, __m512i c)
{
	return _mm512_and_si512(_mm512_mullo_epi64(b, c), _mm512_set1_epi64(0x000fffffffffffffull));
}
__inline static __m512i mul52hi(__m512i b, __m512i c)
{
	__m512d prod1_ld = _mm512_cvtepu64_pd(b);
	__m512d prod2_ld = _mm512_cvtepu64_pd(c);
	prod1_ld = _mm512_fmadd_round_pd(prod1_ld, prod2_ld, dbias, (_MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC));
	return _mm512_sub_epi64(castepu(prod1_ld), vbias1);
}
__inline static void mul52lohi(__m512i b, __m512i c, __m512i* l, __m512i* h)
{
	__m512d prod1_ld = _mm512_cvtepu64_pd(b);
	__m512d prod2_ld = _mm512_cvtepu64_pd(c);
	__m512d prod1_hd = _mm512_fmadd_round_pd(prod1_ld, prod2_ld, dbias, (_MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC));
	*h = _mm512_sub_epi64(castepu(prod1_hd), vbias1);
	prod1_hd = _mm512_sub_pd(_mm512_castsi512_pd(vbias2), prod1_hd);
	prod1_ld = _mm512_fmadd_round_pd(prod1_ld, prod2_ld, prod1_hd, (_MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC));
	*l = _mm512_castpd_si512(prod1_ld);
	*l = _mm512_and_si512(*l, lo52mask);
	*h = _mm512_and_si512(*h, lo52mask);
	return;
}

#endif



#ifdef IFMA
#define _mm512_mullo_epi52(c, a, b) \
    c = _mm512_madd52lo_epu64(_mm512_set1_epi64(0), a, b);

#define VEC_MUL_ACCUM_LOHI_PD(a, b, lo, hi) \
    lo = _mm512_madd52lo_epu64(lo, a, b); \
    hi = _mm512_madd52hi_epu64(hi, a, b);
#else

#define VEC_MUL_ACCUM_LOHI_PD(a, b, lo, hi) \
	prod1_ld = _mm512_cvtepu64_pd(a);		\
	prod2_ld = _mm512_cvtepu64_pd(b);		\
    prod1_hd = _mm512_fmadd_round_pd(prod1_ld, prod2_ld, dbias, (_MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC)); \
    hi = _mm512_add_epi64(hi, _mm512_sub_epi64(castepu(prod1_hd), vbias1)); \
    prod1_hd = _mm512_sub_pd(castpd(vbias2), prod1_hd); \
	prod1_ld = _mm512_fmadd_round_pd(prod1_ld, prod2_ld, prod1_hd, (_MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC)); \
	lo = _mm512_add_epi64(lo, _mm512_sub_epi64(castepu(prod1_ld), vbias3));
	
#define VEC_MUL2_ACCUM_LOHI_PD(c, a, b, lo1, hi1, lo2, hi2) \
	prod1_ld = _mm512_cvtepu64_pd(a);		\
	prod2_ld = _mm512_cvtepu64_pd(b);		\
	prod3_ld = _mm512_cvtepu64_pd(c);		\
    prod1_hd = _mm512_fmadd_round_pd(prod1_ld, prod3_ld, dbias, (_MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC)); \
	prod2_hd = _mm512_fmadd_round_pd(prod2_ld, prod3_ld, dbias, (_MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC)); \
    hi1 = _mm512_add_epi64(hi1, _mm512_sub_epi64(castepu(prod1_hd), vbias1)); \
	hi2 = _mm512_add_epi64(hi2, _mm512_sub_epi64(castepu(prod2_hd), vbias1)); \
    prod1_hd = _mm512_sub_pd(castpd(vbias2), prod1_hd); \
	prod2_hd = _mm512_sub_pd(castpd(vbias2), prod2_hd); \
	prod1_ld = _mm512_fmadd_round_pd(prod1_ld, prod3_ld, prod1_hd, (_MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC)); \
	prod2_ld = _mm512_fmadd_round_pd(prod2_ld, prod3_ld, prod2_hd, (_MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC)); \
	lo1 = _mm512_add_epi64(lo1, _mm512_sub_epi64(castepu(prod1_ld), vbias3)); \
	lo2 = _mm512_add_epi64(lo2, _mm512_sub_epi64(castepu(prod2_ld), vbias3));

#define _mm512_mullo_epi52(c, a, b) \
    c = _mm512_and_si512(_mm512_mullo_epi64(a, b), _mm512_set1_epi64(0x000fffffffffffffull));
#endif

void printvec(char* msg, __m512i v)
{
	uint64_t m[8];
	storeu64(m, v);
	int i;
	printf("%s: ", msg);
	for (i = 0; i < 8; i++)
		printf("%016lx ", m[i]);
	printf("\n");
	return;
}


double _difftime(struct timeval* start, struct timeval* end)
{
    double secs;
    double usecs;

    if (start->tv_sec == end->tv_sec) {
        secs = 0;
        usecs = end->tv_usec - start->tv_usec;
    }
    else {
        usecs = 1000000 - start->tv_usec;
        secs = end->tv_sec - (start->tv_sec + 1);
        usecs += end->tv_usec;
        if (usecs >= 1000000) {
            usecs -= 1000000;
            secs += 1;
        }
    }

    return secs + usecs / 1000000.;
}

#define carryprop(lo, hi, mask) \
	{ __m512i carry = _mm512_srli_epi64(lo, 52);	\
	hi = _mm512_add_epi64(hi, carry);		\
	lo = _mm512_and_epi64(mask, lo); }

__m512i _mm512_addsetc_epi52(__m512i a, __m512i b, __mmask8* cout)
{
	__m512i t = _mm512_add_epi64(a, b);
	*cout = _mm512_cmpgt_epu64_mask(t, _mm512_set1_epi64(0xfffffffffffffULL));
	t = _mm512_and_epi64(t, _mm512_set1_epi64(0xfffffffffffffULL));
	return t;
}
__m512i _mm512_mask_addsetc_epi52(__m512i c, __mmask8 mask, __m512i a, __m512i b, __mmask8* cout)
{
	__m512i t = _mm512_add_epi64(a, b);
	*cout = _mm512_mask_cmpgt_epu64_mask(mask, t, _mm512_set1_epi64(0xfffffffffffffULL));
	t = _mm512_mask_and_epi64(c, mask, t, _mm512_set1_epi64(0xfffffffffffffULL));
	return t;
}
__m512i _mm512_subsetc_epi52(__m512i a, __m512i b, __mmask8* cout)
{
	__m512i t = _mm512_sub_epi64(a, b);
	*cout = _mm512_cmpgt_epu64_mask(b, a);
	t = _mm512_and_epi64(t, _mm512_set1_epi64(0xfffffffffffffULL));
	return t;
}
__m512i _mm512_mask_subsetc_epi52(__m512i c, __mmask8 mask, __m512i a, __m512i b, __mmask8* cout)
{
	__m512i t = _mm512_sub_epi64(a, b);
	*cout = _mm512_mask_cmpgt_epu64_mask(mask, b, a);
	t = _mm512_mask_and_epi64(c, mask, t, _mm512_set1_epi64(0xfffffffffffffULL));
	return t;
}
__m512i _mm512_adc_epi52(__m512i a, __mmask8 c, __m512i b, __mmask8* cout)
{
	__m512i t = _mm512_add_epi64(a, b);
	t = _mm512_add_epi64(t, _mm512_maskz_set1_epi64(c, 1));
	*cout = _mm512_cmpgt_epu64_mask(t, _mm512_set1_epi64(0xfffffffffffffULL));
	t = _mm512_and_epi64(t, _mm512_set1_epi64(0xfffffffffffffULL));
	return t;
}
__m512i _mm512_mask_adc_epi52(__m512i a, __mmask8 m, __mmask8 c, __m512i b, __mmask8* cout)
{
	__m512i t = _mm512_add_epi64(a, b);
	t = _mm512_mask_add_epi64(a, m, t, _mm512_maskz_set1_epi64(c, 1));
	*cout = _mm512_cmpgt_epu64_mask(t, _mm512_set1_epi64(0xfffffffffffffULL));
	t = _mm512_and_epi64(t, _mm512_set1_epi64(0xfffffffffffffULL));
	return t;
}
__m512i _mm512_sbb_epi52(__m512i a, __mmask8 c, __m512i b, __mmask8* cout)
{
	__m512i t = _mm512_sub_epi64(a, b);
	*cout = _mm512_cmpgt_epu64_mask(b, a);
	__m512i t2 = _mm512_sub_epi64(t, _mm512_maskz_set1_epi64(c, 1));
	*cout = _mm512_kor(*cout, _mm512_cmpgt_epu64_mask(t2, t));
	t2 = _mm512_and_epi64(t2, _mm512_set1_epi64(0xfffffffffffffULL));
	return t2;
}
__m512i _mm512_mask_sbb_epi52(__m512i a, __mmask8 m, __mmask8 c, __m512i b, __mmask8* cout)
{
	__m512i t = _mm512_mask_sub_epi64(a, m, a, b);
	*cout = _mm512_mask_cmpgt_epu64_mask(m, b, a);
	__m512i t2 = _mm512_mask_sub_epi64(a, m, t, _mm512_maskz_set1_epi64(c, 1));
	*cout = _mm512_kor(*cout, _mm512_mask_cmpgt_epu64_mask(m, t2, t));
	t2 = _mm512_and_epi64(t2, _mm512_set1_epi64(0xfffffffffffffULL));
	return t2;
}


__inline static void mulredc52_mask_add_vec(__m512i* c0, __mmask8 addmsk, __m512i a0, __m512i b0, __m512i n0, __m512i vrho)
{
	// CIOS modular multiplication with normal (negative) single-word nhat
	__m512i m;
	__m512i t0, t1, C1;

#ifndef IFMA
	__m512d prod1_hd, prod2_hd;
	__m512d prod1_ld, prod2_ld;
	__m512i i0, i1;
#endif

	__m512i zero = _mm512_set1_epi64(0);
	__m512i one = _mm512_set1_epi64(1);
	__mmask8 scarry2;
	__mmask8 scarry;

	t0 = t1 = C1 = zero;

	VEC_MUL_ACCUM_LOHI_PD(a0, b0, t0, t1);

	// m0
	m = mul52lo(t0, vrho);

	VEC_MUL_ACCUM_LOHI_PD(m, n0, t0, t1);

	// adding m*n0 will generate exactly one carry from t0.
	t0 = _mm512_add_epi64(t1, one);

	__mmask8 bmsk = _mm512_cmpge_epu64_mask(t0, n0);
	t0 = _mm512_mask_sub_epi64(t0, bmsk, t0, n0);
	
	// conditional addmod (double result)
	t0 = _mm512_mask_slli_epi64(t0, addmsk, t0, 1);
	bmsk = _mm512_mask_cmpgt_epu64_mask(addmsk,t0, n0);
	*c0 = _mm512_mask_sub_epi64(t0, bmsk & addmsk, t0, n0);
	// _mm512_and_epi64(t0, lo52mask);
	
	return;
}

//#define DEBUG_SQRMASKADD

__inline static void mask_mulredc104_vec(__m512i* c1, __m512i* c0, __mmask8 mulmsk,
	__m512i a1, __m512i a0, __m512i b1, __m512i b0, __m512i n1, __m512i n0, __m512i vrho)
{
	// CIOS modular multiplication with normal (negative) single-word nhat
	__m512i m;
	__m512i t0, t1, t2, t3, C1, C2;

#ifndef IFMA
	__m512d prod1_hd, prod2_hd, prod3_hd, prod4_hd;                 // 23
	__m512d prod1_ld, prod2_ld, prod3_ld, prod4_ld, prod5_ld;        // 28
	__m512d dbias = _mm512_castsi512_pd(_mm512_set1_epi64(0x4670000000000000ULL));
	__m512i vbias1 = _mm512_set1_epi64(0x4670000000000000ULL);  // 31
	__m512i vbias2 = _mm512_set1_epi64(0x4670000000000001ULL);  // 31
	__m512i vbias3 = _mm512_set1_epi64(0x4330000000000000ULL);  // 31
	int biascount = 0;
	__m512i i0, i1;
#endif

	__m512i zero = _mm512_set1_epi64(0);
	__m512i one = _mm512_set1_epi64(1);
	__m512i lo52mask = _mm512_set1_epi64(0x000fffffffffffffull);
	__mmask8 scarry2;
	__mmask8 scarry;

	t0 = t1 = t2 = t3 = C1 = C2 = zero;

	VEC_MUL_ACCUM_LOHI_PD(a0, b0, t0, t1);
	VEC_MUL_ACCUM_LOHI_PD(a1, b0, t1, t2);
	//VEC_MUL2_ACCUM_LOHI_PD(b0, a0, a1, t0, t1, C1, t2);
	//t1 = _mm512_add_epi64(t1, C1);

	// m0
	m = mul52lo(t0, vrho);

	VEC_MUL_ACCUM_LOHI_PD(m, n0, t0, C1);
	VEC_MUL_ACCUM_LOHI_PD(m, n1, t1, C2);
	//VEC_MUL2_ACCUM_LOHI_PD(m, n0, n1, t0, C1, t1, C2);

	t1 = _mm512_add_epi64(t1, C1);
	t2 = _mm512_add_epi64(t2, C2);
	// we throw t0 away after this so first propagate its carry.
	t0 = _mm512_add_epi64(t1, _mm512_srli_epi64(t0, 52));
	t1 = t2;
	t2 = C1 = zero;

	VEC_MUL_ACCUM_LOHI_PD(a0, b1, t0, C1);
	VEC_MUL_ACCUM_LOHI_PD(a1, b1, t1, t2);
	//VEC_MUL2_ACCUM_LOHI_PD(b1, a0, a1, t0, C1, t1, t2);

	t1 = _mm512_add_epi64(t1, C1);
	C1 = C2 = zero;

	// m1
	m = mul52lo(t0, vrho);

	VEC_MUL_ACCUM_LOHI_PD(m, n0, t0, C1);
	VEC_MUL_ACCUM_LOHI_PD(m, n1, t1, t2);
	//VEC_MUL2_ACCUM_LOHI_PD(m, n0, n1, t0, C1, t1, t2);

	t1 = _mm512_add_epi64(t1, C1);

	// final carryprop
	carryprop(t0, t1, lo52mask);
	carryprop(t1, t2, lo52mask);
	carryprop(t2, C2, lo52mask);

	scarry = _mm512_cmp_epu64_mask(C2, zero, _MM_CMPINT_GT);

	if (scarry > 0) {
		// conditionally subtract when needed (AMM - only on overflow)
		C1 = _mm512_mask_set1_epi64(zero, _mm512_cmpgt_epi64_mask(n0, t1), 1);
		t1 = _mm512_mask_sub_epi64(t1, scarry, t1, n0);
		t2 = _mm512_mask_sub_epi64(t2, scarry, t2, n1);
		t2 = _mm512_mask_sub_epi64(t2, scarry, t2, C1);
	}
	
	// conditionally subtract when needed.  we only check against overflow,
	// so this is almost-montgomery multiplication
	// *c0 = _mm512_mask_subsetc_epi52(t1, scarry2, t1, n0, &scarry);
	// *c1 = _mm512_mask_sbb_epi52(t2, scarry2, scarry, n1, &scarry);

	// on Zen4-epyc it is slower to do this:
	// *c0 = _mm512_mask_and_epi64(a0, mulmsk, lo52mask, t1);
	// *c1 = _mm512_mask_and_epi64(a1, mulmsk, lo52mask, t2);

	// than this:
	*c0 = _mm512_and_epi64(lo52mask, t1);
	*c1 = _mm512_and_epi64(lo52mask, t2);
	*c0 = _mm512_mask_mov_epi64(*c0, ~mulmsk, a0);
	*c1 = _mm512_mask_mov_epi64(*c1, ~mulmsk, a1);

	return;
}
__inline static void sqrredc104_vec(__m512i* c1, __m512i* c0,
	__m512i a1, __m512i a0, __m512i n1, __m512i n0, __m512i vrho)
{
	// CIOS modular multiplication with normal (negative) single-word nhat
	__m512i m;
	__m512i t0, t1, t2, t3, C1, C2, sqr_lo, sqr_hi;

#ifndef IFMA
	__m512d prod1_hd, prod2_hd, prod3_hd, prod4_hd;                 // 23
	__m512d prod1_ld, prod2_ld, prod3_ld, prod4_ld, prod5_ld;        // 28
	__m512d dbias = _mm512_castsi512_pd(_mm512_set1_epi64(0x4670000000000000ULL));
	__m512i vbias1 = _mm512_set1_epi64(0x4670000000000000ULL);  // 31
	__m512i vbias2 = _mm512_set1_epi64(0x4670000000000001ULL);  // 31
	__m512i vbias3 = _mm512_set1_epi64(0x4330000000000000ULL);  // 31
	int biascount = 0;
	__m512i i0, i1;
#endif

	__m512i zero = _mm512_set1_epi64(0);
	__m512i one = _mm512_set1_epi64(1);
	__m512i lo52mask = _mm512_set1_epi64(0x000fffffffffffffull);
	__mmask8 scarry2;
	__mmask8 scarry;

	t0 = t1 = t2 = t3 = C1 = C2 = sqr_lo = sqr_hi = zero;

	VEC_MUL_ACCUM_LOHI_PD(a1, a0, sqr_lo, sqr_hi);
	t1 = sqr_lo;
	t2 = sqr_hi;
	VEC_MUL_ACCUM_LOHI_PD(a0, a0, t0, t1);

#ifdef DEBUG_SQRMASKADD
	printvec("sqrlo", sqr_lo);
	printvec("sqrhi", sqr_hi);
#endif

#ifdef DEBUG_SQRMASKADD
	printvec("t0 = lo^2 lo", t0);
	printvec("t1 = lo^2 hi + sqrlo", t1);
#endif

	// m0
	m = mul52lo(t0, vrho);

#ifdef DEBUG_SQRMASKADD
	printvec("m = t0 * rho", m);
#endif

	VEC_MUL_ACCUM_LOHI_PD(m, n0, t0, C1);
	VEC_MUL_ACCUM_LOHI_PD(m, n1, t1, C2);

#ifdef DEBUG_SQRMASKADD
	printvec("m*n0 lo", t0);
	printvec("m*n0 hi", C1);
	printvec("m*n1 lo", t1);
	printvec("m*n1 hi", C2);
#endif


	t1 = _mm512_add_epi64(t1, C1);
	t2 = _mm512_add_epi64(t2, C2);
	// we throw t0 away after this so first propagate its carry.
	// t0 = _mm512_add_epi64(t1, one);
	// t0 = _mm512_mask_add_epi64(t1, _mm512_cmpgt_epu64_mask(m, zero), t1, one);
	t0 = _mm512_add_epi64(t1, _mm512_srli_epi64(t0, 52));
	t1 = t2;
	t2 = C1 = C2 = zero;

#ifdef DEBUG_SQRMASKADD
	printvec("t0", t0);
	printvec("t1", t1);
#endif


	VEC_MUL_ACCUM_LOHI_PD(a1, a1, t1, t2);

	t0 = _mm512_add_epi64(t0, sqr_lo);
	t1 = _mm512_add_epi64(t1, sqr_hi);

#ifdef DEBUG_SQRMASKADD
	printvec("t0 = accum(a1*a1<<52+sqrterm)", t0);
	printvec("t1 = accum(a1*a1<<52+sqrterm)", t1);
	printvec("t2 = accum(a1*a1<<52+sqrterm)", t2);
#endif


	// m1
	m = mul52lo(t0, vrho);

#ifdef DEBUG_SQRMASKADD
	printvec("m = t0 * rho", m);
#endif

	VEC_MUL_ACCUM_LOHI_PD(m, n0, t0, C1);
	VEC_MUL_ACCUM_LOHI_PD(m, n1, t1, t2);

#ifdef DEBUG_SQRMASKADD
	printvec("m*n0 lo", t0);
	printvec("m*n0 hi", C1);
	printvec("m*n1 lo", t1);
	printvec("m*n1 hi", C2);
#endif

	t1 = _mm512_add_epi64(t1, C1);

	// final carryprop
	carryprop(t0, t1, lo52mask);
	carryprop(t1, t2, lo52mask);
	carryprop(t2, C2, lo52mask);

#ifdef DEBUG_SQRMASKADD
	printvec("t0 = after carryprop", t0);
	printvec("t1 = after carryprop", t1);
	printvec("t2 = after carryprop", t2);
#endif

	scarry = _mm512_cmp_epu64_mask(C2, zero, _MM_CMPINT_GT);

	if (scarry > 0) {
		// conditionally subtract when needed (AMM - only on overflow)
		__mmask8 bmsk;
		bmsk = _mm512_cmpgt_epi64_mask(n0, t1);
		t1 = _mm512_mask_sub_epi64(t1, scarry, t1, n0);
		t2 = _mm512_mask_sub_epi64(t2, scarry, t2, n1);
		t2 = _mm512_mask_sub_epi64(t2, scarry, t2, _mm512_mask_set1_epi64(zero, bmsk, 1));
	}
	*c0 = _mm512_and_epi64(lo52mask, t1);
	*c1 = _mm512_and_epi64(lo52mask, t2);


#ifdef DEBUG_SQRMASKADD
	printvec("t0 = after modsub (overflow)", t1);
	printvec("t1 = after modsub (overflow)", t2);
#endif

	return;
}
__inline static void mask_sqrredc104_vec(__m512i* c1, __m512i* c0, __mmask8 mulmsk,
	__m512i a1, __m512i a0, __m512i n1, __m512i n0, __m512i vrho)
{
	// CIOS modular multiplication with normal (negative) single-word nhat
	__m512i m;
	__m512i t0, t1, t2, C3, C1, C2, sqr_lo, sqr_hi;

#ifndef IFMA
	__m512d prod1_hd, prod2_hd, prod3_hd, prod4_hd;                 // 23
	__m512d prod1_ld, prod2_ld, prod3_ld, prod4_ld, prod5_ld;        // 28
	__m512d dbias = _mm512_castsi512_pd(_mm512_set1_epi64(0x4670000000000000ULL));
	__m512i vbias1 = _mm512_set1_epi64(0x4670000000000000ULL);  // 31
	__m512i vbias2 = _mm512_set1_epi64(0x4670000000000001ULL);  // 31
	__m512i vbias3 = _mm512_set1_epi64(0x4330000000000000ULL);  // 31
	int biascount = 0;
	__m512i i0, i1;
#endif

	__m512i zero = _mm512_set1_epi64(0);
	__m512i one = _mm512_set1_epi64(1);
	__m512i lo52mask = _mm512_set1_epi64(0x000fffffffffffffull);
	__mmask8 scarry2;
	__mmask8 scarry;

	t0 = t1 = t2 = C1 = C2 = C3 = sqr_lo = sqr_hi = zero;

	VEC_MUL_ACCUM_LOHI_PD(a1, a0, sqr_lo, sqr_hi);
	t1 = sqr_lo;
	t2 = sqr_hi;
	VEC_MUL_ACCUM_LOHI_PD(a0, a0, t0, t1);

	// m0
	m = mul52lo(t0, vrho);

	VEC_MUL_ACCUM_LOHI_PD(m, n0, t0, C1);
	VEC_MUL_ACCUM_LOHI_PD(m, n1, t1, C2);

	t1 = _mm512_add_epi64(t1, C1);
	t2 = _mm512_add_epi64(t2, C2);
	// we throw t0 away after this so first propagate its carry.
	// t0 = _mm512_add_epi64(t1, one);
	// t0 = _mm512_mask_add_epi64(t1, _mm512_cmpgt_epu64_mask(m, zero), t1, one);
	t0 = _mm512_add_epi64(t1, _mm512_srli_epi64(t0, 52));
	t1 = t2;
	t2 = C1 = C2 = zero;

	VEC_MUL_ACCUM_LOHI_PD(a1, a1, t1, t2);

	t0 = _mm512_add_epi64(t0, sqr_lo);
	t1 = _mm512_add_epi64(t1, sqr_hi);

	// m1
	m = mul52lo(t0, vrho);

	VEC_MUL_ACCUM_LOHI_PD(m, n0, t0, C1);
	VEC_MUL_ACCUM_LOHI_PD(m, n1, t1, t2);

	t1 = _mm512_add_epi64(t1, C1);

	// final carryprop
	carryprop(t0, t1, lo52mask);
	carryprop(t1, t2, lo52mask);

	carryprop(t2, C2, lo52mask);
	scarry = _mm512_cmp_epu64_mask(C2, zero, _MM_CMPINT_GT);

	//scarry = _mm512_cmpge_epu64_mask(t2, n1);

	if (scarry > 0) {
		// conditionally subtract when needed (AMM - only on overflow)
		C1 = _mm512_mask_set1_epi64(zero, _mm512_cmpgt_epi64_mask(n0, t1), 1);
		t1 = _mm512_mask_sub_epi64(t1, scarry, t1, n0);
		t2 = _mm512_mask_sub_epi64(t2, scarry, t2, n1);
		t2 = _mm512_mask_sub_epi64(t2, scarry, t2, C1);
	}

	// on Zen4-epyc it is slower to do this:
	// *c0 = _mm512_mask_and_epi64(a0, mulmsk, lo52mask, t1);
	// *c1 = _mm512_mask_and_epi64(a1, mulmsk, lo52mask, t2);

	// than this:
	*c0 = _mm512_and_epi64(lo52mask, t1);
	*c1 = _mm512_and_epi64(lo52mask, t2);
	*c0 = _mm512_mask_mov_epi64(*c0, ~mulmsk, a0);
	*c1 = _mm512_mask_mov_epi64(*c1, ~mulmsk, a1);

	return;
}
__inline static void sqrredc_maskadd_vec(__m512i* a1, __m512i* a0, 
	__mmask8 addmsk, __mmask8 protectmsk,
	__m512i n1, __m512i n0, __m512i vrho)
{
	// CIOS modular multiplication with normal (negative) single-word nhat
	__m512i m, t0, t1, t2, t3, C1, C2, sqr_lo, sqr_hi;

#ifndef IFMA
	__m512d prod1_hd;
	__m512d prod1_ld, prod2_ld;
#endif

	__m512i zero = _mm512_set1_epi64(0);
	__m512i one = _mm512_set1_epi64(1);
	__m512i lo52mask = _mm512_set1_epi64(0x000fffffffffffffull);
	__mmask8 scarry;

	t0 = t3 = C1 = C2 = sqr_lo = sqr_hi = zero;

	VEC_MUL_ACCUM_LOHI_PD(*a1, *a0, sqr_lo, sqr_hi);
	t1 = sqr_lo;
	t2 = sqr_hi;

#ifdef DEBUG_SQRMASKADD
	printvec("sqrlo", sqr_lo);
	printvec("sqrhi", sqr_hi);
#endif
	
	VEC_MUL_ACCUM_LOHI_PD(*a0, *a0, t0, t1);

#ifdef DEBUG_SQRMASKADD
	printvec("t0 = lo^2 lo", t0);
	printvec("t1 = lo^2 hi + sqrlo", t1);
#endif
	

	// m0
	//_mm512_mullo_epi52(m, vrho, t0);
	m = mul52lo(t0, vrho);

#ifdef DEBUG_SQRMASKADD
	printvec("m = t0 * rho", m);
#endif
	

	VEC_MUL_ACCUM_LOHI_PD(m, n0, t0, C1);
	VEC_MUL_ACCUM_LOHI_PD(m, n1, t1, C2);

#ifdef DEBUG_SQRMASKADD
	printvec("m*n0 lo", t0);
	printvec("m*n0 hi", C1);
	printvec("m*n1 lo", t1);
	printvec("m*n1 hi", C2);
#endif
	

	t1 = _mm512_add_epi64(t1, C1);
	t2 = _mm512_add_epi64(t2, C2);
	// we throw t0 away after this so first propagate its carry.
	t0 = _mm512_add_epi64(t1, one);
	t1 = t2;
	t2 = C1 = C2 = zero;

#ifdef DEBUG_SQRMASKADD
	printvec("t0", t0);
	printvec("t1", t1);
#endif
	

	VEC_MUL_ACCUM_LOHI_PD(*a1, *a1, t1, t2);

	t0 = _mm512_add_epi64(t0, sqr_lo);
	t1 = _mm512_add_epi64(t1, sqr_hi);

#ifdef DEBUG_SQRMASKADD
	printvec("t0 = accum(a1*a1<<52+sqrterm)", t0);
	printvec("t1 = accum(a1*a1<<52+sqrterm)", t1);
	printvec("t2 = accum(a1*a1<<52+sqrterm)", t2);
#endif
	

	// m1
	m = mul52lo(t0, vrho);

#ifdef DEBUG_SQRMASKADD
	printvec("m = t0 * rho", m);
#endif
	

	VEC_MUL_ACCUM_LOHI_PD(m, n0, t0, C1);
	VEC_MUL_ACCUM_LOHI_PD(m, n1, t1, t2);

#ifdef DEBUG_SQRMASKADD
	printvec("m*n0 lo", t0);
	printvec("m*n0 hi", C1);
	printvec("m*n1 lo", t1);
	printvec("m*n1 hi", C2);
#endif
	

	t1 = _mm512_add_epi64(t1, C1);

	// final carryprop
	carryprop(t0, t1, lo52mask);
	carryprop(t1, t2, lo52mask);
	carryprop(t2, C2, lo52mask);

#ifdef DEBUG_SQRMASKADD
	printvec("t0 = after carryprop", t0);
	printvec("t1 = after carryprop", t1);
	printvec("t2 = after carryprop", t2);
#endif

	scarry = _mm512_cmp_epu64_mask(C2, zero, _MM_CMPINT_GT);

	if (protectmsk)
	{
		scarry |=
			_mm512_mask_cmpgt_epu64_mask(protectmsk, t2, n1) |
			(_mm512_mask_cmpeq_epu64_mask(protectmsk, t2, n1) & 
				_mm512_mask_cmpge_epu64_mask(protectmsk, t1, n0));
	}

	// conditionally subtract when needed (AMM - only on overflow)
	t1 = _mm512_mask_sub_epi64(t1, scarry, t1, n0);
	t2 = _mm512_mask_sub_epi64(t2, scarry, t2, n1);
	C1 = _mm512_srli_epi64(t1, 63);
	t2 = _mm512_mask_sub_epi64(t2, scarry, t2, C1);
	t1 = _mm512_and_epi64(lo52mask, t1);
	t2 = _mm512_and_epi64(lo52mask, t2);

#ifdef DEBUG_SQRMASKADD
	printvec("t0 = after modsub (overflow)", t1);
	printvec("t1 = after modsub (overflow)", t2);
#endif

	// conditional addmod (double result)
	// add
	t1 = _mm512_mask_slli_epi64(t1, addmsk, t1, 1);
	t2 = _mm512_mask_slli_epi64(t2, addmsk, t2, 1);
	// when doubling, it is safe to check both carries before adding
	// in the previous carry, because the shift makes room for
	// the previous carry.  So either the upper word shift generates
	// a carry or doesn't, the addition won't cause one.
	C1 = _mm512_srli_epi64(t1, 52);
	C2 = _mm512_srli_epi64(t2, 52);
	t2 = _mm512_add_epi64(t2, C1);
	t1 = _mm512_and_epi64(lo52mask, t1);
	t2 = _mm512_and_epi64(lo52mask, t2);

#ifdef DEBUG_SQRMASKADD
	printvec("t0 = after double", t1);
	printvec("t1 = after double", t2);
	printvec("t2 = after double", C2);
#endif

	// We double a result that could be larger than N, therefore
	// the result could be more than twice as large as N and may
	// have two overflow bits.  So we need to conditionally sub 
	// up to two times.
	__mmask8 bmsk = _mm512_mask_cmpgt_epu64_mask(addmsk, C2, zero);

	// conditionally subtract N (AMM - on overflow only)
	t1 = _mm512_mask_sub_epi64(t1, bmsk & addmsk, t1, n0);
	t2 = _mm512_mask_sub_epi64(t2, bmsk & addmsk, t2, n1);
	C1 = _mm512_srli_epi64(t1, 63);
	t2 = _mm512_mask_sub_epi64(t2, bmsk & addmsk, t2, C1);

	if (protectmsk)
	{
		//t1 = _mm512_mask_subsetc_epi52(t1, bmsk & addmsk, t1, n0, &scarry);
		//t2 = _mm512_mask_sbb_epi52(t2, bmsk & addmsk, scarry, n1, &scarry);
		//C2 = _mm512_mask_sbb_epi52(C2, bmsk & addmsk, scarry, zero, &scarry);
		
		C2 = _mm512_mask_sub_epi64(C2, bmsk & addmsk & protectmsk, C2, _mm512_srli_epi64(t2, 63));
		t2 = _mm512_mask_and_epi64(t2, bmsk & addmsk & protectmsk, lo52mask, t2);
		bmsk = _mm512_mask_cmpgt_epu64_mask(addmsk & protectmsk, C2, zero);
		t1 = _mm512_mask_subsetc_epi52(t1, bmsk & addmsk & protectmsk, t1, n0, &scarry);
		t2 = _mm512_mask_sbb_epi52(t2, bmsk & addmsk & protectmsk, scarry, n1, &scarry);

#ifdef DEBUG_SQRMASKADD
		C2 = _mm512_mask_sbb_epi52(C2, bmsk & addmsk, scarry, zero, &scarry);
		if (_mm512_cmpgt_epi64_mask(C2, zero) & bmsk & addmsk)
		{
			printf("assert failed, still a carry after conditional subtract\n");
			printvec("t0 ", t1);
			printvec("t1 ", t2);
			printvec("t2 ", C2);
			printvec("in1", *a1);
			printvec("in0", *a0);
			printvec("n1 ", n1);
			printvec("n0 ", n0);
			printvec("rho", vrho);
			exit(1);
		}
#endif
	}

	//C2 = _mm512_sub_epi64(C2, _mm512_srli_epi64(t2, 63));
	//bmsk = _mm512_mask_cmpgt_epu64_mask(addmsk, C2, zero);
	//
	//// conditionally subtract N (AMM - on overflow only)
	//t1 = _mm512_mask_sub_epi64(t1, bmsk & addmsk, t1, n0);
	//t2 = _mm512_mask_sub_epi64(t2, bmsk & addmsk, t2, n1);
	//C1 = _mm512_srli_epi64(t1, 63);
	//t2 = _mm512_mask_sub_epi64(t2, bmsk & addmsk, t2, C1);

	*a0 = _mm512_and_epi64(lo52mask, t1);
	*a1 = _mm512_and_epi64(lo52mask, t2);

#ifdef DEBUG_SQRMASKADD
	printvec("t0 = after modsub (overflow)", t1);
	printvec("t1 = after modsub (overflow)", t2);
#endif
	

	return;
}
__inline static void addmod104_x8(__m512i* c1, __m512i* c0, __m512i a1, __m512i a0, 
	__m512i b1, __m512i b0, __m512i n1, __m512i n0)
{
	// add
	__mmask8 bmsk;
	//a0 = _mm512_addsetc_epi52(a0, b0, &bmsk);
	//a1 = _mm512_adc_epi52(a1, bmsk, b1, &bmsk);
	a0 = _mm512_add_epi64(a0, b0);
	a1 = _mm512_add_epi64(a1, b1);
	a1 = _mm512_add_epi64(a1, _mm512_srli_epi64(a0, 52));
	a0 = _mm512_and_epi64(a0, lo52mask);

	// compare
	//__mmask8 msk = bmsk | _mm512_cmpgt_epu64_mask(a1, n1);
	__mmask8 msk = _mm512_cmpgt_epu64_mask(a1, n1);
	msk |= (_mm512_cmpeq_epu64_mask(a1, n1) & _mm512_cmpge_epu64_mask(a0, n0));

	// conditionally subtract N
	*c0 = _mm512_mask_subsetc_epi52(a0, msk, a0, n0, &bmsk);
	*c1 = _mm512_mask_sbb_epi52(a1, msk, bmsk, n1, &bmsk);
	// *c0 = _mm512_mask_sub_epi64(a0, msk, a0, n0);
	// *c1 = _mm512_mask_sub_epi64(a1, msk, a1, n1);
	// *c1 = _mm512_mask_sub_epi64(*c1, msk, *c1, _mm512_srli_epi64(*c0, 63));
	return;
}
__inline static void mask_addmod104_x8(__m512i* c1, __m512i* c0, __mmask8 addmsk, 
	__m512i a1, __m512i a0, __m512i b1, __m512i b0, __m512i n1, __m512i n0)
{
	// add
	__mmask8 bmsk;
	a0 = _mm512_mask_addsetc_epi52(a0, addmsk, a0, b0, &bmsk);
	a1 = _mm512_mask_adc_epi52(a1, addmsk, bmsk, b1, &bmsk);

	// compare
	__mmask8 msk = bmsk | _mm512_cmpgt_epu64_mask(a1, n1);
	msk |= (_mm512_cmpeq_epu64_mask(a1, n1) & _mm512_cmpge_epu64_mask(a0, n0));

	// conditionally subtract N
	*c0 = _mm512_mask_subsetc_epi52(a0, addmsk & msk, a0, n0, &bmsk);
	*c1 = _mm512_mask_sbb_epi52(a1, addmsk & msk, bmsk, n1, &bmsk);
	return;
}
__inline static void mask_dblmod104_x8(__m512i* c1, __m512i* c0, __mmask8 addmsk,
	__m512i a1, __m512i a0, __m512i n1, __m512i n0)
{
	// add
	__mmask8 bmsk;
	//a0 = _mm512_mask_addsetc_epi52(a0, addmsk, a0, b0, &bmsk);
	//a1 = _mm512_mask_adc_epi52(a1, addmsk, bmsk, b1, &bmsk);

	a0 = _mm512_mask_slli_epi64(a0, addmsk, a0, 1);
	a1 = _mm512_mask_slli_epi64(a1, addmsk, a1, 1);
	// when doubling, it is safe to check both carries before adding
	// in the previous carry, because the shift makes room for
	// the previous carry.  So either the upper word shift generates
	// a carry or doesn't, the addition won't cause one.
	a1 = _mm512_add_epi64(a1, _mm512_srli_epi64(a0, 52));
	a0 = _mm512_and_epi64(lo52mask, a0);

	// compare
	__mmask8 msk = _mm512_cmpgt_epu64_mask(a1, n1);
	msk |= (_mm512_cmpeq_epu64_mask(a1, n1) & _mm512_cmpge_epu64_mask(a0, n0));

	// conditionally subtract N
	*c0 = _mm512_mask_subsetc_epi52(a0, addmsk & msk, a0, n0, &bmsk);
	*c1 = _mm512_mask_sbb_epi52(a1, addmsk & msk, bmsk, n1, &bmsk);
	return;
}
__inline static void mask_redsub104_x8(__m512i* c1, __m512i* c0, __mmask8 addmsk,
	__m512i a1, __m512i a0, __m512i n1, __m512i n0)
{
	__mmask8 bmsk;

	// compare
	__mmask8 msk = _mm512_cmpgt_epu64_mask(a1, n1);
	msk |= (_mm512_cmpeq_epu64_mask(a1, n1) & _mm512_cmpge_epu64_mask(a0, n0));

	// conditionally subtract N
	*c0 = _mm512_mask_subsetc_epi52(a0, addmsk & msk, a0, n0, &bmsk);
	*c1 = _mm512_mask_sbb_epi52(a1, addmsk & msk, bmsk, n1, &bmsk);
	return;
}
__inline static void redsub104_x8(__m512i* c1, __m512i* c0, 
	__m512i a1, __m512i a0, __m512i n1, __m512i n0)
{
	__mmask8 bmsk;

	// compare
	__mmask8 msk = _mm512_cmpgt_epu64_mask(a1, n1);
	msk |= (_mm512_cmpeq_epu64_mask(a1, n1) & _mm512_cmpge_epu64_mask(a0, n0));

	// conditionally subtract N
	*c0 = _mm512_mask_subsetc_epi52(a0, msk, a0, n0, &bmsk);
	*c1 = _mm512_mask_sbb_epi52(a1, msk, bmsk, n1, &bmsk);

	return;
}
__inline static void submod104_x8(__m512i* c1, __m512i* c0, __m512i a1, __m512i a0,
	__m512i b1, __m512i b0, __m512i n1, __m512i n0)
{
	// compare
	__mmask8 msk = _mm512_cmplt_epu64_mask(a1, b1);
	msk |= _mm512_cmpeq_epu64_mask(a1, b1) & _mm512_cmplt_epu64_mask(a0, b0);

	// subtract
	__mmask8 bmsk;
	a0 = _mm512_subsetc_epi52(a0, b0, &bmsk);
	a1 = _mm512_sbb_epi52(a1, bmsk, b1, &bmsk);

	// conditionally add N
	*c0 = _mm512_mask_addsetc_epi52(a0, msk, a0, n0, &bmsk);
	*c1 = _mm512_mask_adc_epi52(a1, msk, bmsk, n1, &bmsk);
	return;
}

static uint64_t multiplicative_inverse(uint64_t a)
{
	// compute the 64-bit inverse of a mod 2^64
	//    assert(a%2 == 1);  // the inverse (mod 2<<64) only exists for odd values
	uint64_t x0 = (3 * a) ^ 2;
	uint64_t y = 1 - a * x0;
	uint64_t x1 = x0 * (1 + y);
	y *= y;
	uint64_t x2 = x1 * (1 + y);
	y *= y;
	uint64_t x3 = x2 * (1 + y);
	y *= y;
	uint64_t x4 = x3 * (1 + y);
	return x4;
}

static __m512i multiplicative_inverse104_x8(uint64_t* a)
{
	//    assert(a%2 == 1);  // the inverse (mod 2<<64) only exists for odd values
	__m512i x0, x1, x2, x3, x4, x5, y, n, i0, i1;
	__m512i three = _mm512_set1_epi64(3), two = _mm512_set1_epi64(2), one = _mm512_set1_epi64(1);
	__m512i lo52mask = _mm512_set1_epi64(0x000fffffffffffffull);

	n = loadu64(a);
	_mm512_mullo_epi52(x0, n, three);
	x0 = _mm512_xor_epi64(x0, two);
	_mm512_mullo_epi52(y, n, x0);
	y = _mm512_sub_epi64(one, y);
	y = _mm512_and_epi64(lo52mask, y);

	x1 = _mm512_add_epi64(y, one);
	_mm512_mullo_epi52(x1, x0, x1);
	_mm512_mullo_epi52(y, y, y);

	x2 = _mm512_add_epi64(y, one);
	_mm512_mullo_epi52(x2, x1, x2);
	_mm512_mullo_epi52(y, y, y);

	x3 = _mm512_add_epi64(y, one);
	_mm512_mullo_epi52(x3, x2, x3);
	_mm512_mullo_epi52(y, y, y);

	x4 = _mm512_add_epi64(y, one);
	_mm512_mullo_epi52(x4, x3, x4);

	return x4;
}

#if defined(INTEL_COMPILER) || defined(INTEL_LLVM_COMPILER)
#define ROUNDING_MODE (_MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC)
#else
#define ROUNDING_MODE _MM_FROUND_CUR_DIRECTION
#endif


__m512i rem_epu64_x8(__m512i n, __m512i d)
{
	// DANGER: I haven't proven this works for every possible input.
	__m512d d1pd = _mm512_cvtepu64_pd(d);
	__m512d n1pd = _mm512_cvtepu64_pd(n);
	__m512i q, q2, r;

	//n1pd = _mm512_div_pd(n1pd, d1pd);
	//q = _mm512_cvt_roundpd_epu64(n1pd, (_MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC));
	
	n1pd = _mm512_div_round_pd(n1pd, d1pd, ROUNDING_MODE);
	q = _mm512_cvttpd_epu64(n1pd);
	
	__m512i qd = _mm512_mullox_epi64(q, d);
	r = _mm512_sub_epi64(n, qd);

	// fix q too big by a little, with special check for
	// numerators close to 2^64 and denominators close to 1
	// DANGER: the special check is unused for 64-bits, only for 32-bits.
	// This routine is only used in modmul32 and input numerators
	// shouldn't get that large in normal cases.  The factor base
	// would need to be close to 2^32...
	__mmask8 err = _mm512_cmpgt_epu64_mask(r, n); // |
		//(_mm512_cmpgt_epu64_mask(r, d) & _mm512_cmplt_epu64_mask(
		//	_mm512_sub_epi64(_mm512_set1_epi64(0), r), _mm512_set1_epi64(1024)));
	if (err)
	{
		n1pd = _mm512_cvtepu64_pd(_mm512_sub_epi64(_mm512_set1_epi64(0), r));
		
		//n1pd = _mm512_div_pd(n1pd, d1pd);
		//q2 = _mm512_add_epi64(_mm512_set1_epi64(1), _mm512_cvt_roundpd_epu64(n1pd,
		//	(_MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC)));

		n1pd = _mm512_div_round_pd(n1pd, d1pd, ROUNDING_MODE);
		q2 = _mm512_add_epi64(_mm512_set1_epi64(1), _mm512_cvttpd_epu64(n1pd));

		q = _mm512_mask_sub_epi64(q, err, q, q2);
		r = _mm512_mask_add_epi64(r, err, r, _mm512_mullox_epi64(q2, d));
	}

	// fix q too small by a little bit
	err = _mm512_cmpge_epu64_mask(r, d);
	if (err)
	{
		n1pd = _mm512_cvtepu64_pd(r);
		
		//n1pd = _mm512_div_pd(n1pd, d1pd);
		//q2 = _mm512_cvt_roundpd_epu64(n1pd,
		//	(_MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC));

		n1pd = _mm512_div_round_pd(n1pd, d1pd, ROUNDING_MODE);
		q2 = _mm512_cvttpd_epu64(n1pd);

		q = _mm512_mask_add_epi64(q, err, q, q2);
		r = _mm512_mask_sub_epi64(r, err, r, _mm512_mullox_epi64(q2, d));
	}

	return r;
}

// a Fermat PRP test on 8x 52-bit inputs
uint8_t fermat_prp_52x8(uint64_t* n)
{
	// assumes has no small factors.  assumes n <= 52 bits.
	// assumes n is a list of 8 52-bit integers
	// do a base-2 fermat prp test on each using LR binexp.
	__m512i vrho = multiplicative_inverse104_x8(n);
	__m512i unity;
	__m512i r;
	__m512i nvec;
	__m512i evec;
	__m512i m;
	__m512i zero = _mm512_setzero_si512();
	__m512i one = _mm512_set1_epi64(1);

	vrho = _mm512_and_epi64(_mm512_sub_epi64(zero, vrho), lo52mask);
	nvec = loadu64(n);
	evec = _mm512_sub_epi64(nvec, one);

#if defined(INTEL_COMPILER) || defined(INTEL_LLVM_COMPILER)
	r = _mm512_rem_epu64(_mm512_set1_epi64(1ULL<<52), nvec);
#else
	r = rem_epu64_x8(_mm512_set1_epi64(1ULL<<52), nvec);
#endif

	// penultimate-hi-bit mask
	m = _mm512_sub_epi64(_mm512_set1_epi64(62), _mm512_lzcnt_epi64(evec));
	m = _mm512_sllv_epi64(_mm512_set1_epi64(1), m);

	// we know the first bit is set and the first squaring is of unity,
	// so we can do the first iteration manually with no squaring.
	unity = r;

	r = _mm512_add_epi64(r, r);
	__mmask8 ge = _mm512_cmpge_epi64_mask(r, nvec);
	r = _mm512_mask_sub_epi64(r, ge, r, nvec);

	while (_mm512_cmpgt_epu64_mask(m, zero))
	{
		__mmask8 bitcmp = _mm512_test_epi64_mask(m, evec);
		mulredc52_mask_add_vec(&r, bitcmp, r, r, nvec, vrho);
		m = _mm512_srli_epi64(m, 1);
	}

	// AMM possibly needs a final correction by n
	ge = _mm512_cmpge_epi64_mask(r, nvec);
	r = _mm512_mask_sub_epi64(r, ge, r, nvec);

	return _mm512_cmpeq_epu64_mask(unity, r);
}

// a Fermat PRP test on 8x 104-bit inputs
uint8_t fermat_prp_104x8(uint64_t* n)
{
	// assumes has no small factors.  assumes n >= 54 bits.
	// assumes n is a list of 8 104-bit integers (16 52-bit words)
	// in the format: 8 lo-words, 8 hi-words.
	// do a base-2 fermat prp test on each using LR binexp.
	__m512i vrho = multiplicative_inverse104_x8(n);
	vec_u104_t unity;
	vec_u104_t r;
	__m512i nvec[2];
	__m512i evec[2];
	__m512i m;
	__m512i zero = _mm512_setzero_si512();
	__m512i one = _mm512_set1_epi64(1);
	__m512i lo52mask = _mm512_set1_epi64(0x000fffffffffffffull);
	uint64_t tmp = 0;

	vrho = _mm512_and_epi64(_mm512_sub_epi64(zero, vrho), lo52mask);

	nvec[0] = loadu64(&n[0]);
	nvec[1] = loadu64(&n[8]);
	submod104_x8(&evec[1], &evec[0], 
		nvec[1], nvec[0], zero, one, nvec[1], nvec[0]);

#ifdef DEBUG_SQRMASKADD
	printvec("n'", vrho);
	printvec("n1", nvec[1]);
	printvec("n0", nvec[0]);
#endif
	
	// the 128-bit division we do the slow way
	int i;
	for (i = 0; i < 8; i++)
	{
		uint128_t mod = ((uint128_t)n[i + 8] << 52) + n[i];
		uint128_t t = (uint128_t)1 << 104;
		t %= mod;

		unity.data[0][i] = (uint64_t)t & 0xfffffffffffffULL;
		unity.data[1][i] = (uint64_t)(t >> 52);

		r.data[0][i] = unity.data[0][i];
		r.data[1][i] = unity.data[1][i];
	}

	// penultimate-hi-bit mask
	m = _mm512_sub_epi64(_mm512_set1_epi64(62), _mm512_lzcnt_epi64(evec[1]));
	m = _mm512_sllv_epi64(_mm512_set1_epi64(1), m);
	m = _mm512_mask_set1_epi64(m, _mm512_cmple_epu64_mask(evec[1], one), 0);

	__mmask8 protect = _mm512_cmpgt_epi64_mask(_mm512_srli_epi64(evec[1], 49), zero);

	// we know the first bit is set and the first squaring is of unity,
	// so we can do the first iteration manually with no squaring.
	// Note: the first 5 iterations can be done much more cheaply in
	// single precision and then converted into montgomery representation,
	// but that would require a 208-bit division; not worth it.
	__m512i r0 = loadu64(r.data[0]);
	__m512i r1 = loadu64(r.data[1]);

#ifdef DEBUG_SQRMASKADD
	printvec("one1", r1);
	printvec("one0", r0);
#endif

	addmod104_x8(&r1, &r0, r1, r0, r1, r0, nvec[1], nvec[0]);

#ifdef DEBUG_SQRMASKADD
	printvec("two1", r1);
	printvec("two0", r0);
#endif
	
	__mmask8 done;
	if (protect)
	{
		done = _mm512_cmpeq_epu64_mask(m, zero);
		while (done != 0xff)
		{
			__mmask8 bitcmp = _mm512_test_epi64_mask(m, evec[1]);

#ifdef DEBUG_SQRMASKADD
			printf("mask = %u", bitcmp & 1);
			for (i = 1; i < 8; i++) printf(",%u", bitcmp & (1 << i));
			printf("\n");
#endif

			//sqrredc_maskadd_vec(&r1, &r0, bitcmp, protect, nvec[1], nvec[0], vrho);
			mask_sqrredc104_vec(&r1, &r0, ~done, r1, r0, nvec[1], nvec[0], vrho);
			mask_redsub104_x8(&r1, &r0, (~done) & protect, r1, r0, nvec[1], nvec[0]);
#ifdef DEBUG_SQRMASKADD
				printvec("r1 after protect redsub", r1);
				printvec("r0 after protect redsub", r0);
#endif
			mask_dblmod104_x8(&r1, &r0, (~done) & bitcmp, r1, r0, nvec[1], nvec[0]);

#ifdef DEBUG_SQRMASKADD
			printvec("r1 after dblmod", r1);
			printvec("r0 after dblmod", r0);
#endif

			m = _mm512_srli_epi64(m, 1);
			done = _mm512_cmpeq_epu64_mask(m, zero);
		}
	}
	else
	{

		__mmask8 done = _mm512_cmpeq_epu64_mask(m, zero);
		while (done != 0xff)
		{
			__mmask8 bitcmp = _mm512_test_epi64_mask(m, evec[1]);

#ifdef DEBUG_SQRMASKADD
			printf("mask = %u", bitcmp & 1);
			for (i = 1; i < 8; i++) printf(",%u", bitcmp & (1 << i));
			printf("\n");
#endif

			//sqrredc_maskadd_vec(&r1, &r0, bitcmp, protect, nvec[1], nvec[0], vrho);
			mask_sqrredc104_vec(&r1, &r0, ~done, r1, r0, nvec[1], nvec[0], vrho);
			mask_dblmod104_x8(&r1, &r0, (~done) & bitcmp, r1, r0, nvec[1], nvec[0]);

#ifdef DEBUG_SQRMASKADD
			printvec("r1 after dblmod", r1);
			printvec("r0 after dblmod", r0);
#endif

			m = _mm512_srli_epi64(m, 1);
			done = _mm512_cmpeq_epu64_mask(m, zero);
		}
	}

	m = _mm512_sub_epi64(_mm512_set1_epi64(62), _mm512_lzcnt_epi64(evec[0]));
	m = _mm512_sllv_epi64(_mm512_set1_epi64(1), m);
	m = _mm512_mask_set1_epi64(m, _mm512_cmpge_epu64_mask(evec[1], one), 1ULL << 51);

	if (protect)
	{
		done = _mm512_cmpeq_epu64_mask(m, zero);
		while (done != 0xff)
		{
			__mmask8 bitcmp = _mm512_test_epi64_mask(m, evec[0]);

#ifdef DEBUG_SQRMASKADD
			printf("mask = %u", bitcmp & 1);
			for (i = 1; i < 8; i++) printf(",%u", bitcmp & (1 << i));
			printf("\n");
#endif

			//sqrredc_maskadd_vec(&r1, &r0, bitcmp, protect, nvec[1], nvec[0], vrho);
			mask_sqrredc104_vec(&r1, &r0, ~done, r1, r0, nvec[1], nvec[0], vrho);
			mask_redsub104_x8(&r1, &r0, (~done) & protect, r1, r0, nvec[1], nvec[0]);
#ifdef DEBUG_SQRMASKADD
				printvec("r1 after protect redsub", r1);
				printvec("r0 after protect redsub", r0);
#endif
			mask_dblmod104_x8(&r1, &r0, (~done) & bitcmp, r1, r0, nvec[1], nvec[0]);

#ifdef DEBUG_SQRMASKADD
			printvec("r1 after dblmod", r1);
			printvec("r0 after dblmod", r0);
#endif

			m = _mm512_srli_epi64(m, 1);
			done = _mm512_cmpeq_epu64_mask(m, zero);
		}
		}
	else
	{

		__mmask8 done = _mm512_cmpeq_epu64_mask(m, zero);
		while (done != 0xff)
		{
			__mmask8 bitcmp = _mm512_test_epi64_mask(m, evec[0]);

#ifdef DEBUG_SQRMASKADD
			printf("mask = %u", bitcmp & 1);
			for (i = 1; i < 8; i++) printf(",%u", bitcmp & (1 << i));
			printf("\n");
#endif

			//sqrredc_maskadd_vec(&r1, &r0, bitcmp, protect, nvec[1], nvec[0], vrho);
			mask_sqrredc104_vec(&r1, &r0, ~done, r1, r0, nvec[1], nvec[0], vrho);
			mask_dblmod104_x8(&r1, &r0, (~done) & bitcmp, r1, r0, nvec[1], nvec[0]);

#ifdef DEBUG_SQRMASKADD
			printvec("r1 after dblmod", r1);
			printvec("r0 after dblmod", r0);
#endif

			m = _mm512_srli_epi64(m, 1);
			done = _mm512_cmpeq_epu64_mask(m, zero);
	}
	}

#ifdef DEBUG_SQRMASKADD
	printvec("r1", r1);
	printvec("r0", r0);
#endif

	// AMM possibly needs a final correction by n
	mask_redsub104_x8(&r1, &r0, 0xff, r1, r0, nvec[1], nvec[0]);

#ifdef DEBUG_SQRMASKADD
	printvec("r1", r1);
	printvec("r0", r0);
#endif
	
	uint8_t isprp = 
		_mm512_cmpeq_epu64_mask(loadu64(unity.data[0]), r0) &
		_mm512_cmpeq_epu64_mask(loadu64(unity.data[1]), r1);

#ifdef DEBUG_SQRMASKADD
	printf("result mask = %02x\n", isprp);
#endif
	
	
	return isprp;
}

// a Miller-Rabin SPRP test on 8x 52-bit inputs using base 2
uint8_t MR_2sprp_52x8(uint64_t* n)
{
	// assumes has no small factors.  assumes n <= 52 bits.
	// assumes n is a list of 8 52-bit integers
	// do a base-2 MR sprp test on each using LR binexp.
	__m512i vrho = multiplicative_inverse104_x8(n);
	__m512i unity;
	__m512i r;
	__m512i nvec;
	__m512i evec;
	__m512i m;
	__m512i zero = _mm512_setzero_si512();
	__m512i one = _mm512_set1_epi64(1);

	vrho = _mm512_and_epi64(_mm512_sub_epi64(zero, vrho), lo52mask);
	nvec = loadu64(n);
	evec = _mm512_sub_epi64(nvec, one);

#if defined(INTEL_COMPILER) || defined(INTEL_LLVM_COMPILER)
	r = _mm512_rem_epu64(_mm512_set1_epi64(1ULL << 52), nvec);
#else
	r = rem_epu64_x8(_mm512_set1_epi64(1ULL << 52), nvec);
#endif

	// penultimate-hi-bit mask
	m = _mm512_sub_epi64(_mm512_set1_epi64(62), _mm512_lzcnt_epi64(evec));
	m = _mm512_sllv_epi64(_mm512_set1_epi64(1), m);

	// we know the first bit is set and the first squaring is of unity,
	// so we can do the first iteration manually with no squaring.
	unity = r;

	r = _mm512_add_epi64(r, r);
	__mmask8 ge = _mm512_cmpge_epi64_mask(r, nvec);
	r = _mm512_mask_sub_epi64(r, ge, r, nvec);

	while (_mm512_cmpgt_epu64_mask(m, zero))
	{
		__mmask8 bitcmp = _mm512_test_epi64_mask(m, evec);
		mulredc52_mask_add_vec(&r, bitcmp, r, r, nvec, vrho);
		m = _mm512_srli_epi64(m, 1);
	}

	// AMM possibly needs a final correction by n
	ge = _mm512_cmpge_epi64_mask(r, nvec);
	r = _mm512_mask_sub_epi64(r, ge, r, nvec);

	return _mm512_cmpeq_epu64_mask(unity, r);
}

// a Miller-Rabin SPRP test on 8x 104-bit inputs using base 2
uint8_t MR_2sprp_104x8(uint64_t* n)
{
	// assumes has no small factors.  assumes n >= 54 bits.
	// assumes n is a list of 8 104-bit integers (16 52-bit words)
	// in the format: 8 lo-words, 8 hi-words.
	// do a Miller-Rabin sprp test using base 2.
	__m512i vrho = multiplicative_inverse104_x8(n);
	__m512i mone[2];
	vec_u104_t rvec;
	vec_u104_t onevec;
	vec_u104_t twovec;
	__m512i nv[2];
	__m512i dv[2];
	__m512i rv[2];
	__m512i bv[2];
	__m512i n1v[2];
	__m512i tv[2];
	__m512i m;
	__m512i zerov = _mm512_setzero_si512();
	__m512i onev = _mm512_set1_epi64(1);
	__m512i lo52mask = _mm512_set1_epi64(0x000fffffffffffffull);
	uint64_t tmp = 0;

	vrho = _mm512_and_epi64(_mm512_sub_epi64(zerov, vrho), lo52mask);

	nv[0] = loadu64(&n[0]);
	nv[1] = loadu64(&n[8]);

	// the 128-bit division we do one at a time
	int i;
	for (i = 0; i < 8; i++)
	{
		uint128_t mod = ((uint128_t)n[i + 8] << 52) + n[i];
		uint128_t one = (uint128_t)1 << 104;
		one %= mod;

		onevec.data[0][i] = (uint64_t)one & 0xfffffffffffffULL;
		onevec.data[1][i] = (uint64_t)(one >> 52) & 0xfffffffffffffULL;
	}

	mone[0] = loadu64(onevec.data[0]);
	mone[1] = loadu64(onevec.data[1]);

	// compute d and tzcnt
	submod104_x8(&n1v[1], &n1v[0], nv[1], nv[0], zerov, onev, nv[1], nv[0]);

	__mmask8 done = 0;
	dv[1] = n1v[1];
	dv[0] = n1v[0];
	__m512i tzcntv = zerov;
	while (done != 0xff)
	{
		__m512i c = _mm512_mask_slli_epi64(dv[1], ~done, dv[1], 51);
		dv[0] = _mm512_mask_srli_epi64(dv[0], ~done, dv[0], 1);
		dv[0] = _mm512_mask_or_epi64(dv[0], ~done, c, dv[0]);
		dv[1] = _mm512_mask_srli_epi64(dv[1], ~done, dv[1], 1);
		tzcntv = _mm512_mask_add_epi64(tzcntv, ~done, tzcntv, onev);
		done = done | _mm512_cmpeq_epi64_mask(_mm512_and_epi64(dv[0], onev), onev);
	}
	dv[0] = _mm512_and_epi64(dv[0], lo52mask);

	// penultimate-hi-bit mask based on d
	m = _mm512_sub_epi64(_mm512_set1_epi64(62), _mm512_lzcnt_epi64(dv[1]));
	m = _mm512_sllv_epi64(_mm512_set1_epi64(1), m);
	m = _mm512_mask_set1_epi64(m, _mm512_cmple_epi64_mask(dv[1], onev), 0);

	// we know the first bit is set and the first squaring is of unity,
	// so we can do the first iteration manually (and hence the penultimate mask bit)
	addmod104_x8(&rv[1], &rv[0], mone[1], mone[0], mone[1], mone[0], nv[1], nv[0]);

	__mmask8 protect = _mm512_cmpgt_epi64_mask(_mm512_srli_epi64(n1v[1], 49), zerov);

	// compute b^d
	if (protect)
	{
		done = _mm512_cmpeq_epu64_mask(m, zerov);
		while (done != 0xff)
		{
			__mmask8 bitcmp = _mm512_test_epi64_mask(m, dv[1]);

			mask_sqrredc104_vec(&rv[1], &rv[0], ~done, rv[1], rv[0], nv[1], nv[0], vrho);
			mask_redsub104_x8(&rv[1], &rv[0], (~done) & protect, rv[1], rv[0], nv[1], nv[0]);
			mask_dblmod104_x8(&rv[1], &rv[0], (~done) & bitcmp, rv[1], rv[0], nv[1], nv[0]);

			m = _mm512_srli_epi64(m, 1);
			done = _mm512_cmpeq_epu64_mask(m, zerov);
		}
	}
	else
	{
		done = _mm512_cmpeq_epu64_mask(m, zerov);
		while (done != 0xff)
		{
			__mmask8 bitcmp = _mm512_test_epi64_mask(m, dv[1]);

			mask_sqrredc104_vec(&rv[1], &rv[0], ~done, rv[1], rv[0], nv[1], nv[0], vrho);
			mask_dblmod104_x8(&rv[1], &rv[0], (~done) & bitcmp, rv[1], rv[0], nv[1], nv[0]);

			m = _mm512_srli_epi64(m, 1);
			done = _mm512_cmpeq_epu64_mask(m, zerov);
		}
	}

	m = _mm512_sub_epi64(_mm512_set1_epi64(62), _mm512_lzcnt_epi64(dv[0]));
	m = _mm512_sllv_epi64(_mm512_set1_epi64(1), m);
	m = _mm512_mask_set1_epi64(m, _mm512_cmpge_epi64_mask(dv[1], onev), 1ULL << 51);

	if (protect)
	{
		done = _mm512_cmpeq_epu64_mask(m, zerov);
		while (done != 0xff)
		{
			__mmask8 bitcmp = _mm512_test_epi64_mask(m, dv[0]);

			mask_sqrredc104_vec(&rv[1], &rv[0], ~done, rv[1], rv[0], nv[1], nv[0], vrho);
			mask_redsub104_x8(&rv[1], &rv[0], (~done) & protect, rv[1], rv[0], nv[1], nv[0]);
			mask_dblmod104_x8(&rv[1], &rv[0], (~done) & bitcmp, rv[1], rv[0], nv[1], nv[0]);

			m = _mm512_srli_epi64(m, 1);
			done = _mm512_cmpeq_epu64_mask(m, zerov);
		}
	}
	else
	{
		done = _mm512_cmpeq_epu64_mask(m, zerov);
		while (done != 0xff)
		{
			__mmask8 bitcmp = _mm512_test_epi64_mask(m, dv[0]);

			mask_sqrredc104_vec(&rv[1], &rv[0], ~done, rv[1], rv[0], nv[1], nv[0], vrho);
			mask_dblmod104_x8(&rv[1], &rv[0], (~done) & bitcmp, rv[1], rv[0], nv[1], nv[0]);

			m = _mm512_srli_epi64(m, 1);
			done = _mm512_cmpeq_epu64_mask(m, zerov);
		}
	}

	// AMM possibly needs a final correction by n
	addmod104_x8(&rv[1], &rv[0], zerov, zerov, rv[1], rv[0], nv[1], nv[0]);

	// check current result == 1
	__mmask8 is1prp = _mm512_cmpeq_epu64_mask(rv[1], mone[1]) &
		_mm512_cmpeq_epu64_mask(rv[0], mone[0]);

	// now compute b^(2^s*d) and check for congruence to -1 as we go.
	// check while tzcnt is > 1 for all inputs or all are already not prp.
	done = is1prp;
	__mmask8 ism1prp = 0;

	submod104_x8(&n1v[1], &n1v[0], zerov, zerov, mone[1], mone[0], nv[1], nv[0]);

	while (done != 0xff)
	{
		tzcntv = _mm512_mask_sub_epi64(tzcntv, ~done, tzcntv, onev);

		// prp by -1 check
		ism1prp = (_mm512_mask_cmpeq_epu64_mask(~is1prp, rv[1], n1v[1]) &
			_mm512_mask_cmpeq_epu64_mask(~is1prp, rv[0], n1v[0]));

		is1prp |= ism1prp;	// stop checking it if we've found a prp criteria.
		done = (is1prp | ism1prp);

		sqrredc104_vec(&rv[1], &rv[0], rv[1], rv[0], nv[1], nv[0], vrho);
		addmod104_x8(&rv[1], &rv[0], zerov, zerov, rv[1], rv[0], nv[1], nv[0]);

		// definitely not prp by 1 check, stop checking
		done |= (_mm512_mask_cmpeq_epu64_mask(~is1prp, rv[1], zerov) &
			_mm512_mask_cmpeq_epu64_mask(~is1prp, rv[0], onev));

		done |= _mm512_mask_cmple_epu64_mask(~done, tzcntv, onev);
	}

	addmod104_x8(&rv[1], &rv[0], zerov, zerov, rv[1], rv[0], nv[1], nv[0]);

	// check current result == m-1
	ism1prp |= (_mm512_mask_cmpeq_epu64_mask(~is1prp, rv[1], n1v[1]) &
		_mm512_mask_cmpeq_epu64_mask(~is1prp, rv[0], n1v[0]));

	return (is1prp | ism1prp);
}

// a Miller-Rabin SPRP test on 8x 104-bit inputs using an
// independent base on each
uint8_t MR_sprp_104x8(uint64_t* n, uint64_t *bases)
{
	// assumes has no small factors.  assumes n >= 54 bits.
	// assumes n is a list of 8 104-bit integers (16 52-bit words)
	// in the format: 8 lo-words, 8 hi-words.
	// assume bases is a list of 8 small bases, one for each input n.
	// do a Miller-Rabin sprp test on each using the supplied bases.
	__m512i vrho = multiplicative_inverse104_x8(n);
	__m512i mone[2];
	vec_u104_t rvec;
	vec_u104_t onevec;
	vec_u104_t twovec;
	__m512i nv[2];
	__m512i dv[2];
	__m512i rv[2];
	__m512i bv[2];
	__m512i n1v[2];
	__m512i tv[2];
	__m512i m;
	__m512i zerov = _mm512_setzero_si512();
	__m512i onev = _mm512_set1_epi64(1);
	__m512i lo52mask = _mm512_set1_epi64(0x000fffffffffffffull);
	uint64_t tmp = 0;

	vrho = _mm512_and_epi64(_mm512_sub_epi64(zerov, vrho), lo52mask);

	nv[0] = loadu64(&n[0]);
	nv[1] = loadu64(&n[8]);

	// the 128-bit division we do one at a time
	int i;
	for (i = 0; i < 8; i++)
	{
		uint128_t mod = ((uint128_t)n[i + 8] << 52) + n[i];
		uint128_t one = (uint128_t)1 << 104;
		one %= mod;

		onevec.data[0][i] = (uint64_t)one & 0xfffffffffffffULL;
		onevec.data[1][i] = (uint64_t)(one >> 52);

	}

	mone[0] = loadu64(onevec.data[0]);
	mone[1] = loadu64(onevec.data[1]);

	// get bases into Monty rep
	bv[0] = loadu64(bases);
	bv[1] = zerov;

	__m512i mpow[2];
	mpow[0] = mone[0];
	mpow[1] = mone[1];

	rv[0] = mone[0];
	rv[1] = mone[1];

	bv[0] = _mm512_srli_epi64(bv[0], 1);
	__mmask8 done = _mm512_cmpeq_epi64_mask(bv[0], zerov);
	while (done != 0xff)
	{
		addmod104_x8(&mpow[1], &mpow[0], mpow[1], mpow[0], mpow[1], mpow[0], nv[1], nv[0]);
		__mmask8 bitcmp = _mm512_test_epi64_mask(onev, bv[0]);
		mask_addmod104_x8(&rv[1], &rv[0], (~done) & bitcmp, rv[1], rv[0], mpow[1], mpow[0], nv[1], nv[0]);

		bv[0] = _mm512_srli_epi64(bv[0], 1);
		done = _mm512_cmpeq_epi64_mask(bv[0], zerov);
	}

	bv[0] = rv[0];
	bv[1] = rv[1];

	// compute d and tzcnt
	submod104_x8(&n1v[1], &n1v[0], nv[1], nv[0], zerov, onev, nv[1], nv[0]);

	done = 0;
	dv[1] = n1v[1];
	dv[0] = n1v[0];
	__m512i tzcntv = zerov;
	while (done != 0xff)
	{
		__m512i c = _mm512_mask_slli_epi64(dv[1], ~done, dv[1], 51);
		dv[0] = _mm512_mask_srli_epi64(dv[0], ~done, dv[0], 1);
		dv[0] = _mm512_mask_or_epi64(dv[0], ~done, c, dv[0]);
		dv[1] = _mm512_mask_srli_epi64(dv[1], ~done, dv[1], 1);
		tzcntv = _mm512_mask_add_epi64(tzcntv, ~done, tzcntv, onev);
		done = done | _mm512_cmpeq_epi64_mask(_mm512_and_epi64(dv[0], onev), onev);
	}
	dv[0] = _mm512_and_epi64(dv[0], lo52mask);

	// penultimate-hi-bit mask based on d
	m = _mm512_sub_epi64(_mm512_set1_epi64(62), _mm512_lzcnt_epi64(dv[1]));
	m = _mm512_sllv_epi64(_mm512_set1_epi64(1), m);
	m = _mm512_mask_set1_epi64(m, _mm512_cmple_epi64_mask(dv[1], onev), 0);

	// we know the first bit is set and the first squaring is of unity,
	// so we can do the first iteration manually (and hence the penultimate mask bit)
	rv[0] = bv[0];
	rv[1] = bv[1];

	// compute b^d
	done = _mm512_cmpeq_epu64_mask(m, zerov);
	while (done != 0xff)
	{
		__mmask8 bitcmp = _mm512_test_epi64_mask(m, dv[1]);

		mask_sqrredc104_vec(&rv[1], &rv[0], ~done, rv[1], rv[0], nv[1], nv[0], vrho);
		mask_mulredc104_vec(&rv[1], &rv[0], (~done) & bitcmp, rv[1], rv[0], bv[1], bv[0], nv[1], nv[0], vrho);

		m = _mm512_srli_epi64(m, 1);
		done = _mm512_cmpeq_epu64_mask(m, zerov);
	}

	m = _mm512_sub_epi64(_mm512_set1_epi64(62), _mm512_lzcnt_epi64(dv[0]));
	m = _mm512_sllv_epi64(_mm512_set1_epi64(1), m);
	m = _mm512_mask_set1_epi64(m, _mm512_cmpge_epi64_mask(dv[1], onev), 1ULL << 51);

	done = _mm512_cmpeq_epu64_mask(m, zerov);
	while (done != 0xff)
	{
		__mmask8 bitcmp = _mm512_test_epi64_mask(m, dv[0]);

		mask_sqrredc104_vec(&rv[1], &rv[0], ~done, rv[1], rv[0], nv[1], nv[0], vrho);
		mask_mulredc104_vec(&rv[1], &rv[0], (~done) & bitcmp, rv[1], rv[0], bv[1], bv[0], nv[1], nv[0], vrho);

		m = _mm512_srli_epi64(m, 1);
		done = _mm512_cmpeq_epu64_mask(m, zerov);
	}

	// AMM possibly needs a final correction by n
	addmod104_x8(&rv[1], &rv[0], zerov, zerov, rv[1], rv[0], nv[1], nv[0]);

	// check current result == 1
	__mmask8 is1prp = _mm512_cmpeq_epu64_mask(rv[1], mone[1]) &
		_mm512_cmpeq_epu64_mask(rv[0], mone[0]);

	// now compute b^(2^s*d) and check for congruence to -1 as we go.
	// check while tzcnt is > 1 for all inputs or all are already not prp.
	done = is1prp;
	__mmask8 ism1prp = 0;

	submod104_x8(&n1v[1], &n1v[0], zerov, zerov, mone[1], mone[0], nv[1], nv[0]);

	while (done != 0xff)
	{
		tzcntv = _mm512_mask_sub_epi64(tzcntv, ~done, tzcntv, onev);

		// prp by -1 check
		ism1prp = (_mm512_mask_cmpeq_epu64_mask(~is1prp, rv[1], n1v[1]) &
			_mm512_mask_cmpeq_epu64_mask(~is1prp, rv[0], n1v[0]));

		is1prp |= ism1prp;	// stop checking it if we've found a prp criteria.
		done = (is1prp | ism1prp);

		sqrredc104_vec(&rv[1], &rv[0], rv[1], rv[0], nv[1], nv[0], vrho);
		addmod104_x8(&rv[1], &rv[0], rv[1], rv[0], zerov, zerov, nv[1], nv[0]);

		// definitely not prp by 1 check, stop checking
		done |= (_mm512_mask_cmpeq_epu64_mask(~is1prp, rv[1], zerov) &
			_mm512_mask_cmpeq_epu64_mask(~is1prp, rv[0], onev));

		done |= _mm512_mask_cmple_epu64_mask(~done, tzcntv, onev);
	}

	addmod104_x8(&rv[1], &rv[0], rv[1], rv[0], zerov, zerov, nv[1], nv[0]);

	// check current result == m-1
	ism1prp |= (_mm512_mask_cmpeq_epu64_mask(~is1prp, rv[1], n1v[1]) &
		_mm512_mask_cmpeq_epu64_mask(~is1prp, rv[0], n1v[0]));

	return (is1prp | ism1prp);
}

// a Miller-Rabin SPRP test on 1 104-bit input using 8x
// different bases
uint8_t MR_sprp_104x8base(uint64_t* n, uint64_t* one, uint64_t *bases)
{
	// assumes has no small factors and is odd.  assumes n >= 54 bits.
	// assumes n is a 104-bit integer with two 52 bit words: [lo,hi]
	// assume bases is a list of 8 small bases
	// do a Miller-Rabin sprp test on the input to each supplied base.
	// uint128_t n128 = ((uint128_t)n[1] << 52) + (uint128_t)n[0];
	__m512i vrho = _mm512_set1_epi64(multiplicative_inverse(n[0]));
	__m512i mone[2];
	__m512i nv[2];
	__m512i dv[2];
	__m512i rv[2];
	__m512i bv[2];
	__m512i n1v[2];
	__m512i tv[2];
	__m512i zerov = _mm512_setzero_si512();
	__m512i onev = _mm512_set1_epi64(1);
	__m512i lo52mask = _mm512_set1_epi64(0x000fffffffffffffull);
	uint64_t tmp = 0;

	vrho = _mm512_and_epi64(_mm512_sub_epi64(zerov, vrho), lo52mask);

	nv[0] = _mm512_set1_epi64(n[0]);
	nv[1] = _mm512_set1_epi64(n[1]);
	
	mone[0] = _mm512_set1_epi64(one[0]);
	mone[1] = _mm512_set1_epi64(one[1]);

	// get bases into Monty rep
	bv[0] = loadu64(bases);
	bv[1] = zerov;

	__m512i mpow[2];
	mpow[0] = mone[0];
	mpow[1] = mone[1];

	rv[0] = mone[0];
	rv[1] = mone[1];

	bv[0] = _mm512_srli_epi64(bv[0], 1);

	__mmask8 done = _mm512_cmpeq_epi64_mask(bv[0], zerov);
	while (done != 0xff)
	{
		addmod104_x8(&mpow[1], &mpow[0], mpow[1], mpow[0], mpow[1], mpow[0], nv[1], nv[0]);
		__mmask8 bitcmp = _mm512_test_epi64_mask(onev, bv[0]);
		mask_addmod104_x8(&rv[1], &rv[0], (~done) & bitcmp, rv[1], rv[0], mpow[1], mpow[0], nv[1], nv[0]);

		bv[0] = _mm512_srli_epi64(bv[0], 1);
		done = _mm512_cmpeq_epi64_mask(bv[0], zerov);
	}

	bv[0] = rv[0];
	bv[1] = rv[1];

	// compute d and tzcnt
	uint64_t d[2];
	d[1] = n[1];
	d[0] = n[0] - 1;			// n odd, so this won't carry
	int ntz = my_ctz104(d[0], d[1]);
	n1v[0] = _mm512_set1_epi64(d[0]);
	n1v[1] = _mm512_set1_epi64(d[1]);
	if (ntz < 52)
	{
		uint64_t shift = d[1] & ((1ULL << ntz) - 1);
		d[0] = (d[0] >> ntz) + (shift << (52 - ntz));
		d[1] >>= ntz;
	}
	else
	{
		d[0] = d[1];
		d[1] = 0;
		d[0] >>= (ntz - 52);
	}


#if 0
	// LR
	uint64_t m;
	if (d[1] <= 1) m = 0;
	else m = (1ull << (62 - my_clz52(d[1])));
	
	while (m > 0)
	{
		sqrredc104_vec(&rv[1], &rv[0], rv[1], rv[0], nv[1], nv[0], vrho);
		if (m & d[1])
			mask_mulredc104_vec(&rv[1], &rv[0], 0xff, rv[1], rv[0], bv[1], bv[0], nv[1], nv[0], vrho);
		m >>= 1;
	}
	
	if (d[1] == 0) m = (1ull << (62 - my_clz52(d[0])));
	else m = (1ull << 51);
	
	while (m > 0)
	{
		sqrredc104_vec(&rv[1], &rv[0], rv[1], rv[0], nv[1], nv[0], vrho);
		if (m & d[0])
			mask_mulredc104_vec(&rv[1], &rv[0], 0xff, rv[1], rv[0], bv[1], bv[0], nv[1], nv[0], vrho);
		m >>= 1;
	}
#elif 0
	// RL-104
	uint128_t d128 = ((uint128_t)d[1] << 52) + (uint128_t)d[0];
	rv[0] = mone[0];
	rv[1] = mone[1];
	while (d128 > 0)
	{
		if (d128 & 1)
			mask_mulredc104_vec(&rv[1], &rv[0], 0xff,
				rv[1], rv[0], bv[1], bv[0], nv[1], nv[0], vrho);

		d128 >>= 1;

		if (d128)
			sqrredc104_vec(&bv[1], &bv[0], bv[1], bv[0], nv[1], nv[0], vrho);
	}
#elif 0
	// LR-kary
	uint64_t g[256];
	// 
	// precomputation
	rv[0] = bv[0];
	rv[1] = bv[1];

	int i;
	storeu64(&g[0 * 8], mone[0]);		// g0 = 1
	storeu64(&g[1 * 8], mone[1]);		// g0 = 1
	storeu64(&g[2 * 8], rv[0]);			// g1 = g 
	storeu64(&g[3 * 8], rv[1]);			// g1 = g 
	sqrredc104_vec(&rv[1], &rv[0], rv[1], rv[0], nv[1], nv[0], vrho);
	storeu64(&g[4 * 8], rv[0]);			// g2 = g^2
	storeu64(&g[5 * 8], rv[1]);			// g2 = g^2
	for (i = 3; i < 16; i++)
	{
		mask_mulredc104_vec(&rv[1], &rv[0], 0xff, rv[1], rv[0], bv[1], bv[0], nv[1], nv[0], vrho);
		storeu64(&g[(i * 2) * 8], rv[0]);			// gi = g^i
		storeu64(&g[(i * 2 + 1) * 8], rv[1]);		// gi = g^i
	}

	uint128_t d128 = ((uint128_t)d[1] << 52) + (uint128_t)d[0];
	rv[0] = mone[0];
	rv[1] = mone[1];
	int lz = my_clz104(d[0], d[1]);
	int msb = 112 - lz;
	int m;

	m = (d128 >> msb) & 0xf;
	rv[0] = loadu64(&g[(2 * m) * 8]);
	rv[1] = loadu64(&g[(2 * m + 1) * 8]);
	msb -= 4;

	while (msb > 0)
	{
		sqrredc104_vec(&rv[1], &rv[0], rv[1], rv[0], nv[1], nv[0], vrho);
		sqrredc104_vec(&rv[1], &rv[0], rv[1], rv[0], nv[1], nv[0], vrho);
		sqrredc104_vec(&rv[1], &rv[0], rv[1], rv[0], nv[1], nv[0], vrho);
		sqrredc104_vec(&rv[1], &rv[0], rv[1], rv[0], nv[1], nv[0], vrho);
		m = (d128 >> msb) & 0xf;

		if (m > 0)
			mask_mulredc104_vec(&rv[1], &rv[0], 0xff,
				loadu64(&g[(2 * m + 1) * 8]), loadu64(&g[(2 * m) * 8]),
				rv[1], rv[0], nv[1], nv[0], vrho);

		msb -= 4;
	}

	msb += 4;
	m = (int)d128 & ((1 << msb) - 1);
	while (msb > 0)
	{
		sqrredc104_vec(&rv[1], &rv[0], rv[1], rv[0], nv[1], nv[0], vrho);
		msb--;
	}

	mask_mulredc104_vec(&rv[1], &rv[0], 0xff,
		loadu64(&g[(2 * m + 1) * 8]), loadu64(&g[(2 * m) * 8]),
		rv[1], rv[0], nv[1], nv[0], vrho);
#else
	// RL-52x2
	rv[0] = mone[0];
	rv[1] = mone[1];
	int i = 0;
	while (d[0] > 0)
	{
		if (d[0] & 1)
			mask_mulredc104_vec(&rv[1], &rv[0], 0xff,
				rv[1], rv[0], bv[1], bv[0], nv[1], nv[0], vrho);

		d[0] >>= 1;
		i++;
		
		sqrredc104_vec(&bv[1], &bv[0], bv[1], bv[0], nv[1], nv[0], vrho);
	}

	for (; i < 52; i++)
	{
		sqrredc104_vec(&bv[1], &bv[0], bv[1], bv[0], nv[1], nv[0], vrho);
	}

	while (d[1] > 1)
	{
		if (d[1] & 1)
			mask_mulredc104_vec(&rv[1], &rv[0], 0xff,
				rv[1], rv[0], bv[1], bv[0], nv[1], nv[0], vrho);

		d[1] >>= 1;

		sqrredc104_vec(&bv[1], &bv[0], bv[1], bv[0], nv[1], nv[0], vrho);
	}
	mask_mulredc104_vec(&rv[1], &rv[0], 0xff,
		rv[1], rv[0], bv[1], bv[0], nv[1], nv[0], vrho);

#endif
	

	

	// AMM possibly needs a final correction by n
	addmod104_x8(&rv[1], &rv[0], zerov, zerov, rv[1], rv[0], nv[1], nv[0]);

	// check current result == 1
	__mmask8 is1prp = _mm512_cmpeq_epu64_mask(rv[1], mone[1]) &
		_mm512_cmpeq_epu64_mask(rv[0], mone[0]);

	// now compute b^(2^s*d) and check for congruence to -1 as we go.
	// check while tzcnt is > 1 for all inputs or all are already not prp.
	done = is1prp;
	__mmask8 ism1prp = 0;

	submod104_x8(&n1v[1], &n1v[0], zerov, zerov, mone[1], mone[0], nv[1], nv[0]);

	while ((done != 0xff) && (ntz > 0))
	{
		ntz--;

		// prp by -1 check
		ism1prp = (_mm512_mask_cmpeq_epu64_mask(~is1prp, rv[1], n1v[1]) &
			_mm512_mask_cmpeq_epu64_mask(~is1prp, rv[0], n1v[0]));

		is1prp |= ism1prp;	// stop checking it if we've found a prp criteria.
		done = (is1prp | ism1prp);

		sqrredc104_vec(&rv[1], &rv[0], rv[1], rv[0], nv[1], nv[0], vrho);
		addmod104_x8(&rv[1], &rv[0], zerov, zerov, rv[1], rv[0], nv[1], nv[0]);

		// definitely not prp by 1 check, stop checking
		done |= (_mm512_mask_cmpeq_epu64_mask(~is1prp, rv[1], zerov) &
			_mm512_mask_cmpeq_epu64_mask(~is1prp, rv[0], onev));
	}

	addmod104_x8(&rv[1], &rv[0], zerov, zerov, rv[1], rv[0], nv[1], nv[0]);

	// check current result == m-1
	ism1prp |= (_mm512_mask_cmpeq_epu64_mask(~is1prp, rv[1], n1v[1]) &
		_mm512_mask_cmpeq_epu64_mask(~is1prp, rv[0], n1v[0]));

	return (is1prp | ism1prp);
}

int main(int argc, char** argv)
{
	uint64_t prp[16];
	int correct = 0;
	int i, k;
	mpz_t gmp2, gmpn, gmpn1;
	mpz_init(gmp2);
	mpz_init(gmpn);
	mpz_init(gmpn1);

#ifndef IFMA
	dbias = _mm512_castsi512_pd(set64(0x4670000000000000ULL));
	vbias1 = set64(0x4670000000000000ULL);
	vbias2 = set64(0x4670000000000001ULL);
	vbias3 = _mm512_set1_epi64(0x4330000000000000ULL);
#endif

	lo52mask = _mm512_set1_epi64(0x000fffffffffffffull);

	struct timeval start, stop;
	int bits;
	uint64_t elapsed;

	// test of fermat_prp_52x8 on random 6k+1 inputs
	for (bits = 70; bits <= 52; bits+=5)
	{
		uint32_t numprp = 0;
		uint64_t ticks1 = my_rdtsc();
		uint64_t ticks2;
		uint32_t num = 1000000;
		double telapsed = 0;
		int k;
		if (bits > 52) bits = 52;
		
		numprp = 0;
		k = 0;
		elapsed = 0;
		telapsed = 0;
		do {
			for (i = 0; i < 8; i++)
			{
				uint64_t x;
				do {
					x = my_random();
					uint128_t maskAnd = ((uint128_t)1 << (bits - 1)) - 1;	// clear msbits
					uint128_t maskOr = ((uint128_t)1 << (bits - 1)) | ((uint128_t)1 << (bits / 2));	// force msb, force another bit
					x &= maskAnd;
					x |= maskOr;
					x /= 6;
					x *= 6;	// now a multiple of 6
					x += 1;	// number like 6*k + 1
				} while (x >> (bits - 1) != 1);
				prp[i] = (uint64_t)x & 0xfffffffffffffull;
			}

			ticks1 = my_rdtsc();
			gettimeofday(&start, NULL);
		
			uint64_t inc = 4;
			int j;

			for (j = 0; j < num; j++)
			{
				numprp += _mm_popcnt_u32(fermat_prp_52x8(prp));
				for (i = 0; i < 8; i++)
				{
					prp[i] += inc;
				}
				inc = 6 - inc;
			}
			
			k++;
			ticks2 = my_rdtsc();
			elapsed += (ticks2 - ticks1);
			gettimeofday(&stop, NULL);
			telapsed += _difftime(&start, &stop);
		
		} while (elapsed < (1ull<<30));

		printf("total ticks = %lu, ticks per %d-bit input = %lu\n",
			elapsed, bits, (elapsed) / (k * num * 8));
		printf("found %d fermat-prp out of %u %d-bit inputs: %1.2f%%\n",
			numprp, k * num * 8, bits, 100. * (double)numprp / (double)(k * num * 8));
		printf("elapsed time: %1.4f sec, %1.4f us / input\n", telapsed, 1000000. * telapsed / (double)(k * num * 8));
	}
	printf("\n");

	// test of fermat_prp_104x8 on random 6k+1 inputs
	for (bits = 100; bits <= 50; bits+=1)
	{
		uint32_t numprp = 0;
		uint64_t ticks1 = my_rdtsc();
		uint64_t ticks2;
		uint32_t num = 100000;
		double telapsed = 0;
		int k;
		if (bits > 104) bits = 104;

		k = 0;
		int fail = 0;
		elapsed = 0;
		do {
			for (i = 0; i < 8; i++)
			{
				uint128_t x;
				do {
					x = my_random();
					uint128_t maskAnd = ((uint128_t)1 << (bits - 1)) - 1;	// clear msbits
					uint128_t maskOr = ((uint128_t)1 << (bits - 1)) | ((uint128_t)1 << (bits / 2));	// force msb, force another bit
					x &= maskAnd;
					x |= maskOr;
					x /= 6;
					x *= 6;	// now a multiple of 6
					x += 1;	// number like 6*k + 1
				} while (x >> (bits - 1) != 1);
				prp[i] = (uint64_t)x & 0xfffffffffffffull;
				prp[i+8] = (uint64_t)(x >> 52) & 0xfffffffffffffull;
			}

			uint64_t inc = 4;
			int j;

			ticks1 = my_rdtsc();
			gettimeofday(&start, NULL);

			for (j = 0; j < num; j++)
			{
				uint8_t prpmask = fermat_prp_104x8(prp);
				numprp += _mm_popcnt_u32(prpmask);
				for (i = 0; i < 8; i++)
				{
					if (0 && ((prpmask & (1<<i)) == 0))
					{
						mpz_set_ui(gmp2, 2);
						mpz_set_ui(gmpn, prp[8+i]);
						mpz_mul_2exp(gmpn, gmpn, 52);
						mpz_add_ui(gmpn, gmpn, prp[i]);
						mpz_sub_ui(gmpn1, gmpn, 1);
						mpz_powm(gmp2, gmp2, gmpn1, gmpn);
						if (mpz_cmp_ui(gmp2, 1) == 0)
						{
							//printf("prp %016lx%016lx failed in lane %d\n", prp[i+8], prp[i], i);
							//gmp_printf("mpz result = %Zx\n", gmp2);
							//exit(1);
							fail++;
						}
					}
					prp[i] += inc;
				}
				inc = 6 - inc;
			}
			//printf("verified %d prps\n", numprp);

			k++;
			ticks2 = my_rdtsc();
			elapsed += (ticks2 - ticks1);
			gettimeofday(&stop, NULL);
			telapsed = +_difftime(&start, &stop);
		} while (elapsed < (1ull<<30));
		
		printf("total ticks = %lu, ticks per %d-bit input = %lu\n",
			elapsed, bits, (elapsed) / (k * num * 8));
		printf("found %d fermat-prp out of %u %d-bit inputs (%d fails): %1.2f%%\n",
			numprp, k * num * 8, bits, fail, 100. * (double)numprp / (double)(k * num * 8));
		printf("elapsed time: %1.4f sec, %1.4f us / input\n", telapsed, 1000000. * telapsed / (double)(k * num * 8));
	}
	printf("\n");

	// test of MR_2sprp_104x8 on random 6k+1 inputs
	for (bits = 100; bits <= 50; bits+=1)
	{
		uint32_t numprp = 0;
		uint64_t ticks1 = my_rdtsc();
		uint64_t ticks2;
		uint32_t num = 100000;
		double telapsed = 0;
		int k;
		if (bits > 104) bits = 104;

		k = 0;
		elapsed = 0;
		int fail = 0;
		do {
			for (i = 0; i < 8; i++)
			{
				uint128_t x;
				do {
					x = my_random();
					uint128_t maskAnd = ((uint128_t)1 << (bits - 1)) - 1;	// clear msbits
					uint128_t maskOr = ((uint128_t)1 << (bits - 1)) | ((uint128_t)1 << (bits / 2));	// force msb, force another bit
					x &= maskAnd;
					x |= maskOr;
					x /= 6;
					x *= 6;	// now a multiple of 6
					x += 1;	// number like 6*k + 1
				} while (x >> (bits - 1) != 1);
				prp[i] = (uint64_t)x & 0xfffffffffffffull;
				prp[i+8] = (uint64_t)(x >> 52) & 0xfffffffffffffull;
			}

			uint64_t inc = 4;
			int j;
			
			ticks1 = my_rdtsc();
			gettimeofday(&start, NULL);

			for (j = 0; j < num; j++)
			{
				uint8_t prpmask = MR_2sprp_104x8(prp);
				numprp += _mm_popcnt_u32(prpmask);
				for (i = 0; i < 8; i++)
				{
					if (0 && ((prpmask & (1 << i)) == 0))
					{
						mpz_set_ui(gmp2, 2);
						mpz_set_ui(gmpn, prp[8 + i]);
						mpz_mul_2exp(gmpn, gmpn, 52);
						mpz_add_ui(gmpn, gmpn, prp[i]);
						mpz_sub_ui(gmpn1, gmpn, 1);
						mpz_powm(gmp2, gmp2, gmpn1, gmpn);
						if (mpz_cmp_ui(gmp2, 1) == 0)
						{
							//printf("prp %016lx%016lx failed in lane %d\n", prp[i + 8], prp[i], i);
							//gmp_printf("mpz result = %Zx\n", gmp2);
							//exit(1);
							fail++;
						}
					}
					prp[i] += inc;
				}
				inc = 6 - inc;
			}

			k++;
			ticks2 = my_rdtsc();
			elapsed += (ticks2 - ticks1);
			gettimeofday(&stop, NULL);
			telapsed = +_difftime(&start, &stop);
		} while (elapsed < (1ull<<30));
		
		printf("total ticks = %lu, ticks per %d-bit input = %lu\n",
			elapsed, bits, (elapsed) / (k * num * 8));
		printf("found %d MR-2sprp out of %u %d-bit inputs (%d fails): %1.2f%%\n",
			numprp, k * num * 8, bits, fail, 100. * (double)numprp / (double)(k * num * 8));
		printf("elapsed time: %1.4f sec, %1.4f us / input\n", telapsed, 1000000. * telapsed / (double)(k * num * 8));
	}
	printf("\n");

	// bases for MR-sprp check:
	uint64_t bases[24] = {3, 5, 7, 11, 
		13, 17, 19, 23, 
		29, 31, 37, 41,
		43, 47, 53, 59, 
		61, 67, 71, 73, 
		79, 83, 89, 97};
		
	// test of MR_sprp_104x8 on PRP 6k+1 inputs
	for (bits = 100; bits <= 50; bits+=1)
	{
		uint32_t numprp = 0;
		uint64_t ticks1 = my_rdtsc();
		uint64_t ticks2;
		uint32_t num = 100000;
		double telapsed = 0;

		if (bits > 104) bits = 104;

		//printf("commencing test of random 6k+1 %d-bit inputs\n", bits);
		elapsed = 0;

		uint64_t inc[8] = { 4, 4, 4, 4, 4, 4, 4, 4 };
		
		for (i = 0; i < 8; i++)
		{
			uint128_t x;
			do {
				x = my_random();
				uint128_t maskAnd = ((uint128_t)1 << (bits - 1)) - 1;	// clear msbits
				uint128_t maskOr = ((uint128_t)1 << (bits - 1)) | ((uint128_t)1 << (bits / 2));	// force msb, force another bit
				x &= maskAnd;
				x |= maskOr;
				x /= 6;
				x *= 6;	// now a multiple of 6
				x += 1;	// number like 6*k + 1
			} while (x >> (bits - 1) != 1);
			prp[i] = (uint64_t)x & 0xfffffffffffffull;
			prp[i+8] = (uint64_t)(x >> 52);
		}
				
		uint8_t isprp = 0;
		while (isprp != 0xff)
		{
			isprp = fermat_prp_104x8(prp);
			
			for (i = 0; i < 8; i++)
			{			
				//printf("%016lx%016lx : %u (%u)\n", prp[i+8], prp[i], isprp & (1 << i), isprp);
				if ((isprp & (1 << i)) == 0)
				{
					prp[i] += inc[i];
					inc[i] = 6 - inc[i];
				}
			}
		}

		ticks1 = my_rdtsc();

		uint64_t basecount[8];
		uint64_t maxcount[8];
		uint64_t currentbase[8];
		for (i = 0; i < 8; i++)
		{
			//printf("prp%d = %016lx%016lx\n", i, prp[i+8], prp[i]); fflush(stdout);
			basecount[i] = 0;
			currentbase[i] = bases[0];
			if (bits <= 62)
				maxcount[i] = 8;
			else if (bits <= 82)
				maxcount[i] = 16;	// was 12, but need a multiple of 8 (same cost anyway with avx512)
			else if (bits <= 112)
				maxcount[i] = 16;
			else
				maxcount[i] = 24;	// was 20, but need a multiple of 8 (same cost anyway with avx512)
		}

		uint32_t tested = 0;
		gettimeofday(&start, NULL);
		
		elapsed = 0;
		int fail = 0;
		while ((elapsed < (1ull<<30)))
		{
			uint8_t prpmask = MR_sprp_104x8(prp, currentbase);
			for (i = 0; i < 8; i++)
			{
				if (1 && ((prpmask & (1 << i)) == 0))
				{
					mpz_set_ui(gmpn, prp[8 + i]);
					mpz_mul_2exp(gmpn, gmpn, 52);
					mpz_add_ui(gmpn, gmpn, prp[i]);
					if (mpz_probab_prime_p(gmpn, 1) > 0)
					{
						//printf("prp %016lx%016lx failed in lane %d\n", prp[i + 8], prp[i], i);
						//gmp_printf("mpz result = %Zx\n", gmp2);
						//exit(1);
						fail++;
					}
				}

				if (prpmask & (1 << i))
				{
					// the input in position i could be prp, increment the
					// base if there are more of them
					if (basecount[i] < maxcount[i])
					{
						basecount[i]++;
						currentbase[i] = bases[basecount[i]];
					}
					else
					{
						// we've tested enough bases to know this is prime
						tested++;
						numprp++;
						//prp[i] += inc[i];
						//inc[i] = 6 - inc[i];
						currentbase[i] = bases[0];
						basecount[i] = 0;
					}
				}
				else
				{
					// the input in position i is definitely not prime,
					// replace it and increment num tested
					tested++;
					//prp[i] += inc[i];
					//inc[i] = 6 - inc[i];
					currentbase[i] = bases[0];
					basecount[i] = 0;
				}
			}
			
			ticks2 = my_rdtsc();
			elapsed = (ticks2 - ticks1);
		}

		gettimeofday(&stop, NULL);
		telapsed = +_difftime(&start, &stop);
		
		printf("total ticks = %lu, ticks per %d-bit input = %lu\n",
			elapsed, bits, (elapsed) / tested);
		printf("found %d MR-sprp out of %u %d-bit inputs (%u fails): %1.2f%%\n",
			numprp, tested, bits, fail, 100. * (double)numprp / (double)tested);
		printf("elapsed time: %1.4f sec, %1.4f us / input\n", telapsed, 1000000. * telapsed / (double)tested);
	}
	printf("\n");

	// test of MR_sprp_104x8base on PRP 6k+1 inputs
	for (bits = 100; bits <= 104; bits+=1)
	{
		uint32_t numprp = 0;
		uint64_t ticks1;
		uint64_t ticks2;
		uint32_t num = 100000;
		double telapsed = 0;
		uint64_t one[16];

		//printf("commencing test of random 6k+1 %d-bit inputs\n", bits);
		elapsed = 0;

		if (bits > 104) bits = 104;

		uint64_t inc[8] = { 4, 4, 4, 4, 4, 4, 4, 4 };
		
		for (i = 0; i < 8; i++)
		{
			uint128_t x;
			do {
				x = my_random();
				uint128_t maskAnd = ((uint128_t)1 << (bits - 1)) - 1;	// clear msbits
				uint128_t maskOr = ((uint128_t)1 << (bits - 1)) | ((uint128_t)1 << (bits / 2));	// force msb, force another bit
				x &= maskAnd;
				x |= maskOr;
				x /= 6;
				x *= 6;	// now a multiple of 6
				x += 1;	// number like 6*k + 1
			} while (x >> (bits - 1) != 1);
			prp[i] = (uint64_t)x & 0xfffffffffffffull;
			prp[i+8] = (uint64_t)(x >> 52);
		}
				
		uint8_t isprp = 0;
		while (isprp != 0xff)
		{
			isprp = fermat_prp_104x8(prp);
			
			for (i = 0; i < 8; i++)
			{			
				//printf("%016lx%016lx : %u (%u)\n", prp[i+8], prp[i], isprp & (1 << i), isprp);
				if ((isprp & (1 << i)) == 0)
				{
					prp[i] += inc[i];
					inc[i] = 6 - inc[i];
				}
			}
		}
		
		uint128_t o128;
		uint128_t n128;
		for (i = 0; i < 8; i++)
		{			
			//printf("prp%d = %016lx%016lx\n", i, prp[i+8], prp[i]); fflush(stdout);
			n128 = ((uint128_t)prp[i+8] << 52) + prp[i];
			o128 = (uint128_t)1 << 104;
			o128 = o128 % n128;
			one[i] = (uint64_t)o128 & 0xfffffffffffffull;
			one[i+8] = (uint64_t)(o128 >> 52);
			//printf("one%d = %016lx%016lx\n", i, one[i + 8], one[i]); fflush(stdout);
		}

		uint64_t basecount = 0;
		uint64_t maxcount;
		uint64_t currentbase[8];
		
		if (bits <= 62)
			maxcount = 8;
		else if (bits <= 82)
			maxcount = 16;	// was 12, but need a multiple of 8 (same cost anyway with avx512)
		else if (bits <= 112)
			maxcount = 16;
		else
			maxcount = 24;	// was 20, but need a multiple of 8 (same cost anyway with avx512)
			
		for (i = 0; i < 8; i++)
		{
			currentbase[i] = bases[i];
		}

		uint32_t tested = 0;
		gettimeofday(&start, NULL);

		ticks1 = my_rdtsc();
		
		elapsed = 0;
		int tnum = 0;
		while ((elapsed < (1ull<<30)))
		{
			uint64_t ntest[2], otest[2];
			ntest[1] = prp[tnum+8];
			ntest[0] = prp[tnum];
			otest[1] = one[tnum+8];
			otest[0] = one[tnum];
			
			// so far, simple RL is faster than kary-LR
			uint8_t prpmask = MR_sprp_104x8base(ntest, otest, currentbase);

			if (prpmask == 0xff)
			{
				// the input is prp to all current bases, increment the
				// base if there are more of them
				if ((basecount + 8) <= maxcount)
				{
					for (i = 0; i < 8; i++)
					{
						currentbase[i] = bases[basecount + i];
					}
					basecount += 8;
				}
				else
				{
					// we've tested enough bases to know this is prime
					tested++;
					numprp++;
					basecount = 0;
					for (i = 0; i < 8; i++)
					{
						currentbase[i] = bases[basecount + i];
					}
					tnum = (tnum + 1) & 7;
				}
			}
			else
			{
				// the input in position i is definitely not prime,
				// replace it and increment num tested
				tested++;
				basecount = 0;
				for (i = 0; i < 8; i++)
				{
					currentbase[i] = bases[basecount + i];
				}
				tnum = (tnum + 1) & 7;
			}

			ticks2 = my_rdtsc();
			elapsed = (ticks2 - ticks1);
		}

		gettimeofday(&stop, NULL);
		telapsed = +_difftime(&start, &stop);
		
		printf("total ticks = %lu, ticks per %d-bit input = %lu\n",
			elapsed, bits, (elapsed) / tested);
		printf("found %d MR-sprp out of %u %d-bit inputs: %1.2f%%\n",
			numprp, tested, bits, 100. * (double)numprp / (double)tested);
		printf("elapsed time: %1.4f sec, %1.4f us / input\n", telapsed, 1000000. * telapsed / (double)tested);
	}
	printf("\n");

	mpz_clear(gmpn);
	mpz_clear(gmpn1);
	mpz_clear(gmp2);

	return 0;
}