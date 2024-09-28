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

	// m0
	m = mul52lo(t0, vrho);

	VEC_MUL_ACCUM_LOHI_PD(m, n0, t0, C1);
	VEC_MUL_ACCUM_LOHI_PD(m, n1, t1, C2);

	t1 = _mm512_add_epi64(t1, C1);
	t2 = _mm512_add_epi64(t2, C2);
	// we throw t0 away after this so first propagate its carry.
	t0 = _mm512_add_epi64(t1, one);
	t1 = t2;
	t2 = C1 = zero;

	VEC_MUL_ACCUM_LOHI_PD(a0, b1, t0, C1);
	VEC_MUL_ACCUM_LOHI_PD(a1, b1, t1, t2);

	t1 = _mm512_add_epi64(t1, C1);
	C1 = C2 = zero;

	// m1
	m = mul52lo(t0, vrho);

	VEC_MUL_ACCUM_LOHI_PD(m, n0, t0, C1);
	VEC_MUL_ACCUM_LOHI_PD(m, n1, t1, t2);

	t1 = _mm512_add_epi64(t1, C1);

	// final carryprop
	carryprop(t0, t1, lo52mask);
	carryprop(t1, t2, lo52mask);
	carryprop(t2, C2, lo52mask);

	scarry2 = _mm512_cmp_epu64_mask(C2, zero, _MM_CMPINT_GT);
	
	// conditionally subtract when needed.  we only check against overflow,
	// so this is almost-montgomery multiplication
	*c0 = _mm512_mask_subsetc_epi52(t1, scarry2, t1, n0, &scarry);
	*c1 = _mm512_mask_sbb_epi52(t2, scarry2, scarry, n1, &scarry);

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

	// m0
	m = mul52lo(t0, vrho);

	VEC_MUL_ACCUM_LOHI_PD(m, n0, t0, C1);
	VEC_MUL_ACCUM_LOHI_PD(m, n1, t1, C2);

	t1 = _mm512_add_epi64(t1, C1);
	t2 = _mm512_add_epi64(t2, C2);
	// we throw t0 away after this so first propagate its carry.
	t0 = _mm512_add_epi64(t1, one);
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

	if (scarry > 0) {
		// conditionally subtract when needed (AMM - only on overflow)
		t1 = _mm512_mask_sub_epi64(t1, scarry, t1, n0);
		t2 = _mm512_mask_sub_epi64(t2, scarry, t2, n1);
		C1 = _mm512_srli_epi64(t1, 63);
		t2 = _mm512_mask_sub_epi64(t2, scarry, t2, C1);
	}
	*c0 = _mm512_and_epi64(lo52mask, t1);
	*c1 = _mm512_and_epi64(lo52mask, t2);

	return;
}
__inline static void mask_sqrredc104_vec(__m512i* c1, __m512i* c0, __mmask8 mulmsk,
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

	// m0
	m = mul52lo(t0, vrho);

	VEC_MUL_ACCUM_LOHI_PD(m, n0, t0, C1);
	VEC_MUL_ACCUM_LOHI_PD(m, n1, t1, C2);

	t1 = _mm512_add_epi64(t1, C1);
	t2 = _mm512_add_epi64(t2, C2);
	// we throw t0 away after this so first propagate its carry.
	t0 = _mm512_add_epi64(t1, one);
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

	if (scarry > 0) {
		// conditionally subtract when needed (AMM - only on overflow)
		// with a few bits of headroom (inputs <= 100 bits) we 
		// empirically never get carries generated here.  And it's 
		// quite a bit faster to remove this sub (here and elsewhere).
		// consider special case code.
		t1 = _mm512_mask_sub_epi64(t1, scarry, t1, n0);
		t2 = _mm512_mask_sub_epi64(t2, scarry, t2, n1);
		C1 = _mm512_srli_epi64(t1, 63);
		t2 = _mm512_mask_sub_epi64(t2, scarry, t2, C1);
	}
	*c0 = _mm512_and_epi64(lo52mask, t1);
	*c1 = _mm512_and_epi64(lo52mask, t2);

	*c0 = _mm512_mask_mov_epi64(*c0, ~mulmsk, a0);
	*c1 = _mm512_mask_mov_epi64(*c1, ~mulmsk, a1);

	return;
}
__inline static void sqrredc_maskadd_vec(__m512i* a1, __m512i* a0, __mmask8 addmsk, __m512i n1, __m512i n0, __m512i vrho)
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
	VEC_MUL_ACCUM_LOHI_PD(*a0, *a0, t0, t1);

	// m0
	//_mm512_mullo_epi52(m, vrho, t0);
	m = mul52lo(t0, vrho);

	VEC_MUL_ACCUM_LOHI_PD(m, n0, t0, C1);
	VEC_MUL_ACCUM_LOHI_PD(m, n1, t1, C2);

	t1 = _mm512_add_epi64(t1, C1);
	t2 = _mm512_add_epi64(t2, C2);
	// we throw t0 away after this so first propagate its carry.
	t0 = _mm512_add_epi64(t1, one);
	t1 = t2;
	t2 = C1 = zero;

	VEC_MUL_ACCUM_LOHI_PD(*a1, *a1, t1, t2);

	t0 = _mm512_add_epi64(t0, sqr_lo);
	t1 = _mm512_add_epi64(t1, sqr_hi);

	// m1
	m = mul52lo(t0, vrho);

	VEC_MUL_ACCUM_LOHI_PD(m, n0, t0, C1);
	VEC_MUL_ACCUM_LOHI_PD(m, n1, t1, t2);

	t1 = _mm512_add_epi64(t1, C1);

	// final carryprop: parallel vs. sequential carry generation 
	// that potentially ignores a carry?
	C1 = _mm512_srli_epi64(t0, 52);
	C2 = _mm512_srli_epi64(t1, 52);
	t1 = _mm512_add_epi64(t1, C1);			 // can this generate another carry?
	t2 = _mm512_add_epi64(t2, C2);
	C2 = _mm512_srli_epi64(t2, 52);
	t1 = _mm512_and_epi64(lo52mask, t1);
	t2 = _mm512_and_epi64(lo52mask, t2);

	scarry = _mm512_cmp_epu64_mask(C2, zero, _MM_CMPINT_GT);

	// conditionally subtract when needed (AMM - only on overflow)
	t1 = _mm512_mask_sub_epi64(t1, scarry, t1, n0);
	t2 = _mm512_mask_sub_epi64(t2, scarry, t2, n1);
	C1 = _mm512_srli_epi64(t1, 63);
	t2 = _mm512_mask_sub_epi64(t2, scarry, t2, C1);
	t1 = _mm512_and_epi64(lo52mask, t1);
	t2 = _mm512_and_epi64(lo52mask, t2);

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
	__mmask8 bmsk = _mm512_mask_cmpgt_epu64_mask(addmsk, C2, zero);

	// conditionally subtract N (AMM - on overflow only)
	t1 = _mm512_mask_sub_epi64(t1, bmsk & addmsk, t1, n0);
	t2 = _mm512_mask_sub_epi64(t2, bmsk & addmsk, t2, n1);
	C1 = _mm512_srli_epi64(t1, 63);
	t2 = _mm512_mask_sub_epi64(t2, bmsk & addmsk, t2, C1);
	*a0 = _mm512_and_epi64(lo52mask, t1);
	*a1 = _mm512_and_epi64(lo52mask, t2);

	return;
}
__inline static void addmod104_x8(__m512i* c1, __m512i* c0, __m512i a1, __m512i a0, 
	__m512i b1, __m512i b0, __m512i n1, __m512i n0)
{
	// add
	__mmask8 bmsk;
	a0 = _mm512_addsetc_epi52(a0, b0, &bmsk);
	a1 = _mm512_adc_epi52(a1, bmsk, b1, &bmsk);

	// compare
	__mmask8 msk = bmsk | _mm512_cmpgt_epu64_mask(a1, n1);
	msk |= (_mm512_cmpeq_epu64_mask(a1, n1) & _mm512_cmpge_epu64_mask(a0, n0));

	// conditionally subtract N
	*c0 = _mm512_mask_subsetc_epi52(a0, msk, a0, n0, &bmsk);
	*c1 = _mm512_mask_sbb_epi52(a1, msk, bmsk, n1, &bmsk);
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

uint8_t fermat_prp_x8(uint64_t* n)
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

	// the 128-bit division we do the slow way
	int i;
	for (i = 0; i < 8; i++)
	{
		uint128_t mod = ((uint128_t)n[i + 8] << 52) + n[i];
		uint128_t t = (uint128_t)1 << 104;
		t %= mod;

		unity.data[0][i] = (uint64_t)t & 0xfffffffffffffULL;
		unity.data[1][i] = (uint64_t)(t >> 52) & 0xfffffffffffffULL;

		r.data[0][i] = unity.data[0][i];
		r.data[1][i] = unity.data[1][i];
	}

	// penultimate-hi-bit mask
	m = _mm512_sub_epi64(_mm512_set1_epi64(62), _mm512_lzcnt_epi64(evec[1]));
	m = _mm512_sllv_epi64(_mm512_set1_epi64(1), m);

	// we know the first bit is set and the first squaring is of unity,
	// so we can do the first iteration manually with no squaring.
	// Note: the first 5 iterations can be done much more cheaply in
	// single precision and then converted into montgomery representation,
	// but that would require a 208-bit division; not worth it.
	__m512i r0 = loadu64(r.data[0]);
	__m512i r1 = loadu64(r.data[1]);

	addmod104_x8(&r1, &r0, r1, r0, r1, r0, nvec[1], nvec[0]);

	while (_mm512_cmpgt_epu64_mask(m, zero))
	{
		__mmask8 bitcmp = _mm512_test_epi64_mask(m, evec[1]);
		sqrredc_maskadd_vec(&r1, &r0, bitcmp, nvec[1], nvec[0], vrho);
		m = _mm512_srli_epi64(m, 1);
	}

	m = _mm512_set1_epi64(1ULL << 51);
	while (_mm512_cmpgt_epu64_mask(m, zero))
	{
		__mmask8 bitcmp = _mm512_test_epi64_mask(m, evec[0]);
		sqrredc_maskadd_vec(&r1, &r0, bitcmp, nvec[1], nvec[0], vrho);
		m = _mm512_srli_epi64(m, 1);
	}

	storeu64(r.data[0], r0);
	storeu64(r.data[1], r1);

	// AMM possibly needs a final correction by n
	addmod104_x8(&r1, &r0, zero, zero, r1, r0, nvec[1], nvec[0]);

	uint8_t isprp = 
		_mm512_cmpeq_epu64_mask(loadu64(unity.data[0]), r0) &
		_mm512_cmpeq_epu64_mask(loadu64(unity.data[1]), r1);
	
	return isprp;
}

uint8_t MR_sprp_x8(uint64_t* n, uint64_t *bases)
{
	// assumes has no small factors.  assumes n >= 54 bits.
	// assumes n is a list of 8 104-bit integers (16 52-bit words)
	// in the format: 8 lo-words, 8 hi-words.
	// assume bases is a list of 8 small bases, one for each input n.
	// do a Miller-Rabin sprp test on each using the supplied bases.
	__m512i vrho = multiplicative_inverse104_x8(n);
	__m512i mone[2];
	__m512i mtwo[2];
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
	submod104_x8(&n1v[1], &n1v[0],
		nv[1], nv[0], zerov, onev, nv[1], nv[0]);

	// the 128-bit division we do one at a time
	int i;
	for (i = 0; i < 8; i++)
	{
		uint128_t mod = ((uint128_t)n[i + 8] << 52) + n[i];
		uint128_t one = (uint128_t)1 << 104;
		one %= mod;

		onevec.data[0][i] = (uint64_t)one & 0xfffffffffffffULL;
		onevec.data[1][i] = (uint64_t)(one >> 52) & 0xfffffffffffffULL;

		uint128_t mtwo = one;
		mtwo += one;
		if (mtwo >= mod)
		{
			mtwo -= mod;
		}

		twovec.data[0][i] = (uint64_t)mtwo & 0xfffffffffffffULL;
		twovec.data[1][i] = (uint64_t)(mtwo >> 52) & 0xfffffffffffffULL;
	}

	mone[0] = loadu64(onevec.data[0]);
	mone[1] = loadu64(onevec.data[1]);

	mtwo[0] = loadu64(twovec.data[0]);
	mtwo[1] = loadu64(twovec.data[1]);

	// get bases into Monty rep
	bv[0] = loadu64(bases);

	__m512i mpow[2];
	mpow[0] = mone[0];
	mpow[1] = mone[1];

	rv[1] = mone[1];
	rv[0] = mone[0];

	bv[0] = _mm512_srli_epi64(bv[0], 1);
	__mmask8 done = _mm512_cmpeq_epi64_mask(bv[0], zerov);
	while (done != 0xff)
	{
		addmod104_x8(&mpow[1], &mpow[0], mpow[1], mpow[0], mpow[1], mpow[0], nv[1], nv[0]);
		__mmask8 bitcmp = _mm512_test_epi64_mask(onev, bv[0]);
		mask_addmod104_x8(&rv[1], &rv[0], ~done, rv[1], rv[1], mpow[1], mpow[0], nv[1], nv[0]);

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

	m = _mm512_set1_epi64(1ULL << 51);
	done = _mm512_cmpeq_epu64_mask(m, zerov);
	while (done != 0xff)
	{
		__mmask8 bitcmp = _mm512_test_epi64_mask(m, dv[0]);

		mask_sqrredc104_vec(&rv[1], &rv[0], ~done, rv[1], rv[0], nv[1], nv[0], vrho);
		mask_mulredc104_vec(&rv[1], &rv[0], (~done)& bitcmp, rv[1], rv[0], bv[1], bv[0], nv[1], nv[0], vrho);

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

	submod104_x8(&n1v[1], &n1v[0], mone[1], mone[0], mtwo[1], mtwo[0], nv[1], nv[0]);

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

int main(int argc, char** argv)
{
	uint64_t prp[16];
	int correct = 0;
	int i, k;

#ifndef IFMA
	dbias = _mm512_castsi512_pd(set64(0x4670000000000000ULL));
	vbias1 = set64(0x4670000000000000ULL);
	vbias2 = set64(0x4670000000000001ULL);
	vbias3 = _mm512_set1_epi64(0x4330000000000000ULL);
#endif

	for (k = 0; k < 4; k++)
	{
		correct = 0;
		
		int bits = 100;

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

		uint64_t ticks1 = my_rdtsc();

		uint64_t inc = 4;
		int num = 1000000;
		int j;

		for (j = 0; j < num; j++)
		{
			correct += _mm_popcnt_u32(fermat_prp_x8(prp));
			for (i = 0; i < 8; i++)
			{
				prp[i] += inc;
			}
			inc = 6 - inc;
		}

		uint64_t ticks2 = my_rdtsc();
		printf("total ticks = %lu, ticks per input = %lu\n",
			ticks2 - ticks1, (ticks2 - ticks1) / (num * 8));
		printf("found %d fermat-prp out of %u inputs: %1.2f%%\n",
			correct, num * 8, 100. * (double)correct / (double)(num * 8));
	}

	// bases for MR-sprp check:
	uint64_t bases[24] = {3, 5, 7, 11, 
		13, 17, 19, 23, 
		29, 31, 37, 41,
		43, 47, 53, 59, 
		61, 67, 71, 73, 
		79, 83, 89, 97};

	for (k = 0; k < 4; k++)
	{
		int num = 1000000;
		int bits = 103;
		printf("commencing test %d of %lu random 6k+1 %d-bit inputs\n", k, num, bits);

		correct = 0;
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
			prp[i + 8] = (uint64_t)(x >> 52) & 0xfffffffffffffull;
		}

		uint64_t ticks1 = my_rdtsc();

		uint64_t inc[8] = { 4, 4, 4, 4, 4, 4, 4, 4 };
		int j;

		uint64_t basecount[8];
		uint64_t maxcount[8];
		uint64_t currentbase[8];
		for (i = 0; i < 8; i++)
		{
			basecount[i] = 0;
			currentbase[i] = bases[0];
			maxcount[i] = 16;
		}

		uint32_t tested = 0;
		while (tested < num)
		{
			uint8_t prpmask = MR_sprp_x8(prp, currentbase);
			for (i = 0; i < 8; i++)
			{
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
						correct++;
						prp[i] += inc[i];
						inc[i] = 6 - inc[i];
						currentbase[i] = bases[0];
						basecount[i] = 0;
					}
				}
				else
				{
					// the input in position i is definitely not prime,
					// replace it and increment num tested
					tested++;
					prp[i] += inc[i];
					inc[i] = 6 - inc[i];
					currentbase[i] = bases[0];
					basecount[i] = 0;
				}
			}
		}

		uint64_t ticks2 = my_rdtsc();
		printf("total ticks = %lu, ticks per input = %lu\n",
			ticks2 - ticks1, (ticks2 - ticks1) / (tested));
		printf("found %d MR-sprp out of %u inputs: %1.2f%%\n",
			correct, tested, 100. * (double)correct / (double)(tested));
	}

	return 0;
}