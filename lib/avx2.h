#include <gmp.h>
#include "sail.h"


void avx_mm256_adds_epi8(lbits *out, lbits vector1, lbits vector2);

void avx_mm256_adds_epi16(lbits *out, lbits vector1, lbits vector2);

void avx_mm256_subs_epi8(lbits *out, lbits vector1, lbits vector2);

void avx_mm256_subs_epi16(lbits *out, lbits vector1, lbits vector2);
