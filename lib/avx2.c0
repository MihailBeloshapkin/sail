#include <gmp.h>
#include <immintrin.h>
#include "sail.h"
#include "avx2.h"


// GMP import/export parameters to operate with bitvectors 
const int WORD_ORDER = 1;
const int ENDIAN = 1;
const int NAILS = 0;


__m256i to_epi8_rev(int8_t *vect) {
  return _mm256_setr_epi8(vect[0], vect[1], vect[2], vect[3], vect[4],
    vect[5], vect[6], vect[7], vect[8], vect[9],
    vect[10], vect[11], vect[12], vect[13], vect[14],
    vect[15], vect[16], vect[17], vect[18], vect[19],
    vect[20], vect[21], vect[22], vect[23], vect[24],
    vect[25], vect[26], vect[27], vect[28], vect[29], 
    vect[30], vect[31]);
}


__m256i to_epi16_rev(int16_t *vect) {
  return _mm256_setr_epi16(vect[0], vect[1], vect[2], vect[3],
    vect[4], vect[5], vect[6], vect[7], vect[8],
    vect[9], vect[10], vect[11], vect[12], vect[13], vect[14],
    vect[15]);
}


void avx_mm256_adds_epi8(lbits *out, lbits vector1, lbits vector2) {
  size_t element_size = 8;
  int vector_length = 32;

  int8_t *vect1 = malloc(vector_length * sizeof(int8_t));
  size_t v1_size = -1;
  int8_t *vect2 = malloc(vector_length * sizeof(int8_t));
  size_t v2_size = -1;
  mpz_export(vect1, &v1_size, WORD_ORDER, element_size, ENDIAN, NAILS, *vector1.bits);
  mpz_export(vect2, &v2_size, WORD_ORDER, element_size, ENDIAN, NAILS, *vector2.bits);
  __m256i v1 = to_epi8_rev(vect1);
  __m256i v2 = to_epi8_rev(vect2);
  __m256i result = _mm256_adds_epi8(v1, v2);
  int8_t *c = (int8_t *) & result;
  mpz_t *r = out->bits;
  mpz_import(r, vector_length, WORD_ORDER, sizeof(c[0]), ENDIAN, NAILS, c);
  free(vect1);
  free(vect2);
  return;
}


void avx_mm256_adds_epi16(lbits *out, lbits vector1, lbits vector2) {
  size_t element_size = 16;
  int vector_length = 16;
  int16_t *vect1 = malloc(vector_length * sizeof(int16_t));
  size_t v1_size = -1;
  int16_t *vect2 = malloc(vector_length * sizeof(int16_t));
  size_t v2_size = -1;
  mpz_export(vect1, &v1_size, WORD_ORDER, element_size, ENDIAN, NAILS, *vector1.bits);
  mpz_export(vect2, &v2_size, WORD_ORDER, element_size, ENDIAN, NAILS, *vector2.bits);
  __m256i v1 = to_epi8_rev(vect1);
  __m256i v2 = to_epi8_rev(vect2);
  __m256i result = _mm256_adds_epi16(v1, v2);
  int16_t *c = (int16_t *)&result;
  mpz_t *r = out->bits;
  mpz_import(r, vector_length, WORD_ORDER, sizeof(c[0]), ENDIAN, NAILS, c);
  free(vect1);
  free(vect2);
  return;
}


void avx_mm256_subs_epi8(lbits *out, lbits vector1, lbits vector2) {
  size_t element_size = 8;
  int vector_length = 32;
  int8_t *vect1 = malloc(vector_length * sizeof(int8_t));
  size_t v1_size = -1;
  int8_t *vect2 = malloc(vector_length * sizeof(int8_t));
  size_t v2_size = -1;
  mpz_export(vect1, &v1_size, WORD_ORDER, element_size, ENDIAN, NAILS, *vector1.bits);
  mpz_export(vect2, &v2_size, WORD_ORDER, element_size, ENDIAN, 0, *vector2.bits);
  __m256i v1 = to_epi8_rev(vect1);
  __m256i v2 = to_epi8_rev(vect2);
  __m256i result = _mm256_subs_epi8(v1, v2);
  int8_t *c = (int8_t *)&result;
  mpz_t *r = out->bits;
  mpz_import(r, vector_length, WORD_ORDER, sizeof(c[0]), ENDIAN, NAILS, c);
  free(vect1);
  free(vect2);
  return;
}


void avx_mm256_subs_epi16(lbits *out, lbits vector1, lbits vector2) {
  size_t element_size = 16;
  int vector_length = 16;
  int16_t *vect1 = malloc(vector_length * sizeof(int16_t));
  size_t v1_size = -1;
  int16_t *vect2 = malloc(vector_length * sizeof(int16_t));
  size_t v2_size = -1;
  mpz_export(vect1, &v1_size, WORD_ORDER, element_size, ENDIAN, NAILS, *vector1.bits);
  mpz_export(vect2, &v2_size, WORD_ORDER, element_size, ENDIAN, NAILS, *vector2.bits);
  __m256i v1 = to_epi8_rev(vect1);
  __m256i v2 = to_epi8_rev(vect2);
  __m256i result = _mm256_subs_epi16(v1, v2);
  int16_t *c = (int16_t *)&result;
  mpz_t *r = out->bits;
  mpz_import(r, vector_length, WORD_ORDER, sizeof(c[0]), ENDIAN, NAILS, c);
  free(vect1);
  free(vect2);
  return;
}