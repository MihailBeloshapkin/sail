$include <vector.sail>

val new_avx = pure "new_avx" :  (bits(256), bits(256)) -> bits(256)

val avx_mm256_adds_epi8 = pure "avx_mm256_adds_epi8" :  (bits(256), bits(256)) -> bits(256)

val avx_mm256_adds_epi16 = pure "avx_mm256_adds_epi16" :  (bits(256), bits(256)) -> bits(256)

val avx_mm256_subs_epi8 = pure "avx_mm256_subs_epi8" :  (bits(256), bits(256)) -> bits(256)

val avx_mm256_subs_epi16 = pure "avx_mm256_subs_epi16" :  (bits(256), bits(256)) -> bits(256)

