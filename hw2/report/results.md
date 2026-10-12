# Результаты экспериментов

| конфигурация | настройки | длительность, с | токенов/с | ускорение к бейзлайну | пик allocated, GiB | пик reserved, GiB | val loss | шагов оптимизатора | с/шаг |
|---|---|---|---|---|---|---|---|---|---|
| baseline | sdpa, bs 32×2 | 30 | 6,661 | 1.00× | 41.2 | 43.4 | 10.207 | 33 | 0.82 |
| fa2_padding_free_bs64_liger_compile | flash_attention_2, bs 64×1, padding-free, liger, compile | 30 | нет замера |  | 11.4 | 11.8 | 11.235 | 1 |  |
| fa2_padding_free_bs64_compile_dynamic | flash_attention_2, bs 64×1, padding-free, compile, dynamic | 120 | 28,123 | 3.94× | 15.2 | 20.0 | 8.407 | 60 | 0.20 |
| fa2_padding_free_bs64_liger_compile_dynamic | flash_attention_2, bs 64×1, padding-free, liger, compile, dynamic | 120 | 24,507 | 3.43× | 16.5 | 20.7 | 7.117 | 264 | 0.23 |
| fa2_padding_free_bs256_liger | flash_attention_2, bs 256×1, padding-free, liger | 120 | 24,415 | 3.42× | 34.4 | 48.0 | 7.776 | 126 | 0.94 |
| fa2_padding_free_bs128_liger | flash_attention_2, bs 128×1, padding-free, liger | 120 | 23,170 | 3.24× | 21.1 | 35.6 | 6.841 | 238 | 0.49 |
| fa2_packing | flash_attention_2, bs 32×2, packing | 120 | 21,110 | 2.95× | 40.2 | 76.2 | 8.124 | 79 | 1.52 |
| fa2_padding_free_bs64_liger | flash_attention_2, bs 64×1, padding-free, liger | 120 | 19,713 | 2.76× | 14.1 | 24.8 | 6.462 | 403 | 0.29 |
| fa2_padding_free_bs64 | flash_attention_2, bs 64×1, padding-free | 120 | 19,293 | 2.70× | 20.1 | 35.5 | 6.463 | 403 | 0.30 |
| fa2_padding_free | flash_attention_2, bs 32×2, padding-free | 120 | 18,625 | 2.61× | 15.1 | 21.2 | 6.485 | 389 | 0.31 |
| compile_dynamic | sdpa, bs 32×2, compile, dynamic | 120 | 10,918 | 1.53× | 31.2 | 32.1 | 8.387 | 62 | 0.51 |
| fa2_padding_free_bs128_liger_compile_dynamic | flash_attention_2, bs 128×1, padding-free, liger, compile, dynamic | 120 | 8,787 | 1.23× | 22.7 | 25.4 | 10.538 | 27 | 1.26 |
| bs16_ga4 | sdpa, bs 16×4 | 120 | 7,861 | 1.10× | 24.2 | 26.0 | 7.826 | 164 | 0.72 |
| fa2 | flash_attention_2, bs 32×2 | 120 | 7,353 | 1.03× | 39.3 | 41.6 | 7.909 | 153 | 0.77 |
| liger | sdpa, bs 32×2, liger | 120 | 7,235 | 1.01× | 26.8 | 28.5 | 7.945 | 144 | 0.78 |
| baseline | sdpa, bs 32×2 | 120 | 7,146 | 1.00× | 41.2 | 43.4 | 7.928 | 149 | 0.79 |
| checkpointing | sdpa, bs 32×2, checkpointing | 120 | 5,754 | 0.81× | 17.4 | 24.0 | 8.093 | 119 | 0.99 |
| fa2_offloading | flash_attention_2, bs 32×2, offloading | 120 | 693 | 0.10× | 16.4 | 23.6 | 11.196 | 13 | 7.80 |
| compile | sdpa, bs 32×2, compile | 120 | нет замера |  | 26.0 | 28.7 | 11.234 | 3 |  |
| fa2_padding_free_bs64_liger_compile | flash_attention_2, bs 64×1, padding-free, liger, compile | 120 | нет замера |  | 16.7 | 20.5 | 11.220 | 9 |  |
| final | flash_attention_2, bs 256×1, padding-free, compile, dynamic | 300 | 29,003 | 4.21× | 39.8 | 57.7 | 6.064 | 276 | 0.79 |
| fa2_padding_free_bs256_compile_dynamic | flash_attention_2, bs 256×1, padding-free, compile, dynamic | 300 | 28,666 | 4.16× | 39.8 | 57.9 | 6.138 | 269 | 0.80 |
| fa2_padding_free_bs128_compile_dynamic | flash_attention_2, bs 128×1, padding-free, compile, dynamic | 300 | 27,629 | 4.01× | 23.9 | 36.1 | 5.341 | 521 | 0.41 |
| fa2_padding_free_bs64_compile_dynamic | flash_attention_2, bs 64×1, padding-free, compile, dynamic | 300 | 26,426 | 3.84× | 15.4 | 22.7 | 5.097 | 1005 | 0.22 |
| fa2_padding_free_bs256_liger | flash_attention_2, bs 256×1, padding-free, liger | 300 | 23,848 | 3.46× | 34.6 | 61.9 | 5.751 | 311 | 0.96 |
| baseline | sdpa, bs 32×2 | 300 | 6,885 | 1.00× | 41.2 | 43.4 | 6.641 | 361 | 0.83 |

**S = 29,003 / 6,885 = 4.21** (итоговый запуск final против бейзлайна, по 300 с)

## Упавшие запуски

- bs64_ga1_120s: torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 6.14 GiB. GPU 0 has a total capacity of 79.10 GiB of which 3.72 GiB is free. Including non-PyTorch
- fa2_packing_liger_compile_120s: FloatingPointError: Значение loss не валидное
- fa2_packing_liger_compile_dynamic_120s: FloatingPointError: Значение loss не валидное
- fa2_padding_free_bs256_liger_compile_dynamic_120s: FloatingPointError: Значение loss не валидное
- offloading_120s: RuntimeError: attn_bias is not correctly aligned (strideH) .attn_bias.stride(1) = 35721, and should be a multiple of 4.

![tokens per second](tokens_per_second.png)

![peak memory](peak_memory.png)
