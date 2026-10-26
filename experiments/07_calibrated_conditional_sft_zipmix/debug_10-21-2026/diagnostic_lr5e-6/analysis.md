# Diagnostic analysis: lr 5e-06, 15/15 cells complete

## cal_target (N=500; base 399/500 = 79.80%)

| method | seeds | mean % | sd pp | by seed % | flip vs base | fixes base-wrong % | breaks base-right % |
|---|---:|---:|---:|---|---:|---:|---:|
| conditional_aligned | 5 | 81.16 | 0.85 | 80.4, 82.0, 82.0, 81.2, 80.2 | 11.9% | 27.7 | 5.3 |
| stratum_uniform | 5 | 80.12 | 1.41 | 79.2, 81.2, 80.4, 78.2, 81.6 | 11.0% | 23.8 | 5.6 |
| conditional_shuffled | 5 | 79.88 | 2.22 | 78.2, 77.8, 80.2, 83.4, 79.8 | 11.4% | 23.0 | 5.7 |

| contrast (aligned minus control) | paired seeds | mean pp [95% t-interval] | positive / ties | exact sign p | Holm p (2 target contrasts) | item-paired SE pp | discordant items win/loss (pooled seeds) | binomial p |
|---|---:|---|---|---:|---:|---:|---|---:|
| conditional_aligned − stratum_uniform | 5 | +1.04 [-0.94, +3.02] | 4 / 0 | 0.3750 | 0.7500 | 0.697 | 132/106 | 0.105 |
| conditional_aligned − conditional_shuffled | 5 | +1.28 [-1.67, +4.23] | 4 / 0 | 0.3750 | 0.7500 | 0.643 | 126/94 | 0.036 |

## cal_retention (N=500; base 274/500 = 54.80%)

| method | seeds | mean % | sd pp | by seed % | flip vs base | fixes base-wrong % | breaks base-right % |
|---|---:|---:|---:|---|---:|---:|---:|
| conditional_aligned | 5 | 57.20 | 1.36 | 56.0, 55.6, 58.4, 58.6, 57.4 | 24.3% | 20.6 | 12.6 |
| stratum_uniform | 5 | 57.20 | 1.24 | 57.4, 58.0, 58.4, 57.0, 55.2 | 25.5% | 21.6 | 13.4 |
| conditional_shuffled | 5 | 55.88 | 0.76 | 55.4, 54.8, 56.4, 56.6, 56.2 | 25.7% | 20.4 | 14.9 |

| contrast (aligned minus control) | paired seeds | mean pp [95% t-interval] | positive / ties | exact sign p | Holm p (2 target contrasts) | item-paired SE pp | discordant items win/loss (pooled seeds) | binomial p |
|---|---:|---|---|---:|---:|---:|---|---:|
| conditional_aligned − stratum_uniform | 5 | +0.00 [-2.41, +2.41] | 2 / 1 | 1.0000 | 1.0000 | 0.776 | 203/203 | 1.000 |
| conditional_aligned − conditional_shuffled | 5 | +1.32 [+0.50, +2.14] | 5 / 0 | 0.0625 | 0.1250 | 0.792 | 222/189 | 0.114 |

Original Stage B advance rule applied descriptively to `cal_target` (≥2 pp mean gain and ≥4/5 positive pairs against both controls): **not met**.
