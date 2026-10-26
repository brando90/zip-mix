# Exp 02 Phase 2: CR distributions (20k-doc pilots, committed CSVs)

Scope: pilot n=20,000 per corpus (README plans 500K); one seed (42); LZ4 frame default; docs < 512 B dropped.

## Summary

|             |     n |   mean |    std |     q1 |   median |     q3 |    iqr |    skew |   frac_in_band |
|:------------|------:|-------:|-------:|-------:|---------:|-------:|-------:|--------:|---------------:|
| fineweb_edu | 20000 | 0.7294 | 0.0814 | 0.6745 |   0.7257 | 0.783  | 0.1085 |  0.1214 |         0.6465 |
| dclm        | 20000 | 0.7491 | 0.0844 | 0.6904 |   0.7445 | 0.804  | 0.1137 |  0.1936 |         0.6191 |
| pile        | 20000 | 0.6893 | 0.1313 | 0.6186 |   0.7096 | 0.7782 | 0.1597 | -0.7872 |         0.5    |

## Quartiles vs Compel

| corpus      |   q1_ours |   q1_compel |   med_ours |   med_compel |   q3_ours |   q3_compel |
|:------------|----------:|------------:|-----------:|-------------:|----------:|------------:|
| fineweb_edu |     0.674 |        0.67 |      0.726 |         0.73 |     0.783 |        0.78 |
| dclm        |     0.69  |        0.69 |      0.744 |         0.75 |     0.804 |        0.82 |

## Pairwise KS (D = effect size) and Cohen's d

| a           | b    |   ks_D |   p |   cohens_d |
|:------------|:-----|-------:|----:|-----------:|
| fineweb_edu | dclm | 0.0931 |   0 |    -0.2376 |
| fineweb_edu | pile | 0.172  |   0 |     0.3671 |
| dclm        | pile | 0.2012 |   0 |     0.5418 |

## Bucket populations (counts per bucket)

|   M | kind     | corpus      | counts                                            |   min_share |
|----:|:---------|:------------|:--------------------------------------------------|------------:|
|   3 | uniform  | fineweb_edu | 0 10731 9269                                      |      0      |
|   3 | uniform  | dclm        | 2 8915 11083                                      |      0.0001 |
|   3 | uniform  | pile        | 416 11399 8185                                    |      0.0208 |
|   3 | quantile | fineweb_edu | 6415 7342 6243                                    |      0.3122 |
|   3 | quantile | dclm        | 4960 7074 7966                                    |      0.248  |
|   3 | quantile | pile        | 8625 5584 5791                                    |      0.2792 |
|   5 | uniform  | fineweb_edu | 0 1 3853 15372 774                                |      0      |
|   5 | uniform  | dclm        | 0 3 2889 15640 1468                               |      0      |
|   5 | uniform  | pile        | 59 885 5886 12268 902                             |      0.003  |
|   5 | quantile | fineweb_edu | 3235 4713 4411 4045 3596                          |      0.1618 |
|   5 | quantile | dclm        | 2383 3957 4206 4490 4964                          |      0.1192 |
|   5 | quantile | pile        | 6382 3330 3383 3463 3442                          |      0.1665 |
|   7 | uniform  | fineweb_edu | 0 0 5 1983 13190 4736 86                          |      0      |
|   7 | uniform  | dclm        | 0 0 4 1369 12295 6042 290                         |      0      |
|   7 | uniform  | pile        | 20 210 1110 4036 10102 4379 143                   |      0.001  |
|   7 | quantile | fineweb_edu | 1932 3334 3363 3115 2971 2823 2462                |      0.0966 |
|   7 | quantile | dclm        | 1317 2688 2901 3040 3165 3234 3655                |      0.0659 |
|   7 | quantile | pile        | 5323 2549 2307 2417 2435 2514 2455                |      0.1154 |
|  10 | uniform  | fineweb_edu | 0 0 0 1 246 3607 10227 5145 771 3                 |      0      |
|  10 | uniform  | dclm        | 0 0 1 2 105 2784 9480 6160 1444 24                |      0      |
|  10 | uniform  | pile        | 7 52 223 662 2035 3851 7617 4651 891 11           |      0.0004 |
|  10 | quantile | fineweb_edu | 1015 2220 2369 2344 2259 2152 2085 1960 1955 1641 |      0.0508 |
|  10 | quantile | dclm        | 585 1798 1917 2040 2058 2148 2232 2258 2294 2670  |      0.0292 |
|  10 | quantile | pile        | 4400 1982 1714 1616 1683 1700 1683 1780 1753 1689 |      0.0808 |
