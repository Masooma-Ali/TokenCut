# Iteration 1 evaluation summary

Tasks: 40

## All repos (40 tasks)

| tier | variant_tokens | reduction_vs_full | pool_has_gold_file | gold_file_recall | gold_chunk_recall | gold_chunk_skeleton_recall | t_parse_s | t_graph_s | t_variants_s |
|---|---|---|---|---|---|---|---|---|---|
| aggressive | 4,752 | 0.851 | 0.750 | 0.725 | 0.563 | 0.563 | 2.36 | 1.95 | 2.30 |
| moderate | 11,085 | 0.651 | 0.750 | 0.750 | 0.678 | 0.678 | 2.36 | 1.95 | 2.30 |
| conservative | 18,996 | 0.403 | 0.750 | 0.750 | 0.697 | 0.697 | 2.36 | 1.95 | 2.30 |
| full | 31,806 | 0.000 | 0.750 | 0.750 | 0.769 | 0.769 | 2.36 | 1.95 | 2.30 |

## django/django (15 tasks)

| tier | variant_tokens | reduction_vs_full | pool_has_gold_file | gold_file_recall | gold_chunk_recall | gold_chunk_skeleton_recall | t_parse_s | t_graph_s | t_variants_s |
|---|---|---|---|---|---|---|---|---|---|
| aggressive | 4,739 | 0.851 | 0.733 | 0.733 | 0.589 | 0.589 | 5.22 | 4.30 | 5.12 |
| moderate | 11,055 | 0.652 | 0.733 | 0.733 | 0.768 | 0.768 | 5.22 | 4.30 | 5.12 |
| conservative | 18,942 | 0.404 | 0.733 | 0.733 | 0.786 | 0.786 | 5.22 | 4.30 | 5.12 |
| full | 31,756 | 0.000 | 0.733 | 0.733 | 0.786 | 0.786 | 5.22 | 4.30 | 5.12 |

## pallets/flask (3 tasks)

| tier | variant_tokens | reduction_vs_full | pool_has_gold_file | gold_file_recall | gold_chunk_recall | gold_chunk_skeleton_recall | t_parse_s | t_graph_s | t_variants_s |
|---|---|---|---|---|---|---|---|---|---|
| aggressive | 4,777 | 0.850 | 0.667 | 0.667 | 0.333 | 0.333 | 0.21 | 0.15 | 0.19 |
| moderate | 11,154 | 0.650 | 0.667 | 0.667 | 0.333 | 0.333 | 0.21 | 0.15 | 0.19 |
| conservative | 19,109 | 0.401 | 0.667 | 0.667 | 0.500 | 0.500 | 0.21 | 0.15 | 0.19 |
| full | 31,914 | 0.000 | 0.667 | 0.667 | 0.667 | 0.667 | 0.21 | 0.15 | 0.19 |

## psf/requests (6 tasks)

| tier | variant_tokens | reduction_vs_full | pool_has_gold_file | gold_file_recall | gold_chunk_recall | gold_chunk_skeleton_recall | t_parse_s | t_graph_s | t_variants_s |
|---|---|---|---|---|---|---|---|---|---|
| aggressive | 4,763 | 0.850 | 1.000 | 0.833 | 0.750 | 0.750 | 0.65 | 0.42 | 0.19 |
| moderate | 11,114 | 0.651 | 1.000 | 1.000 | 0.917 | 0.917 | 0.65 | 0.42 | 0.19 |
| conservative | 19,042 | 0.402 | 1.000 | 1.000 | 0.917 | 0.917 | 0.65 | 0.42 | 0.19 |
| full | 31,839 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.65 | 0.42 | 0.19 |

## pytest-dev/pytest (16 tasks)

| tier | variant_tokens | reduction_vs_full | pool_has_gold_file | gold_file_recall | gold_chunk_recall | gold_chunk_skeleton_recall | t_parse_s | t_graph_s | t_variants_s |
|---|---|---|---|---|---|---|---|---|---|
| aggressive | 4,755 | 0.851 | 0.688 | 0.688 | 0.512 | 0.512 | 0.73 | 0.67 | 0.85 |
| moderate | 11,091 | 0.651 | 0.688 | 0.688 | 0.575 | 0.575 | 0.73 | 0.67 | 0.85 |
| conservative | 19,008 | 0.403 | 0.688 | 0.688 | 0.575 | 0.575 | 0.73 | 0.67 | 0.85 |
| full | 31,821 | 0.000 | 0.688 | 0.688 | 0.688 | 0.688 | 0.73 | 0.67 | 0.85 |

