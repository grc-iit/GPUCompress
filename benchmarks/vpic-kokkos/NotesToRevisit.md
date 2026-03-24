# Benchmark Notes to Revisit

## GPU-Time Overhead Percentages vs Wall-Clock

The overhead breakdown table in the console output computes percentages relative to the
**sum of per-chunk GPU times** (stats + nn + preproc + comp + explore + sgd), NOT wall-clock
`write_ms`. Because 8 compression workers run in parallel, this sum can exceed wall-clock
time by up to 8x. The table labels already state "cumulative across chunks, 8 concurrent
workers" and print `wall: {write_ms} ms` alongside, but a reader skimming the numbers
could misinterpret them as wall-clock fractions.

No code fix needed — labels are accurate. Just be aware when citing these percentages in
the paper that they represent GPU-time composition, not wall-clock budget allocation.

## MAPE and MAE Both Use Clamped Timing Values (All Drivers)

Applies to: Gray-Scott, SDRBench, VPIC

Both MAPE and MAE for compression/decompression time use `compression_ms` /
`decompression_ms` (clamped to 5ms floor), not the unclamped `_raw` values.
This ensures both metrics measure prediction error against the same baseline
that the cost model itself uses internally.

The unclamped `_raw` values are still used in CSV `actual_comp_ms_raw` columns
and in the timing breakdown (comp_ms, decomp_ms in the aggregate CSV) for
accurate latency reporting.
