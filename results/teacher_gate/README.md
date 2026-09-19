# Teacher gate / probe results

**Invalid (fp16 inference was corrupted — kept only as the record of the bug):**
`qwen2b_gate.json`, `probe_x1.json`, `probe_results_x2.json`, `probe_results_x3.json`,
`Qwen3-VL-8B-Instruct_fp16.json`. See `../teacher_diagnosis/diagnose_fp16_vs_fp32.json`.

**Valid (fp32 compute):**
| file | what |
|---|---|
| `random_gate.json` | uniform-random baseline, 30 episodes per scenario |
| `probe_2b_fp32.json` | pre-registered probe, Qwen3-VL-2B fp32 |
| `qwen2b_fp32_cal_lane{0,1}.json` | gate, 2B fp32 calibrated: DTC 1.73, HG/DC/HGS ~ random |
| `Qwen3-VL-4B-Instruct_fp32.json` | small check, 4B fp32 (probe miss by 0.01) |
| `Qwen3-VL-8B-Instruct_fp32_nf4.json` | small check, 8B NF4 weights + fp32 compute (pass) |
| `qwen8b_nf4_cal_dtc_lane{0,1}.json` | gate, 8B DTC 15+15 episodes: 1.67 (bar 3.0 not met) |
