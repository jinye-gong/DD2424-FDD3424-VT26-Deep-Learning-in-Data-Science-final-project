# 报告实验数据（自动生成）

> 由 `scripts/generate_report_assets.py` 生成；训练完成后重新运行以更新。

## 表 2 — 默认 width=64（未对齐 FLOPs）

| Model | Params | FLOPs |
|-------|--------|-------|
| vgg_baseline | 2,197,706 | 309.5M |
| depthwise_small | 1,186,149 | 40.4M |
| convnext_small | 1,662,666 | 462.4M |

## 表 4 — 三档位 FLOPs 对齐（profile，待训练填精度）

| Budget | Model | width | FLOPs | Params |
|--------|-------|-------|-------|--------|
| low | vgg_baseline | 48 | 175.1M | 1,237,402 |
| low | depthwise_small | 128 | 152.1M | 4,723,365 |
| low | convnext_small | 32 | 122.1M | 429,930 |
| med | vgg_baseline | 64 | 309.5M | 2,197,706 |
| med | depthwise_small | 175 | 279.6M | 8,818,287 |
| med | convnext_small | 52 | 309.2M | 1,106,310 |
| high | vgg_baseline | 80 | 481.9M | 3,431,930 |
| high | depthwise_small | 224 | 453.4M | 14,437,509 |
| high | convnext_small | 64 | 462.4M | 1,662,666 |

## 已完成训练（含 test accuracy）

| run_tag | epochs | budget | width | FLOPs | Best Val | Test | 备注 |
|--------|--------|--------|-------|-------|----------|------|------|
| vgg_baseline | 1 | — | — | 309.5M | 23.94% | 23.92% |  |
| convnext_small | 200 | — | — | 462.4M | 91.40% | 90.12% |  |
| _smoke_depthwise_w64 | 1 | baseline | 64 | 40.4M | 20.94% | 21.17% |  |
| vgg_baseline_w64_seed42_ep200 | 200 |  | 64 | 309.5M | 93.92% | 92.78% | ckpt 197/200 |
| convnext_small_w64_seed42_ep200 | 200 |  | 64 | 462.4M | 91.40% | 90.12% |  |
| depthwise_small_w64_seed42_ep200 | 200 | baseline | 64 | 40.4M | 91.20% | 89.58% | ckpt 197/200 |
| vgg_baseline_w48_seed42_ep200 | 200 | low | 48 | 175.1M | 92.58% | 92.05% | ckpt 192/200 |
| depthwise_small_w128_seed42_ep200 | 200 | low | 128 | 152.1M | 92.42% | 90.86% | ckpt 170/200 |
| convnext_small_w32_seed42_ep200 | 200 | low | 32 | 122.1M | 89.80% | 88.85% | ckpt 186/200 |
| depthwise_small_w175_seed42_ep200 | 200 | med | 175 | 279.6M | 92.46% | 92.18% | ckpt 195/200 |
| convnext_small_w52_seed42_ep200 | 200 | med | 52 | 309.2M | 90.70% | 90.34% | ckpt 189/200 |
| vgg_baseline_w80_seed42_ep200 | 200 | high | 80 | 481.9M | 93.40% | 92.46% | ckpt 161/200 |
| depthwise_small_w224_seed42_ep200 | 200 | high | 224 | 453.4M | 92.88% | 92.15% | ckpt 189/200 |
| convnext_small_w52_k3_seed42_ep200 | 200 | med_kernel | 52 | 294.3M | 92.46% | 91.85% | ckpt 176/200 |
| convnext_small_w52_k5_seed42_ep200 | 200 | med_kernel | 52 | 300.3M | 92.06% | 90.95% | ckpt 185/200 |
| convnext_small_w52_k11_seed42_ep200 | 200 | med_kernel | 52 | 336.1M | 90.04% | 88.21% | ckpt 188/200 |
| depthwise_small_w175_k5_seed42_ep200 | 200 | med_kernel | 175 | 291.9M | 92.16% | 91.47% | ckpt 199/200 |
| convnext_small_w43_d3-3-3_seed42_ep200 | 200 | depth_deep | 43 | 312.9M | 91.08% | 90.27% | ckpt 198/200 |
| convnext_small_w64_d1-1-1_seed42_ep200 | 200 | depth_shallow | 64 | 249.8M | 90.90% | 89.16% | ckpt 187/200 |
| depthwise_small_w175_k7_seed42_ep200 | 200 | med_kernel | 175 | 310.3M | 91.08% | 90.28% | ckpt 191/200 |
| depthwise_small_w175_k11_seed42_ep200 | 200 | med_kernel | 175 | 365.6M | 89.66% | 88.49% | ckpt 190/200 |
