# PPT Presentation Script (English)

**Project:** Scaling Laws and Accuracy–Efficiency Trade-offs in Modern ConvNets on CIFAR-10  
**Group 148:** Boyi Shi · Puhao Zhu · Jinye Gong  
**Course:** DD2424 — Deep Learning in Data Science  
**Duration:** ~4 minutes · **Slides:** 9 · **Speakers:** 3

---

## Speaker Allocation

| Slides | Speaker | Time |
|--------|---------|------|
| 1 – 3 | **Boyi Shi** | ~80s |
| 4 – 5 | **Puhao Zhu** | ~75s |
| 6 – 8 | **Jinye Gong** | ~85s |
| 9 | (any) | ~5s |

**Handoffs:** Boyi ends Slide 3 → Puhao · Puhao ends Slide 5 → Jinye

---

## Boyi Shi · Slides 1–3 (~80s)

### Slide 1 — Title (~10s)

> Hi everyone, we're Group 148. Our project asks a simple question: do modern ConvNet designs like ConvNeXt actually win on small images like CIFAR-10, once we compare them fairly under the same compute budget?

---

### Slide 2 — Motivation & Research Questions (~35s)

> Modern blocks were designed for ImageNet-scale images, but CIFAR-10 is only 32 by 32. Also, comparing default configs is misleading — at the same width equals 64, VGG uses 310 million FLOPs but Depthwise only uses 40 million. So we ask two research questions.
>
> First, under matched FLOPs, does ConvNeXt actually beat a strong VGG baseline?
>
> Second, which architectural axis — width, kernel size, or depth — gives us the most accuracy per extra FLOP?

---

### Slide 3 — Three Model Families + Fair-Comparison Protocol (~35s)

> We implement three block families: a VGG baseline with standard 3-by-3 convolutions, a Depthwise variant that splits convolution into depthwise plus pointwise, and a ConvNeXt-style block with large depthwise kernels, LayerNorm, and an inverted bottleneck.
>
> All three are trained from scratch in a unified PyTorch pipeline, and we align them to three FLOPs tiers — low at 150 million, medium at 310 million, and high at 460 million — by tuning the width.
>
> Same data split, same seed, same recipe. Puhao will walk through the setup.

**Handoff:** *"Puhao, over to you."*

---

## Puhao Zhu · Slides 4–5 (~75s)

### Slide 4 — Training Setup & Experiment Matrix (~30s)

> Our training recipe is fixed across every run: AdamW with cosine annealing and a 5-epoch warmup, 200 epochs, batch size 128. We deliberately turn off MixUp and AMP so the comparison stays clean.
>
> In total we ran 17 experiments: 9 for the block-versus-block matrix at three FLOPs tiers, 7 for kernel scans where we sweep k equals 3, 5, 7, and 11, and 2 for ConvNeXt depth variants. All 17 finished a full 200 epochs.

---

### Slide 5 — Result 1: Matched-FLOPs Block Comparison (~45s)

> This is our first headline result. After aligning all three families to the same FLOPs budget, VGG wins every single tier.
>
> At the low tier, VGG scores 92.05 percent, beating Depthwise by 1.2 points and ConvNeXt by 3.2 points. At the medium tier, VGG hits 92.78 percent — that's our global best. At the high tier, VGG is still on top.
>
> ConvNeXt loses by 2 to 3 points in every tier. Depthwise is the efficiency champion — at the medium tier it reaches 99.4 percent of VGG's accuracy using 10 percent fewer FLOPs.
>
> So the answer to RQ1 is clear: under matched compute, ConvNeXt does not win on CIFAR-10.

**Handoff:** *"Jinye will show the scaling results and conclusions."*

---

## Jinye Gong · Slides 6–8 (~85s)

### Slide 6 — Result 2: Width Scaling + Pareto Frontier (~40s)

> Looking at the full accuracy-versus-FLOPs plot — every dot here is a 200-epoch run. The best trade-off curve, the upper envelope, is owned entirely by VGG and Depthwise. Every ConvNeXt point is dominated — there's always a VGG or DW model that is both cheaper and more accurate.
>
> Also notice VGG saturates after width 64: pushing to width 80 actually drops accuracy to 92.46 percent. Depthwise gives the steepest gain in the low-to-medium region. So just scaling FLOPs isn't free — the marginal return drops fast once you cross the medium tier.

---

### Slide 7 — Result 3: Kernel & Depth Scans (~35s)

> So why does ConvNeXt struggle? The kernel scan is the smoking gun. Both Depthwise and ConvNeXt drop monotonically as the kernel grows — losing nearly 4 points when we go from k equals 3 to k equals 11.
>
> The reason is geometric: after three pooling stages, the feature map is only 4 by 4. The receptive field is already saturated, so larger kernels just over-smooth the signal. The large-kernel trick from ImageNet simply does not transfer to 32 by 32 images.
>
> Depth shows the same plateau — going from 2 blocks per stage to 3 barely helps, going down to 1 hurts.

---

### Slide 8 — Conclusions + Budget-Aware Design Rules (~25s)

> To wrap up, three takeaways.
>
> First, VGG is the best family on CIFAR-10, with our top result at 92.78 percent. Second, if compute is tight, use Depthwise with 3-by-3 kernels. Third, do not copy ConvNeXt-style large kernels to small images — they hurt more than they help.
>
> Concretely: under 180 million FLOPs use VGG width 48, around 300 million use VGG width 64, and above 430 million the returns diminish so don't bother scaling further.

---

## Slide 9 — Thank You (~5s, any speaker)

> Thank you — we're happy to take any questions.

---

## Q&A — Anticipated Questions

| Question | Short answer |
|----------|--------------|
| **Why no MixUp?** | We turned off MixUp and AMP to keep the recipe identical across all 17 runs — the goal was a clean architectural comparison, not maximum accuracy. |
| **Does VGG win just because it has more parameters?** | No — see the Pareto plot. Depthwise has fewer params and still beats ConvNeXt at every budget. Block design matters more than param count. |
| **ConvNeXt is SOTA on ImageNet — why does it lose here?** | Because the feature map after three pools is just 4×4. Large depthwise kernels were designed to capture long-range context at 224×224, but at 32×32 the receptive field is saturated and they over-smooth. |
| **Single seed — how reliable are the gaps?** | We report a single seed (42) per config. The 2–3 point gaps between families are well above typical CIFAR-10 seed variance (~0.3 pt), so the ranking is robust; absolute numbers may shift slightly. |
| **Why FLOPs = 2 × MACs?** | One MAC is one multiply plus one add, which is 2 floating-point operations. `thop` returns MAC counts, so we multiply by 2 to report true FLOPs and stay comparable across papers. |

### Q&A speaker hints

- **Boyi:** architecture, why MixUp/AMP off  
- **Puhao:** FLOPs, MACs, parameters, figures  
- **Jinye:** training pipeline, seed, experiment count  

---

## Rehearsal checklist

- [ ] Boyi: finish Slides 1–3 within ~80s; hand off to Puhao on Slide 3  
- [ ] Puhao: finish Slides 4–5 within ~75s; numbers match slides (92.05 / 92.78 / 99.4%)  
- [ ] Jinye: finish Slides 6–8 within ~85s; point at Pareto and kernel plots  
- [ ] Group: decide who speaks Slide 9  
- [ ] Group: each member owns 1–2 Q&A rows  

---

*Reference: `REPORT_zh.md`, `figures/`, `logs/results.csv` · Chinese script: `PPT_SCRIPT_zh.md`*
