# DD2424 Peer Review — Group 22

**Reviewer:** Group 148 (Boyi Shi, Puhao Zhu, Jinye Gong)  
**Reviewed group:** Group 22 — Erik Smit, Filip Dimitrijević, Jonatan Bölenius  
**Project title:** Transfer Learning and Semi-Supervised Learning for Pet Breed Classification  

---

## 1. In one sentence summarize the main point of this project.

The group uses a pretrained ResNet18 on the Oxford-IIIT Pet dataset to compare transfer-learning options (linear probing, fine-tuning, gradual unfreezing) and simple pseudo-labeling, especially when labels are limited or classes are imbalanced.

---

## 2. Which part of the report was most clearly written?

**Section 4 (Methods)** worked best for us. **§4.4** explains the two fine-tuning strategies in a step-by-step way, and **§4.7** makes the pseudo-label setup easy to follow, including the loss formula and how the unlabeled weight changes over training.

---

## 3. Which part/section of the report was least clear? Was it possible to understand the material presented in this section?

**§5.1 and Table 1** took a bit more effort. On full data, gradual unfreezing gets **89.53%**, while simultaneous fine-tuning reaches **90.27%**, but later the report argues gradual unfreezing is better mainly when labels are scarce (Figure 2). It would help to state this more directly in §5.1. Also, §4.7 mentions “three fractions” but lists four values—likely a small typo. We could still follow the section after reading it twice.

---

## 4. Did the report give you a better understanding of the problem/deep learning concept investigated?

**Yes.**

- **§5.2 (limited labels):** helped us see that gradual unfreezing can be more stable than fine-tuning everything at once when very little labeled data is available.
- **§5.4 (pseudo-labeling):** was useful because the group reports that pseudo-labeling did not beat the supervised baseline and explains why that may happen on this dataset (strong pretrained features, similar-looking breeds).

---

## 5. What was the most impressive experimental result presented in the project and why?

**Linear probing already reaches 89.32% test accuracy** on 37 pet breeds while training only the final layer. That is a strong result and a good starting point for the rest of the project—it shows how much ImageNet pretraining already helps before additional fine-tuning.

---

## 6. What was the most interesting or surprising experimental result presented in the project and why?

We found it interesting that **pseudo-labeling did not improve over the supervised baseline** in their tests (Table 4). That goes against the common assumption that unlabeled data always helps, and the group discusses possible reasons instead of ignoring the result. We also liked that with only **1% labels**, data augmentation plus regularization improved accuracy more clearly than regularization alone (Table 2).

---

## 7. Which experiment(s) would you like the project group to complete if they were to continue with this project?

If they continue, we would mainly like to see:

1. A short note or plot clarifying **when gradual unfreezing helps vs. hurts** compared with simultaneous fine-tuning (full data vs. low-data), since Table 1 and Figure 2 tell slightly different stories.
2. For the imbalance experiments, a bit more discussion using **per-class or F1 results** (they already report some of this in Table 3) to show how much the cat classes are affected.

These feel like natural next steps rather than a full new project.

---

## 8. Mention two things you liked about the project report and/or the video presentation.

1. **Good project structure:** the report connects several related ideas—transfer learning, limited data, imbalance, and semi-supervision—in one clear storyline instead of stopping after a single fine-tuning experiment.

2. **Balanced reporting:** the group discusses results that did not improve performance (pseudo-labeling, limited gains from full fine-tuning) and still draws reasonable conclusions from them.

---

## 9. (Optional) Reference recommendation

They already cite **FixMatch** for the confidence-threshold idea. If they want to explore semi-supervised learning further, that paper is a natural follow-up read. Otherwise, the current references already support the main methods well.

---

*Peer review — Group 148, May 2026*
