# Statistics Assignment — Full Worked Solutions

---

## Question 1: Correlation between Exercise Duration and Resting Heart Rate

### (a) Computational Table

To evaluate the linear association between Exercise Duration (X) and Resting Heart Rate (Y), we compute XY, X², and Y² for each participant.

| Participant | X (min) | Y (bpm) | XY | X² | Y² |
|---|---|---|---|---|---|
| 1 | 15 | 85 | 1275 | 225 | 7225 |
| 2 | 20 | 82 | 1640 | 400 | 6724 |
| 3 | 25 | 81 | 2025 | 625 | 6561 |
| 4 | 30 | 77 | 2310 | 900 | 5929 |
| 5 | 35 | 76 | 2660 | 1225 | 5776 |
| 6 | 40 | 74 | 2960 | 1600 | 5476 |
| 7 | 45 | 73 | 3285 | 2025 | 5329 |
| 8 | 50 | 71 | 3550 | 2500 | 5041 |
| 9 | 55 | 69 | 3795 | 3025 | 4761 |
| 10 | 60 | 67 | 4020 | 3600 | 4489 |
| 11 | 65 | 65 | 4225 | 4225 | 4225 |
| **Σ** | **440** | **820** | **31745** | **20350** | **61536** |

### (b) Aggregate Statistics

From the table (n = 11):

- ΣX = 440, ΣY = 820
- ΣXY = 31745
- ΣX² = 20350, ΣY² = 61536
- X̄ = ΣX/n = 440/11 = **40**
- Ȳ = ΣY/n = 820/11 = **74.545**

These five sums (n, ΣX, ΣY, ΣXY, ΣX², ΣY²) are exactly the quantities needed for Karl Pearson's coefficient of correlation.

### (c) Karl Pearson's Coefficient of Correlation

**Formula:**

$$r = \dfrac{n\Sigma XY - (\Sigma X)(\Sigma Y)}{\sqrt{\left[n\Sigma X^{2}-(\Sigma X)^{2}\right]\left[n\Sigma Y^{2}-(\Sigma Y)^{2}\right]}}$$

**Step 1 — Numerator:**

n·ΣXY = 11 × 31745 = 349195
ΣX·ΣY = 440 × 820 = 360800

Numerator = 349195 − 360800 = **−11605**

**Step 2 — Denominator (X part):**

n·ΣX² = 11 × 20350 = 223850
(ΣX)² = 440² = 193600

n·ΣX² − (ΣX)² = 223850 − 193600 = **30250**

**Step 3 — Denominator (Y part):**

n·ΣY² = 11 × 61536 = 676896
(ΣY)² = 820² = 672400

n·ΣY² − (ΣY)² = 676896 − 672400 = **4496**

**Step 4 — Combine:**

Denominator = √(30250 × 4496) = √136,004,000 ≈ **11662.07**

**Step 5 — Final value:**

$$r = \frac{-11605}{11662.07} \approx -0.995$$

**r ≈ −0.995**

### (d) Interpretation of the Coefficient

- **Direction:** r is negative, indicating an *inverse* relationship — as daily exercise duration increases, resting heart rate decreases.
- **Magnitude:** |r| = 0.995 is extremely close to 1, indicating a **very strong linear relationship** (on the standard scale, |r| > 0.8 is considered a strong correlation).
- **Coefficient of determination:** r² = (−0.995)² ≈ **0.990**, meaning about **99%** of the variation in resting heart rate among these participants can be statistically explained by variation in exercise duration, leaving only about 1% attributable to other factors.

### (e) Recommendation to the Wellness Program Coordinator

The data provide strong statistical evidence (r ≈ −0.995, r² ≈ 0.99) that longer daily exercise duration is associated with lower resting heart rate, a recognized marker of better cardiovascular fitness. It is therefore reasonable to recommend that the program coordinator **encourage participants to increase their regular daily exercise duration**, as this is strongly associated with improved cardiovascular indicators in this sample.

*Caveat for the coordinator:* correlation does not by itself establish causation, and the sample size (n = 11) is small. A larger, longer-term, and ideally randomized study — along with tracking of other cardiovascular markers — would strengthen the basis for program-wide policy changes.

---

## Question 2: One-Way ANOVA on Branch Waiting Times

### (a) Summary Statistics

| | Branch A | Branch B | Branch C |
|---|---|---|---|
| Data | 18, 20, 19, 21, 22, 20 | 15, 16, 17, 18, 16, 17 | 12, 13, 14, 15, 13, 14 |
| Sample size (nᵢ) | 6 | 6 | 6 |
| Group total (Tᵢ) | 120 | 99 | 81 |
| Group mean (X̄ᵢ) | 20.00 | 16.50 | 13.50 |

- Total sample size: N = n_A + n_B + n_C = 6 + 6 + 6 = **18**
- Grand total: G = 120 + 99 + 81 = **300**
- Grand mean: X̄ = G/N = 300/18 = **16.667**

### (b) Correction Factor and Sums of Squares

**Correction Factor (CF):**

$$CF = \frac{G^{2}}{N} = \frac{300^{2}}{18} = \frac{90000}{18} = 5000$$

**Total Sum of Squares (SST):**

First find ΣΣX² (sum of squares of every individual observation, across all branches):

- Branch A: 18²+20²+19²+21²+22²+20² = 324+400+361+441+484+400 = 2410
- Branch B: 15²+16²+17²+18²+16²+17² = 225+256+289+324+256+289 = 1639
- Branch C: 12²+13²+14²+15²+13²+14² = 144+169+196+225+169+196 = 1099

ΣΣX² = 2410 + 1639 + 1099 = **5148**

$$SST = \Sigma\Sigma X^{2} - CF = 5148 - 5000 = \mathbf{148}$$

**Sum of Squares Between Groups (SSB):**

$$SSB = \sum \frac{T_i^{2}}{n_i} - CF$$

$$= \frac{120^{2}}{6}+\frac{99^{2}}{6}+\frac{81^{2}}{6} - 5000 = 2400 + 1633.5 + 1093.5 - 5000$$

$$= 5127 - 5000 = \mathbf{127}$$

**Sum of Squares Within Groups (SSW):**

$$SSW = SST - SSB = 148 - 127 = \mathbf{21}$$

*(Check: this matches directly summing squared deviations from each group mean: Branch A = 10, Branch B = 5.5, Branch C = 5.5 → total 21 ✓)*

### (c) One-Way ANOVA Table

| Source of Variation | Sum of Squares (SS) | df | Mean Square (MS) | F |
|---|---|---|---|---|
| Between Branches | 127 | k − 1 = 2 | 127/2 = 63.5 | 63.5 / 1.4 = **45.36** |
| Within Branches (Error) | 21 | N − k = 15 | 21/15 = 1.4 | |
| **Total** | **148** | **N − 1 = 17** | | |

### (d) Hypothesis Test (Critical Value Approach)

**Hypotheses:**
- H₀: μ_A = μ_B = μ_C (mean waiting time is the same at all three branches)
- H₁: At least one branch mean differs from the others

**Level of significance:** α = 0.05
**Degrees of freedom:** df₁ = 2 (numerator), df₂ = 15 (denominator)
**Critical value:** F₀.₀₅(2, 15) = **3.68** (from F-distribution table)

**Decision rule:** Reject H₀ if F_calculated > F_critical

**Comparison:** F_calculated = 45.36 > F_critical = 3.68

**Decision:** **Reject H₀**

**Conclusion:** At the 5% level of significance, there is sufficient statistical evidence to conclude that the average customer waiting time differs significantly among the three branches.

### (e) Recommendation to Senior Management

The ANOVA results show a highly significant difference in mean waiting times across branches:

- **Branch C:** 13.5 minutes (lowest)
- **Branch B:** 16.5 minutes
- **Branch A:** 20.0 minutes (highest)

**Branch C recorded the lowest average waiting time**, suggesting that whichever operational strategy is in place there (e.g., additional counters, token-based queueing, or dedicated staff for routine transactions) is currently the most effective at reducing customer wait times. Management should investigate Branch C's specific practices and consider **rolling them out to Branch A and Branch B**, where waiting times are markedly higher.

**Important caveat:** the ANOVA F-test only establishes that *at least one* branch mean differs from the others — it does not by itself identify *which specific pairs* of branches differ significantly. To pinpoint this (e.g., confirm that A vs. C, and B vs. C, are each individually significant), a **post hoc test such as Tukey's HSD** should be conducted before finalizing operational decisions.
