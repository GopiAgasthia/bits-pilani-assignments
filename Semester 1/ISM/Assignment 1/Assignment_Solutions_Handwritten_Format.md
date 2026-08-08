# ISM Assignment Solutions - Hand-Written Format

---

## **Question 1: Defective Items Analysis**

### **Given Data:**
12, 15, 14, 13, 16, 18, 14, 15, 13, 14, 17, 14, 36

---

### **(a) Mean, Median, Mode and Distribution Shape**

**Step 1: Calculate Mean**

n = 13

Sum = 12 + 15 + 14 + 13 + 16 + 18 + 14 + 15 + 13 + 14 + 17 + 14 + 36

Sum = 211

Mean (x̄) = Σx/n = 211/13 = **16.23**

---

**Step 2: Find Median**

Sorted data: 12, 13, 13, 14, 14, 14, 14, 15, 15, 16, 17, 18, 36

Position of median = (n+1)/2 = (13+1)/2 = 7th position

Median = **14**

---

**Step 3: Find Mode**

Frequency count:
- 12 appears 1 time
- 13 appears 2 times
- 14 appears 4 times ← highest frequency
- 15 appears 2 times
- 16 appears 1 time
- 17 appears 1 time
- 18 appears 1 time
- 36 appears 1 time

Mode = **14**

---

**Step 4: Determine Distribution Shape**

Comparison:
- Mean = 16.23
- Median = 14
- Mode = 14

Since: Mean > Median = Mode

**Conclusion:** The distribution is **positively (right) skewed**.

The extreme value (36) pulls the mean to the right, while median and mode remain at the center of the bulk of the data.

---

### **(b) Five-Point Summary and Outliers (IQR Method)**

**Sorted Data:**
12, 13, 13, 14, 14, 14, 14, 15, 15, 16, 17, 18, 36

---

**Step 1: Five-Point Summary**

1. **Minimum** = 12

2. **Q₁ (First Quartile)**
   - Position = (n+1)/4 = 14/4 = 3.5
   - Q₁ = average of 3rd and 4th values
   - Q₁ = (13 + 14)/2 = **13.5**

3. **Median (Q₂)** = 14

4. **Q₃ (Third Quartile)**
   - Position = 3(n+1)/4 = 3(14)/4 = 10.5
   - Q₃ = average of 10th and 11th values
   - Q₃ = (15 + 16)/2 = **15.5**

5. **Maximum** = 36

**Five-Point Summary:** (12, 13.5, 14, 15.5, 36)

---

**Step 2: Calculate IQR**

IQR = Q₃ - Q₁ = 15.5 - 13.5 = **2**

---

**Step 3: Determine Outlier Boundaries**

**Mild Outliers:**

Lower Fence = Q₁ - 1.5(IQR)
            = 13.5 - 1.5(2)
            = 13.5 - 3
            = **10.5**

Upper Fence = Q₃ + 1.5(IQR)
            = 15.5 + 1.5(2)
            = 15.5 + 3
            = **18.5**

**Extreme Outliers:**

Lower Extreme Fence = Q₁ - 3(IQR)
                    = 13.5 - 3(2)
                    = 13.5 - 6
                    = **7.5**

Upper Extreme Fence = Q₃ + 3(IQR)
                    = 15.5 + 3(2)
                    = 15.5 + 6
                    = **21.5**

---

**Step 4: Identify Outliers**

Check each value:
- All values from 12 to 18 lie within [10.5, 18.5] ✓
- Value 36: 36 > 18.5 → **Mild Outlier**
- Value 36: 36 > 21.5 → **Extreme Outlier**

**Conclusion:** 36 is an **EXTREME OUTLIER**

---

### **(c) Should Company Report Mean or Median? (Mathematical Justification)**

**Given:**
- Mean (x̄) = 16.23
- Median (M) = 14
- Outlier present: 36

---

**Mathematical Justification:**

**1. Effect of Outlier on Mean:**

Mean without outlier:
x̄₁₂ = (211 - 36)/12 = 175/12 = 14.58

Difference = 16.23 - 14.58 = 1.65

Percentage increase = (1.65/14.58) × 100 = **11.3%**

The outlier inflates the mean by 11.3%, making it unrepresentative.

---

**2. Effect of Outlier on Median:**

Sorted data without 36:
12, 13, 13, 14, 14, 14, 14, 15, 15, 16, 17, 18

Median₁₂ = (14 + 14)/2 = 14

Difference = 14 - 14 = 0

The median remains **unchanged** (robust to outliers).

---

**3. Sensitivity Analysis:**

**Mean Sensitivity:**
- Mean is calculated using ALL values: x̄ = (Σxᵢ)/n
- A single extreme value affects the sum significantly
- Sensitivity = ∂x̄/∂xᵢ = 1/n (constant for all observations)

**Median Sensitivity:**
- Median depends only on middle position(s)
- Extreme values don't affect middle positions
- Breakdown point = 50% (can tolerate up to 50% outliers)

---

**4. Coefficient of Variation:**

With outlier:
- Standard deviation (s) ≈ 6.47
- CV = (s/x̄) × 100 = (6.47/16.23) × 100 = 39.9%

Without outlier:
- Standard deviation (s) ≈ 1.73
- CV = (s/x̄) × 100 = (1.73/14.58) × 100 = 11.9%

High CV (39.9%) indicates high variability, suggesting mean is unreliable.

---

**5. Relative Difference:**

|Mean - Median|/Median = |16.23 - 14|/14 = 2.23/14 = 0.159 = **15.9%**

A difference > 10% indicates significant skewness and suggests median is more appropriate.

---

**Mathematical Conclusion:**

The company should report the **MEDIAN (14)** because:

1. **Robustness:** Median is unaffected by the extreme outlier (36)
2. **Representativeness:** 92% of months (12/13) have defects ≤ 18
3. **Stability:** Mean changes by 11.3% with outlier; median unchanged
4. **Central Tendency:** Median better represents typical performance
5. **Client Communication:** Median (14) reflects normal operations; mean (16.23) is misleading

**Formula Summary:**
- Median is resistant: Breakdown point = 50%
- Mean is sensitive: Influenced by every observation
- For skewed data with outliers: **Median > Mean** as measure of center

---

---

## **Question 2: Naïve Bayes Classification**

### **Given Training Data:**

| Visitor | Device  | Subscribe |
|---------|---------|-----------|
| 1       | Mobile  | No        |
| 2       | Mobile  | Yes       |
| 3       | Desktop | Yes       |
| 4       | Desktop | No        |
| 5       | Tablet  | Yes       |

---

### **(i) Compute Prior Probabilities**

**Step 1: Count Outcomes**

Total observations (n) = 5

Subscribe = Yes: Visitors 2, 3, 5 → Count = 3

Subscribe = No: Visitors 1, 4 → Count = 2

---

**Step 2: Calculate Priors**

P(Subscribe = Yes) = n(Yes)/n(Total)
                   = 3/5
                   = **0.6**

P(Subscribe = No) = n(No)/n(Total)
                  = 2/5
                  = **0.4**

---

**Verification:** P(Yes) + P(No) = 0.6 + 0.4 = 1.0 ✓

---

### **(ii) Predict for New Mobile Visitor**

**Given:** New visitor uses Mobile device

**Task:** Predict Subscribe (Yes or No)

---

**Step 1: Calculate Likelihoods**

**For Subscribe = Yes:**

Mobile users who subscribed: Visitor 2 → Count = 1
Total Yes subscribers: 3

P(Mobile | Yes) = n(Mobile ∩ Yes)/n(Yes)
                = 1/3
                = **0.333**

---

**For Subscribe = No:**

Mobile users who didn't subscribe: Visitor 1 → Count = 1
Total No subscribers: 2

P(Mobile | No) = n(Mobile ∩ No)/n(No)
               = 1/2
               = **0.5**

---

**Step 2: Apply Naïve Bayes Formula**

**Posterior for Yes:**

P(Yes | Mobile) ∝ P(Yes) × P(Mobile | Yes)

Score(Yes) = (3/5) × (1/3)
           = 3/15
           = 1/5
           = **0.20**

---

**Posterior for No:**

P(No | Mobile) ∝ P(No) × P(Mobile | No)

Score(No) = (2/5) × (1/2)
          = 2/10
          = 1/5
          = **0.20**

---

**Step 3: Compare Posteriors**

Score(Yes) = 0.20
Score(No) = 0.20

Since: Score(Yes) = Score(No)

**Result:** The classifier produces a **TIE**

---

**Step 4: Tie-Breaking Rule**

When posteriors are equal, use majority class:

Majority class = Yes (appears 3 times vs No appears 2 times)

---

**Final Prediction:** Subscribe = **Yes**

**Note:** This is a tie-breaking decision. The model cannot distinguish between Yes and No based on the Mobile device feature alone with the given training data.

---

---

## **Question 3: Course Enrollment Probability**

### **Given Information:**

Total students (n) = 500

**Course Enrollments:**
- Data Science (D) = 230
- Cyber Security (C) = 180
- Business Analytics (B) = 150

**Intersections:**
- D ∩ C = 90
- D ∩ B = 70
- C ∩ B = 60
- D ∩ C ∩ B = 35

---

### **Venn Diagram Calculations**

**Step 1: Calculate Exclusive Regions**

**Only D (D only):**
= |D| - |D∩C| - |D∩B| + |D∩C∩B|
= 230 - 90 - 70 + 35
= **105**

**Only C (C only):**
= |C| - |D∩C| - |C∩B| + |D∩C∩B|
= 180 - 90 - 60 + 35
= **65**

**Only B (B only):**
= |B| - |D∩B| - |C∩B| + |D∩C∩B|
= 150 - 70 - 60 + 35
= **55**

---

**Step 2: Calculate Pairwise Exclusive Regions**

**D ∩ C only (not B):**
= |D∩C| - |D∩C∩B|
= 90 - 35
= **55**

**D ∩ B only (not C):**
= |D∩B| - |D∩C∩B|
= 70 - 35
= **35**

**C ∩ B only (not D):**
= |C∩B| - |D∩C∩B|
= 60 - 35
= **25**

---

**Step 3: All Three Courses**

**D ∩ C ∩ B = 35**

---

**Step 4: Calculate Students in No Course**

Total in at least one course:
= 105 + 65 + 55 + 55 + 35 + 25 + 35
= **375**

Students in no course:
= 500 - 375
= **125**

---

**Venn Diagram Summary:**

```
         D                C
    ┌─────────┐      ┌─────────┐
    │         │      │         │
    │   105   │  55  │   65    │
    │         │      │         │
    │    ┌────┴──────┴────┐    │
    │    │       35        │    │
    └────┤                 ├────┘
         │   35       25   │
         │                 │
         └────────┬────────┘
                  │   55    │
                  │    B    │
                  └─────────┘

Outside all: 125
```

---

### **(1) Probability of At Least One Course**

**Method: Inclusion-Exclusion Principle**

|D ∪ C ∪ B| = |D| + |C| + |B| - |D∩C| - |D∩B| - |C∩B| + |D∩C∩B|

|D ∪ C ∪ B| = 230 + 180 + 150 - 90 - 70 - 60 + 35

|D ∪ C ∪ B| = 560 - 220 + 35

|D ∪ C ∪ B| = **375**

---

**Probability:**

P(at least one) = |D ∪ C ∪ B|/n

P(at least one) = 375/500

P(at least one) = 3/4

P(at least one) = **0.75 or 75%**

---

### **(2) Probability of Exactly Two Courses**

**Students in exactly two courses:**

= (D∩C only) + (D∩B only) + (C∩B only)

= 55 + 35 + 25

= **115**

---

**Probability:**

P(exactly two) = 115/500

P(exactly two) = 23/100

P(exactly two) = **0.23 or 23%**

---

### **(3) Conditional Probability P(C | D)**

**Formula:**

P(C | D) = P(C ∩ D)/P(D)

---

**Step 1: Calculate P(C ∩ D)**

P(C ∩ D) = |D ∩ C|/n = 90/500 = 0.18

---

**Step 2: Calculate P(D)**

P(D) = |D|/n = 230/500 = 0.46

---

**Step 3: Apply Formula**

P(C | D) = 0.18/0.46

P(C | D) = 18/46

P(C | D) = 9/23

P(C | D) ≈ **0.3913 or 39.13%**

---

### **(4) Are D and C Independent? (Mathematical Justification)**

**Definition:** Events D and C are independent if and only if:

**P(D ∩ C) = P(D) × P(C)**

---

**Step 1: Calculate P(D)**

P(D) = 230/500 = 23/50 = **0.46**

---

**Step 2: Calculate P(C)**

P(C) = 180/500 = 9/25 = **0.36**

---

**Step 3: Calculate P(D) × P(C)**

P(D) × P(C) = 0.46 × 0.36

P(D) × P(C) = (23/50) × (9/25)

P(D) × P(C) = 207/1250

P(D) × P(C) = **0.1656**

---

**Step 4: Calculate P(D ∩ C)**

P(D ∩ C) = 90/500 = 9/50 = **0.18**

---

**Step 5: Test Independence**

Compare: P(D ∩ C) vs P(D) × P(C)

0.18 ≠ 0.1656

Since: P(D ∩ C) ≠ P(D) × P(C)

---

**Conclusion:** Events D and C are **NOT INDEPENDENT**

---

**Additional Analysis:**

Difference = P(D ∩ C) - P(D) × P(C)
          = 0.18 - 0.1656
          = 0.0144

Percentage difference = (0.0144/0.1656) × 100 ≈ 8.7%

The observed joint probability is 8.7% higher than expected under independence, indicating a **positive association** between D and C.

---

### **(5) Interpretation of P(C | D)**

**Result:** P(C | D) = 0.3913 ≈ 39.13%

---

**Interpretation:**

1. **Conditional Enrollment:**
   - Among students enrolled in Data Science (D), approximately **39.13%** are also enrolled in Cyber Security (C)
   - This means about **2 out of every 5** Data Science students also take Cyber Security

2. **Comparison with Overall Enrollment:**
   - Overall enrollment in C: P(C) = 36%
   - Conditional enrollment: P(C | D) = 39.13%
   - Difference: 39.13% - 36% = 3.13%

3. **Positive Association:**
   - Since P(C | D) > P(C), there is a **positive association**
   - Students enrolled in Data Science are **more likely** to enroll in Cyber Security than the general student population
   - This suggests complementary skills or overlapping interests

4. **Practical Implications:**
   - The university could offer combined D-C tracks or joint projects
   - Marketing C to D students may be more effective
   - Course scheduling should accommodate students taking both
   - Faculty could develop integrated curriculum materials

5. **Statistical Significance:**
   - The 3.13 percentage point increase represents an 8.7% relative increase
   - This is consistent with the non-independence finding in part (4)
   - The overlap (90 students) is statistically meaningful given the sample size

---

**Summary Statement:**

The conditional probability P(C | D) = 39.13% reveals that Data Science students show a higher propensity for Cyber Security enrollment compared to the general population (36%). This positive association suggests that students interested in data-driven fields also recognize the importance of security skills, indicating natural synergies between these domains that the university could leverage in program design and student advising.

---

## **END OF SOLUTIONS**