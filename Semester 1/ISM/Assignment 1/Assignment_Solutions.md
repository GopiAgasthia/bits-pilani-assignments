# Assignment Solutions

## Question 1: Defective Items Analysis

### Data
12, 15, 14, 13, 16, 18, 14, 15, 13, 14, 17, 14, 36

### (a) Mean, Median, Mode

Number of observations, n = 13

Sum of observations:
= 12 + 15 + 14 + 13 + 16 + 18 + 14 + 15 + 13 + 14 + 17 + 14 + 36
= 211

Mean:
= 211 / 13
= 16.23

Sorted data:
12, 13, 13, 14, 14, 14, 14, 15, 15, 16, 17, 18, 36

Median:
= 7th observation
= 14

Mode:
= 14

Conclusion:
Mean (16.23) > Median (14) = Mode (14)

Therefore, the distribution is positively (right) skewed.

---

### (b) Five-Point Summary and Outliers

Sorted data:
12, 13, 13, 14, 14, 14, 14, 15, 15, 16, 17, 18, 36

Five-point summary:

- Minimum = 12
- Q1 = 13.5
- Median = 14
- Q3 = 15.5
- Maximum = 36

IQR:
= Q3 - Q1
= 15.5 - 13.5
= 2

Mild outlier limits:

Lower Fence:
= Q1 - 1.5(IQR)
= 13.5 - 3
= 10.5

Upper Fence:
= Q3 + 1.5(IQR)
= 15.5 + 3
= 18.5

Since 36 > 18.5, 36 is an outlier.

Extreme outlier limits:

Lower Extreme Fence:
= Q1 - 3(IQR)
= 13.5 - 6
= 7.5

Upper Extreme Fence:
= Q3 + 3(IQR)
= 15.5 + 6
= 21.5

Since 36 > 21.5, 36 is an EXTREME outlier.

---

### (c) Mean or Median?

Mean = 16.23

Median = 14

The value 36 is an extreme outlier and inflates the mean considerably.

Median is resistant to outliers because it depends only on the position of observations.

Therefore, the company should report the MEDIAN (14) because it better represents the typical monthly number of defects.

---

# Question 2: Naïve Bayes Classification

Training Data:

| Visitor | Device | Subscribe |
|----------|---------|------------|
| 1 | Mobile | No |
| 2 | Mobile | Yes |
| 3 | Desktop | Yes |
| 4 | Desktop | No |
| 5 | Tablet | Yes |

### (i) Prior Probabilities

Number of Yes outcomes = 3

Number of No outcomes = 2

Total observations = 5

P(Subscribe = Yes)
= 3/5
= 0.6

P(Subscribe = No)
= 2/5
= 0.4

---

### (ii) Prediction for a Mobile Visitor

Likelihoods:

P(Mobile | Yes)
= 1/3

P(Mobile | No)
= 1/2

Naïve Bayes scores:

For Yes:

P(Yes) × P(Mobile | Yes)
= (3/5)(1/3)
= 1/5
= 0.20

For No:

P(No) × P(Mobile | No)
= (2/5)(1/2)
= 1/5
= 0.20

Result:

Both posterior scores are equal.

P(Yes | Mobile) = P(No | Mobile)

Therefore, the classifier cannot uniquely distinguish between Yes and No using the given data.

A practical implementation would either:
1. Report a tie, or
2. Choose the majority class (Yes).

Predicted class: Subscribe = Yes (using majority-class tie break).

---

# Question 3: Course Enrollment Probability

Given:

Total students = 500

| Set | Count |
|------|-------|
| D | 230 |
| C | 180 |
| B | 150 |
| D ∩ C | 90 |
| D ∩ B | 70 |
| C ∩ B | 60 |
| D ∩ C ∩ B | 35 |

## Venn Diagram Region Values

Only D:
= 230 − 90 − 70 + 35
= 105

Only C:
= 180 − 90 − 60 + 35
= 65

Only B:
= 150 − 70 − 60 + 35
= 55

D ∩ C only:
= 90 − 35
= 55

D ∩ B only:
= 70 − 35
= 35

C ∩ B only:
= 60 − 35
= 25

All three:
= 35

Outside all sets:
= 500 − 375
= 125

Textual Venn representation:

- Only D = 105
- Only C = 65
- Only B = 55
- D∩C only = 55
- D∩B only = 35
- C∩B only = 25
- D∩C∩B = 35
- None = 125

---

### (1) Probability of At Least One Course

Using Inclusion-Exclusion:

n(D ∪ C ∪ B)

= 230 + 180 + 150 − 90 − 70 − 60 + 35

= 375

P(at least one)

= 375 / 500

= 0.75

---

### (2) Probability of Exactly Two Courses

Exactly two:

(D∩C only) + (D∩B only) + (C∩B only)

= 55 + 35 + 25

= 115

P(exactly two)

= 115 / 500

= 0.23

---

### (3) Conditional Probability

P(C | D)

= P(C ∩ D) / P(D)

= 90 / 230

= 0.3913

---

### (4) Are D and C Independent?

P(D)

= 230/500
= 0.46

P(C)

= 180/500
= 0.36

P(D)P(C)

= 0.46 × 0.36

= 0.1656

P(D ∩ C)

= 90/500

= 0.18

Since:

P(D ∩ C) ≠ P(D)P(C)

0.18 ≠ 0.1656

Therefore, D and C are NOT independent.

---

### (5) Interpretation

P(C | D) = 0.3913

Among students enrolled in Data Science, approximately 39.13% are also enrolled in Cyber Security.

Since D and C are not independent, enrollment in one course influences the likelihood of enrollment in the other. The observed overlap is slightly higher than what would be expected under independence, suggesting a positive association between student interest in Data Science and Cyber Security.
