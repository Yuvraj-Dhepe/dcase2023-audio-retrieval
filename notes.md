### Evaluation

In the formula for **Average Precision (AP)**, the index $i$ goes from 1 to $N$, where $N$ is the number of **relevant items**. However, the positions $ P()$ refer to the positions of the **relevant items** in the ranked list, not just every position in the ranked list. Let me clarify this in more detail.

---

### Average Precision (AP) Formula

Links:
https://medium.com/@neri.vvo/mean-average-precision-made-simple-complete-guide-with-example-5f13331cce14
https://sdsawtelle.github.io/blog/output/mean-average-precision-MAP-for-recommender-systems.html

$$
\text{AP} = \frac{1}{N} \sum_{k=1}^{N} \text{Precision@k}
$$

- $N$ is the total number of relevant items in the entire dataset (e.g., if there are 5 relevant items, then $ N =5$).
- $\text{Precision@k}$ is the precision calculated **at the position of the k-th relevant item**.

---

### Why Only Relevant Items Are Considered in the Sum

In the sum, we are calculating precision only at the positions where relevant items appear. This is because **AP** focuses on how well the system retrieves relevant items and how early they appear in the ranked list.

Let’s explain with a quick example.

---

### Example

Let’s say you have 10 ranked items with the following relevance (targets):

```
Ranked list positions:   1   2   3   4   5   6   7   8   9   10
Relevance (targets):     0   1   0   0   1   0   1   0   1   0
```

In this case, there are 4 relevant items (where `target == 1`), so $ N =4$. The relevant items are located at positions 2, 5, 7, and 9.

---

#### Calculating Precision at Relevant Positions

1. **At position 2** (the first relevant item):

   - The number of relevant items retrieved so far: 1.
   - Total retrieved items: 2.
   - Precision = $\frac{1}{2}$

2. **At position 5** (the second relevant item):

   - The number of relevant items retrieved so far: 2.
   - Total retrieved items: 5.
   - Precision = $\frac{2}{5}$

3. **At position 7** (the third relevant item):

   - The number of relevant items retrieved so far: 3.
   - Total retrieved items: 7.
   - Precision = $\frac{3}{7}$

4. **At position 9** (the fourth relevant item):
   - The number of relevant items retrieved so far: 4.
   - Total retrieved items: 9.
   - Precision = $\frac{4}{9}$

---

#### Computing Average Precision (AP)

Now, we sum up the precision values at the relevant positions:

$$\text{AP} = \frac{1}{N} \sum_{k=1}^{N} \text{Precision@k}$$

$$
\text{AP} = \frac{1}{4} \left( \frac{1}{2} + \frac{2}{5} + \frac{3}{7} + \frac{4}{9} \right)
$$

$$
\text{AP} = \frac{1}{4} \left( 0.5 + 0.4 + 0.4286 + 0.4444 \right)
$$

$$
\text{AP} = \frac{1}{4} \times 1.773
$$

$$
\text{AP} = 0.443
$$

---

### Why Do We Only Use Relevant Items?

The reason we only consider **relevant items** when calculating Average Precision is that AP is designed to measure how well a system ranks **relevant items** and how early they are retrieved in the list. The positions of irrelevant items don’t contribute to the score directly; instead, they affect the denominator in the precision calculations.

If we included all positions (relevant and irrelevant) in the sum, the metric would lose its focus on relevance and give an inaccurate representation of the ranking system's performance.

---

### Key Points

- The sum in the AP formula runs from $ i =1$ to $N$ (where $N$ is the number of relevant items).
- Precision is calculated **at the positions where relevant items appear** in the ranked list.
- Irrelevant items are not included in the precision calculation but do influence precision because they increase the denominator (i.e., the number of retrieved items).

This approach ensures that **AP** reflects the system's ability to retrieve relevant items early in the ranked list.
