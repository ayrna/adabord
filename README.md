# 🌳 Welcome to `ADABORD`

This repository provides the official implementation of `ADABORD`, a novel AdaBoost-based ensemble method designed for **ordinal classification** tasks. This repository contains the code accompanying the scientific article:

> **AdaBORD: Adaptive Boosting for Ordinal Classification with Decision Trees**

* 🚀 Built on top of **scikit-learn**, extending it with ordinal-aware boosting strategies.
* 🚀 Introduces a novel **AdaBoost formulation** tailored to exploit the natural order of class labels.
* 🚀 Designed for **reproducibility** — all experiments from the article can be replicated with the provided code.

✨ `ADABORD` is a **top-performing method** on the [TOCUCO](https://github.com/ayrna/tocuco) benchmark, the standard repository for ordinal classification comparison.

---

### Custom scikit-learn dependency for decision tree classifier with ordinal gini criterion.

`ADABORD` relies on a modified version of scikit-learn that introduces ordinal splitting criteria. Install it directly from the source branch:

```bash
pip install git+ssh://git@github.com/RafaAyGar/scikit-learn.git@feature_ogini_split
```

Or with `uv`:

```bash
uv pip install "scikit-learn @ git+ssh://git@github.com/RafaAyGar/scikit-learn.git@feature_ogini_split"
```

---

# 📚 Citation

If you use `ADABORD` in your research, we would appreciate a citation for the following work:

*citation not available yet*
