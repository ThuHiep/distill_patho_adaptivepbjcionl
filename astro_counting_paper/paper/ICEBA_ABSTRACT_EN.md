# ICEBA2026 — Abstract Submission (English)

**Journal choice:** (1) The Proceedings of ICEBA2026 — Springer Nature Book Series (Scopus-indexed)
**Session / subject area:** *[choose 1 of 8 — prefer Data Analytics / Applied Statistics / Computational Science]*

---

## Title

**Online Conformal Calibration for Astronomical Source Counting with Guaranteed Coverage under
Survey-Depth Shift**

## Authors

*[Full name¹, Full name², … — Affiliation(s)]*
**Corresponding author:** *[name, email]*

## Abstract (~200 words)

Many scientific tasks require counting objects by class together with trustworthy prediction intervals,
not a bare number. When data-acquisition conditions change (instrument, depth, environment), the data
distribution shifts and statically calibrated intervals silently lose coverage. We apply **Adaptive
PB-JCI Online**—a lightweight online conformal calibration layer that preserves joint coverage for
multi-class count vectors under shift—to a scientific domain far from its original one (histopathology
nucleus counting): counting **stars and galaxies** in sky-survey images. Using **61,423 real sources**
from the Sloan Digital Sky Survey (SDSS DR17), with the official SDSS photometric pipeline as the
backbone and a controlled survey-depth shift, Adaptive PB-JCI is the **only** method that attains valid
joint coverage (≥90%) at the **best interval (Winkler) score**, and holds ~91% across all shift
magnitudes while static calibration collapses from 85% to 9%. Against a full suite of modern
online-conformal baselines (ACI, NexCP, FACI, SAOCP, COP, Rolling-Origin), it uniquely combines validity
and tightness: over-covering methods inflate the interval score ~10×, while tighter methods under-cover.
The results demonstrate cross-domain generality of the method—from life science to astrophysics—without
changing a single formula.

## Keywords (5)

conformal prediction; online calibration; count uncertainty quantification; distribution shift;
astronomical source counting

---

*Word count of abstract ≈ 190 (fits ~200 words / 8–10 lines / A4). English required for journal (1).*
