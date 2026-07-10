# Cover Letter — Journal of Combinatorial Optimization (CSoNet 2026 Journal Track)

**Prof. My T. Thai**
Editor-in-Chief, *Journal of Combinatorial Optimization*

**Date:** July 1, 2026

Dear Prof. My T. Thai,

**This manuscript is submitted to the journal track of CSoNet 2026.**

We are pleased to submit the manuscript entitled **"Adaptive Poisson–Binomial Joint
Conformal Intervals for Reliable Cell Counting under Distribution Shift"** for
consideration of publication by the prestigious *Journal of Combinatorial
Optimization*.

This manuscript proposes a comprehensive framework that advances reliable,
uncertainty-aware multi-class cell counting in histopathology images under
distribution shift, through improvements at the uncertainty-scoring,
joint-calibration, and online-adaptation levels. The manuscript offers several key
contributions:

An analytic uncertainty-normalization scale derived from the Poisson–Binomial
variance of the per-instance detection confidences, adaptively distributing
prediction-interval widths according to the difficulty and ambiguity of each image.

A max-statistic joint calibration that constructs prediction intervals with
simultaneous coverage across all cell types, directly calibrating the joint coverage
event rather than relying on a conservative per-class (Bonferroni) correction.

An adaptive online calibration mechanism whose window size is driven by recently
observed coverage, recovering target coverage under distribution shift while keeping
intervals substantially narrower than adaptive conformal inference.

A lightweight, backbone-agnostic calibration layer that equips existing frozen
cell-counting backbones with reliable prediction intervals, without retraining or
modifying the segmentation architecture.

Extensive experiments are conducted on three public histopathology datasets (PanNuke,
NuInsSeg, MoNuSAC) and two backbones (SAM 3, PathoSAM); in a cross-dataset shift where
static calibration retains only about 41% coverage, the adaptive mechanism restores
coverage to 90.0 ± 0.6% and attains the lowest interval (Winkler) score among recent
online conformal baselines.

We confirm that this manuscript is not under consideration by any other journal. We
know of no conflicts of interest associated with this publication, and there has
been no significant financial support for this work that could have influenced its
outcome. As Corresponding Author, I confirm that the manuscript has been read and
approved for submission by all the named authors.

Thank you for your consideration of our manuscript. I look forward to hearing from
you.

Sincerely,

Dr. Viet Hang Duong
Computer Science Faculty,
The University of Information Technology, VietNam National University HCM — http://en.uit.edu.vn/
Email: hangdv@uit.edu.vn; Tel: +884-0919.196.708
ORCID: 0000-0002-9728-4438
