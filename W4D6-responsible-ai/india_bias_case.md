# India AI Bias Case: Facial Recognition Technology

A published study discussed by Basheer (2025) examined gender
bias in commercial facial processing tools tested on Indian faces.
The underlying audit, conducted by Jain and Parsheera, evaluated
32,184 images of electoral candidates published by the Election
Commission of India between May 2019 and November 2020. Four tools
were examined, including FaceX, developed by an Indian startup.

The reported results showed higher gender-classification error
rates for female subjects. Face++ recorded an error rate of 14.68%
for female subjects, while FaceX recorded 10.47%. The corresponding
rates for male subjects were substantially lower. These findings
illustrate why overall accuracy can hide differences between
demographic groups.

Possible root causes include limitations in training data,
algorithm design, image quality, and insufficient representation
of diverse facial characteristics. The results also show that
using locally collected images alone does not guarantee fairness.
The audit used binary gender categories, so it could not assess
performance for people outside those categories.

Proposed fixes include evaluating models on representative Indian
datasets, publishing subgroup-specific false-positive and
false-negative rates, and conducting independent audits before
deployment. Developers should also document limitations, protect
sensitive biometric information, and establish human review and
appeal procedures when system outputs affect people. High-stakes
use should be restricted unless reliability and safeguards are
demonstrated.

This case shows that responsible AI requires more than high
overall accuracy. Fairness testing must reflect the people affected
by the system and examine who experiences errors.

Source: Basheer, I. P. (2025). Bias in the Algorithm: Issues
Raised Due to Use of Facial Recognition in India. Journal of
Development Policy and Practice, 10(1), 61-79.
https://doi.org/10.1177/24551333241283992
