# Error analysis — LoRA 1.5B r16 on the test split

Source: `results/predictions/ft_1p5b_r16_test.jsonl` (206 rows, frozen model, greedy decoding).
All candidates with full note sentences: `results/error_candidates_ft_1p5b_r16_test.md`
(`python scripts/error_analysis.py --run ft_1p5b_r16 --split test`).

## Overview

61 of 206 labels are wrong (accuracy 0.704). Output validity is perfect: 0 parse failures, 0 invalid
labels, 0 out-of-range sentence IDs. Label confusions (criterion type, gold → predicted):

| confusion | rows |
|---|---|
| exclusion: not enough information → not excluded | 28 |
| exclusion: not applicable → not excluded | 12 |
| inclusion: not included → not enough information | 6 |
| inclusion: included → not enough information | 6 |
| inclusion: not enough information → included | 3 |
| inclusion: not applicable → not enough information | 3 |
| inclusion: not included → included | 1 |
| exclusion: not excluded → excluded | 1 |
| inclusion: not enough information → not included | 1 |

Two thirds of the errors (40/61) are one pattern: an exclusion criterion that the experts marked
"not enough information" or "not applicable" is predicted "not excluded". The model never predicts
"not applicable" on test and predicts "not included" once. This follows the training distribution:
"not excluded" is 75% of the training exclusion rows (291/387), "not applicable" 4.7% of all
training rows (28/602) and "not included" 2.2% (13/602).

Besides the label errors, 33 rows have the right label but different evidence from the experts.

## Analysed errors

**1. Missing-information confusion: "not mentioned" read as "ruled out"** (annotation 169, patient
sigir-201410, exclusion).
Criterion: *estimated pre-operative GFR < 30 ml/min/1.73m²*. Gold: not enough information [].
Predicted: not excluded [].
The note (a 67-year-old woman after cardiac catheterisation) gives no renal function. The experts
say the record cannot decide the criterion; the model says the patient is not excluded. This is
the largest error group (28 rows). It matters for screening: "not excluded" passes the criterion,
while "not enough information" asks for more data. In the app's aggregation, the first can turn a
"Needs more information" outcome into "Potentially eligible".

**2. Missing-information confusion, contestable gold** (annotation 193, sigir-201411, exclusion).
Criterion: *Patients were moribund*. Gold: not enough information, citing all five sentences.
Predicted: not excluded [].
The patient is hypotensive (80/60), tachycardic and tachypnoeic but conscious and in "moderate
discomfort". The experts cite the whole note as relevant but undecided. The model treats the absence
of a statement that she is dying as a negative. The label is debatable, but citing no evidence is
clearly wrong: every sentence bears on the question.

**3. Inclusion "not included" hedged to "not enough information"** (annotations 161–163,
sigir-201410).
Criteria: *Patients undergoing elective carotid endarterectomy / open AAA repair / endovascular
AAA repair*. Gold: not included []. Predicted: not enough information [0].
Sentence 0 says she underwent cardiac catheterisation via the femoral artery. The experts conclude
from the documented procedure that she is not undergoing the listed ones; the model will not infer
the negative. This is the inclusion-side counterpart of error 1, in the opposite direction: the
model is cautious on inclusion criteria and permissive on exclusion criteria. "Not included" was
never predicted correctly on test (F1 0.0, 7 rows).

**4. Wrong label with the right evidence: qualitative term vs numeric threshold** (annotations
186–187, sigir-201411, inclusion).
Criteria: *Heart rate of at least 90/min*; *respiratory rate of at least 20/min …*. Gold: included
[2]. Predicted: not enough information [2].
The model cites exactly the right sentence ("…tachypneic and tachycardic") but does not map
"tachycardic" to HR > 100 ≥ 90 or "tachypneic" to RR > 20. Retrieval is right; the clinical
inference from a qualitative finding to a numeric threshold is missing. Annotation 766 (*Fever*,
gold included [0]; the sentence lists "fever") is the same error without any threshold.

**5. Wrong label with the right evidence: over-reading a partial match** (annotation 678,
sigir-20158, inclusion).
Criterion: mild sleep-disordered breathing, defined as habitual snoring on ≥ 3 nights/week for ≥ 3
months. Gold: not enough information [0]. Predicted: included [0].
Sentence 0 reports nighttime snoring and pauses, but no frequency or duration. The model matches the
topic and ignores the quantitative parts of the definition. Annotation 762 (sore throat "together
with at least 2 of the following clinical signs", with the signs list cut off) is the same pattern.

**6. "Not applicable" vs "not excluded"** (annotations 165, 750, 751; patients sigir-201410,
sigir-201513, exclusion).
Criteria: *Pregnancy* for a 67-year-old woman; *pregnancy*, *breast-feeding* for a 5-year-old boy.
Gold: not applicable []. Predicted: not excluded [].
For the experts a criterion that cannot apply to the patient's age or sex is "not applicable". The
model gives the practically equivalent "not excluded": the outcome is the same, but the label is
wrong. All 15 "not applicable" test rows are missed (12 → not excluded, 3 → not enough
information). The training labels are not consistent either: of the 23 training exclusion
criteria about pregnancy or breast-feeding, 12 are "not applicable", 6 "not enough information"
and 5 "not excluded".

**7. Exclusion polarity flip: wrong body region** (annotation 166, sigir-201410, exclusion).
Criterion: *Significant upper limb peripheral arterial disease*. Gold: not excluded [].
Predicted: **excluded** [1, 2].
The cited sentences describe a cool right *foot* and loss of distal pulses after femoral access:
lower-limb arterial compromise. The model matches "peripheral arterial disease" and ignores "upper
limb". This is the only "excluded" prediction on test, and the most harmful error type for a
screener: it would mark an eligible patient "Likely ineligible". Because the test set has no gold
"excluded" rows, this single prediction adds an "excluded" class with F1 = 0 to the overall
macro-F1 (0.369 over six labels; 0.443 over the five labels with test support).

**8. Inclusion polarity flip: anticipating a future procedure** (annotation 164, sigir-201410,
inclusion).
Criterion: *Patients undergoing surgical lower limb revascularisation*. Gold: not included [].
Predicted: included [0].
The note describes a femoral pseudoaneurysm/ischaemia after catheterisation. A clinician might
expect revascularisation, but the note documents only catheterisation. The model cites the
procedure sentence as if it satisfied the criterion.

**9. Right label, wrong evidence: boiler-plate sentence cited** (annotation 381, sigir-201421,
inclusion).
Criterion: *Age > 18 years*. Gold: included [0]. Predicted: included [4].
The label is right, but the model cites the generic sentence "The patient will provide informed
consent, and will comply with the trial protocol…" (present in several notes) instead of sentence 0
with the age ("A 21-year-old female…"). Annotations 310 and 311 cite irrelevant sentences for
"uncontrolled hypertension" / "NYHA IV". The explanation shown in the app would quote an
unsupportive sentence, which a reviewer can spot because the evidence is shown verbatim.

**10. Right label, missing evidence** (annotations 196, 197, 322; sigir-201411, sigir-201417,
exclusion).
Criteria: *Less than 18 years of age*, *Decreased level of consciousness*, *Pregnant women…* (for a
male patient). Gold: not excluded with the age/sex sentence. Predicted: not excluded [].
The model learned that "not excluded" usually comes with no evidence: 217 of the 291 training
"not excluded" rows (75%) have empty gold evidence. When the decision does rest on a stated fact
(age, sex), it still often omits the citation. This is the main source of evidence false
negatives: evidence recall is 0.61 against a precision of 0.85 on rows with non-empty gold.

**11. Contradictory / ambiguous annotation** (annotation 176, sigir-201410, inclusion).
Criterion: *No medical history of central venous catheterization*. Gold: included [0]. Predicted:
not enough information [0].
Both cite sentence 0 (cardiac catheterisation via the femoral *artery*). The experts read the
arterial procedure as evidence of no central-venous history; the model's reading (an absent history
is not documented) is equally defensible. Annotation 422 (*If female, using birth control*: gold
not applicable, model not enough information, same evidence) is a similar case where the gold
labels a criterion "not applicable" although it applies to a female patient. Some of the 61 errors
are therefore disagreements with debatable gold labels rather than model failures.

## Categories with no occurrences on test

* **Unsupported citations:** 0 rows cite a sentence ID outside the note (`records_with_unsupported_ids
  = 0`), in both the base and fine-tuned runs.
* **Invalid output / labels:** 0 parse failures, 0 labels outside the criterion type's label set.
* **Truncation:** none. The longest test prompt is 525 tokens (621 over all splits) against a
  1,536-token budget, so no sentence or gold evidence ID is cut (`results/token_stats.json`).

## Takeaways

1. The dominant failure is calibration between "not excluded" and "not enough information" on
   exclusion criteria (28 rows). For screening it is the riskiest because it turns "unknown" into
   "pass". Class re-weighting or oversampling the minority labels would be the first fix.
2. Rare labels ("not included", "not applicable", "excluded") are effectively not learned from
   13–28 training examples each. That explains most of the val/test macro-F1 gap, since test has
   different minority-class counts (no "excluded", 7 "not included", 15 "not applicable").
3. Evidence errors are mostly omissions (recall 0.61, precision 0.85 on rows with non-empty gold),
   not hallucinations: there are no out-of-range IDs, and wrong citations are rare (errors 9, 7).
