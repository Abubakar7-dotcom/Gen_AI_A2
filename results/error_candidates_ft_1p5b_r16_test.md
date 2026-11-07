# Error candidates: ft_1p5b_r16 on test (206 records, 61 wrong labels)

| category | records |
|---|---|
| missing_info_confusion | 44 |
| right_label_wrong_evidence | 33 |
| not_applicable | 15 |
| wrong_label_right_evidence | 8 |
| polarity_flip | 2 |

## Label confusions (type, gold -> predicted)

- exclusion: not enough information -> not excluded: 28
- exclusion: not applicable -> not excluded: 12
- inclusion: not included -> not enough information: 6
- inclusion: included -> not enough information: 6
- inclusion: not enough information -> included: 3
- inclusion: not applicable -> not enough information: 3
- inclusion: not included -> included: 1
- exclusion: not excluded -> excluded: 1
- inclusion: not enough information -> not included: 1

## missing_info_confusion (44)

### annotation 161 — patient sigir-201410, inclusion
- criterion: Patients undergoing elective carotid endarterectomy
- gold: **not included** [] | predicted: **not enough information** [0]
- raw output: `{"label": "not enough information", "evidence": [0]}`
  - [0] (cited) A physician is called to see a 67-year-old woman who underwent cardiac catheterization via the right femoral artery earlier in the morning.

### annotation 162 — patient sigir-201410, inclusion
- criterion: Patients undergoing open abdominal aortic aneurysm repair
- gold: **not included** [] | predicted: **not enough information** [0]
- raw output: `{"label": "not enough information", "evidence": [0]}`
  - [0] (cited) A physician is called to see a 67-year-old woman who underwent cardiac catheterization via the right femoral artery earlier in the morning.

### annotation 163 — patient sigir-201410, inclusion
- criterion: Patients undergoing endovascular abdominal aneurysm repair
- gold: **not included** [] | predicted: **not enough information** [0]
- raw output: `{"label": "not enough information", "evidence": [0]}`
  - [0] (cited) A physician is called to see a 67-year-old woman who underwent cardiac catheterization via the right femoral artery earlier in the morning.

### annotation 169 — patient sigir-201410, exclusion
- criterion: Patients with an estimated pre-operative glomerular filtration rate < 30mls/min/1.73m2
- gold: **not enough information** [] | predicted: **not excluded** []
- raw output: `{"label": "not excluded", "evidence": []}`

### annotation 174 — patient sigir-201410, exclusion
- criterion: Patients with previous axillary surgery
- gold: **not enough information** [] | predicted: **not excluded** []
- raw output: `{"label": "not excluded", "evidence": []}`

### annotation 176 — patient sigir-201410, inclusion
- criterion: No medical history of central vena catheterization.
- gold: **included** [0] | predicted: **not enough information** [0]
- raw output: `{"label": "not enough information", "evidence": [0]}`
  - [0] (gold) (cited) A physician is called to see a 67-year-old woman who underwent cardiac catheterization via the right femoral artery earlier in the morning.

### annotation 177 — patient sigir-201410, inclusion
- criterion: Maintenance hemodialysis after central vena catheterization.
- gold: **not included** [] | predicted: **not enough information** []
- raw output: `{"label": "not enough information", "evidence": []}`

### annotation 185 — patient sigir-201411, inclusion
- criterion: Patients were included if they were diagnosed with Shock, Acute respiratory distress syndrome (ARDS), or both.
- gold: **included** [2, 3] | predicted: **not enough information** [1, 2, 3, 4]
- raw output: `{"label": "not enough information", "evidence": [1, 2, 3, 4]}`
  - [1] (cited) She denies trauma.
  - [2] (gold) (cited) On examination she is pale and in moderate discomfort, as well as tachypneic and tachycardic.
  - [3] (gold) (cited) Her body temperature is normal and her blood pressure is 80/60.
  - [4] (cited) Her right arm has no discoloration or movement limitation.

### annotation 186 — patient sigir-201411, inclusion
- criterion: Heart rate of at least 90/min;
- gold: **included** [2] | predicted: **not enough information** [2]
- raw output: `{"label": "not enough information", "evidence": [2]}`
  - [2] (gold) (cited) On examination she is pale and in moderate discomfort, as well as tachypneic and tachycardic.

### annotation 187 — patient sigir-201411, inclusion
- criterion: A respiratory rate of at least 20/min or a PaCO2 of 32mmHg or lower or the use of mechanical ventilation;
- gold: **included** [2] | predicted: **not enough information** [2]
- raw output: `{"label": "not enough information", "evidence": [2]}`
  - [2] (gold) (cited) On examination she is pale and in moderate discomfort, as well as tachypneic and tachycardic.

### annotation 188 — patient sigir-201411, inclusion
- criterion: The use of vasopressors to maintain a systolic blood pressure of at least 90mmHg despite fluid resuscitation, low dose of dopamine (≤ 5 μg/kg per minute), or dobutamine;
- gold: **not included** [3] | predicted: **not enough information** []
- raw output: `{"label": "not enough information", "evidence": []}`
  - [3] (gold) Her body temperature is normal and her blood pressure is 80/60.

### annotation 193 — patient sigir-201411, exclusion
- criterion: Patients were moribund.
- gold: **not enough information** [0, 1, 2, 3, 4] | predicted: **not excluded** []
- raw output: `{"label": "not excluded", "evidence": []}`
  - [0] (gold) A 40-year-old woman with no past medical history presents to the ER with excruciating pain in her right arm that had started 1 hour prior to her admission.
  - [1] (gold) She denies trauma.
  - [2] (gold) On examination she is pale and in moderate discomfort, as well as tachypneic and tachycardic.
  - [3] (gold) Her body temperature is normal and her blood pressure is 80/60.
  - [4] (gold) Her right arm has no discoloration or movement limitation.

## not_applicable (15)

### annotation 165 — patient sigir-201410, exclusion
- criterion: Pregnancy
- gold: **not applicable** [] | predicted: **not excluded** []
- raw output: `{"label": "not excluded", "evidence": []}`

### annotation 422 — patient sigir-201422, inclusion
- criterion: If female, using birth control
- gold: **not applicable** [3] | predicted: **not enough information** [3]
- raw output: `{"label": "not enough information", "evidence": [3]}`
  - [3] (gold) (cited) She has had no sexual partners and her menses are regular.

### annotation 683 — patient sigir-20158, exclusion
- criterion: Previous tonsillectomy, including partial tonsillectomy
- gold: **not applicable** [] | predicted: **not excluded** []
- raw output: `{"label": "not excluded", "evidence": []}`

### annotation 693 — patient sigir-20158, exclusion
- criterion: Suspected diagnosis of a sleep disorder other than OSA (i.e., periodic leg movements, narcolepsy, insomnia, central sleep apnea, sleep hypoventilation syndrome)
- gold: **not applicable** [] | predicted: **not excluded** []
- raw output: `{"label": "not excluded", "evidence": []}`

### annotation 697 — patient sigir-20158, exclusion
- criterion: Pregnant
- gold: **not applicable** [] | predicted: **not excluded** []
- raw output: `{"label": "not excluded", "evidence": []}`

### annotation 745 — patient sigir-201513, inclusion
- criterion: female patients have adequate birth-control method
- gold: **not applicable** [] | predicted: **not enough information** []
- raw output: `{"label": "not enough information", "evidence": []}`

### annotation 750 — patient sigir-201513, exclusion
- criterion: pregnancy
- gold: **not applicable** [] | predicted: **not excluded** []
- raw output: `{"label": "not excluded", "evidence": []}`

### annotation 751 — patient sigir-201513, exclusion
- criterion: breast-feeding
- gold: **not applicable** [] | predicted: **not excluded** []
- raw output: `{"label": "not excluded", "evidence": []}`

### annotation 754 — patient sigir-201513, exclusion
- criterion: alcoholism (drunk at least once a week)
- gold: **not applicable** [] | predicted: **not excluded** []
- raw output: `{"label": "not excluded", "evidence": []}`

### annotation 758 — patient sigir-201513, exclusion
- criterion: army recruit
- gold: **not applicable** [] | predicted: **not excluded** []
- raw output: `{"label": "not excluded", "evidence": []}`

### annotation 767 — patient sigir-201513, exclusion
- criterion: Symptoms that collectively suggest nonstreptococcal T/P (eg, laryngitis, coryza, conjunctivitis, diarrhea, cough)
- gold: **not applicable** [0] | predicted: **not excluded** [0]
- raw output: `{"label": "not excluded", "evidence": [0]}`
  - [0] (gold) (cited) A 5-year-old boy presents to the emergency department with complaints of progressively worsening dysphagia, drooling, fever and vocal changes.

### annotation 771 — patient sigir-201513, exclusion
- criterion: Females of childbearing potential (ie, have reached menarche)
- gold: **not applicable** [0] | predicted: **not excluded** []
- raw output: `{"label": "not excluded", "evidence": []}`
  - [0] (gold) A 5-year-old boy presents to the emergency department with complaints of progressively worsening dysphagia, drooling, fever and vocal changes.

## polarity_flip (2)

### annotation 164 — patient sigir-201410, inclusion
- criterion: Patients undergoing surgical lower limb revascularisation (suprainguinal or infrainguinal)
- gold: **not included** [] | predicted: **included** [0]
- raw output: `{"label": "included", "evidence": [0]}`
  - [0] (cited) A physician is called to see a 67-year-old woman who underwent cardiac catheterization via the right femoral artery earlier in the morning.

### annotation 166 — patient sigir-201410, exclusion
- criterion: Significant upper limb peripheral arterial disease
- gold: **not excluded** [] | predicted: **excluded** [1, 2]
- raw output: `{"label": "excluded", "evidence": [1, 2]}`
  - [1] (cited) She is now complaining of a cool right foot.
  - [2] (cited) Upon examination she has a pulsatile mass in her right groin with loss of distal pulses, and auscultation reveals a bruit over the point at which the right femoral artery was entered.

## right_label_wrong_evidence (33)

### annotation 184 — patient sigir-201410, exclusion
- criterion: Other inappropriate situation
- gold: **not excluded** [] | predicted: **not excluded** [0]
- raw output: `{"label": "not excluded", "evidence": [0]}`
  - [0] (cited) A physician is called to see a 67-year-old woman who underwent cardiac catheterization via the right femoral artery earlier in the morning.

### annotation 189 — patient sigir-201411, inclusion
- criterion: at least 1 of 3 signs of hypoperfusion (urine output < 0.5mL/kg of body weight per hour for 1 hour or more; neurologic dysfunction defined by confusion, psychosis, or a Glasgow coma scale score of ≤ 6; plasma lactate higher than the upper limit of the normal value).
- gold: **not enough information** [] | predicted: **not enough information** [2]
- raw output: `{"label": "not enough information", "evidence": [2]}`
  - [2] (cited) On examination she is pale and in moderate discomfort, as well as tachypneic and tachycardic.

### annotation 196 — patient sigir-201411, exclusion
- criterion: Less than 18 years of age
- gold: **not excluded** [0] | predicted: **not excluded** []
- raw output: `{"label": "not excluded", "evidence": []}`
  - [0] (gold) A 40-year-old woman with no past medical history presents to the ER with excruciating pain in her right arm that had started 1 hour prior to her admission.

### annotation 197 — patient sigir-201411, exclusion
- criterion: Decreased level of consciousness
- gold: **not excluded** [0, 5] | predicted: **not excluded** []
- raw output: `{"label": "not excluded", "evidence": []}`
  - [0] (gold) A 40-year-old woman with no past medical history presents to the ER with excruciating pain in her right arm that had started 1 hour prior to her admission.
  - [5] (gold) The patient will provide informed consent, and will comply with the trial protocol without any practical issues.

### annotation 305 — patient sigir-201417, inclusion
- criterion: Standard analytical parameters, defined by: Leukocytes ≥ 3000 Neutrophils ≥ 1500 Platelets ≥ 100,000 Aspartate aminotransferase (AST) / Alanine aminotransferase (ALT) ≤ 1.5 standard range institution creatinine ≤ 1.5 mg / dl
- gold: **not enough information** [] | predicted: **not enough information** [0]
- raw output: `{"label": "not enough information", "evidence": [0]}`
  - [0] (cited) A 48-year-old white male with history of common variable immunodeficiency (CVID) with acute abdominal pain, fever, dehydration, HR of 132 bpm, BP 80/40.

### annotation 307 — patient sigir-201417, inclusion
- criterion: Patient selection should be cautious, covering basically 5 types of liver damage which must be submitted prior to liver volumetry: Metastatic Disease subsidiary right hepatectomy extended to segment IV Metastatic Disease subsidiary right hepatectomy with suspected diseased liver (neoadjuvant chemotherapy) (in cases of doubt may be used liver function test indocyanine green) Bilobar liver metastases with multiple nodules in the right lobe and more than 3 nodules greater than 30 mm in the left hepatic lobe (LHI) will perform lumpectomies the LHI + right portal branch ligation (or postoperative percutaneous embolization) in order to make right hepatectomy 4-6 weeks (two stage surgery) Subsidiary Hepatocarcinoma extended right hepatectomy Liver Injury benign / malignant (Hemangiomas, hydatid cysts or liver tumors / primary bile hepatoblastoma), which by extension threatens the viability of the remaining liver tissue.
- gold: **not enough information** [3] | predicted: **not enough information** []
- raw output: `{"label": "not enough information", "evidence": []}`
  - [3] (gold) Exploratory laparotomy reveals a ruptured liver abscess, which is then surgically drained.

### annotation 310 — patient sigir-201417, exclusion
- criterion: Patients with uncontrolled hypertension.
- gold: **not excluded** [] | predicted: **not excluded** [5]
- raw output: `{"label": "not excluded", "evidence": [5]}`
  - [5] (cited) The patient will provide informed consent, and will comply with the trial protocol without any practical issues.

### annotation 311 — patient sigir-201417, exclusion
- criterion: Severe heart failure (NYHA IV).
- gold: **not excluded** [] | predicted: **not excluded** [0]
- raw output: `{"label": "not excluded", "evidence": [0]}`
  - [0] (cited) A 48-year-old white male with history of common variable immunodeficiency (CVID) with acute abdominal pain, fever, dehydration, HR of 132 bpm, BP 80/40.

### annotation 319 — patient sigir-201417, exclusion
- criterion: Difficulty in monitoring.
- gold: **not excluded** [5] | predicted: **not excluded** []
- raw output: `{"label": "not excluded", "evidence": []}`
  - [5] (gold) The patient will provide informed consent, and will comply with the trial protocol without any practical issues.

### annotation 322 — patient sigir-201417, exclusion
- criterion: Pregnant women or women of childbearing age who do not have adequate contraception.
- gold: **not excluded** [0] | predicted: **not excluded** []
- raw output: `{"label": "not excluded", "evidence": []}`
  - [0] (gold) A 48-year-old white male with history of common variable immunodeficiency (CVID) with acute abdominal pain, fever, dehydration, HR of 132 bpm, BP 80/40.

### annotation 381 — patient sigir-201421, inclusion
- criterion: Age > 18 years at the time of signing the informed consent form
- gold: **included** [0] | predicted: **included** [4]
- raw output: `{"label": "included", "evidence": [4]}`
  - [0] (gold) A 21-year-old female is evaluated for progressive arthralgias and malaise.
  - [4] (cited) The patient will provide informed consent, and will comply with the trial protocol without any practical issues.

### annotation 382 — patient sigir-201421, inclusion
- criterion: Fulfil at least 4 criteria for SLE as defined by the American College of Rheumatology (ACR)
- gold: **included** [0, 1, 2] | predicted: **included** [1, 2, 3]
- raw output: `{"label": "included", "evidence": [1, 2, 3]}`
  - [0] (gold) A 21-year-old female is evaluated for progressive arthralgias and malaise.
  - [1] (gold) (cited) On examination she is found to have alopecia, a rash mainly distributed on the bridge of her nose and her cheeks, a delicate non-palpable purpura on her calves, and swelling and tenderness of her wrists and ankles.
  - [2] (gold) (cited) Her lab shows normocytic anemia, thrombocytopenia, a 4/4 positive ANA and anti-dsDNA.
  - [3] (cited) Her urine is positive for protein and RBC casts.

## wrong_label_right_evidence (8)

### annotation 176 — patient sigir-201410, inclusion
- criterion: No medical history of central vena catheterization.
- gold: **included** [0] | predicted: **not enough information** [0]
- raw output: `{"label": "not enough information", "evidence": [0]}`
  - [0] (gold) (cited) A physician is called to see a 67-year-old woman who underwent cardiac catheterization via the right femoral artery earlier in the morning.

### annotation 186 — patient sigir-201411, inclusion
- criterion: Heart rate of at least 90/min;
- gold: **included** [2] | predicted: **not enough information** [2]
- raw output: `{"label": "not enough information", "evidence": [2]}`
  - [2] (gold) (cited) On examination she is pale and in moderate discomfort, as well as tachypneic and tachycardic.

### annotation 187 — patient sigir-201411, inclusion
- criterion: A respiratory rate of at least 20/min or a PaCO2 of 32mmHg or lower or the use of mechanical ventilation;
- gold: **included** [2] | predicted: **not enough information** [2]
- raw output: `{"label": "not enough information", "evidence": [2]}`
  - [2] (gold) (cited) On examination she is pale and in moderate discomfort, as well as tachypneic and tachycardic.

### annotation 422 — patient sigir-201422, inclusion
- criterion: If female, using birth control
- gold: **not applicable** [3] | predicted: **not enough information** [3]
- raw output: `{"label": "not enough information", "evidence": [3]}`
  - [3] (gold) (cited) She has had no sexual partners and her menses are regular.

### annotation 678 — patient sigir-20158, inclusion
- criterion: Diagnosis of mild sleep-disordered breathing (MSDB) defined as meeting all of the following criteria: Caregiver report of habitual snoring that occurs most of the night on at least three nights per week, and has been present for at least three months (on average occurring > 3 nights per week or more half of sleep time)
- gold: **not enough information** [0] | predicted: **included** [0]
- raw output: `{"label": "included", "evidence": [0]}`
  - [0] (gold) (cited) A 10 yo boy with nighttime snoring, pauses in breathing, and restlessness with nighttime awakenings.

### annotation 762 — patient sigir-201513, inclusion
- criterion: A sore and scratchy throat and/or pain on swallowing (odynophagia) together with at least 2 of the following clinical signs:
- gold: **not enough information** [0] | predicted: **included** [0]
- raw output: `{"label": "included", "evidence": [0]}`
  - [0] (gold) (cited) A 5-year-old boy presents to the emergency department with complaints of progressively worsening dysphagia, drooling, fever and vocal changes.

### annotation 766 — patient sigir-201513, inclusion
- criterion: Fever
- gold: **included** [0] | predicted: **not enough information** [0]
- raw output: `{"label": "not enough information", "evidence": [0]}`
  - [0] (gold) (cited) A 5-year-old boy presents to the emergency department with complaints of progressively worsening dysphagia, drooling, fever and vocal changes.

### annotation 767 — patient sigir-201513, exclusion
- criterion: Symptoms that collectively suggest nonstreptococcal T/P (eg, laryngitis, coryza, conjunctivitis, diarrhea, cough)
- gold: **not applicable** [0] | predicted: **not excluded** [0]
- raw output: `{"label": "not excluded", "evidence": [0]}`
  - [0] (gold) (cited) A 5-year-old boy presents to the emergency department with complaints of progressively worsening dysphagia, drooling, fever and vocal changes.
