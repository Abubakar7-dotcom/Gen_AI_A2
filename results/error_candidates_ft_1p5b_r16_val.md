# Error candidates: ft_1p5b_r16 on val (206 records, 53 wrong labels)

| category | records |
|---|---|
| right_label_wrong_evidence | 45 |
| missing_info_confusion | 41 |
| not_applicable | 11 |
| wrong_label_right_evidence | 6 |
| polarity_flip | 1 |

## Label confusions (type, gold -> predicted)

- exclusion: not enough information -> not excluded: 26
- exclusion: not applicable -> not excluded: 10
- inclusion: included -> not enough information: 9
- inclusion: not enough information -> included: 4
- inclusion: not included -> not enough information: 2
- inclusion: not applicable -> not enough information: 1
- inclusion: not included -> included: 1

## missing_info_confusion (41)

### annotation 8 — patient sigir-20141, exclusion
- criterion: Unstable angina
- gold: **not enough information** [0, 1, 2] | predicted: **not excluded** [0]
- raw output: `{"label": "not excluded", "evidence": [0]}`
  - [0] (gold) (cited) A 58-year-old African-American woman presents to the ER with episodic pressing/burning anterior chest pain that began two days earlier for the first time in her life.
  - [1] (gold) The pain started while she was walking, radiates to the back, and is accompanied by nausea, diaphoresis and mild dyspnea, but is not increased on inspiration.
  - [2] (gold) The latest episode of pain ended half an hour prior to her arrival.

### annotation 15 — patient sigir-20141, exclusion
- criterion: Pregnant and nursing women
- gold: **not enough information** [] | predicted: **not excluded** []
- raw output: `{"label": "not excluded", "evidence": []}`

### annotation 18 — patient sigir-20141, exclusion
- criterion: Patients not eligible for Tc 99m SPECT
- gold: **not enough information** [] | predicted: **not excluded** []
- raw output: `{"label": "not excluded", "evidence": []}`

### annotation 19 — patient sigir-20141, exclusion
- criterion: Patients in whom beta blockers are contraindicated
- gold: **not enough information** [] | predicted: **not excluded** []
- raw output: `{"label": "not excluded", "evidence": []}`

### annotation 22 — patient sigir-20141, inclusion
- criterion: Clinical diagnosis of one or more risk factors, such as diabetes, smoking habit, lipid metabolism abnormality, history of ischemic heart disease (IHD) or cerebrovascular disease, obesity (BMI>25), chronic heart failure (NYHA II-III), and electrocardiogram (ECG) abnormality (LVH)
- gold: **included** [3, 4] | predicted: **not enough information** [3, 4]
- raw output: `{"label": "not enough information", "evidence": [3, 4]}`
  - [3] (gold) (cited) She is known to have hypertension and obesity.
  - [4] (gold) (cited) She denies smoking, diabetes, hypercholesterolemia, or a family history of heart disease.

### annotation 26 — patient sigir-20141, exclusion
- criterion: Pregnant women and women of childbearing potential
- gold: **not enough information** [0] | predicted: **not excluded** []
- raw output: `{"label": "not excluded", "evidence": []}`
  - [0] (gold) A 58-year-old African-American woman presents to the ER with episodic pressing/burning anterior chest pain that began two days earlier for the first time in her life.

### annotation 31 — patient sigir-20142, inclusion
- criterion: patients have fever more than one week
- gold: **not included** [0] | predicted: **not enough information** [0]
- raw output: `{"label": "not enough information", "evidence": [0]}`
  - [0] (gold) (cited) An 8-year-old male presents in March to the ER with fever up to 39 C, dyspnea and cough for 2 days.

### annotation 35 — patient sigir-20142, exclusion
- criterion: we find that the patient selected does not meet the selection criteria within the observation period
- gold: **not enough information** [] | predicted: **not excluded** [0, 1, 2, 3, 4, 5]
- raw output: `{"label": "not excluded", "evidence": [0, 1, 2, 3, 4, 5]}`
  - [0] (cited) An 8-year-old male presents in March to the ER with fever up to 39 C, dyspnea and cough for 2 days.
  - [1] (cited) He has just returned from a 5 day vacation in Colorado.
  - [2] (cited) Parents report that prior to the onset of fever and cough, he had loose stools.
  - [3] (cited) He denies upper respiratory tract symptoms.
  - [4] (cited) On examination he is in respiratory distress and has bronchial respiratory sounds on the left.
  - [5] (cited) A chest x-ray shows bilateral lung infiltrates.

### annotation 36 — patient sigir-20142, exclusion
- criterion: patients leave with automatic discharge
- gold: **not enough information** [] | predicted: **not excluded** []
- raw output: `{"label": "not excluded", "evidence": []}`

### annotation 235 — patient sigir-201415, inclusion
- criterion: Menorrhagia and/or bulk symptoms associated with uterine fibroids
- gold: **not enough information** [0, 1, 4, 5] | predicted: **included** [0, 1, 2, 3, 4]
- raw output: `{"label": "included", "evidence": [0, 1, 2, 3, 4]}`
  - [0] (gold) (cited) A 36-year-old woman presents to the emergency department with 12 weeks of amenorrhea, vaginal spotting that has increased since yesterday, lower abdominal tenderness, nausea and vomiting.
  - [1] (gold) (cited) The physical exam reveals overall tender abdomen, 18-week sized uterus, and cervical dilation of 2 cm.
  - [2] (cited) The complete blood count and biochemical profile are normal.
  - [3] (cited) Point of care pregnancy test with cut-off sensitivity of 25 mIU/ml Beta-HCG is negative.
  - [4] (gold) (cited) The ultrasound reports enlarged uterus (12 cm x 9 cm x 7 cms) with multiple cystic areas in the interior.
  - [5] (gold) The differential diagnosis includes vesicular mole vs fibroid degeneration.

### annotation 253 — patient sigir-201415, exclusion
- criterion: 3. Patients who are considered high risk pregnancy due to uterine factors (e.g. abnormal uterus, uterine scars, cerclage) except fibroids.
- gold: **not enough information** [4] | predicted: **not excluded** [1, 4]
- raw output: `{"label": "not excluded", "evidence": [1, 4]}`
  - [1] (cited) The physical exam reveals overall tender abdomen, 18-week sized uterus, and cervical dilation of 2 cm.
  - [4] (gold) (cited) The ultrasound reports enlarged uterus (12 cm x 9 cm x 7 cms) with multiple cystic areas in the interior.

### annotation 254 — patient sigir-201415, exclusion
- criterion: 4. Patients with fibroid that is more than 50% sub-mucosal or with hysteroscopically resectable
- gold: **not enough information** [] | predicted: **not excluded** []
- raw output: `{"label": "not excluded", "evidence": []}`

## not_applicable (11)

### annotation 24 — patient sigir-20141, exclusion
- criterion: Patients with IHD within 6 months after percutaneous coronary intervention(PCI), and who are stable but are going to implement PCI or coronary artery bypass grafting(CABG)
- gold: **not applicable** [] | predicted: **not excluded** []
- raw output: `{"label": "not excluded", "evidence": []}`

### annotation 33 — patient sigir-20142, inclusion
- criterion: full of physical examination and laboratory examination have been carried out after one week，but still cannot make a definite diagnosis
- gold: **not applicable** [] | predicted: **not enough information** [4, 5]
- raw output: `{"label": "not enough information", "evidence": [4, 5]}`
  - [4] (cited) On examination he is in respiratory distress and has bronchial respiratory sounds on the left.
  - [5] (cited) A chest x-ray shows bilateral lung infiltrates.

### annotation 42 — patient sigir-20142, exclusion
- criterion: Pregnant women
- gold: **not applicable** [0] | predicted: **not excluded** []
- raw output: `{"label": "not excluded", "evidence": []}`
  - [0] (gold) An 8-year-old male presents in March to the ER with fever up to 39 C, dyspnea and cough for 2 days.

### annotation 513 — patient sigir-201427, exclusion
- criterion: Women who are pregnant or breast feeding.
- gold: **not applicable** [] | predicted: **not excluded** []
- raw output: `{"label": "not excluded", "evidence": []}`

### annotation 558 — patient sigir-201430, exclusion
- criterion: Inability to provide informed consent.
- gold: **not applicable** [8] | predicted: **not excluded** [8]
- raw output: `{"label": "not excluded", "evidence": [8]}`
  - [8] (gold) (cited) The patient will provide informed consent, and will comply with the trial protocol without any practical issues.

### annotation 568 — patient sigir-201430, exclusion
- criterion: Pregnant women or women planning to become pregnant or unwilling to use appropriate birth control methods before and 2 months after cell infusion.
- gold: **not applicable** [] | predicted: **not excluded** []
- raw output: `{"label": "not excluded", "evidence": []}`

### annotation 801 — patient sigir-201515, exclusion
- criterion: Definite small-vessel etiology by history or imaging
- gold: **not applicable** [] | predicted: **not excluded** [1, 2]
- raw output: `{"label": "not excluded", "evidence": [1, 2]}`
  - [1] (cited) At the time, computed tomography was negative for brain hemorrhage and she was given thrombolytic therapy with resolution of her symptoms.
  - [2] (cited) Transesophageal echocardiogram and magnetic resonance angiogram of brain and great vessels found no evidence of abnormalities.

### annotation 817 — patient sigir-201515, exclusion
- criterion: Subjects who have a non-vascular origin of the neurological symptoms after brain imaging (CT scan or MRI)
- gold: **not applicable** [0] | predicted: **not excluded** [1]
- raw output: `{"label": "not excluded", "evidence": [1]}`
  - [0] (gold) Karen is a 72-year-old woman with hypertension and type 2 diabetes, who was hospitalized for cryptogenic stroke two weeks ago.
  - [1] (cited) At the time, computed tomography was negative for brain hemorrhage and she was given thrombolytic therapy with resolution of her symptoms.

### annotation 948 — patient sigir-201522, exclusion
- criterion: The patient is pregnant or lactating. If the patient is female and of childbearing potential, the patient must have a negative pregnancy test and avoid becoming pregnant while receiving study drug. A pregnancy test should be performed within 14 days prior to the first dose of study drug.
- gold: **not applicable** [] | predicted: **not excluded** []
- raw output: `{"label": "not excluded", "evidence": []}`

### annotation 950 — patient sigir-201522, exclusion
- criterion: The patient has hepatic cirrhosis.
- gold: **not applicable** [] | predicted: **not excluded** []
- raw output: `{"label": "not excluded", "evidence": []}`

### annotation 954 — patient sigir-201522, exclusion
- criterion: The patient has previously enrolled into this study.
- gold: **not applicable** [] | predicted: **not excluded** []
- raw output: `{"label": "not excluded", "evidence": []}`

## polarity_flip (1)

### annotation 935 — patient sigir-201522, inclusion
- criterion: Thought likely to be Mycobacterium culture positive (including NTM infected for the NIH CC) by enrolling physician
- gold: **not included** [3] | predicted: **included** [3]
- raw output: `{"label": "included", "evidence": [3]}`
  - [3] (gold) (cited) Culture of the sputum revealed an organism with septated, low-angle branching hyphae that had straight, parallel walls.

## right_label_wrong_evidence (45)

### annotation 0 — patient sigir-20141, inclusion
- criterion: Patients of chronic stable angina with abnormal Exercise Myocardial Perfusion Spect Scan with reversible and partially reversible ischemic changes.
- gold: **not enough information** [0, 1, 2] | predicted: **not enough information** []
- raw output: `{"label": "not enough information", "evidence": []}`
  - [0] (gold) A 58-year-old African-American woman presents to the ER with episodic pressing/burning anterior chest pain that began two days earlier for the first time in her life.
  - [1] (gold) The pain started while she was walking, radiates to the back, and is accompanied by nausea, diaphoresis and mild dyspnea, but is not increased on inspiration.
  - [2] (gold) The latest episode of pain ended half an hour prior to her arrival.

### annotation 5 — patient sigir-20141, exclusion
- criterion: Hypertension of > 170/100 mm of Hg
- gold: **not excluded** [] | predicted: **not excluded** [3]
- raw output: `{"label": "not excluded", "evidence": [3]}`
  - [3] (cited) She is known to have hypertension and obesity.

### annotation 6 — patient sigir-20141, exclusion
- criterion: Valvular heart disease and cardiomyopathy
- gold: **not excluded** [] | predicted: **not excluded** [4]
- raw output: `{"label": "not excluded", "evidence": [4]}`
  - [4] (cited) She denies smoking, diabetes, hypercholesterolemia, or a family history of heart disease.

### annotation 13 — patient sigir-20141, exclusion
- criterion: IDDM (Type-1 diabetes mellitus)
- gold: **not excluded** [4] | predicted: **not excluded** []
- raw output: `{"label": "not excluded", "evidence": []}`
  - [4] (gold) She denies smoking, diabetes, hypercholesterolemia, or a family history of heart disease.

### annotation 17 — patient sigir-20141, exclusion
- criterion: On calcium channel blockers
- gold: **not excluded** [5] | predicted: **not excluded** []
- raw output: `{"label": "not excluded", "evidence": []}`
  - [5] (gold) She currently takes no medications.

### annotation 20 — patient sigir-20141, exclusion
- criterion: Geographical inaccessibility for treatment or follow-up evaluations
- gold: **not excluded** [8] | predicted: **not excluded** []
- raw output: `{"label": "not excluded", "evidence": []}`
  - [8] (gold) The patient will provide informed consent, and will comply with the trial protocol without any practical issues.

### annotation 23 — patient sigir-20141, exclusion
- criterion: Patients who have already been administered ARB
- gold: **not excluded** [5] | predicted: **not excluded** []
- raw output: `{"label": "not excluded", "evidence": []}`
  - [5] (gold) She currently takes no medications.

### annotation 34 — patient sigir-20142, exclusion
- criterion: fever for non-infectious diseases such as rheumatic autoimmune disease or with tumor
- gold: **not excluded** [] | predicted: **not excluded** [0]
- raw output: `{"label": "not excluded", "evidence": [0]}`
  - [0] (cited) An 8-year-old male presents in March to the ER with fever up to 39 C, dyspnea and cough for 2 days.

### annotation 38 — patient sigir-20142, inclusion
- criterion: Patients with cough or shortness of breath
- gold: **included** [0] | predicted: **included** [0, 4]
- raw output: `{"label": "included", "evidence": [0, 4]}`
  - [0] (gold) (cited) An 8-year-old male presents in March to the ER with fever up to 39 C, dyspnea and cough for 2 days.
  - [4] (cited) On examination he is in respiratory distress and has bronchial respiratory sounds on the left.

### annotation 236 — patient sigir-201415, exclusion
- criterion: Malignancy
- gold: **not excluded** [0, 1, 4, 5] | predicted: **not excluded** []
- raw output: `{"label": "not excluded", "evidence": []}`
  - [0] (gold) A 36-year-old woman presents to the emergency department with 12 weeks of amenorrhea, vaginal spotting that has increased since yesterday, lower abdominal tenderness, nausea and vomiting.
  - [1] (gold) The physical exam reveals overall tender abdomen, 18-week sized uterus, and cervical dilation of 2 cm.
  - [4] (gold) The ultrasound reports enlarged uterus (12 cm x 9 cm x 7 cms) with multiple cystic areas in the interior.
  - [5] (gold) The differential diagnosis includes vesicular mole vs fibroid degeneration.

### annotation 237 — patient sigir-201415, exclusion
- criterion: Current or planned pregnancy
- gold: **not excluded** [0, 3] | predicted: **not excluded** []
- raw output: `{"label": "not excluded", "evidence": []}`
  - [0] (gold) A 36-year-old woman presents to the emergency department with 12 weeks of amenorrhea, vaginal spotting that has increased since yesterday, lower abdominal tenderness, nausea and vomiting.
  - [3] (gold) Point of care pregnancy test with cut-off sensitivity of 25 mIU/ml Beta-HCG is negative.

### annotation 238 — patient sigir-201415, exclusion
- criterion: Small submucous fibroids suitable for hysteroscopic resection
- gold: **not excluded** [4] | predicted: **not excluded** []
- raw output: `{"label": "not excluded", "evidence": []}`
  - [4] (gold) The ultrasound reports enlarged uterus (12 cm x 9 cm x 7 cms) with multiple cystic areas in the interior.

## wrong_label_right_evidence (6)

### annotation 22 — patient sigir-20141, inclusion
- criterion: Clinical diagnosis of one or more risk factors, such as diabetes, smoking habit, lipid metabolism abnormality, history of ischemic heart disease (IHD) or cerebrovascular disease, obesity (BMI>25), chronic heart failure (NYHA II-III), and electrocardiogram (ECG) abnormality (LVH)
- gold: **included** [3, 4] | predicted: **not enough information** [3, 4]
- raw output: `{"label": "not enough information", "evidence": [3, 4]}`
  - [3] (gold) (cited) She is known to have hypertension and obesity.
  - [4] (gold) (cited) She denies smoking, diabetes, hypercholesterolemia, or a family history of heart disease.

### annotation 31 — patient sigir-20142, inclusion
- criterion: patients have fever more than one week
- gold: **not included** [0] | predicted: **not enough information** [0]
- raw output: `{"label": "not enough information", "evidence": [0]}`
  - [0] (gold) (cited) An 8-year-old male presents in March to the ER with fever up to 39 C, dyspnea and cough for 2 days.

### annotation 558 — patient sigir-201430, exclusion
- criterion: Inability to provide informed consent.
- gold: **not applicable** [8] | predicted: **not excluded** [8]
- raw output: `{"label": "not excluded", "evidence": [8]}`
  - [8] (gold) (cited) The patient will provide informed consent, and will comply with the trial protocol without any practical issues.

### annotation 800 — patient sigir-201515, inclusion
- criterion: Onset of stroke or TIA symptoms within the previous 60 days
- gold: **included** [0] | predicted: **not enough information** [0]
- raw output: `{"label": "not enough information", "evidence": [0]}`
  - [0] (gold) (cited) Karen is a 72-year-old woman with hypertension and type 2 diabetes, who was hospitalized for cryptogenic stroke two weeks ago.

### annotation 935 — patient sigir-201522, inclusion
- criterion: Thought likely to be Mycobacterium culture positive (including NTM infected for the NIH CC) by enrolling physician
- gold: **not included** [3] | predicted: **included** [3]
- raw output: `{"label": "included", "evidence": [3]}`
  - [3] (gold) (cited) Culture of the sputum revealed an organism with septated, low-angle branching hyphae that had straight, parallel walls.

### annotation 938 — patient sigir-201522, inclusion
- criterion: Willing to provide blood samples
- gold: **included** [4] | predicted: **not enough information** [4]
- raw output: `{"label": "not enough information", "evidence": [4]}`
  - [4] (gold) (cited) The patient will provide informed consent, and will comply with the trial protocol without any practical issues.
