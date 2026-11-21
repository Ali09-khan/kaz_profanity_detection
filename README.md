# Kazakh Profanity Detection

Given one Kazakh product review or comment, predict `0` for no Kazakh profanity or `1` for Kazakh profanity present. The output includes the model's estimated probability for label `1`.
Full Name: Alikhan Rakhman
ID: 25MD0197

## Data and labels

The supplied `kazakh_profanity_share.xlsx` workbook contains 3,061 KazSAnDRA review IDs with provisional profanity labels (1,523 negative and 1,538 positive). Each ID exists in the public KazSAnDRA corpus. The workbook contains 2,724 app reviews, 191 marketplace reviews, 95 map and navigation reviews, and 51 book reviews. Apps are digital products, and the marketplace subset includes reviews of physical goods. KazSAnDRA is by Rustem Yeshpanov and Huseyin Atakan Varol and is available under CC BY 4.0. [Paper](https://aclanthology.org/2024.lrec-main.844/) · [Original data](https://github.com/IS2AI/KazSAnDRA/tree/d769cf7f7bd62b59f88915508d4bc062310b54d6/dataset)

The project starts from the workbook's profanity annotations. The upstream star and sentiment scores are unused. A contextual review of the validation and test labels corrected Russian-only swearing, ordinary insults, and missed vulgarities. Training positives were screened with spelling and profanity patterns and reviewed overrides. Training negatives retain their workbook labels except for clear cases identified during review. This produced a single set of project labels in [labels.csv](labels.csv), with the original labels and review status preserved for comparison. Training labels remain approximate, and the reviewed labels have no independent second annotator.

| Label | Rule |
|---|---|
| 1 | A recognizable Kazakh vulgar word or expression occurs, including when quoted. Common misspellings, transliteration, and masked spellings count when the intended expression is clear. |
| 1 | A mixed-language review contains identifiable Kazakh profanity. Established Kazakh shorthand such as `шшс` counts when the usage is recognizable. |
| 0 | The review contains ordinary criticism, a non-vulgar insult such as `мал`, a literal term in a harmless context, or a reference to swearing without an actual swear word. |
| 0 | Any recognizable profanity is solely from another language. This includes Russian swearing with Kazakh inflections when there is no Kazakh profane expression. |

Unresolvable masks and borderline cases can be excluded during review. The following examples were written to illustrate the rules and are outside all three splits.

| Synthetic example | Label | Reason |
|---|---:|---|
| Тауар кешікті, сапасы нашар. | 0 | Delivery and quality complaint |
| Сапасы жаман, бля. | 0 | Russian-only profanity |
| Мына боқ дұрыс жұмыс істемейді. | 1 | Kazakh vulgarity in a complaint |
| Мына қо*ақ қосымша ашылмайды. | 1 | Recognizable masked Kazakh vulgarity |

[prepare.py](prepare.py) downloads three pinned source archives (about 9 MB) into an ignored local cache and joins them with the label manifest. The repository contains review IDs and labels. Training uses public source text locally. The workbook and raw reviews stay outside the tracked files. No synthetic examples were added to the data.

### Splits

The preparation script preserves the workbook's train, validation, and test assignments. The workbook's split seed is unavailable. Preparation is deterministic and the model uses seed `42`.

Exact duplicates are matched after Unicode and case normalization, contact-pattern removal, collapsing runs to two characters, and stripping punctuation and spaces. For near-duplicate grouping, the code also applies common Kazakh spelling folds, reduces repeats to one, and links short strings sharing a single-deletion form or strings with character bigram/trigram cosine similarity of at least `0.84`. When a group crosses splits, test takes priority over validation, and validation takes priority over train. Lower-priority copies are discarded to preserve the reviewed holdouts.

| Split | Label 0 | Label 1 | Total |
|---|---:|---:|---:|
| Train | 1,920 | 538 | 2,458 |
| Validation | 181 | 82 | 263 |
| Test | 172 | 94 | 266 |
| **Total** | **2,273** | **714** | **2,987** |

Preparation excluded 24 ambiguous records, seven workbook texts that differed from the pinned source, one review without alphanumeric text after normalization, six exact duplicate copies, and 36 lower-priority near-duplicates.

## Model

A CPU-trained logistic regression uses hashed character 2–5-grams with sublinear TF-IDF and L2 normalization. Character fragments are useful for a small dataset with inflections, spelling variants, and mixed scripts. The hash assigns each fragment to one of 131,072 bins, so there is no vocabulary to build or unknown-token placeholder. A new word can share fragments with training words. A bin without training evidence has no learned weight, and different fragments can hash to the same bin.

The model normalizes Unicode and case, removes common contact patterns, compresses repeated characters to at most two copies, and folds whitespace. It keeps internal punctuation such as asterisks and processes the full comment. A single vulgar word can have less influence when it appears in a long review.

The classifier minimizes L2-regularized logistic loss with the LIBLINEAR solver and up to 1,000 iterations. TF-IDF and the classifier are fit on training rows only. Validation selects classifier `C` from `{1, 4, 16}` and a probability threshold from `0.25` to `0.65` in steps of `0.05`. Selection maximizes class-1 F1, then precision, then uses the first candidate in grid order if still tied. The final choice is evaluated on the held-out test set.

## Run locally

Run from the repository directory with Python 3.13.

```sh
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Predict with the saved [model.joblib](model.joblib), which includes the hash settings, fitted IDF, classifier, and threshold. [predict.py](predict.py) supplies the text normalization. Prediction needs only these files and the installed packages.

```sh
python predict.py "Тауар кешікті, сапасы нашар."
```

Download and check the prepared splits, then reproduce training on CPU.

```sh
python prepare.py
python train.py
```

Training writes `model.joblib` and [results.json](results.json) and prints the IDs of several held-out mistakes for inspection.

## Held-out results

Validation selected `C=16` and a decision threshold of `0.25`. The model flags a review when its estimated probability is at least that threshold.

| Split | Reviews | Precision (class 1) | Recall (class 1) | F1 (class 1) |
|---|---:|---:|---:|---:|
| Validation | 263 | 0.944 | 0.817 | 0.876 |
| Test | 266 | 0.975 | 0.840 | 0.903 |

Test confusion matrix

| Actual / predicted | 0 | 1 |
|---|---:|---:|
| 0 | 170 | 2 |
| 1 | 15 | 79 |

The model flagged 81 test reviews, of which 79 contained Kazakh profanity. It caught 79 of the 94 profane reviews. A false positive can hide legitimate product criticism, while a false negative leaves profanity visible. These scores support using the model to prioritize a review queue. Automatic removal would put more weight on the cost of false positives.

### Example predictions

The saved model predicts `0` (`0.026`) for the synthetic complaint “Тауар кешікті, сапасы нашар.” and `1` (`0.953`) for the synthetic comment “Бұл қотақбас қосымша ашылмай тұр.” The masked rule example “Мына қо*ақ қосымша ашылмайды.” receives `0` (`0.106`), even though its intended label is `1`.

The following examples are excerpts from the test set. Scores are probabilities for label `1`.

| Review ID | Excerpt | Actual → predicted | Score | Likely reason |
|---|---|---:|---:|---|
| `pla041071pla` | Актиф керегі жок праграма тағы керегі жок симкарта | 0 → 1 | 0.302 | Common fragments such as `гі` and `та` raised the score of an ordinary SIM-card complaint. |
| `pla019876pla` | Ешқандай регистрация жасалмайд, бәрін дұрыс толтырып отырмын | 0 → 1 | 0.297 | The learned weight for a short fragment such as `от` contributed to flagging a registration complaint. |
| `pla131713pla` | Сіккккееннім барғооое | 1 → 0 | 0.101 | Compressing repeated characters still leaves an unusual form with fragments the model has seen less often. |
| `pla126879pla` | … aman bolyńdar dep jazsam am deıdi … | 1 → 0 | 0.087 | A short quoted vulgarity in a Latin-script comment carries little evidence across the rest of the review. |
| `pla062626pla` | Сігіліп қалған банк. Халықты зар қақсатқан. | 1 → 0 | 0.094 | The vulgar inflection raises the score, but it is surrounded by ordinary banking vocabulary. |

### Limitations and improvement

App reviews account for 247 of the 266 test reviews and 93 of the 94 test positives. The eight marketplace test reviews all have label `0`. The test therefore cannot establish recall for physical-goods reviews. The workbook was sampled for annotation, so the observed class balance and precision can differ from the rate in a real moderation queue. The scores have no separate calibration. The near-duplicate rules can miss paraphrases, and labels for slang and masking remain judgment calls.

A useful next dataset would include more profanity-positive physical-goods reviews, ordinary negative complaints, and realistic masked and Latin-script variants. A second Kazakh-speaking annotator could resolve disputed cases and independently label a new holdout. This would let the same model be checked for both missed profanity and unfair blocking of clean reviews.

## Defense walkthrough

In the positive example above, normalization lowercases the comment, then hash buckets receive the counts for fragments such as `қот` and `отақ`. The saved inverse document frequencies and per-review normalization scale those counts. Logistic regression sums the weighted features and maps the sum to a probability with the sigmoid. The final probability `0.953` clears `0.25`, giving label `1`. Training adjusts the weights to reduce log loss and penalizes large weights. `C` is the inverse strength of that penalty. To demonstrate the decision boundary, run the saved classifier on the three synthetic prediction examples and compare their scores with the selected threshold.
