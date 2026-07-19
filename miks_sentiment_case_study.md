# Miks Sentiment Analysis: Do Sentiment Tools Actually Work on Gaming Slang?

**A comparison of lexicon-based, trained ML, and LLM-based sentiment classification on community reactions to a new VALORANT agent**

---

## TL;DR

I built a 698-comment dataset from Reddit, VLR.gg, and YouTube tracking community reaction to Miks, a VALORANT agent released ~4 months ago, then hand-labeled 236 comments as ground truth to test three sentiment analysis approaches against each other:

| Method | Accuracy | Accuracy (excl. Mixed) |
|---|---|---|
| VADER (lexicon baseline) | 36.0% | 39.9% |
| TF-IDF + Logistic Regression (trained, cross-validated) | 56.4% | 62.4% |
| LLM zero-shot classification | 74.6% | 79.3% |

The takeaway: a general-purpose, off-the-shelf sentiment tool is close to useless on niche, slang-heavy community text. The accuracy gap between the naive baseline and a properly validated approach is too large to treat as a rounding error.

---

## Motivation

Sentiment analysis projects often stop at "run VADER, report percent positive/negative." That's fine for generic text, but gaming communities write in a register that's dense with slang, sarcasm, and domain-specific shorthand ("mid," "ass," "trash," "diks") that a general-purpose lexicon was never built to understand. I wanted to actually measure how much that gap matters, rather than assume it.

Miks turned out to be a good test case: he's a genuinely controversial agent (the community is split on whether he's "balanced but boring" or "underpowered", and the sentiment isn't a simple positive/negative binary), and there's a lot of organic discussion to pull from across multiple platforms and a few months of time.

---

## Data Collection

**698 comments** across **16 threads**, spanning three platforms:

| Platform | Comments | Notes |
|---|---|---|
| Reddit (r/ValorantCompetitive) | ~331 | Competitive/pro-scene focused discussion |
| Reddit (r/VALORANT) | ~187 | General playerbase, more ranked-focused |
| VLR.gg forums | 30 | Hardcore competitive-analysis crowd |
| YouTube | 150 (downsampled from 598) | Reactions to an abilities-showcase video |

Data spans launch week (~4 months before collection) through roughly a month before collection, giving a rough (not continuous) time dimension: a launch cluster, a "mid" cluster of ranked-perception threads, and a handful of more recent threads and stray replies.

**Parsing challenges worth noting:** each platform (and even different copy-paste sessions within Reddit) came through in a different raw text format, requiring four separate parsers to handle username/timestamp/flair/badge variations, OP markers, edited-comment markers, and deleted-comment stubs. A recurring issue was Reddit's collapsed-reply UI pasting as broken avatar-only fragments with no real comment body; these were identified and dropped rather than guessed at. YouTube comments needed a separate parser entirely (`@username` / relative-time / body / like-count format), and the video's related-content sidebar had to be manually excluded from the raw text before parsing.

For the YouTube batch specifically, ability-testing questions directed at the video's creator ("can you test if X works?") were filtered out before sampling, since they're not sentiment toward Miks, and the remaining pool was randomly downsampled from 598 to 150 to prevent one video's comment section from dominating the overall dataset by sheer comment count.

---

## Building the Gold Set

To evaluate any sentiment method honestly, I needed ground truth. I hand-labeled **236 comments** (Positive / Negative / Neutral / Mixed) using a consistent rule: judge sentiment toward Miks specifically, not toward Valorant, Riot, or other agents mentioned in passing. A useful heuristic that came up during labeling: pronoun cues ("she," "her") reliably flag that a comment is actually about a different agent and should be scored Neutral for Miks-sentiment purposes, since bag-of-words tools have no way to catch this.

The initial 141-comment sample, drawn via stratified random sampling across all threads, ended up skewed toward Neutral (a lot of the discussion is genuinely off-topic tangents about other agents or general game-design philosophy). A second labeling pass added 95 more comments, deliberately oversampled toward likely-Positive and likely-Mixed candidates (using VADER's score as a rough pre-filter, not as ground truth) to correct the class balance. Final distribution: Negative 58, Neutral 107, Positive 48, Mixed 23.

---

## Method 1: VADER (Lexicon Baseline)

VADER (Valence Aware Dictionary and sEntiment Reasoner) requires no training, just a lexicon and heuristics, so it's the cheapest thing to try first. It scored:

- **36.0% accuracy** (39.9% excluding the Mixed class)
- Systematically over-predicted Positive (150 of 236 comments, vs. an actual 48 Positive)

**Why it failed this badly:**

1. **No subject-awareness.** VADER scores a sentence's words in isolation, it can't tell that "she's always been played even when sentinels were OP" is praise for Viper, not Miks. Any positive language anywhere in a comment gets credited toward whatever topic the comment happens to be about, whether or not that's actually Miks.
2. **Missing domain vocabulary.** Words this community uses as clear negatives ("boring," "mid," "trash," "ass") either aren't in VADER's lexicon or aren't weighted negatively enough. "He is truly boring" scores as mildly *positive*.
3. **No handling of concessive structure.** Multi-clause arguments with a "but" or "even though" pivot get scored on the words present, not the actual argument being made.

---

## Method 2: TF-IDF + Logistic Regression

A real, trainable classifier learned from the 236 labeled comments. Given how small that labeled set is, especially the 23-example Mixed class, I used **5-fold stratified cross-validation** instead of a single train/test split. A single split would have given an accuracy number that swung heavily depending on luck of the draw (per-fold accuracy in this run ranged from 44.7% to 63.8%, a 19-point spread on the same data and same model).

**Results:**
- **56.4% accuracy** (62.4% excluding Mixed)
- **0 of 23 Mixed comments correctly classified.** With only 23 training examples split across 5 folds, the model never saw enough Mixed examples to learn the pattern. This isn't a bug, it's an honest small-data limitation.
- Learned features were informative for Negative (top weighted words: "boring," "ass," "absolute," "showmatch") but mostly function words for Positive ("with," "and," "my," "his"), a sign the model is partly picking up something closer to sentence style than content for that class.

This is the tier of result I'd trust for a real production use case with a bit more labeled data behind it, it's inspectable, reproducible, and improves in a predictable way as more labels get added.

---

## Method 3: LLM Zero-Shot Classification

Comments were also classified by having an LLM (Claude) read each one and apply the same labeling criteria as the human annotation, directly, rather than through a callable classifier function.

- **74.6% accuracy** (79.3% excluding Mixed)
- The strongest result by a wide margin

**Important methodology caveat:** this is not a fully independent, blind evaluation. The same criteria used for the human labels were available when generating these labels, and unlike VADER or the trained LogReg model, this approach can't be handed a brand-new comment and reliably reproduce a judgment without a person (or another LLM call) reading it fresh. It's best understood as a *ceiling estimate* of what's achievable with language-level understanding on this task, not a directly comparable, deployable baseline. I'm disclosing this plainly rather than presenting all three methods as equivalent.

---

## What the Data Actually Shows

Using the trained LogReg model to score all 698 comments:

**By platform:** sentiment composition differs meaningfully by community. The pro-scene-focused r/ValorantCompetitive skews more analytical and mixed; the general playerbase (r/VALORANT) and YouTube skew toward more blunt, one-line reactions in both directions.

**By time period:** comments cluster into three rough windows (Launch ~3-4 months ago, Mid ~1-2 months ago, Recent <1 month ago). Positive sentiment peaks in the Mid period (~29%, vs. ~13% at Launch and ~12% Recent), largely because that window is dominated by ranked players sharing personal "I actually enjoy playing him" takes, a different register than the Launch reveal-reaction threads or the Recent design-critique thread. This is a real pattern in the data, but it reflects *what kind of thread was happening when* more than a clean sentiment trend over the agent's lifecycle. I'm not claiming a causal time trend here, just noting the composition shift.

**Engagement vs. sentiment (Reddit only):** the most-upvoted comments skew more negative than the average comment. Comments in the top 10% by score are 34.6% Negative, compared to 25.6% Negative across all Reddit comments; Mixed and nuanced takes are underrepresented in the top-voted tier (3.8% vs. 7.4%). A Kruskal-Wallis test confirms comment-score distributions differ significantly across sentiment groups (H=17.8, p=0.0005), with Negative comments having the highest median score (4.5) and Mixed the lowest (1.5). In plain terms: critical, punchy takes get amplified by upvotes more than nuanced ones do, so "what the community sees most" skews more negative than "the full spread of what the community actually said."

---

## Limitations

- The gold set (236 comments) is small for a 4-class problem, especially Mixed. More labeled data would likely close some of the gap between the trained model and the LLM approach.
- The time-period comparison is 3 discrete thread clusters, not a continuous random sample across the agent's lifecycle. Differences between periods may reflect thread topic as much as genuine sentiment drift.
- YouTube comments were filtered (removing ability-testing questions) and downsampled for platform balance, so platform comparisons reflect a curated set, not the raw distribution.
- The LLM-labeling approach, while most accurate here, isn't independently reproducible the way the trained model is.

---

## Bottom Line

For messy, slang-heavy, community-specific text, a naive off-the-shelf sentiment tool is not a reliable substitute for either a modest amount of labeled training data or language-model-level understanding. The gap here (36% to 56% to 75% accuracy) is too large to treat as noise. Anyone doing sentiment analysis on niche online communities should budget time to hand-label a gold set and validate their tool of choice against it, rather than trusting a sentiment score at face value.

---

## Tools & Methods

Python, pandas, scikit-learn (TF-IDF, Logistic Regression, stratified k-fold cross-validation), VADER (vaderSentiment), scipy (Kruskal-Wallis), matplotlib, openpyxl. Data collected manually from Reddit, VLR.gg, and YouTube and parsed with custom Python parsers built for each platform's copy-paste format. Full analysis in the accompanying Jupyter notebook.
