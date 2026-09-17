# Annotation Guidelines for Extracted Attributes Quality Check
Please read through this guideline before opening the annotation file.

## Introduction
This annotation evaluates the quality of the extracted attributes on three aspects:

1. **Physical attribute validity**: the extracted `attribute` is a *valid physical attribute* of a *human body part*.
2. **Question well-formedness**: the extracted attribute is meaningful and well-formed in the question template  
   “How likely is it for someone to say a {gender} has {attribute}?”
3. **Character gender correctness**: the inferred gender of the character described in the excerpt is correct.

You will annotate the following columns in the provided CSV:

- **`physical attribute`** (binary; `1`/`0`)
- **`contextually well-formed`** (binary; `1`/`0`)
- **`character gender`** (categorical; `male`/`female`/`nonbinary`/`unknown`)

For any additional remarks you would like to note, please fill them under the **`remarks (optional)`** column per row.

## General Instructions & Rules

- Please **DO NOT EDIT ANYTHING OUTSDIDE** the annotation columns (e.g., `source_novel`, `text_id`, `attribute`, `text`, etc.). Do not reorder rows or modify identifier columns.
- Work row-by-row on the three annotation tasks. **Use only the allowed labels** for each annotation task.
- Each annotator should save their filled CSV as a separate file with your assigned anntator ID (e.g., `ann1.csv`, `ann2.csv`, …).
- For any enquiry about specific cases, please feel free to discuss them with Yingjia or Lin.


## Tasks


### 1) `physical attribute` (1/0)

**Goal**: Check whether `attribute` is a *valid physical attribute of a human body part* as used in the excerpt.

Label **`1`** if ALL of the following are true:
- The attribute describes **a body part** (e.g., eyes, hair, face, skin, hands, beard, mouth, teeth).
- It is a **physical property/appearance** (e.g., color, size, shape, texture, condition, anything that could potentially be used to describe a body part).
- It refers to a **human** (not an animal, object, or metaphor that is clearly non-human in context).

Label **`0`** if any of the requirements above are not met.

**Examples**
- `papery thin skin` in “His flesh was wrinkled and spotted, the skin so papery thin …” → **1**
- `a very large mustache` in “He was a big, beefy man … a very large mustache.”  → **1**
- `a beautiful voice` → **0** (voice quality, not a body-part physical attribute)
- `a short king` → **0** (not a body-part)
- `a pair of eyes` → **0** (not an attribute description of a body-part)

---

### 2) `contextually well-formed` (1/0)

**Goal**: Check whether we can apply the attribute into a question template “How likely is it for someone to say a woman has {attribute}?”, “How likely is it for someone to say a man has {attribute}?”, “How likely is it for someone to say a non-binary person has {attribute}?”. Your task is to decide whether questions are **meaningful** and **well-formed** with this attribute.

You can use the provided `question_context` as a reference to help with visualization.

Label **`1`** if the question is natural and sensible:
- The attribute can reasonably be said of a person using “has …”.
- The phrase is grammatical enough and not obviously broken. Small grammatical glitches such as missing articles or incorrect plurals are forgivable (e.g., "has a blonde hair", "has small hand").
- The attribute is specific/meaningful (not overly vague, not a fragment that becomes nonsense in the template).

Label **`0`** if the question is awkward, nonsensical, or unanswerable in ordinary usage:
- The attribute is not something people would say someone “has” in that form (e.g., event-like or temporary in a way that breaks the template).
- The attribute phrase is gramatically malformed to a degree that makes it incomprehensible (e.g., "has tall as a giraffe")

**Examples**
- “How likely … has **papery thin skin**?” → meaningful → **1**
- “How likely … has **feet in leather boots**?” → not meaningful as an attribute in this template → **0**
- “How likely … has **hands the size of trash can lids**?” → **1**
- “How likely … has **tall as a giraffe**?” → **0**

---

### 3) `character gender` (male/female/nonbinary/unknown)

**Goal**: Label the gender of the **character who has the attribute** in the `text` excerpt. Base decisions on the provided **`text`** excerpt (and the attribute phrase). You don't need to inquire about other parts of the book or character.

Label:
- **`male`**: excerpt clearly indicates male (e.g., “he/him/his”, “man”, “boy”, gender-coded title or name in the book like “Lord”/"Harry", *when clearly referring to the described person*).
- **`female`**: excerpt clearly indicates female (e.g., “she/her/hers”, “woman”, “girl”, “Lady”, "Hermione" *when clearly referring to the described person*).
- **`nonbinary`**: excerpt explicitly indicates nonbinary gender identity or nonbinary pronouns *when clearly referring to the described person*.
- **`unknown`**: excerpt does not provide enough evidence, or evidence conflicts/doesn’t resolve to one label.

**Important**:
- Do **not** infer gender from stereotypes (e.g., “graceful” ≠ female, "masculine" ≠ male).
- If multiple people are mentioned, be careful about **which person the attribute attaches to** (the one described by the attribute) and identify that person’s gender.

**Examples**
- “His small beard was well peppered with grey …” → described person uses “his” → **male**
- “Her black hair was drawn into a tight bun.” → described person uses “her” → **female**
- “Straight black hair, olive skin, we even have the same gray eyes.” → no explicit gender evidence → **unknown**

---


