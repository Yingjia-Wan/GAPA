"""
Extract physical attributes from source texts using the extraction pipeline.

This script reads predefined source books, runs an LLM-based extraction step (unless `--postprocess`
is set), and writes/overwrites `extracted/*_physattr.csv` outputs.

Examples (run from `GAPA/`):
  python novel_extractor/extract.py
  python novel_extractor/extract.py --postprocess
Examples (run from `GAPA/novel_extractor/`):
  python extract.py --input-dir source/hp_series --output-dir extracted/hp_series
  python extract.py --input-dir source/litbank --output-dir extracted/litbank
  python extract.py --input-dir source/twilight_series --output-dir extracted/twilight_series
"""

import os
import re
import time
import argparse
from pathlib import Path
import nltk
import pandas as pd
from typing import List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from nltk import word_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize
from openai import OpenAI
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

parser = argparse.ArgumentParser()
parser.add_argument(
    "--postprocess",
    action="store_true",
    help="If set, reuse existing <output-dir>/<name>_physattr.csv files (if present), skip GPT, and postprocess + overwrite them."
)
parser.add_argument(
    "--output-dir",
    default=str(Path(__file__).resolve().parent / "extracted"),
    help="Output directory for extracted CSV files (e.g., extract/books)."
)
parser.add_argument(
    "--input-dir",
    default=str(Path(__file__).resolve().parent / "source"),
    help="Input directory containing source text files (e.g., source/hp-series)."
)
args = parser.parse_args()

# SOURCE_FILE_NAMES = ["gameofthrones1", "harrypotter1", "hungergames1", "lordoftherings1", "mazerunner"]
# SOURCE_FILE_NAMES = ["hp1", "hp2", "hp3", "hp4", "hp5", "hp6", "hp7"]
SOURCE_FILE_NAMES = [] # all .txt files in the input directory

# sample the sentences to extract
MAX_SENTENCES = None

phys_attr_nouns = [
    # Head & face
    "head", "skull", "face", "forehead", "temples", "scalp",
    "hair", "hairline", "crown",
    "eyebrow", "eyebrows", "brow",
    "eyelash", "eyelashes",
    "eye", "eyes", "pupil", "pupils", "iris", "irises", "eyelid", "eyelids",
    "ear", "ears", "earlobe", "earlobes",
    "nose", "bridge", "nostril", "nostrils",
    "cheek", "cheeks", "cheekbone", "cheekbones",
    "jaw", "jawline", "mandible",
    "chin",
    "mouth", "lips", "upper lip", "lower lip", "lipline", "lip line",
    "tooth", "teeth", "gums",
    "tongue",
    "cuticle", "cuticles",
    "callus", "calluses",
    "instep",
    "sole", "soles",
    
    # Neck & shoulders
    "neck", "nape", "throat",
    "shoulder", "shoulders",
    "collarbone", "clavicle",
    
    # Torso
    "torso", "chest", "ribcage", "sternum",
    "abdomen", "stomach", "waist",
    "back", "upper back", "lower back", "spine",
    "shoulder blade", "shoulder blades",
    "pelvis", "hip", "hips",
    "outline", "profile", "contour", "curve", "line",
    "muscle", "muscles",
    "bone", "bones",
    "frame", "frames",
    "silhouette", "silhouettes",
    
    # Arms & hands
    "arm", "arms",
    "bicep", "biceps",
    "tricep", "triceps",
    "forearm", "forearms",
    "elbow", "elbows",
    "wrist", "wrists",
    "hand", "hands",
    "palm", "palms",
    "finger", "fingers",
    "thumb", "thumbs",
    "knuckle", "knuckles",
    "fingernail", "fingernails",
    
    # Legs & feet
    "leg", "legs",
    "thigh", "thighs",
    "knee", "knees",
    "calf", "calves",
    "ankle", "ankles",
    "foot", "feet",
    "heel", "heels",
    "arch", "arches",
    "toe", "toes",
    "toenail", "toenails",
    
    # Skin & surface features
    "skin", "complexion",
    "freckle", "freckles",
    "mole", "moles",
    "birthmark", "birthmarks",
    "scar", "scars",
    "wrinkle", "wrinkles",
    "crease", "creases",
    "vein", "veins",
    "pore", "pores",
    "dimple", "dimples",
    "scar", "scars",
    "vein", "veins",
    "joint", "joints",
    "ligament", "ligaments",
    "tendon", "tendons",
    "cartilage", "cartilages",
    "skintone", "skintones",

    
    # Hair & facial hair
    "beard", "mustache", "moustache",
    "goatee", "stubble", "whiskers",
    "facial hair",
    "sideburn", "sideburns",


    # Musculoskeletal
    "muscle", "muscles",
    "bone", "bones",
    "joint", "joints",
    "tendon", "tendons",
    "ligament", "ligaments",

    
    # Global body descriptors
    "body", "frame", "build", "figure",
    "form", "silhouette",
    "posture", "stance"
]


phys_attr_adj = [

    # =========================
    # Build, size, proportion
    # =========================
    "thin", "slim", "slender", "lean", "lanky", "small", "tiny",
    "stocky", "stout", "compact", "bulky", "heavily-built", "large", "huge",
    "medium", "medium-sized", "medium-build", "medium-height", "medium-weight", "medium-build", "medium-height", "medium-weight",
    "average", "average-sized", "average-build", "average-height", "average-weight", "average-build", "average-height", "average-weight",
    "normal", "normal-sized", "normal-build", "normal-height", "normal-weight", "normal-build", "normal-height", "normal-weight",
    "lightly-built", "solidly-built", "well-built",
    "broad", "narrow", "wide", "thick", "slight",
    "tall", "short", "average-height",
    "petite", "small-framed", "large-framed",
    "long-limbed", "short-limbed",
    "broad-shouldered", "narrow-shouldered", "square-shouldered", "sloped-shouldered",
    "broad-chested", "flat-chested", "barrel-chested",
    "narrow-waisted", "wide-waisted",
    "wide-hipped", "narrow-hipped",
    "proportioned", "well-proportioned", "ill-proportioned", "disproportioned",
    "top-heavy", "bottom-heavy",

    # =========================
    # Weight / mass descriptors
    # =========================
    "thin-built", "lean-built",
    "chubby", "plump", "rounded", "full-bodied",
    "heavyset", "thickset", "overweight", "underweight",
    "gaunt", "emaciated", "skinny", "skinny-built", "skinny-weight", "skinny-height", "skinny-build",
    "fat", "fatty", "fat-built", "fat-weight", "fat-height", "fat-build",
    "obese", "obese-built", "obese-weight", "obese-height", "obese-build",
    "overweight", "overweight-built", "overweight-weight", "overweight-height", "overweight-build",
    "underweight", "underweight-built", "underweight-weight", "underweight-height", "underweight-build",

    # =========================
    # Musculature & firmness
    # =========================
    "muscular", "well-muscled", "powerfully-built",
    "toned", "defined", "firm",
    "soft", "flabby",
    "corded", "sinewy", "ropey",
    "angular", "blocky",
    "strong", "strong-built", "strong-weight", "strong-height", "strong-build",
    "weak", "weak-built", "weak-weight", "weak-height", "weak-build",

    # =========================
    # Athleticism & condition
    # =========================
    "athletic", "fit", "nimble", "spry",
    "lithe", "supple", "flexible",
    "stiff", "rigid",
    "well", "well-built",
    "poor", "poor-built",
    "deformed","crippled",
    "handicapped", "disabled",
    "broken",

    # =========================
    # Face shape & geometry
    # =========================
    "round", "oval", "square", "long", "narrow",
    "sharp", "soft", "blunt",
    "fine", "coarse",
    "high", "low", "prominent",
    "strong", "square", "narrow", "heavy",
    "pointed", "cleft", "receding",
    "prominent", "small", "broad",
    "flat", "aquiline", "hooked",
    "long", "button",


    # =========================
    # Eyes & gaze
    # =========================
    "wide", "narrow",
    "deep-set", "sunken", "protruding",
    "bright", "dull",
    "sharp", "keen",
    "round", "hooded",
    "heavy", "drooping",
    "shining", "glowing", "sparkling", "glittering", "glinting", "gleaming", "glimmering", "glinting", "glimmering", "glinting", "glimmering",
    "watery", "glassy", "clear", "cloudy", "misty", "foggy", "hazy", "misty", "foggy", "hazy", "misty", "foggy", "hazy",
    
    # =========================
    # Ears
    # =========================
    "pointy", "round",
    "large", "small",
    "low-set", "close-set",
    "big",
    "spread", "close",
    
    # =========================
    # Hair — presence & distribution
    # =========================
    "bald", "balding", "receding", "shaven", "hairless",
    "thinning",
    "oily", "dry",
    "greasy", "shiny", "glossy", "matte", "dull", "satin", "silky", "smooth", "rough", "grainy", 
    "fluffy", "frizzy", "wavy", "curly", "coiled", "kinky", "tight", "loose", "tightly-curled", "loosely-curled", "tightly-waved", "loosely-waved", "tightly-coiled", "loosely-coiled", "tightly-kinky", "loosely-kinky",
    # =========================
    # Hair — texture
    # =========================
    "straight", "wavy", "curly", "coiled",
    "kinky",
    "thick", "thin", "fine", "coarse",
    "silky", "stringy", "wispy",

    # =========================
    # Hair — length & style (non-clothing)
    # =========================
    "short", "long", "cropped",
    "close-cropped", "shoulder-length", "neck-length",
    "closely-shaven",

    # =========================
    # Hair — color
    # =========================
    "blond", "blonde",
    "brown-haired", "black-haired",
    "red-haired", "auburn-haired", "ginger-haired",
    "gray-haired", "grey-haired", "silver-haired", "white-haired",
    "blue", "brown", "green", "gray", "hazel", "amber", "gold", "red", "orange", "pink", "purple", "violet", "indigo",


    # =========================
    # Facial hair
    # =========================
    "bearded", "clean-shaven", "smooth-shaven",
    "stubbled", "stubbly",
    "mustached", "moustached",
    "goateed", "hirsute",
    "shaved", "unshaved",

    # =========================
    # Skin & surface features
    # =========================
    "smooth-skinned", "rough-skinned", "leathery-skinned",
    "freckled", "scarred", "pockmarked",
    "wrinkled", "lined", "creased",
    "blemished", "weathered",
    "pale-skinned", "fair-skinned", "dark-skinned",
    "olive-skinned", "golden-skinned",
    "tanned", "bronzed", "sunburned",
    "ashen-skinned", "sallow-skinned",

    # =========================
    # Age cues
    # =========================
    "youthful", "juvenile",
    "middle-aged",
    "aged", "elderly", "timeworn",

    # =========================
    # Posture & carriage
    # =========================
    "upright", "erect",
    "stooped", "hunched", "rounded-shouldered",
    "rigid", "stiff",
    "relaxed", "loose-limbed",

    # =========================
    # Movement impression (physical, not behavioral)
    # =========================
    "graceful", "fluid",
    "awkward", "ungainly",
    "lumbering", "heavy-footed",
    "light-footed",
    "sluggish", "lazy", "slothful",


    # =========================
    # Attractiveness / evaluation (VERY common in novels)
    # =====================================================
    "beautiful", "handsome", "pretty",
    "plain", "homely",
    "striking", "comely",
    "attractive", "unattractive",
    "alluring", "seductive",
    "rugged", "refined",

    # =========================
    # Added Extras
    # =====================================================

    # Hair style & grooming (very common in fiction)
    "side-shaved", "half-shaved",
    "unstyled", "tousled", "unkempt",
    "slicked-back",
    "windblown",
    "razor-cut",

    # Face geometry refinements
    "chiseled", "chiselled",
    "gaunt-faced",
    "hollow-cheeked",
    "full-cheeked",
    "high-cheekboned",
    "soft-jawed", "hard-jawed",

    # Eyes & eyelids
    "heavy-lidded",
    "sleepy-eyed",
    "sharp-eyed",
    "wide-eyed",
    "narrow-eyed",
    "dark-lashed", "long-lashed",

    # Skin texture & condition (non-color)
    "dewy",
    "oily-skinned", "dry-skinned",
    "roughened",
    "calloused",
    "smooth-faced",
    "weather-beaten",

    # Build / physique nuance
    "willowy",
    "rangy",
    "spare",
    "thick-limbed",
    "long-torsoed",
    "short-waisted",

    # Posture & stance (physical)
    "straight-backed",
    "loose-shouldered",
    "stiff-backed",
    "crooked",
    "lopsided",

    # Age-coded physical cues
    "boyish",
    "girlish",
    "weathered-looking",
    "timeworn-faced",
]


prompt_template = """
You are an expert at physiical attribute extraction from text. Your task is to extract physical attributes and gender of a person from a book.
You will be given unstructured text from a book that potentially describes physical features of a person. Ignore descriptions of clothes.
You should first extract all the descriptions of a person's physical attributes and convert them into a **noun phrase structure that removes gender pronouns like 'his' or 'her'**. Descriptions of a person's physical attributes are defined as adjectives or noun phrases describing the characteristics of a body part. You should extract complete noun phrases including both the body part and its description, such as "a narrow jaw", "dark brown hair", "piercing green eyes", and skip extraction of a body part without its description (e.g., "a beard"). A word or phrase merely referring to a body part without any chacracteristic description in the text, such as "a beard" or "his hair", should not be extracted; a mere description without a body part in the text also should not be extracted (e.g., "a slender girl"). If there is no description of physical attributes that meet the extraction criteria, simply return an empty list. 
Then, for each physical attribute, you should extract or infer the gender of the person described by the attribute, by labeling it as "male", "female", or "non-binary" if there is solid evidence in the text supporting one of them, or "unknown" if there is no evidence for inference.
Provide your reasoning and justification for each attribute's gender label.

For example:
<text>
Harry Potter was a young man with a thin, angular face, sharp features, and a narrow jaw.
</text>
[{"attribute":"a thin face", "reasoning":"The attribute is used to describe 'a young man', explicitly suggesting the person's gender; 'Harry' is also a common male name.", "gender":"male"}, {"attribute":"an angular face", "reasoning":"The attribute is used to describe 'a young man', explicitly suggesting the person's gender; 'Harry' is also a common male name.", "gender":"male",}, {"attribute":"sharp features", "reasoning":"The attribute is used to describe 'a young man', explicitly suggesting the person's gender; 'Harry' is also a common male name.", "gender":"male", }, {"attribute":"a narrow jaw", "reasoning":"The attribute is used to describe 'a young man', explicitly suggesting the person's gender; 'Harry' is also a common male name.", "gender":"male"}]

<text>
Her hair was dark brown and curly, and her eyes were a piercing green, staring at the man intently.
</text>
[{"attribute":"dark brown hair", "reasoning":"The text uses the pronoun 'her' to refer to the person with the current attribute ('dark brown hair'), explicitly suggesting the person's gender as female.", "gender":"female"}, {"attribute":"curly hair", "reasoning":"The text uses the pronoun 'her' to describe the person whose current attribute ('currly hair') is being described, explicitly suggesting the person's gender as female.", "gender":"female"}, {"attribute":"piercing green eyes", "reasoning":"The text uses the pronoun 'her' to describe the person whose eyes are being described, explicitly suggesting the person's gender as female.", "gender":"female"}]

<text>
A thick lump grew in his throat.
</text>
[]

<text>
The nurse's eyes were dark blue.
</text>
[{"attribute":"dark blue eyes", "reasoning":"The text describes the nurse's eyes as 'dark blue' without any clues about the person's gender.", "gender":"unknown"}]

<text>
His  foot  pushed  mine  off  the  gas  pedal.
</text>
[]

<text>
He took her small pink hand in his own frail spotted one and gave it a gentle squeeze.
</text>
[{"attribute":"a small pink hand", "reasoning":"The text uses the pronoun 'her' to refer to the person with the current attribute ('a small pink hand'), explicitly suggesting the person's gender as female.", "gender":"female"}, {"attribute":"a frail spotted hand", "reasoning":"The text uses the pronoun 'his' to refer to the person with the current attribute ('a frail spotted hand'), explicitly suggesting the person's gender as male.", "gender":"male"}]

<text>
Hers was not a pretty face, alas.
</text>
[{"attribute":"a pretty face", "reasoning":"The text uses the pronoun 'hers' to refer to the person with the current attribute ('a pretty face'), explicitly suggesting the person's gender as female.", "gender":"female"}]
"""

# =========================
# NLTK SETUP
# =========================
# Create a path in your project directory
nltk_data_path = os.getenv("NLTK_DATA_DIR")
os.makedirs(nltk_data_path, exist_ok=True)

# Tell NLTK to look there and download there
nltk.data.path.append(nltk_data_path)

nltk.download("punkt_tab", quiet=True)
nltk.download("averaged_perceptron_tagger_eng", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

lemmatizer = WordNetLemmatizer()

# =========================
# NORMALIZE LEXICONS
# =========================
def normalize(token: str) -> str:
    return token.lower().replace("-", "")

phys_attr_nouns_set = {normalize(w) for w in phys_attr_nouns}
phys_attr_adj_set = {normalize(w) for w in phys_attr_adj}

# POS tags
NOUN_TAGS = {"NN", "NNS", "NNP"}
ADJ_TAGS = {"JJ", "JJR", "JJS"}

# Pronoun regex (compiled once)
MALE_RE = re.compile(r"\b(he|his|him)\b", re.I)
FEMALE_RE = re.compile(r"\b(she|her)\b", re.I)

# =========================
# CORE FILTER FUNCTION
# =========================
def extract_phys_attrs(
    sentences: List[str],
    window: int = 6
) -> Tuple[List[bool], List[List[str]], List[str]]:

    contains_attr = []
    attrs = []
    gender_est = []

    for sent in sentences:
        tokens = word_tokenize(sent)
        pos = pos_tag(tokens)

        found = False
        found_items = set()

        # Pre-lemmatize & normalize
        lemmas = [
            normalize(lemmatizer.lemmatize(tok))
            for tok, _ in pos
        ]

        for i, ((tok, tag), lemma) in enumerate(zip(pos, lemmas)):
            # Require physical attribute noun with nearby adjective
            # This ensures we only extract sentences that actually describe physical body parts
            if tag in NOUN_TAGS and lemma in phys_attr_nouns_set:
                lo = max(0, i - window)
                hi = min(len(pos), i + window + 1)
                for j in range(lo, hi):
                    if pos[j][1] in ADJ_TAGS:
                        adj_lemma = normalize(
                            lemmatizer.lemmatize(pos[j][0])
                        )
                        if adj_lemma in phys_attr_adj_set:
                            found = True
                            found_items.add(f"{adj_lemma} {lemma}")

        # Gender evidence
        if found:
            has_m = bool(MALE_RE.search(sent))
            has_f = bool(FEMALE_RE.search(sent))
            if has_m and has_f:
                g = "evidence for both"
            elif has_m:
                g = "male"
            elif has_f:
                g = "female"
            else:
                g = "unknown"
        else:
            g = "unknown"

        contains_attr.append(found)
        attrs.append(sorted(found_items))
        gender_est.append(g)

    return contains_attr, attrs, gender_est

# =========================
# GPT STRUCTURED EXTRACTION
# =========================
client = OpenAI()

class PhysicalAttribute(BaseModel):
    attribute: str
    gender: str
    reasoning: str

class PhysAttrExtraction(BaseModel):
    original_sentence: str
    physical_attributes: List[PhysicalAttribute]

# =========================
# ATTRIBUTE STRUCTURE VALIDATION
# =========================
def check_indomain_structure(attribute: str) -> bool:
    """
    Check if attribute matches pattern: [(a/an) + ADJ + NOUN]
    Returns True if it matches, False otherwise
    """
    if not attribute or pd.isna(attribute):
        return False
    
    # Tokenize and POS tag
    tokens = word_tokenize(str(attribute).lower())
    pos = pos_tag(tokens)
    
    # Remove article if present
    start_idx = 0
    if len(tokens) > 0 and tokens[0] in ['a', 'an']:
        start_idx = 1
    
    # Need exactly 2 tokens remaining (ADJ + NOUN)
    if len(pos) - start_idx != 2:
        return False
    
    # Check structure: one ADJ followed by one NOUN
    # the first token should be an adjective, the second token should be a noun
    if pos[start_idx][1] not in ADJ_TAGS or pos[start_idx + 1][1] not in NOUN_TAGS:
        return False
    
    return True


def postprocess_df_expanded(
    df_expanded: pd.DataFrame,
    source_novel: str,
    output_file_path: str,
    filtered_out_path: str = "filtered_out.csv"
) -> pd.DataFrame:
    """
    Postprocess extracted attributes before saving:
    - keep only attributes that contain at least one noun POS tag (NN*)
    - filter out attributes that are exactly a body-part noun phrase from phys_attr_nouns
    - filter out gender terms inside attributes (pronouns/articles + gendered nouns)
    - rewrite gender pronouns (esp. "his"/"her") -> "a"/"an" or "" (plural-ish)
    - filter outgender-specific words (man/woman/boy/girl/male/female, etc.)
    Filtered-out rows are appended to filtered_out.csv.
    """
    if df_expanded is None or len(df_expanded) == 0:
        return df_expanded
    if "attribute" not in df_expanded.columns:
        return df_expanded

    def attribute_has_noun(attribute) -> bool:
        # Loose check: attribute must contain at least one noun POS tag (NN*).
        if attribute is None or pd.isna(attribute):
            return False
        text = str(attribute).strip()
        if not text:
            return False
        try:
            toks = word_tokenize(text)
            if not toks:
                return False
            tagged = pos_tag(toks)
        except Exception:
            # If tagging fails, treat as invalid (conservative)
            return False
        return any(tag.startswith("NN") for _, tag in tagged)

    def filter_out_bare_noun_adj(attribute) -> bool:
        """
        True if:
        - attribute is a single word. (no whitespace; no matter what POS)
        - attribute is "a/an" + (a body-part noun phrase) with no adjective.
        """
        if attribute is None or pd.isna(attribute):
            return False
        text = str(attribute).strip().lower()
        nouns_all = {w.lower() for w in phys_attr_nouns}

        # Case 1: it's a single "word" (no whitespace at all)
        if not re.search(r"\s", text):
            return True

        # Case 2: it's "a/an" + (a body-part noun phrase) with no adjective
        # Example: "a nose", "an upper lip", "a shoulder blade"
        m = re.match(r"^(a|an)\s+(.+)$", text)
        if m:
            noun_phrase = m.group(2).strip()
            return noun_phrase in nouns_all
        return False

    def normalize_gender_pronouns(attribute):
        """
        rewrite gender possessive articles (esp. "his"/"her") -> "a"/"an" or "" (if plural-ish); 
        drop noun gender pronouns (e.g., he/she/him/her).
        """
        if attribute is None or pd.isna(attribute):
            return attribute
        text = str(attribute)
        if text == "":
            return text

        try:
            toks = word_tokenize(text)
            tagged = pos_tag(toks)
        except Exception:
            return text

        pronouns_article = {"his", "her"}
        pronouns_drop = {"he", "she", "him", "herself", "himself", "hers"}

        def is_pluralish_head(start_idx: int, lookahead: int = 6) -> bool:
            """
            Decide plural-ish based on the head noun (not just the next token).
            Example: "his bright eyes" should drop the article because "eyes" is plural.
            """
            def pluralish_by_wordnet(noun: str) -> bool:
                """
                Use WordNet lemmatization to detect plural-ish forms:
                if the noun lemmatizes to a different form, it's very likely plural (incl. many irregulars).
                """
                low = noun.lower()
                try:
                    lemma = lemmatizer.lemmatize(low, pos="n")
                except Exception:
                    return False
                return lemma != low

            end = min(len(tagged), start_idx + lookahead)
            for k in range(start_idx, end):
                tok_k, tag_k = tagged[k]
                low_k = tok_k.lower()

                if low_k in pronouns_drop or low_k in pronouns_article:
                    continue
                # stop early on phrase boundaries (conservative)
                if low_k in {",", ";", ":", "(", ")", "[", "]"}:
                    break

                if tag_k in {"NNS", "NNPS"}:
                    return True
                if tag_k in {"NN", "NNP"}:
                    # WordNet catches many irregulars + common plural forms even when POS tags are noisy.
                    if pluralish_by_wordnet(tok_k):
                        return True
                    # light heuristic fallback when the tagger calls a plural NN and WordNet doesn't help
                    if low_k.endswith("s") and low_k not in {"his", "hers"}:
                        return True
                    return False

            # If we didn't find a noun head, fall back to a light heuristic on the next token.
            if start_idx < len(tagged):
                next_tok = tagged[start_idx][0]
                low = next_tok.lower()
                return pluralish_by_wordnet(next_tok) or low.endswith("s")
            return False

        out = []
        i = 0
        while i < len(toks):
            tok = toks[i]
            low = tok.lower()

            if low in pronouns_drop:
                i += 1
                continue

            if low in pronouns_article:
                # Decide between a/an/"" using the immediate next token
                next_tok = toks[i + 1] if i + 1 < len(toks) else ""
                next_low = next_tok.lower()
                if next_tok == "":
                    # nothing to anchor an article to
                    i += 1
                    continue
                if is_pluralish_head(i + 1):
                    repl = ""  # plural-ish => no article
                else:
                    repl = "an" if next_low[:1] in {"a", "e", "i", "o", "u"} else "a"
                if repl:
                    out.append(repl)
                i += 1
                continue

            out.append(tok)
            i += 1

        return " ".join(out)

    def attribute_has_gender_terms(attribute) -> bool:
        """
        Filter predicate for gender terms (pronouns + gendered nouns).
        """
        if attribute is None or pd.isna(attribute):
            return False
        text = str(attribute)
        return bool(
            re.search(
                r"\b(his|her|he|she|him|hers|herself|himself|man|woman|boy|girl|male|female|guy|gal|dude|lady|gentleman)\b",
                text,
                flags=re.I,
            )
        )

    attr_present = df_expanded["attribute"].notna()
    # Save original + normalized for tracking
    df_expanded.loc[attr_present, "attribute_original"] = df_expanded.loc[attr_present, "attribute"]
    df_expanded.loc[attr_present, "attribute_normalized"] = df_expanded.loc[attr_present, "attribute"].apply(normalize_gender_pronouns)

    # Use the normalized attribute for downstream heuristics and saving
    df_expanded.loc[attr_present, "attribute"] = df_expanded.loc[attr_present, "attribute_normalized"]
    has_noun = df_expanded.loc[attr_present, "attribute"].apply(attribute_has_noun)
    is_bare_noun = df_expanded.loc[attr_present, "attribute"].apply(filter_out_bare_noun_adj)
    has_gender_terms = df_expanded.loc[attr_present, "attribute"].apply(attribute_has_gender_terms)

    filtered_out_mask = attr_present.copy()
    filtered_out_mask.loc[attr_present] = (~has_noun) | is_bare_noun | has_gender_terms

    df_filtered_out = df_expanded[filtered_out_mask].copy()
    df_expanded = df_expanded[~filtered_out_mask].copy()

    if len(df_filtered_out) > 0:
        df_filtered_out["source_novel"] = source_novel
        df_filtered_out["output_file_path"] = output_file_path
        df_filtered_out.to_csv(
            filtered_out_path,
            mode="a",
            header=not os.path.exists(filtered_out_path),
            index=False
        )

    print(
        f"Postprocess: filtered out {len(df_filtered_out)} rows "
        f"(no noun, bare body-part noun, or contains gender terms); added to {filtered_out_path}"
    )
    return df_expanded

def gpt_extract(sentence: str, retries: int = 3) -> PhysAttrExtraction:
    for attempt in range(retries):
        try:
            completion = client.beta.chat.completions.parse(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            prompt_template
                        ),
                    },
                    {"role": "user", "content": 
                    f'''<text>\n{sentence}\n</text>\n['''},
                ],
                response_format=PhysAttrExtraction,
            )
            return completion.choices[0].message.parsed
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(5 * (attempt + 1))
            else:
                raise e

def gpt_extract_with_index(sentence_idx: Tuple[int, str]) -> Tuple[int, Optional[PhysAttrExtraction], Optional[str]]:
    """Extract with index for batch processing"""
    idx, sentence = sentence_idx
    try:
        result = gpt_extract(sentence)
        return idx, result, None
    except Exception as e:
        error_msg = f"{type(e).__name__}: {e}"
        print(f"\nSkipping sentence {idx} due to GPT error: {error_msg}")
        print(f"Problematic sentence {idx}: {sentence}")
        return idx, None, error_msg

# =========================
# MAIN PROCESSING LOOP
# =========================

if SOURCE_FILE_NAMES:
    source_file_names_to_process = SOURCE_FILE_NAMES
else:
    try:
        source_file_names_to_process = sorted(
            os.path.splitext(file_name)[0]
            for file_name in os.listdir(args.input_dir)
            if file_name.lower().endswith(".txt")
        )
        print(
            f"SOURCE_FILE_NAMES is empty; discovered {len(source_file_names_to_process)} .txt files in {args.input_dir}"
        )
    except FileNotFoundError:
        print(f"ERROR: input directory not found: {args.input_dir}")
        source_file_names_to_process = []

for SOURCE_FILE_NAME in source_file_names_to_process:
    print("\n" + "="*80)
    print(f"Processing: {SOURCE_FILE_NAME}")
    print("="*80)
    
    SOURCE_FILE_PATH = os.path.join(args.input_dir, SOURCE_FILE_NAME + ".txt")
    OUTPUT_FILE_PATH = os.path.join(args.output_dir, SOURCE_FILE_NAME + "_physattr.csv")
    # create the output directory if it doesn't exist
    os.makedirs(os.path.dirname(OUTPUT_FILE_PATH), exist_ok=True)

    # If postprocess-only mode and we already have an output CSV, reuse it.
    if args.postprocess and os.path.exists(OUTPUT_FILE_PATH):
        print(f"Postprocess-only: loading existing output {OUTPUT_FILE_PATH} (skipping GPT)")
        df_existing = pd.read_csv(OUTPUT_FILE_PATH)
        df_existing = postprocess_df_expanded(
            df_existing,
            source_novel=SOURCE_FILE_NAME,
            output_file_path=OUTPUT_FILE_PATH,
        )
        print("Saving postprocessed results to CSV...")
        df_existing.to_csv(OUTPUT_FILE_PATH, index=False)
        print(f"✅ Saved {len(df_existing)} rows to {OUTPUT_FILE_PATH}")
        continue
    
    # =========================
    # LOAD TEXT
    # =========================
    
    print("Loading text file...")
    try:
        # Try different encodings in order of likelihood
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
        text = None
        for encoding in encodings:
            try:
                with open(SOURCE_FILE_PATH, "r", encoding=encoding) as f:
                    text = f.read().replace("\n", " ")
                print(f"Loaded text file with {encoding} encoding, length: {len(text)} characters")
                break
            except UnicodeDecodeError:
                continue
        if text is None:
            raise ValueError(f"Could not decode file with any of the attempted encodings: {encodings}")
    except FileNotFoundError:
        print(f"ERROR: {SOURCE_FILE_PATH} not found! Skipping...")
        continue
    except Exception as e:
        print(f"ERROR loading file: {e}. Skipping...")
        continue
    
    # Remove chapter headers (robust)
    text = re.sub(r"CHAPTER\s+[A-Z]+\s+[A-Z ]+", " ", text)
    
    print("Tokenizing sentences...")
    sentences = sent_tokenize(text)
    print(f"Found {len(sentences)} sentences")
    
    # get only the first xxx sentences
    if MAX_SENTENCES is not None:
        sentences = sentences[:MAX_SENTENCES]
    else:
        sentences = sentences
    
    # =========================
    # RUN FILTER
    # =========================
    df = pd.DataFrame({"text": sentences})
    
    # we don't use the filter function to produce attributes or gener label anymore
    # df["contains_physattr"], df["physattr"], df["gender_est"] = extract_phys_attrs(sentences)
    df["contains_physattr"], _, df["intext_gender"] = extract_phys_attrs(sentences)
    
    df_phys = df[df["contains_physattr"]].reset_index(drop=True)
    
    print(f"With physical attributes/Total sentences for extraction: {len(df_phys)}/{len(df)}")
    
    # =========================
    # RUN GPT (BATCH PROCESSING)
    # =========================
    if len(df_phys) == 0:
        print("WARNING: No sentences with physical attributes found. Skipping this file.")
        continue
    
    print(f"Starting GPT extraction for {len(df_phys)} sentences...")
    print(f"Using batch processing with concurrent requests...")
    
    # Prepare sentences with indices for batch processing
    sentence_batch = list(enumerate(df_phys["text"]))
    
    # Initialize result lists with None to maintain order
    gpt_outputs = [None] * len(df_phys)
    phys_attrs = [None] * len(df_phys)
    skipped_sentences = []
    
    # Batch processing with ThreadPoolExecutor
    max_workers = 10  # Adjust based on rate limits
    completed = 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_idx = {
            executor.submit(gpt_extract_with_index, (idx, sent)): idx 
            for idx, sent in sentence_batch
        }
        
        # Process completed tasks
        for future in as_completed(future_to_idx):
            idx, result, error_msg = future.result()
            if error_msg is not None:
                gpt_outputs[idx] = None
                phys_attrs[idx] = []
                skipped_sentences.append(
                    {
                        "sentence_idx": idx,
                        "text": df_phys.iloc[idx]["text"],
                        "error": error_msg,
                    }
                )
            else:
                gpt_outputs[idx] = result
                # Handle empty attributes - ensure it's always a list
                attrs = result.physical_attributes if result.physical_attributes is not None else []
                phys_attrs[idx] = attrs

            completed += 1
            print(f"Processed {completed}/{len(df_phys)} sentences...", end="\r")
    
    print(f"\nCompleted GPT extraction for {len(df_phys)} sentences.")
    if len(skipped_sentences) > 0:
        skipped_output_path = os.path.join(args.output_dir, SOURCE_FILE_NAME + "_skipped_sentences.csv")
        pd.DataFrame(skipped_sentences).to_csv(skipped_output_path, index=False)
        print(f"Saved {len(skipped_sentences)} skipped sentences to {skipped_output_path}")

    # Expand the results: one row per attribute
    expanded_rows = []
    text_id = 0
    attribute_text_id = 0
    for idx, row in df_phys.iterrows():
        sentence = row["text"]
        intext_gender = row["intext_gender"]
        
        attrs = phys_attrs[idx]
        llm_response = gpt_outputs[idx]
        
        if len(attrs) == 0:
            # Keep sentences with no attributes as single rows
            expanded_rows.append({
                "text_id": text_id,
                "attribute_text_id": None,
                "text": sentence,
                "attribute": None,
                "intext_gender": intext_gender,
                "llm_gender": None,
                "llm_reasoning": None,
                "gender_match": False,
                "indomain_strct": False,
                "llm_response": str(llm_response) if llm_response else None
            })
            text_id += 1
        else:
            # Create one row per attribute
            for attr in attrs:
                expanded_rows.append({
                    "text_id": int(text_id),
                    "attribute_text_id": int(attribute_text_id),
                    "text": sentence,
                    "attribute": attr.attribute,
                    "intext_gender": intext_gender,
                    "llm_gender": attr.gender,
                    "llm_reasoning": attr.reasoning,
                    "gender_match": False,  # Will be computed below
                    "indomain_strct": check_indomain_structure(attr.attribute),
                    "llm_response": str(llm_response) if llm_response else None
                })
            text_id += 1
            attribute_text_id += 1
    
    df_expanded = pd.DataFrame(expanded_rows)
    
    # Reorder columns
    df_expanded = df_expanded[["text_id", "attribute_text_id", "text", "attribute", "intext_gender", "llm_gender", "llm_reasoning", "gender_match", "indomain_strct", "llm_response"]]

    # Function to normalize gender values for comparison
    def normalize_gender(gender):
        if pd.isna(gender) or gender is None:
            return None
        gender_str = str(gender).lower().strip()
        if gender_str in ["male", "man"]:
            return "male"
        elif gender_str in ["female", "woman"]:
            return "female"
        elif gender_str in ["non-binary", "evidence for both"]:
            return "non-binary"
        elif gender_str in ["unknown", "none found", "na", "n/a", ""]:
            return "unknown"
        else:
            return gender_str

    # Compare intext_gender and llm_gender
    def genders_match(intext, llm):
        intext_norm = normalize_gender(intext)
        llm_norm = normalize_gender(llm)
        if intext_norm is None or llm_norm is None:
            return False
        return intext_norm == llm_norm

    df_expanded["gender_match"] = df_expanded.apply(
        lambda row: genders_match(row["intext_gender"], row["llm_gender"]), axis=1
    )

    # Count sentences with empty attributes
    empty_attr_count = sum(1 for attrs in phys_attrs if len(attrs) == 0)
    print(f"Statistics: {empty_attr_count} sentences with empty attributes out of {len(df_phys)} total")
    
    # Count non-null attributes
    non_null_attrs = df_expanded["attribute"].notna().sum()
    print(f"Statistics: {non_null_attrs} total attributes extracted")
    
    # Count gender matches
    match_count = df_expanded["gender_match"].sum()
    print(f"Statistics: {match_count} attributes with matching genders out of {non_null_attrs} total")
    
    # Count in-domain structure matches
    indomain_count = df_expanded["indomain_strct"].sum()
    print(f"Statistics: {indomain_count} attributes with in-domain structure out of {non_null_attrs} total")

    # =========================
    # POSTPROCESS
    # =========================
    df_expanded = postprocess_df_expanded(
        df_expanded,
        source_novel=SOURCE_FILE_NAME,
        output_file_path=OUTPUT_FILE_PATH,
    )

    # =========================
    # SAVE
    # =========================
    print("Saving results to CSV...")
    df_expanded.to_csv(OUTPUT_FILE_PATH, index=False)
    print(f"✅ Saved {len(df_expanded)} rows to {OUTPUT_FILE_PATH}")

print("\n" + "="*80)
print("All files processed successfully!")
print("="*80)
