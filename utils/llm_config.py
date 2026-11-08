"""
llm_config.py

Configuration, taxonomies, and output-naming constants for the
emotion annotation pipeline.

Usage
-----
    import llm_config as cfg
    print(cfg.MODEL_NAME, cfg.CARTESIAN_TAXONOMY)

or import specific names:

    from llm_config import MODEL_NAME, CARTESIAN_TAXONOMY, EMOTION_SCHEMES
"""

# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_NAME = "phi4"   # change to "phi4" or "gemma3:12b"
OLLAMA_MODEL = MODEL_NAME


# Runtime / determinism
TAXONOMY_MODEL_OPTIONS = {
    "temperature": 0,
    "seed": 42,
    "num_predict": 8192,  # llama3.1:8b is verbose
}

ANNOTATION_MODEL_OPTIONS = {
    "temperature": 0,
    "seed": 42,
    "num_predict": 512,   # annotation response is just 3 fields
}

AGNOSTIC_LABEL_CAP = {
    "phi4": 999,
    "gemma3:12b": 999,
}

MIN_LARGETAX_SUPPORT = 1     # No prunning, just removing labels with 0 assignment
TEXT_LEN = 250               # truncate aria texts to keep prompts stable

FAST_MODE = False
if FAST_MODE:
    MAX_RETRIES = 1
    N_FAST_MODE = 5

DEV_MODE = False
if DEV_MODE:
    N_INIT = 5
    MAX_RETRIES = 1

# ============================================================================
# TAXONOMIES
# ============================================================================

TAXONOMY_6 = ["admiration", "anger", "fear", "joy", "love", "sadness"]
BANNED_EMOTIONS = set(TAXONOMY_6)  # prevent agnostic collapse to basic emotions

# Your dataframe has exactly one human scheme column
EMOTION_SCHEMES = ["AriEmozione"]

CARTESIAN_TAXONOMY = [
    'love/desire',
    'hope/sadness',
    'remorse',
    'sadness/contempt',
    'irresolution/despair',
    'humility',
    'contempt',
    'confidence/friendship',
    'disdain/hatred',
    'sadness',
    'love/jealousy',
    'boldness',
    'admiration/love',
    'desire',
    'hope/confidence',
    'courage',
    'anger',
    'sadness/humility',
    'shame/joy',
    'disdain/love',
    'generosity',
    'boldness/disdain',
    'love/esteem',
    'repentance',
    'love/contempt/sadness',
    'irresolution',
    'fear/love',
    'contempt/pity',
    'pride',
    'disdain/hope',
    'sadness/shame',
    'anger/pity',
    'repulsion',
    'admiration',
    'esteem',
    'abjectness',
    'regret',
    'confidence',
    'disdain/fear',
    'repentance/shame',
    'love/pity',
    'veneration/boldness',
    'envy',
    'veneration/contempt',
    'disdain',
    'love/remorse',
    'veneration',
    'irresolution/fear',
    'fear/hope',
    'horror',
    'love',
    'distaste',
    'sadness/love',
    'hatred/boldness',
    'sadness/repulsion',
    'fear',
    'pity/generosity',
    'disdain/jealousy',
    'contempt/aversion',
    'glory',
    'indignation/sadness',
    'irresolution/despair/sadness',
    'hope/love',
    'hope',
    'humility/abjectness',
    'hatred',
    'hatred/pity',
    'joy',
    'abjectness/courage',
    'despair/sadness',
    'anger/love',
    'sadness/repentance',
    'pride/abjectness',
    'astonishment',
    'pride/boldness',
    'indignation/hatred',
    'despair/anger',
    'devotion',
    'abjectness/repentance',
    'despair',
    'love/sadness',
    'indignation',
    'jealousy',
    'self-satisfaction',
    'shame',
    'sadness/irresolution',
    'love/hope',
    'gratitude',
    'desire/abjectness',
    'pity',
    'anger/contempt',
    'mockery',
]






# ============================================================================
# LOGGING
# ============================================================================

def log(msg):
    print(msg, flush=True)


# ============================================================================
# OUTPUT NAMING NORMALIZATION
# ============================================================================

# Based or Inspired on Annotators
COND_SMALL = "LLM-"
COND_AGNOSTIC_SMALL = "LLM-Prop_Small"

# Agnostic of Annotators, only based on the Descartes taxonomy
COND_LARGE = "LLM-Descartes"
COND_AGNOSTIC_LARGE = "LLM-Prop_Large"

PER_HUMAN_CONDITIONS = {COND_SMALL, COND_AGNOSTIC_SMALL}
