"""Topic identification: predefined theme keywords + data-driven term discovery."""
import math
import re

from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS

from sentiment import get_analyzer

# Regex fragments per theme (matched on whole words, case-insensitive)
TOPIC_PATTERNS = {
    "Teaching & Faculty": r"teach\w*|faculty|professor\w*|lectur\w*|instructor\w*|mentor\w*|syllabus|tutor\w*|doubts?",
    "Infrastructure": r"infrastructure|building\w*|classrooms?|campus|hostel\w*|furniture|benches|bench|projector\w*|air ?condition\w*|ac|fans?|washrooms?|toilets?|restrooms?|parking|ventilation|electricity|power cuts?|lifts?|elevators?|roads?",
    "Laboratories": r"labs?|laborator(?:y|ies)|experiments?|practicals?|apparatus|workshops?|hands-on",
    "Administration": r"admin\w*|office|fees?|scholarships?|registration|admissions?|clerks?|paperwork|bureaucra\w*|certificates?|documents?|principal|dean|hod|notices?|timetables?|schedul\w*|grievances?|counsel\w*",
    "Library": r"librar\w*|books?|journals?|reading rooms?|e-?books?|study rooms?",
    "Internet & Wi-Fi": r"wi-?fi|internet|network\w*|bandwidth|broadband|connectivity|routers?|signal",
    "Canteen": r"canteens?|cafeterias?|food|mess|meals?|snacks?|lunch|breakfast|hygien\w*|unhygien\w*|drinking water|tea|coffee|menu|tiffin",
    "Examinations": r"exams?|examinations?|tests?|testing|assessments?|evaluations?|marks?|grad(?:e|es|ing)|results?|question papers?|papers?|internals?|viva|assignments?|invigilat\w*|re-?evaluation",
}
_COMPILED = {
    name: re.compile(r"\b(?:" + pattern + r")\b", re.IGNORECASE)
    for name, pattern in TOPIC_PATTERNS.items()
}

# Generic words that should never become "emerging topics"
_EXTRA_STOP = {
    "college", "university", "students", "student", "feedback", "need", "needs", "should",
    "would", "could", "like", "also", "much", "more", "many", "better", "best", "worst",
    "please", "every", "always", "never", "sometimes", "time", "times", "make", "made",
    "well", "just", "thing", "things", "even", "still", "provide", "provided", "improve",
    "improved", "improvement", "quality", "facility", "facilities", "class", "classes",
    "course", "courses", "overall", "experience", "average", "nothing", "special",
    "really", "very", "good", "great", "bad", "poor", "nice", "regularly", "rarely",
}
_STOP = set(ENGLISH_STOP_WORDS) | _EXTRA_STOP


def discover_emerging_terms(texts, n_terms=3, min_docs=3):
    """Find frequent words (document frequency) not covered by predefined themes."""
    texts = [t for t in texts if isinstance(t, str) and t.strip()]
    if len(texts) < 8:               # too little data to call anything "recurring"
        return []
    threshold = max(min_docs, math.ceil(0.10 * len(texts)))
    try:
        vectorizer = CountVectorizer(
            token_pattern=r"(?u)\b[a-zA-Z]{4,}\b", binary=True, lowercase=True
        )
        matrix = vectorizer.fit_transform(texts)
    except ValueError:               # empty vocabulary
        return []

    vocab = vectorizer.get_feature_names_out()
    doc_freq = matrix.sum(axis=0).A1
    sentiment_words = set(get_analyzer().lexicon)

    candidates = []
    for term, count in zip(vocab, doc_freq):
        if count < threshold or term in _STOP or term in sentiment_words:
            continue
        if any(rx.fullmatch(term) for rx in _COMPILED.values()):
            continue                 # already covered by a predefined theme
        candidates.append((term, int(count)))
    candidates.sort(key=lambda x: (-x[1], x[0]))
    return [term for term, _ in candidates[:n_terms]]


def extract_topics(texts, n_emerging=3):
    """Return (list_of_topic_lists, emerging_terms). A text can have several topics."""
    emerging = discover_emerging_terms(texts, n_terms=n_emerging)
    results = []
    for text in texts:
        found = [name for name, rx in _COMPILED.items() if rx.search(text)]
        for term in emerging:
            if re.search(rf"\b{re.escape(term)}\b", text, re.IGNORECASE):
                found.append(f"Emerging: {term.title()}")
        results.append(found or ["General"])
    return results, emerging