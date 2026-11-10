"""
Small built-in vocabulary used for (a) word auto-complete while fingerspelling and
(b) an OFFLINE English -> Hindi / Marathi glossary.

The glossary is hand-written and intentionally small (everyday, health and school words).
Anything not in it is translated online (see translate.py). Have a native speaker review it
before using it in a real deployment.
"""

# Suggestions are matched on the first letters typed. Keep words lowercase.
COMMON_WORDS = """
a about after again all also am and any are ask at bad be because bed big book boy bring bus but buy
by call can cold come cook cry day deaf dinner do doctor doing door down drink eat emergency end evening
family far father fast fever find fine food for friend from girl give go good goodbye happy hard have he
hear hello help her here home hospital hot house how hungry i if in is it job kind know learn left
like listen little look love lunch make man medicine meet money morning mother my name need new night
no not now of off ok old on one open our out pain pen please police read ready rest right run sad
school see she sick sign sit sleep slow small sorry speak stop student study sugar sun take talk teacher
tell thank thanks that the their there they think this time tired to today tomorrow train understand
up us very wait walk want water we welcome well what when where which who why will with work write
yes yesterday you your
""".split()

# English -> (Hindi, Marathi)
GLOSSARY = {
    "hello": ("नमस्ते", "नमस्कार"), "thank you": ("धन्यवाद", "धन्यवाद"), "thanks": ("धन्यवाद", "धन्यवाद"),
    "please": ("कृपया", "कृपया"), "sorry": ("माफ़ कीजिए", "माफ करा"), "welcome": ("स्वागत है", "स्वागत आहे"),
    "goodbye": ("अलविदा", "निरोप"), "yes": ("हाँ", "हो"), "no": ("नहीं", "नाही"), "ok": ("ठीक है", "ठीक आहे"),
    "help": ("मदद", "मदत"), "water": ("पानी", "पाणी"), "food": ("खाना", "जेवण"), "eat": ("खाओ", "खा"),
    "drink": ("पियो", "प्या"), "sleep": ("सोना", "झोपणे"), "stop": ("रुको", "थांबा"), "come": ("आओ", "ये"),
    "go": ("जाओ", "जा"), "wait": ("रुको", "थांबा"), "need": ("चाहिए", "हवे"), "work": ("काम", "काम"),
    "doctor": ("डॉक्टर", "डॉक्टर"), "hospital": ("अस्पताल", "रुग्णालय"), "medicine": ("दवा", "औषध"),
    "pain": ("दर्द", "वेदना"), "sick": ("बीमार", "आजारी"), "fever": ("बुखार", "ताप"),
    "emergency": ("आपातकाल", "आणीबाणी"), "police": ("पुलिस", "पोलीस"),
    "school": ("स्कूल", "शाळा"), "teacher": ("शिक्षक", "शिक्षक"), "student": ("विद्यार्थी", "विद्यार्थी"),
    "book": ("किताब", "पुस्तक"), "pen": ("कलम", "पेन"), "friend": ("दोस्त", "मित्र"),
    "mother": ("माँ", "आई"), "father": ("पिता", "वडील"), "brother": ("भाई", "भाऊ"), "sister": ("बहन", "बहीण"),
    "family": ("परिवार", "कुटुंब"), "home": ("घर", "घर"), "house": ("घर", "घर"), "name": ("नाम", "नाव"),
    "good": ("अच्छा", "चांगला"), "bad": ("बुरा", "वाईट"), "big": ("बड़ा", "मोठा"), "small": ("छोटा", "लहान"),
    "hot": ("गर्म", "गरम"), "cold": ("ठंडा", "थंड"), "happy": ("खुश", "आनंदी"), "sad": ("दुखी", "दुःखी"),
    "morning": ("सुबह", "सकाळ"), "night": ("रात", "रात्र"), "today": ("आज", "आज"), "tomorrow": ("कल", "उद्या"),
    "yesterday": ("कल", "काल"), "time": ("समय", "वेळ"), "money": ("पैसा", "पैसे"), "love": ("प्यार", "प्रेम"),
    "bus": ("बस", "बस"), "train": ("रेलगाड़ी", "रेल्वे"),
    "who": ("कौन", "कोण"), "what": ("क्या", "काय"), "where": ("कहाँ", "कुठे"), "when": ("कब", "केव्हा"),
    "why": ("क्यों", "का"), "how": ("कैसे", "कसे"), "i": ("मैं", "मी"), "you": ("आप", "तुम्ही"),
    "we": ("हम", "आम्ही"), "deaf": ("बधिर", "कर्णबधिर"),
}

# Letter names written in Devanagari (used when a single letter is spoken in Hindi/Marathi)
LETTER_NAMES = {
    "A": "ए", "B": "बी", "C": "सी", "D": "डी", "E": "ई", "F": "एफ़", "G": "जी", "H": "एच", "I": "आई",
    "J": "जे", "K": "के", "L": "एल", "M": "एम", "N": "एन", "O": "ओ", "P": "पी", "Q": "क्यू", "R": "आर",
    "S": "एस", "T": "टी", "U": "यू", "V": "वी", "W": "डब्ल्यू", "X": "एक्स", "Y": "वाई", "Z": "ज़ेड",
}


def suggest(prefix: str, limit: int = 5) -> list:
    """Return up to `limit` common words that start with `prefix` (case-insensitive, shortest first)."""
    p = prefix.strip().lower()
    if len(p) < 2:
        return []
    hits = [w for w in COMMON_WORDS if w.startswith(p) and w != p]
    return sorted(set(hits), key=lambda w: (len(w), w))[:limit]
