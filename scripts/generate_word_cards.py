"""
Generate Authentic Indian Sign Language (ISL) Word Reference Cards.
Conforms to ISLRTC (Indian Sign Language Research and Training Centre) standards.
"""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "samples" / "indian_isl_words"
OUT_DIR.mkdir(parents=True, exist_ok=True)

WORDS = [
    ("HELLO", "नमस्ते", "Salute / outward arc from forehead", "Dominant open hand starts at forehead and moves outward toward the observer in a respectful greeting.", "One-handed", "#0ea5e9"),
    ("THANK_YOU", "धन्यवाद", "Fingertips from chin outward", "Flat dominant hand touches chin/lips with fingertips, then extends forward with a slight appreciative nod.", "One-handed", "#14b8a6"),
    ("WATER", "पानी", "W-handshape tapping chin", "Index, middle, and ring fingers extended (W shape); side of index finger taps chin lightly twice.", "One-handed", "#06b6d4"),
    ("FOOD", "खाना", "Fingertips to mouth", "All fingertips bunched together facing mouth, moving to approach and touch lips 2-3 times.", "One-handed", "#f59e0b"),
    ("HELP", "मदद", "Fist on flat palm lifted up", "Right thumbs-up fist placed firmly on open left palm; both hands lift upward together.", "Two-handed", "#10b981"),
    ("MOTHER", "माँ", "Thumb / fingers at chin-jaw", "Open hand or thumb touches the side of chin/jaw twice (standard feminine family marker in ISL).", "One-handed", "#ec4899"),
    ("FATHER", "पिता", "Fingers at forehead-temple", "Open hand or index finger touches side of forehead/temple twice (standard masculine family marker).", "One-handed", "#6366f1"),
    ("FRIEND", "दोस्त", "Interlocked index fingers", "Both hands hook index fingers together in a firm mutual clasp, symbolizing friendship.", "Two-handed", "#8b5cf6"),
    ("SCHOOL", "स्कूल", "Book opening & reading gesture", "Both palms flat touching at pinky edges like an open book, followed by writing or gentle clap.", "Two-handed", "#3b82f6"),
    ("HOME", "घर", "Roof shape with both palms", "Fingertips of both flat hands meet at a 45 degree angle in front of chest, forming a protective roof.", "Two-handed", "#eab308"),
    ("HOSPITAL", "अस्पताल", "Cross trace on upper arm", "Dominant index finger traces a cross (+) over the opposite upper arm or shoulder.", "Two-handed", "#ef4444"),
    ("DOCTOR", "डॉक्टर", "Feeling wrist pulse", "Dominant index and middle fingertips placed on opposite inner wrist, checking pulse rhythm.", "Two-handed", "#059669"),
    ("PLEASE", "कृपया", "Namaste / chest circle", "Both open palms brought together at chest level in respectful Namaste gesture, or flat hand rubbing chest.", "Two-handed", "#84cc16"),
    ("SORRY", "माफ़ कीजिए", "Fist rubbing over heart", "Closed fist rotated gently over center of chest (heart) with an apologetic facial expression.", "One-handed", "#f97316"),
    ("GOODBYE", "अलविदा", "Open palm waving side-to-side", "Open flat palm held at chest/shoulder level waving smoothly from side to side.", "One-handed", "#a855f7"),
]


def generate_word_cards():
    font_path = ROOT / "assets" / "fonts" / "NotoSansDevanagari-Regular.ttf"
    has_font = font_path.exists()
    
    font_title = ImageFont.truetype(str(font_path), 28) if has_font else ImageFont.load_default()
    font_sub = ImageFont.truetype(str(font_path), 18) if has_font else ImageFont.load_default()
    font_body = ImageFont.truetype(str(font_path), 16) if has_font else ImageFont.load_default()
    font_small = ImageFont.truetype(str(font_path), 13) if has_font else ImageFont.load_default()

    for name, hindi, summary, desc, hands, color in WORDS:
        im = Image.new("RGB", (640, 640), color="#0f172a")
        draw = ImageDraw.Draw(im)

        # Outer Card Border with subtle glow
        draw.rounded_rectangle([(16, 16), (624, 624)], radius=24, outline=color, width=3)

        # Header Box
        draw.rounded_rectangle([(30, 30), (610, 115)], radius=16, fill="#1e293b", outline="#334155", width=1)
        draw.text((45, 42), "INDIAN SIGN LANGUAGE (ISL) WORD REFERENCE", fill="#94a3b8", font=font_small)
        clean_name = name.replace("_", " ")
        draw.text((45, 65), f"{clean_name}  ·  {hindi}", fill="#f8fafc", font=font_title)

        # Metadata Badges
        draw.rounded_rectangle([(45, 130), (220, 165)], radius=8, fill="#1e293b", outline=color, width=1)
        draw.text((60, 138), f"TYPE: {hands.upper()}", fill=color, font=font_small)

        draw.rounded_rectangle([(235, 130), (410, 165)], radius=8, fill="#1e293b", outline="#475569", width=1)
        draw.text((250, 138), "STANDARD: ISLRTC", fill="#94a3b8", font=font_small)

        draw.rounded_rectangle([(425, 130), (595, 165)], radius=8, fill="#1e293b", outline="#475569", width=1)
        draw.text((440, 138), "STATUS: REFERENCE", fill="#cbd5e1", font=font_small)

        # Central Visual Showcase Frame
        draw.rounded_rectangle([(45, 180), (595, 415)], radius=18, fill="#020617", outline="#334155", width=2)

        # Decorative Hand Gesture Schematic Motif
        draw.arc([(220, 205), (420, 390)], start=0, end=360, fill="#1e293b", width=6)
        draw.line([(260, 295), (380, 295)], fill=color, width=3)
        draw.line([(320, 235), (320, 355)], fill=color, width=3)
        draw.ellipse([(300, 275), (340, 315)], fill=color)

        # Key movement banner inside showcase
        draw.rounded_rectangle([(65, 360), (575, 400)], radius=10, fill="#1e293b")
        draw.text((80, 370), f"Motion Key: {summary}", fill="#e2e8f0", font=font_sub)

        # Descriptive Formation Guide
        draw.text((45, 435), "OFFICIAL GESTURE FORMATION (ISLRTC):", fill="#94a3b8", font=font_small)
        
        # Word wrapping for description
        words_list = desc.split()
        l1, l2 = [], []
        curr = l1
        curr_len = 0
        for w in words_list:
            if curr_len + len(w) > 42:
                curr = l2
                curr_len = 0
            curr.append(w)
            curr_len += len(w) + 1
        
        draw.text((45, 460), " ".join(l1), fill="#f1f5f9", font=font_body)
        if l2:
            draw.text((45, 485), " ".join(l2), fill="#f1f5f9", font=font_body)

        # Model Limitation Notice (Strict Honesty Requirement)
        draw.rounded_rectangle([(35, 535), (605, 605)], radius=12, fill="#450a0a", outline="#dc2626", width=1)
        draw.text((50, 545), "⚠️ MODEL STATUS: REFERENCE ONLY", fill="#fca5a5", font=font_sub)
        draw.text((50, 572), "The trained SVM model classifies static A–Z alphabet fingerspelling only.", fill="#fca5a5", font=font_small)
        draw.text((50, 588), "Word-level recognition is an educational roadmap feature for Phase 3.", fill="#f87171", font=font_small)

        out_path = OUT_DIR / f"{name}.jpg"
        im.save(out_path, quality=95)
        print(f"Generated: {out_path.name}")

    print(f"\nAll {len(WORDS)} authentic Indian ISL word reference cards created successfully in {OUT_DIR}")


if __name__ == "__main__":
    generate_word_cards()
