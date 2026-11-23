"""
Quick start: extract voting information with the released DeBERTa-CRF-VotIE model.

    python scripts/quick_start.py

Replace `text` with your own Portuguese meeting-minutes segment.
"""

from transformers import AutoModel, AutoTokenizer

model_name = "Evans2424/DeBERTa-CRF-VotIE"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name, trust_remote_code=True).eval()

text = ("A proposta foi aprovada por maioria, com os votos a favor dos eleitos do PS "
        "e a abstenção dos vereadores do PSD.")

# Entities with character offsets, grouped by type
for etype, spans in model.extract_entities(text, tokenizer).items():
    for span in spans:
        print(f"{etype:<20} {span['text']!r} ({span['start']}-{span['end']})")

# Word-level BIO labels with confidences
print()
for w in model.decode(tokenizer=tokenizer, text=text):
    print(f"{w['word']:<15} {w['label']:<22} {w['confidence']:.2f}")
