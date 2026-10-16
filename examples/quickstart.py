"""Offline quickstart with the bundled pretrained model; nothing is downloaded."""

from banglatok import Tokenizer

texts = ["আমি বাংলায় গান গাই।", "ami banglay gaan gai", "I speak Bangla", "meetingটা ১০টায় 🙂"]

with Tokenizer.from_pretrained() as tokenizer:
    print(tokenizer)
    for text, ids in zip(texts, tokenizer.encode_batch(texts), strict=True):
        assert tokenizer.decode(ids) == text
        print(f"{text!r}: {len(ids)} tokens {tokenizer.tokenize(text)}")
    batch = tokenizer(texts, max_length=16, padding=True)
    print("encoder input shape:", len(batch["input_ids"]), "x", len(batch["input_ids"][0]))
