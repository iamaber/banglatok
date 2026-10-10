"""Offline API demo with a tiny hand-authored vocabulary, not a trained model."""

from importlib.resources import as_file, files

from bn_tokenizers_embedding import Tokenizer

demo = files("bn_tokenizers_embedding").joinpath("data/demo-unigram.json")
with as_file(demo) as model, Tokenizer(model) as tokenizer:
    texts = ["আমি বাংলা বলি।", "ami bangla boli", "I speak Bangla", "meetingটা 🙂"]
    print("Toy vocabulary: API demonstration only; not a trained tokenizer benchmark.")
    for text, ids in zip(texts, tokenizer.encode_batch(texts)):
        print(f"{text!r} -> {ids} -> {tokenizer.decode(ids)!r}")
