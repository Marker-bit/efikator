from utils import cut_context, encode_context, alphabet, CONTEXT_SIZE
from models import SimpleNN  # noqa
import torch
import argparse
import re
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("input", type=str, help="input file")
parser.add_argument("output", type=str, help="output file")
parser.add_argument(
    "--model", type=str, default="model.pth", help="Path to the model file"
)
args = parser.parse_args()

model = torch.load(args.model, weights_only=False, map_location=torch.device("cpu"))

with open(args.input, "r", encoding="utf-8") as f:
    text = f.read()

new_text = list(text)

for word in re.finditer(r"[а-яё]+", text, re.IGNORECASE):
    s, e = word.span()
    word = word.group()
    new_contexts = cut_context(CONTEXT_SIZE, word.lower())  # Example new context
    for c in new_contexts:
        encoded_new = encode_context(c[0], alphabet)
        encoded_new = torch.tensor(encoded_new, dtype=torch.float32).view(1, -1)
        prediction = model(encoded_new)
        tqdm.write(
            f"Probability of replacing 'е' with 'ё' in '{word}' on index {c[1]}: {prediction.item():.4f}"
        )
        if prediction.item() > 0.5:
            tqdm.write(f"Replacing 'е' with 'ё' in '{word}' on index {c[1]}")
            up = word[c[1]].upper() == word[c[1]]
            new_text[s + c[1]] = "Ё" if up else "ё"
            new_w = word[: c[1]] + "Ё" if up else "ё" + word[c[1] + 1 :]
            tqdm.write(f"New word: {new_w}")

with open(args.output, "w", encoding="utf-8") as f:
    f.write("".join(new_text))
    


# tests = [
#     "перегноёнными",
#     "дрель",
#     "теремок",
#     "амёба",
#     "берёза",
#     "математика",
#     "ёжик",
#     "лён",
#     "крысёнок",
#     "неё",
#     "ковёр",
# ]
# for i in tests:
#     new_contexts = cut_context(CONTEXT_SIZE, i)  # Example new context
#     for c in new_contexts:
#         encoded_new = encode_context(c[0], alphabet)
#         encoded_new = torch.tensor(encoded_new, dtype=torch.float32).view(1, -1)
#         prediction = model(encoded_new)
#         print(
#             f"Probability of replacing 'е' with 'ё' in '{i}'({c[0]}): {prediction.item():.4f}"
#         )
