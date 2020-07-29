from tokenizers import SentencePieceBPETokenizer
from tqdm.auto import tqdm

from utils import dataset_config, read_lines, root_dir, tokenizer_config

source_train = read_lines(f"{root_dir}/{dataset_config.raw.source_train}")
target_train = read_lines(f"{root_dir}/{dataset_config.raw.target_train}")
assert len(source_train) == len(target_train)

tokenizer = SentencePieceBPETokenizer(
    f"{tokenizer_config.tokenizer_name}-vocab.json",
    f"{tokenizer_config.tokenizer_name}-merges.txt",
)

results = tokenizer.encode_batch(source_train)
with open(f"{root_dir}/{dataset_config.tokenized.source_train}", "w") as f:
    for line in tqdm(results, desc="source"):
        f.write(" ".join(line.tokens) + "\n")

results = tokenizer.encode_batch(target_train)
with open(f"{root_dir}/{dataset_config.tokenized.target_train}", "w") as f:
    for line in tqdm(results, desc="target"):
        f.write(" ".join(line.tokens) + "\n")
