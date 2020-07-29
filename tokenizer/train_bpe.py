from tokenizers import SentencePieceBPETokenizer

from utils import dataset_config, root_dir, tokenizer_config

tokenizer = SentencePieceBPETokenizer()
tokenizer.train(
    [
        f"{root_dir}/{dataset_config.raw.source_train}",
        f"{root_dir}/{dataset_config.raw.target_train}",
    ],
    vocab_size=tokenizer_config.vocab_size,
    min_frequency=tokenizer_config.min_frequency,
    special_tokens=list(tokenizer_config.special_tokens),
    limit_alphabet=tokenizer_config.limit_alphabet,
)
tokenizer.save_model(
    directory=f"{root_dir}/tokenizer", name=tokenizer_config.tokenizer_name
)
