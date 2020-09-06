from pathlib import Path

from tokenizers import SentencePieceBPETokenizer
from omegaconf import OmegaConf

root_dir = Path('..')
config_dir = root_dir / 'configs'
dataset_config = OmegaConf.load(config_dir / 'dataset' / 'wmt14.de-en.yaml')
tokenizer_config = OmegaConf.load(config_dir / 'tokenizer' / 'sentencepiece_bpe_wmt14_deen.yaml')


tokenizer = SentencePieceBPETokenizer()
tokenizer.train(
    [
        str(root_dir / dataset_config.path.source_train),
        str(root_dir / dataset_config.path.target_train)
    ],
    vocab_size=tokenizer_config.vocab_size,
    min_frequency=tokenizer_config.min_frequency,
    special_tokens=list(tokenizer_config.special_tokens),
    limit_alphabet=tokenizer_config.limit_alphabet,
)
tokenizer.save_model(
    directory='.', name=tokenizer_config.tokenizer_name
)
