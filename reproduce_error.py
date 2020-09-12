from pathlib import Path

from omegaconf import OmegaConf

config_dir = Path('configs')
dataset_config_path = config_dir / 'dataset' / 'wmt14.en-de.omegaconf-bug.yaml'
tokenizer_config_path = config_dir / 'tokenizer' / 'sentencepiece_bpe_wmt14_en-de.yaml'

configs = OmegaConf.create()
configs.update(dataset=dataset_config_path, tokenizer=tokenizer_config_path)


def read_lines(path):
    with open(path, 'r', encoding='utf-8') as f:
        lines = [line.rstrip('\n') for line in f]
    return lines


def write_lines(lines, path):
    with open(path, 'w', encoding='utf-8') as f:
        for line in lines:
            f.write(line + '\n')


source_train = read_lines(configs.dataset.path.source_train)
write_lines(source_train, 'source_train.txt')
