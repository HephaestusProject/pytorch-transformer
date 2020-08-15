import os

from omegaconf import OmegaConf

root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../")
config_dir = f"{root_dir}/configs"

dataset_config = OmegaConf.load(f"{config_dir}/dataset/wmt14.de-en.yaml")
tokenizer_config = OmegaConf.load(
    f"{config_dir}/tokenizer/sentencepiece_bpe_wmt14_deen.yaml"
)


def read_lines(path):
    with open(path, encoding="utf-8") as f:
        return [line.rstrip() for line in f.read().rstrip().split('\n')]
