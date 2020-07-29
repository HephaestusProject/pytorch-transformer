#!/bin/bash
#
# Download WMT14 parallel corpus for de-en translation model.

DOWNLOAD_PATH='dataset'

wget https://www.statmt.org/wmt13/training-parallel-europarl-v7.tgz --directory-prefix ${DOWNLOAD_PATH}  # same as previous year
wget https://www.statmt.org/wmt13/training-parallel-commoncrawl.tgz --directory-prefix ${DOWNLOAD_PATH}  # same as previous year
wget https://www.statmt.org/wmt14/training-parallel-nc-v9.tgz --directory-prefix ${DOWNLOAD_PATH}

# Extract corpus
cd ${DOWNLOAD_PATH}
# Extract europarl. This will generate training/europarl-v7.de-en.{de, en}
tar -xzvf training-parallel-europarl-v7.tgz
# Extract news-commentary. This will generate training/news-commentary-v9.de-en.{de, en}
tar -xzvf training-parallel-nc-v9.tgz
# Extract common crawl. This will generate training/commoncraw.de-en.{de, en}
tar -xzvf training-parallel-commoncrawl.tgz --directory training

# Concatenate de
cat training/europarl-v7.de-en.de training/news-commentary-v9.de-en.de training/commoncrawl.de-en.de > wmt14.de-en.train.de
# Concatenate en
cat training/europarl-v7.de-en.en training/news-commentary-v9.de-en.en training/commoncrawl.de-en.en > wmt14.de-en.train.en

# Clear tmp
rm training-parallel-commoncrawl.tgz training-parallel-europarl-v7.tgz training-parallel-nc-v9.tgz
rm -r training
