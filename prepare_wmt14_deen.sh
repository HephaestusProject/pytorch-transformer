#!/bin/bash
#
# Download WMT14 parallel corpus for de-en Transformers.
# Note that testset used in "Attention is all you need" is newstest2013 in development corpus.

DOWNLOAD_PATH='dataset'

# 1. Download corpus from https://www.statmt.org/wmt14/translation-task.html

## 1.1 Training corpus
wget https://www.statmt.org/wmt13/training-parallel-europarl-v7.tgz --directory-prefix ${DOWNLOAD_PATH}  # same as previous year
wget https://www.statmt.org/wmt13/training-parallel-commoncrawl.tgz --directory-prefix ${DOWNLOAD_PATH}  # same as previous year
wget https://www.statmt.org/wmt14/training-parallel-nc-v9.tgz --directory-prefix ${DOWNLOAD_PATH}

## 1.2 Validation corpus
wget https://www.statmt.org/wmt14/dev.tgz --directory-prefix ${DOWNLAOD_PATH}


# 2. Extract corpus
cd ${DOWNLOAD_PATH}

## 2.1 Training corpus
### Extract europarl. This will generate training/europarl-v7.de-en.{de, en}
tar -xzvf training-parallel-europarl-v7.tgz
### Extract news-commentary. This will generate training/news-commentary-v9.de-en.{de, en}
tar -xzvf training-parallel-nc-v9.tgz
### Extract common crawl. This will generate training/commoncraw.de-en.{de, en}
tar -xzvf training-parallel-commoncrawl.tgz --directory training

## 2.2 Validation corpus
tar -xzvf dev.tgz


# 3. Processing corpus: Concatenate and Normalize
wget https://www.statmt.org/wmt11/normalize-punctuation.perl
chmod +x normalize-punctuation.perl

## 3.1 Training corpus
cat training/europarl-v7.de-en.de training/news-commentary-v9.de-en.de training/commoncrawl.de-en.de > wmt14.deen.train.de
cat training/europarl-v7.de-en.en training/news-commentary-v9.de-en.en training/commoncrawl.de-en.en > wmt14.deen.train.en
./normalize-punctuation.perl de < wmt14.deen.train.de > wmt14.deen.train.norm.de
./normalize-punctuation.perl en < wmt14.deen.train.en > wmt14.deen.train.norm.en

## 3.2 Validation corpus
cat dev/newssyscomb2009.de dev/news-test2008.de dev/newstest2009.de dev/newstest2010.de dev/newstest2011.de dev/newstest2012.de > wmt14.deen.dev.de
cat dev/newssyscomb2009.en dev/news-test2008.en dev/newstest2009.en dev/newstest2010.en dev/newstest2011.en dev/newstest2012.en > wmt14.deen.dev.en
./normalize-punctuation.perl de < wmt14.deen.dev.de > wmt14.deen.dev.norm.de
./normalize-punctuation.perl en < wmt14.deen.dev.en > wmt14.deen.dev.norm.en

## 3.3 Test corpus
cat dev/newstest2013.de > wmt14.deen.test.de
cat dev/newstest2013.en > wmt14.deen.test.en
./normalize-punctuation.perl de < wmt14.deen.test.de > wmt14.deen.test.norm.de
./normalize-punctuation.perl en < wmt14.deen.test.en > wmt14.deen.test.norm.en


# 4. Remove tgz and directories
rm training-parallel-commoncrawl.tgz training-parallel-europarl-v7.tgz training-parallel-nc-v9.tgz dev.tgz
rm -r training dev
