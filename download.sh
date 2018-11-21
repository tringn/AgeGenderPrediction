#! /usr/bin/env bash
FILENAME=UTKFace_AsianOnly.tar.gz
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1VKeUymGHYh701vDqvOJvYuYIsI9fAE7S' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1VKeUymGHYh701vDqvOJvYuYIsI9fAE7S" -O $FILENAME && rm -rf /tmp/cookies.txt 
mkdir -p data
tar -xvC data/ -f $FILENAME 
