#!/usr/bin/env bash

source .env

export DYLD_LIBRARY_PATH=/Users/vejmelka/Projects/ai/piper-phonemize/install/lib/
stdbuf -oL ./command -l cs -vth 0.4 -c 0 -m ../whisper.cpp/models/ggml-large-v3-q5_0.bin -pms 150 -p "Myšáku" -cms 15000 2>&1 | python -u src/socrates.py


