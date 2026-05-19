#!/bin/sh
set -e
cl65 -t atari -O -Cl --standard c99 -o atari-ai.xex atari-ai.c
echo "Built: atari-ai.xex"