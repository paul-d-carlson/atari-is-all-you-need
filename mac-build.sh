#!/bin/sh
set -e
clang -std=c11 -O2 -Wall -Wextra -Wpedantic -o mac_ai mac-ai.c
echo "Built: mac_ai"