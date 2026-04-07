#!/bin/bash

MODEL_NAME="mistralai/Mistral-7B-v0.3"
METHOD="ssn"   # ssn or ddf
MODE="latter"  # latter or whole

for MEASURE in entropy confidence_score; do
    echo "▶ Running method=$METHOD measure=$MEASURE"

    python3 main.py \
        --model_name "$MODEL_NAME" \
        --method "$METHOD" \
        --mode "$MODE" \
        --measure "$MEASURE"

    echo "✅ Finished $MEASURE"
done
