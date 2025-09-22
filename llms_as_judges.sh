python src/verify_edits.py \
    --input_dir reviseqa_data/nl \
    --model_names x-ai/grok-code-fast-1 \
    openai/o1-mini \
    google/gemini-2.5-flash \
    qwen/qwen-plus-2025-07-28 \
    openai/gpt-5-mini \
    --batch_size 8
