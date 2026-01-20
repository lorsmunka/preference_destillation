# Preference Distillation

University thesis project. Proof of concept: distill domain-specific knowledge from large models (Gemma 3 4B) into small transformers using reduced vocabulary.

Classification could be done more easily with other model types (BERT, simple classifiers), but autoregressive generation is the point. This architecture proves the technique works with available resources. Different architectures will be explored later.

## Purpose

Demonstrate that task-specific models with minimal vocabularies can replace expensive LLM calls. Applications: email classification, agentic workflow nodes, any structured output task. Potential 10x cost reduction.

Example extension: a "math model" using only mathematical characters.

## Architecture

- **Teacher**: Gemma 3 4B generates training data with logits
- **Student**: Small transformer (~50M params) with reduced output vocabulary (~400 tokens)
- **Task**: Reddit comment classification (tone, sentiment, safety, toxicity)
- **Output**: Deterministic JSON with fixed schema

## Vocabulary Sections (in order)

1. **Example tokens**: JSON structure and label values
2. **Whitespace tokens**: Formatting variations
3. **Prompt tokens**: Evaluation prompt vocabulary
4. **Auxiliary tokens**: Common English fallback tokens

## Training

- KL divergence loss (match teacher distribution) + Cross-entropy loss (match predicted token)
- Autoregressive generation is intentional for the proof of concept
- Future: mask deterministic tokens at inference, only predict uncertain positions

## Code Style

- Write full variable names, no abbreviations (use `index` not `idx`, `value` not `val`)
- Self-documenting code structure over comments
- Simple and beginner-friendly
- No over-engineering
