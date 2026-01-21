# Preference Distillation

University thesis proof of concept: distill domain-specific knowledge from Gemma 3 4B into a small transformer using reduced vocabulary. Autoregressive generation is intentional—simpler architectures (BERT, classifiers) could classify, but this proves the technique works for generation.

## Task

Reddit comment classification (tone, sentiment, safety, toxicity) → deterministic JSON output.

## Architecture

- **Teacher**: Gemma 3 4B generates training data with logits
- **Student**: Small transformer with reduced output vocabulary (~525 tokens)
- **Loss**: KL divergence (match teacher distribution) + Cross-entropy (match predicted token), ratio anneals from 0.9→0.1

## Project Structure

```
training/       Model, trainer, entry point (run from here)
shared/         Config, logger, utilities
analysis/       Log visualization, model evaluation
batches/        Training data (JSONL)
logs/           Training logs, state persistence
checkpoints/    Model checkpoints per epoch
```

## Running

```
cd training && python main.py
```

Press `%` for graceful exit (saves temp checkpoint).

## Configuration

All in `shared/config.py`. Key params:
- `EPOCH_COUNT`, `BATCH_SIZE`, `LEARNING_RATE`
- `KL_RATIO_START/END` - Loss blend annealing
- `DISTILLATION_TEMPERATURE` - Softens logits

## Evaluation

Two accuracy types:
- **Teacher-forced**: Ground truth as context (optimistic)
- **Student**: Own predictions as context (realistic)

## Code Style

- Full variable names (`index` not `idx`)
- Self-documenting over comments
- Simple, beginner-friendly, no over-engineering
