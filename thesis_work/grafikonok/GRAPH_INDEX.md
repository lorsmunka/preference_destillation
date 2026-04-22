# Grafikonok indexe

A script chart-only PNG-ket general. A cim es magyarazat a thesis szovegeben legyen, ne a kepben.

1. `01_vocabulary_reduction.png` - Vocabulary reduction: Theoretical reduced input/output vocabulary sizes by domain.
2. `02_parameter_allocation.png` - Parameter allocation: No reduction vs input-only vs input+output reduction for sentiment and post generation.
3. `03_sentiment_scaling_curve.png` - Sentiment scaling: Accuracy saturation across model sizes.
4. `04_reduced_vs_full_vocab.png` - Reduced vs full input vocab: Full and reduced-input student/task accuracy at comparable parameter budgets.
5. `05_structured_loss_spread.png` - Structured loss spread: Seed spread and standard deviation for task accuracy.
6. `06_accuracy_gap_seed_lines.png` - Teacher-forced/student/task gap: Thin lines are individual seeds; thick lines are strategy means.
7. `07a_pure_ce_training_curves.png` - Pure CE training curves: Exact mid-training plotting style for combined loss, KL loss, CE loss, and accuracy.
8. `07b_pure_kl_training_curves.png` - Pure KL training curves: Exact mid-training plotting style for combined loss, KL loss, CE loss, and accuracy.
9. `07c_klce_annealing_training_curves.png` - KL/CE annealing training curves: Exact mid-training plotting style for combined loss, KL loss, CE loss, and accuracy.
10. `08_reddit_confusion_matrices.png` - Confusion matrices: Sentiment category mistakes for the 5M-scale run.
11. `09_postgen_topk_metrics.png` - Post generation top-k: Free-form generation top-k tradeoff.
12. `10_domain_data_cost.png` - Domain data cost: Why post generation is much more expensive to compile.