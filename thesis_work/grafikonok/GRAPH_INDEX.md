# Grafikonok indexe

A script tobbnyire chart-only PNG-ket general. A 06a-c abrakban a modellmeret is latszik.

1. `01_vocabulary_reduction.png` - Vocabulary reduction: Theoretical reduced input/output vocabulary sizes by domain.
2. `02_parameter_allocation.png` - Parameter allocation: No reduction vs input-only vs input+output reduction for sentiment and post generation.
3. `03_sentiment_scaling_curve.png` - Sentiment scaling: Accuracy saturation across model sizes.
4. `04_reduced_vs_full_vocab.png` - Reduced vs full input vocab: Full and reduced-input student/task accuracy at comparable parameter budgets.
5. `05_structured_loss_spread.png` - Structured loss spread: Seed spread and standard deviation for task accuracy.
6. `06a_accuracy_gap_seed_lines_ce.png` - CE accuracy gap: Teacher-forced/student/task gap for sentiment CE seeds.
7. `06b_accuracy_gap_seed_lines_kl.png` - KL accuracy gap: Teacher-forced/student/task gap for sentiment KL seeds.
8. `06c_accuracy_gap_seed_lines_klce_annealing.png` - KL/CE annealing accuracy gap: Teacher-forced/student/task gap for sentiment annealing seeds.
9. `07a_pure_ce_training_curves.png` - Pure CE training curves: Exact mid-training plotting style for combined loss, KL loss, CE loss, and accuracy.
10. `07b_pure_kl_training_curves.png` - Pure KL training curves: Exact mid-training plotting style for combined loss, KL loss, CE loss, and accuracy.
11. `07c_klce_annealing_training_curves.png` - KL/CE annealing training curves: Exact mid-training plotting style for combined loss, KL loss, CE loss, and accuracy.
12. `07d_large_pure_ce_training_curves.png` - Larger pure CE training curves: Same plotting style for the longer sentiment pure CE run.
13. `07e_large_pure_kl_training_curves.png` - Larger pure KL training curves: Same plotting style for the longer sentiment pure KL run.
14. `07f_large_klce_annealing_training_curves.png` - Larger KL/CE annealing training curves: Same plotting style for the longer sentiment KL/CE annealing run.
15. `08_reddit_confusion_matrices.png` - Confusion matrices: Sentiment category mistakes for the 5M-scale run.
16. `09_postgen_topk_metrics.png` - Post generation top-k: Free-form generation top-k tradeoff.
17. `10_domain_data_cost.png` - Domain data cost: Why post generation is much more expensive to compile.
18. `14_math_agent_kl_ce.png` - Thesis figure 9.1: math tutor 16.5M student on the full data, task accuracy and test CE loss per pass, KL vs CE and KL -> CE (3 seeds each); `plot_14_math_agent_kl_ce.py`.
19. `15_math_data_curve.png` - Thesis figure 9.2: math tutor data curve on one test set, the same 16.5M KL student (seed 1, 10 passes) on 6k / 20k / 65k / 195,980 replies, task accuracy and test CE; `plot_15_math_data_curve.py`.
20. `16_sentiment_size.png` - Thesis figure 9.3: sentiment students of 7 sizes (5.3M-129.2M) on ~490k training examples, field-level task accuracy over valid JSON replies, one seed each; `plot_16_sentiment_size.py`.
21. `17_code_agent_overfit.png` - Thesis figure 9.5: coding agent student (12.0M, KL, one seed), CE loss per pass on its training data vs on the test projects (lowest at pass 2, then rising); `plot_17_code_agent_overfit.py`.
