import sys
from pathlib import Path

from experimental_analysis.run_data import load_all_runs, filter_runs, sort_runs
from experimental_analysis.compare_runs import (
    run_comparison,
    interactive_run_selection,
    build_comparison_table,
    print_table,
    print_winner_summary,
    export_csv,
)
from experimental_analysis.compare_plots import generate_all_charts, OUTPUT_DIR
from experimental_analysis.compare_plots import (
    plot_accuracy_vs_params,
    plot_accuracy_over_training,
    plot_loss_over_training,
    plot_epoch_eval_bars,
    plot_training_efficiency,
    plot_vocab_comparison,
    plot_annealing_comparison,
    plot_mini_eval_overlay,
    plot_confusion_comparison,
)
from experimental_analysis.scaling_analysis import run_scaling_analysis
from experimental_analysis.evaluate_runs import evaluate_runs


def print_menu():
    print("\n" + "=" * 50)
    print("  EXPERIMENTAL ANALYSIS")
    print("=" * 50)
    print()
    print("  COMPARE")
    print("    [1] Compare all runs (table)")
    print("    [2] Compare completed runs (table)")
    print("    [3] Select runs to compare (interactive)")
    print("    [4] Export comparison to CSV")
    print()
    print("  PLOTS (all runs)")
    print("    [5] Generate all charts")
    print("    [6] Scaling curve (accuracy vs params)")
    print("    [7] Training accuracy overlay")
    print("    [8] Loss overlay")
    print("    [9] Epoch eval bars")
    print("   [10] Training efficiency (acc vs time)")
    print("   [11] Vocab comparison (full vs reduced)")
    print("   [12] Annealing comparison")
    print("   [13] Mini-eval overlay")
    print("   [14] Confusion matrices")
    print()
    print("  ANALYSIS")
    print("   [15] Scaling analysis (fit + knee + summary)")
    print("   [16] Evaluate all runs (report to txt)")
    print("   [17] Evaluate completed runs (report to txt)")
    print()
    print("    [q] Quit")
    print()


def main():
    # Direct CLI mode
    if len(sys.argv) > 1:
        command = sys.argv[1]
        if command == "compare":
            from experimental_analysis.compare_runs import main as compare_main
            sys.argv = sys.argv[1:]
            compare_main()
        elif command == "plots":
            from experimental_analysis.compare_plots import main as plots_main
            sys.argv = sys.argv[1:]
            plots_main()
        elif command == "scaling":
            from experimental_analysis.scaling_analysis import main as scaling_main
            scaling_main()
        elif command == "evaluate":
            from experimental_analysis.evaluate_runs import main as evaluate_main
            sys.argv = sys.argv[1:]
            evaluate_main()
        else:
            print(f"Unknown command: {command}")
            print("Commands: compare, plots, scaling, evaluate")
        return

    # Interactive menu mode
    print("Loading runs...")
    all_runs = load_all_runs()
    print(f"Found {len(all_runs)} runs ({sum(1 for run in all_runs if run.status == 'completed')} completed).\n")

    while True:
        print_menu()
        choice = input("  Select: ").strip()

        if choice == "q":
            print("Bye!")
            break

        elif choice == "1":
            runs = sort_runs(all_runs, "params")
            table = build_comparison_table(runs)
            print_table(table)
            print_winner_summary(runs)

        elif choice == "2":
            completed = [run for run in all_runs if run.status == "completed"]
            runs = sort_runs(completed, "params")
            table = build_comparison_table(runs)
            print_table(table)
            print_winner_summary(runs)

        elif choice == "3":
            selected = interactive_run_selection(all_runs)
            if selected:
                sort_key = input("  Sort by (params/name/student_accuracy/loss/time) [params]: ").strip() or "params"
                selected = sort_runs(selected, sort_key)
                table = build_comparison_table(selected)
                print_table(table)
                print_winner_summary(selected)

        elif choice == "4":
            output_path = input("  CSV path [comparison.csv]: ").strip() or "comparison.csv"
            completed = [run for run in all_runs if run.status == "completed"]
            runs = sort_runs(completed, "params")
            export_csv(runs, output_path)

        elif choice == "5":
            generate_all_charts(all_runs)

        elif choice == "6":
            plot_accuracy_vs_params(all_runs)

        elif choice == "7":
            plot_accuracy_over_training(all_runs)

        elif choice == "8":
            plot_loss_over_training(all_runs)

        elif choice == "9":
            plot_epoch_eval_bars(all_runs)

        elif choice == "10":
            plot_training_efficiency(all_runs)

        elif choice == "11":
            plot_vocab_comparison(all_runs)

        elif choice == "12":
            plot_annealing_comparison(all_runs)

        elif choice == "13":
            plot_mini_eval_overlay(all_runs)

        elif choice == "14":
            category = input("  Category (tone/sentiment/safety/toxicity) [sentiment]: ").strip() or "sentiment"
            plot_confusion_comparison(all_runs, category=category)

        elif choice == "15":
            run_scaling_analysis(runs=all_runs)

        elif choice == "16":
            evaluate_runs()

        elif choice == "17":
            evaluate_runs(completed_only=True)

        else:
            print("  Invalid option.")


if __name__ == "__main__":
    main()
