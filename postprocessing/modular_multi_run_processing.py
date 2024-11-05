from postprocessing.modular_scores_wandb import run_experiment
from postprocessing.modular_xmodel_data_split import process_data
from postprocessing.modular_retrieval_split_wandb import score_retrieval
import pandas as pd
from tqdm import tqdm
import click


def extract_top_middle_bottom(df, sort_column):
    """
    Extracts the top 10, middle 10, and bottom 10 rows from a DataFrame based on a given column.

    Args:
        df (pd.DataFrame): The input DataFrame.
        sort_column (str): The column name to sort by.

    Returns:
        pd.DataFrame: A new DataFrame containing the top 10, middle 10, and bottom 10 rows.
    """
    # Sort the DataFrame by the given column
    df_sorted = df.sort_values(by=sort_column, ascending=False).reset_index(
        drop=True
    )

    # Extract top 10 rows
    top_10 = df_sorted.head(10)

    # Extract bottom 10 rows
    bottom_10 = df_sorted.tail(10)

    # Calculate the start index for the middle 10 rows
    middle_start = len(df_sorted) // 2 - 5
    middle_10 = df_sorted.iloc[middle_start : middle_start + 10]

    # Combine the top, middle, and bottom 10 rows into a single DataFrame
    combined_df = pd.concat([top_10, middle_10, bottom_10], axis=0)

    return combined_df


@click.command()
@click.option(
    "--config", required=True, help="Path to the configuration YAML file."
)
@click.option("--run_id", default=None, help="Single run ID to process.")
@click.option(
    "--run_num", type=int, default=None, help="Run number for experiment."
)
@click.option(
    "--csv-path",
    default=None,
    help="Path to the CSV file containing multiple runs.",
)
@click.option(
    "--sort-column",
    default="val_obj",
    help="Column to sort by when processing CSV.",
)
def main(config, run_id, csv_path, sort_column, run_num):
    """
    Main function to run experiments for either a single run ID or multiple IDs from a CSV.
    """
    if run_id:
        # Process a single run ID
        try:
            print(f"\nStarting run for ID: {run_id}")

            # Run the experiment
            print("Running experiment...")
            run_experiment(config_path=config, run_id=run_id, run_num=run_num)
            print("Experiment completed.")

            # Process the data
            print("Processing data...")
            process_data(config_path=config, run_id=run_id, run_num=run_num)
            print("Data processing completed.")

            # Score the retrieval
            print("Scoring retrieval...")
            score_retrieval(config_path=config, run_id=run_id, run_num=run_num)
            print("Retrieval scoring completed.")

            print(f"Completed all steps for run ID: {run_id}\n")

        except Exception as ex:
            print(f"Error processing run ID: {run_id}")
            print(f"Error: {ex}")

    elif csv_path:
        # Read the CSV and process top, middle, and bottom 10 runs
        df = pd.read_csv(csv_path)

        # Extract the top, middle, and bottom 10 runs based on the sort column
        top_middle_bottom_runs_df = extract_top_middle_bottom(df, sort_column)

        extract_path = "z_results/top_middle_bottom_runs.csv"
        top_middle_bottom_runs_df.to_csv(extract_path, index=False)

        for _, row in tqdm(
            top_middle_bottom_runs_df.iterrows(),
            total=len(top_middle_bottom_runs_df),
            desc="Processing Runs",
        ):
            try:
                run_id = row["id"]

                print(f"\nStarting run for ID: {run_id}")

                # Run the experiment
                print("Running experiment...")
                run_experiment(config_path=config_path, run_id=run_id)
                print("Experiment completed.")

                # Process the data
                print("Processing data...")
                process_data(config_path=config_path, run_id=run_id)
                print("Data processing completed.")

                # Score the retrieval
                print("Scoring retrieval...")
                score_retrieval(config_path=config_path, run_id=run_id)
                print("Retrieval scoring completed.")

                print(f"Completed all steps for run ID: {run_id}\n")

            except Exception as ex:
                print(f"Error processing run ID: {run_id}")
                print(f"Error: {ex}")
                continue
    else:
        print(
            "Please provide either a single --run-id or a --csv-path with runs to process."
        )


if __name__ == "__main__":
    main()
