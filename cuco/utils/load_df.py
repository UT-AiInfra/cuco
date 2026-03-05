import pandas as pd
import json
import sqlite3
from pathlib import Path
from typing import Optional


def load_programs_to_df(db_path_str: str) -> Optional[pd.DataFrame]:
    """
    Loads the 'programs' table from an SQLite database into a pandas DataFrame.

    Args:
        db_path_str: The path to the SQLite database file.

    Returns:
        A pandas DataFrame containing program data, or None if an error occurs
        or no data is found. The 'metrics' JSON string is parsed and its
        key-value pairs are added as columns to the DataFrame.
    """
    db_file = Path(db_path_str)
    if not db_file.exists():
        print(f"Error: Database file not found at {db_path_str}")
        return None

    conn = None
    try:
        conn = sqlite3.connect(str(db_file))
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM programs")  # Fetch all columns
        all_program_rows = cursor.fetchall()

        if not all_program_rows:
            print(f"No programs found in the database: {db_path_str}")
            return pd.DataFrame()  # Return empty DataFrame if no programs

        # Get column names from cursor.description
        column_names = [description[0] for description in cursor.description]
        # print(column_names)
        programs_data = []
        for row_tuple in all_program_rows:
            # Convert row tuple to dict
            p_dict = dict(zip(column_names, row_tuple))

            # Metrics and metadata are stored as JSON strings
            metrics_json = p_dict.get("metrics", "{}")
            metrics_dict = json.loads(metrics_json) if metrics_json else {}

            # Parse inspiration_ids JSON
            archive_insp_ids_json = p_dict.get("archive_inspiration_ids", "[]")
            archive_insp_ids = (
                json.loads(archive_insp_ids_json) if archive_insp_ids_json else []
            )
            top_k_insp_ids_json = p_dict.get("top_k_inspiration_ids", "[]")
            top_k_insp_ids = (
                json.loads(top_k_insp_ids_json) if top_k_insp_ids_json else []
            )
            metadata_json = p_dict.get("metadata", "{}")
            metadata_dict = json.loads(metadata_json) if metadata_json else {}

            # Parse public_metrics and private_metrics
            public_metrics_raw = p_dict.get("public_metrics", "{}")
            if isinstance(public_metrics_raw, str):
                public_metrics_dict = (
                    json.loads(public_metrics_raw) if public_metrics_raw else {}
                )
            else:
                public_metrics_dict = public_metrics_raw or {}

            private_metrics_raw = p_dict.get("private_metrics", "{}")
            if isinstance(private_metrics_raw, str):
                private_metrics_dict = (
                    json.loads(private_metrics_raw) if private_metrics_raw else {}
                )
            else:
                private_metrics_dict = private_metrics_raw or {}

            embedding = p_dict.get("embedding", [])
            if isinstance(embedding, str):
                embedding = json.loads(embedding)
            # Create a flat dictionary for the DataFrame
            try:
                timestamp = pd.to_datetime(p_dict.get("timestamp"), unit="s")
            except Exception:
                timestamp = None
            flat_data = {
                "id": p_dict.get("id"),
                "code": p_dict.get("code"),
                "language": p_dict.get("language"),
                "parent_id": p_dict.get("parent_id"),
                "archive_inspiration_ids": archive_insp_ids,
                "top_k_inspiration_ids": top_k_insp_ids,
                "generation": p_dict.get("generation"),
                "timestamp": timestamp,
                "complexity": p_dict.get("complexity"),
                "embedding": embedding,
                "code_diff": p_dict.get("code_diff"),
                "correct": bool(p_dict.get("correct", False)),
                "combined_score": p_dict.get("combined_score"),
                **metadata_dict,
                **public_metrics_dict,
                **private_metrics_dict,
                "text_feedback": p_dict.get("text_feedback", ""),
            }
            flat_data.update(metrics_dict)
            programs_data.append(flat_data)

        return pd.DataFrame(programs_data)

    except sqlite3.Error as e:
        print(f"SQLite error while loading {db_path_str}: {e}")
        return None
    except json.JSONDecodeError as e:
        db_path = db_path_str
        print(f"JSON decoding error for metrics/metadata in {db_path}: {e}")
        return None
    finally:
        if conn:
            conn.close()


def get_path_to_best_node(
    df: pd.DataFrame, score_column: str = "combined_score"
) -> pd.DataFrame:
    """
    Finds the chronological path to the node with the highest score.

    Args:
        df: DataFrame containing program data
        score_column: The column name to use for finding the best node
                      (default: "combined_score")

    Returns:
        A DataFrame representing the chronological path to the
        best node, starting from the earliest ancestor and ending with the
        best node.
    """
    if df.empty:
        return pd.DataFrame()

    if score_column not in df.columns:
        raise ValueError(f"Column '{score_column}' not found in DataFrame")

    # Create a dictionary mapping id to row for quick lookups
    id_to_row = {row["id"]: row for _, row in df.iterrows()}

    print(f"Total rows: {len(df)}")
    # Only correct rows
    correct_df = df[df["correct"]]
    print(f"Correct rows: {len(correct_df)}")

    # Find the node with the maximum score
    best_node_row = correct_df.loc[correct_df[score_column].idxmax()]

    # Start building the path with the best node
    path = [best_node_row.to_dict()]
    current_id = best_node_row["parent_id"]

    # Trace back through parent_ids to construct the path
    while current_id is not None and current_id in id_to_row:
        parent_row = id_to_row[current_id]
        path.append(parent_row.to_dict())
        current_id = parent_row["parent_id"]

    # Reverse to get chronological order (oldest first)
    return pd.DataFrame(path[::-1])


def store_best_path(df: pd.DataFrame, results_dir: str):
    best_path = get_path_to_best_node(df)
    path_dir = Path(f"{results_dir}/best_path")
    path_dir.mkdir(exist_ok=True)
    patch_dir = Path(f"{path_dir}/patches")
    patch_dir.mkdir(exist_ok=True)
    code_dir = Path(f"{path_dir}/code")
    code_dir.mkdir(exist_ok=True)
    meta_dir = Path(f"{path_dir}/meta")
    meta_dir.mkdir(exist_ok=True)

    i = 0
    for _, row in best_path.iterrows():
        print(f"\nGeneration {row['generation']} - Score: {row['combined_score']:.2f}")

        if row["code_diff"] is not None:
            patch_path = patch_dir / f"patch_{i}.patch"
            patch_path.write_text(str(row["code_diff"]))
            print(f"Saved patch to {patch_path}")

        base_path = code_dir / f"main_{i}.py"
        base_path.write_text(str(row["code"]))

        # store row data as json, handle non-serializable types
        import datetime

        def default_serializer(obj):
            if isinstance(obj, (datetime.datetime, datetime.date)):
                return obj.isoformat()
            try:
                import pandas as pd

                if isinstance(obj, pd.Timestamp):
                    return obj.isoformat()
            except ImportError:
                pass
            return str(obj)

        row_data_path = meta_dir / f"meta_{i}.json"
        row_data_path.write_text(json.dumps(row.to_dict(), default=default_serializer))
        print(f"Saved meta data to {row_data_path}")
        print(f"Saved base code to {base_path}")
        print(row["patch_name"])
        i += 1
