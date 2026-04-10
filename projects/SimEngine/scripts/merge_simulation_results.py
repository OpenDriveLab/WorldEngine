
import os
import glob
import pandas as pd
import argparse

def main():
    parser = argparse.ArgumentParser(description="Merge distributed reward simulation results.")
    parser.add_argument("--test_path", required=True)
    parser.add_argument("--react_type", required=True, help="Type of reaction (e.g., R, NR).")

    args = parser.parse_args()

    TEST_PATH = args.test_path
    REACT_TYPE = args.react_type

    # test_path is now an absolute path from the calling script
    merged_base = TEST_PATH

    print("Starting to merge results...")

    # Create merged destination directories
    os.makedirs(os.path.join(merged_base, "WE_output/openscene_format/meta_datas"), exist_ok=True)
    os.makedirs(os.path.join(merged_base, "WE_output/openscene_format/sensor_blobs"), exist_ok=True)
    os.makedirs(os.path.join(merged_base, "WE_output/openscene_format/pdms_pkl"), exist_ok=True)
    os.makedirs(os.path.join(merged_base, "plan_traj"), exist_ok=True)

    def merge_plan_csv():
        print("Merging plan idx CSV files...")
        merged_csv_path = os.path.join(merged_base, "plan_traj/plan_idx.csv")
        all_dfs = []
        for i in range(8):
            split_csv = os.path.join(merged_base, f"split_{i}/plan_traj/plan_idx.csv")
            # Read all rows except header and last row (overall_average)
            df_split = pd.read_csv(split_csv)
            all_dfs.append(df_split)
        merged_df = pd.concat(all_dfs, ignore_index=True)
        merged_df.to_csv(merged_csv_path, index=False)
        # Save another csv into pdms_pkl
        merged_csv_path = os.path.join(merged_base, "WE_output/openscene_format/pdms_pkl/plan_idx.csv")
        merged_df.to_csv(merged_csv_path, index=False)

    merge_plan_csv()
    # Merge closed-loop metric CSV files
    print("Merging CSV files...")
    merged_csv_path = os.path.join(merged_base, f"WE_output/openscene_format/all_scenes_pdm_averages_{REACT_TYPE}.csv")
    all_dfs = []
    for i in range(8):
        split_csv = os.path.join(merged_base, f"split_{i}/WE_output/openscene_format/all_scenes_pdm_averages_{REACT_TYPE}.csv")
        df_split = pd.read_csv(split_csv, skipfooter=1, engine='python')
        all_dfs.append(df_split)
    merged_df = pd.concat(all_dfs, ignore_index=True)

    averages = merged_df.mean(numeric_only=True)
    averages['token'] = 'overall_average'
    new_row = pd.DataFrame([averages])
    df = pd.concat([merged_df, new_row])
    df.to_csv(merged_csv_path, float_format='%.5f', index=False)
    print("CSV file merged successfully, new overall_average calculated")

    # merge dense reward csv file
    merged_csv_path = os.path.join(merged_base, f"WE_output/openscene_format/all_scenes_pdm_pkl_paths_{REACT_TYPE}.csv")
    all_dfs = []
    for i in range(8):
        split_csv = os.path.join(merged_base, f"split_{i}/WE_output/openscene_format/all_scenes_pdm_pkl_paths_{REACT_TYPE}.csv")
        # skip if dense reward not computed
        if not os.path.exists(split_csv):
            continue
        df_split = pd.read_csv(split_csv)
        all_dfs.append(df_split)
    if len(all_dfs) > 0:
        merged_df = pd.concat(all_dfs, ignore_index=True)
        merged_df.to_csv(merged_csv_path, index=False)

    def _link_files(src_dir, dst_dir, pattern, symlink=False):
        """Link files from src_dir to dst_dir matching the given glob pattern."""
        for src_file in glob.glob(os.path.join(src_dir, pattern)):
            dst_file = os.path.join(dst_dir, os.path.basename(src_file))
            if os.path.exists(dst_file):
                continue
            if symlink:
                os.symlink(os.path.realpath(src_file), dst_file)
            else:
                os.link(src_file, dst_file)

    def merge_split_files(split_id):
        print(f"Merging files from split {split_id}...")
        split_path = os.path.join(merged_base, f"split_{split_id}")
        openscene_base = "WE_output/openscene_format"

        # (subdir, glob_pattern, use_symlink)
        link_tasks = [
            (f"{openscene_base}/meta_datas", "*.pkl", False),
            (f"{openscene_base}/pdms_pkl", "*.pkl", False),
            (f"{openscene_base}/sensor_blobs", "*", True),
        ]

        for subdir, pattern, use_symlink in link_tasks:
            src_dir = os.path.join(split_path, subdir)
            if os.path.isdir(src_dir):
                _link_files(src_dir, os.path.join(merged_base, subdir), pattern, symlink=use_symlink)

    for i in range(8):
        merge_split_files(i)

    print("All operations completed successfully!")
    print(f"Results available in: {merged_base}")

if __name__ == "__main__":
    main()
