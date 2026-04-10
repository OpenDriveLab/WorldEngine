import argparse
import os
import pickle
import yaml
from typing import Dict, List
from tqdm import tqdm
WORLDENGINE_ROOT = os.getenv('WORLDENGINE_ROOT', os.path.abspath('.'))

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data_root",
    type=str,
    default=f"{WORLDENGINE_ROOT}/data/openscene-v1.1",
    help="root directory of raw openscene data",
)
parser.add_argument(
    "--meta_data_folder",
    type=str,
    default="meta_datas_navformer"
)
parser.add_argument(
    "--output_dir",
    type=str,
    default="merged_infos_navformer",
    help="output directory for merged infos (relative to data_root if not absolute)",
)
args = parser.parse_args()

def get_pkl_filelist(meta_data_dir: str) -> List[str]:

    meta_data_list = os.listdir(meta_data_dir)
    meta_data_list = [
        os.path.join(meta_data_dir, each)
        for each in meta_data_list
        if each.endswith(".pkl")
    ]

    return meta_data_list


def merge_split_infos(
    split_name: str,
    paths: List[str],
    log_filter: List[str],
    scene_filter: List[str],
    history_frame_num: int,
    future_frame_num: int,
    save_path: str,
) -> None:
    if os.path.exists(save_path) or not paths:
        print(f"skipped because {split_name} is saved before")
        return

    data_infos = []
    total_len = 0
    for file in tqdm(paths):
        with open(file, "rb") as f:
            tqdm.write(f"{split_name}: loading {file}")
            data_tmp = pickle.load(f)["infos"]
            total_len += len(data_tmp)
            add = False

            log_name_tmp = data_tmp[0]["log_name"]
            if log_name_tmp not in log_filter:
                continue

            # get the scene_filter for this log
            scene_filter_expanded = set()
            for idx, data_frame in enumerate(data_tmp):
                if data_frame["token"] in scene_filter:
                    start_frame_idx = idx - history_frame_num
                    end_frame_idx = idx + future_frame_num
                    for i in range(start_frame_idx, end_frame_idx + 1):
                        if i < 0 or i >= len(data_tmp):
                            continue
                        scene_filter_expanded.add(data_tmp[i]["token"])

            data_save = []
            for data_frame in data_tmp:
                token = data_frame["token"]
                if token in scene_filter_expanded:
                    add = True
                    data_save.append(data_frame)

            if add:
                data_infos.extend(data_save)

    print(f"{split_name} info len before: {total_len}")
    print(f"{split_name} info len after: {len(data_infos)}")

    save_dir = os.path.dirname(save_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(save_path, "wb") as f:
        pickle.dump(data_infos, f)

if __name__ == "__main__":

    # OpenScene/nuPlan/NAVSIM:
    # navtest: 12136 -> 1.69h
    # navtrain: 102983 -> 14.3h

    meta_data_dir = os.path.join(args.data_root, args.meta_data_folder, "trainval")
    train_paths = get_pkl_filelist(meta_data_dir)
    navtrain_filter = "configs/navsim_splits/navtrain_split/navtrain.yaml"
    with open(navtrain_filter, 'r') as file:
        navtrain_filter = yaml.safe_load(file)
        log_filter_train = navtrain_filter['log_names']
        scene_filter_train = navtrain_filter['tokens']

    meta_data_dir = os.path.join(args.data_root, args.meta_data_folder, "test")
    test_paths = get_pkl_filelist(meta_data_dir)
    navtest_filter = "configs/navsim_splits/navtest_split/navtest.yaml"
    with open(navtest_filter, 'r') as file:
        navtest_filter = yaml.safe_load(file)
        log_filter_test = navtest_filter['log_names']
        scene_filter_test = navtest_filter['tokens']

    print(f"trainval log len: {len(train_paths)}")
    print(f"test log len: {len(test_paths)}")

    output_dir = args.output_dir
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(args.data_root, output_dir)

    save_test = os.path.join(output_dir, "nuplan_openscene_navtest.pkl")
    save_train = os.path.join(output_dir, "nuplan_openscene_navtrain.pkl")

    # load and merge pkl files into train/test
    # only take the infos for now, leave the mapping to be used later if needed
    split_configs = [
        (
            "test",
            test_paths,
            log_filter_test,
            scene_filter_test,
            4,  # history_frame_num (use 4 instead of 3 for NAVSIM v2)
            0,  # future_frame_num
            save_test,
        ),
        (
            "train",
            train_paths,
            log_filter_train,
            scene_filter_train,
            3,  # history_frame_num
            8,  # future_frame_num
            save_train,
        ),
    ]

    for split_name, paths, log_filter, scene_filter, history, future, save_path in split_configs:
        merge_split_infos(
            split_name=split_name,
            paths=paths,
            log_filter=log_filter,
            scene_filter=scene_filter,
            history_frame_num=history,
            future_frame_num=future,
            save_path=save_path
        )
