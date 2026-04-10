import os
import argparse
import pickle
from tqdm import tqdm
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Merge distributed reward simulation results.")
    parser.add_argument("--test_path", required=True)
    parser.add_argument("--appendix", type=str)

    args = parser.parse_args()

    TEST_PATH = args.test_path
    MODEL_NAME = TEST_PATH.split('/')[-2]
    WORLDENGINE_ROOT = os.environ["WORLDENGINE_ROOT"]
    DATA_NAME = f'{MODEL_NAME}_{os.path.basename(TEST_PATH)}' + (f'_{args.appendix}' if args.appendix else '')

    # test_path is now an absolute path from the calling script
    merged_base = TEST_PATH
    export_base = os.path.join(WORLDENGINE_ROOT, "data/alg_engine/openscene-synthetic")

    for folder in ['meta_datas', 'pdms_pkl', 'sensor_blobs']:
        os.symlink(
            os.path.realpath(f'{merged_base}/WE_output/openscene_format/{folder}'),
            f'{export_base}/{folder}/{DATA_NAME}'
        )

    merged_pickle_path = f'{export_base}/meta_datas/{DATA_NAME}/combined.pkl'
    if os.path.exists(merged_pickle_path):
        os.remove(merged_pickle_path)

    all_meta_datas = sorted(os.listdir(f'{export_base}/meta_datas/{DATA_NAME}'))
    data_infos = []
    for pkl_name in tqdm(all_meta_datas, desc="merging data pickles", ncols=80):
        with open(f'{export_base}/meta_datas/{DATA_NAME}/{pkl_name}', 'rb') as f:
            infos = pickle.load(f)
            for info in infos:
                info['syn_name'] = DATA_NAME
                if info['frame_idx'] < 3:
                    continue 

                # use reward existance to filter invalid frame
                pdm_reward_path = f"{export_base}/pdms_pkl/{DATA_NAME}/{info['log_name']}_step_{info['frame_idx']}_scores.pkl"
                if not os.path.exists(pdm_reward_path):
                    info['invalid'] = True
                    continue
                with open(pdm_reward_path, 'rb') as f:
                    pdm = pickle.load(f)
                for k in pdm.keys():
                    pdm[k] = np.array(pdm[k], dtype=np.float32)
                info['pdm'] = pdm
            data_infos.extend(infos)
    pickle.dump(data_infos, open(merged_pickle_path, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    main()
