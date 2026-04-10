import copy
import numpy as np
import mmcv
from mmdet.datasets import DATASETS
from mmdet3d_plugin.datasets import NavSimOpenSceneE2E


@DATASETS.register_module()
class NavSimOpenSceneE2EClosedLoop(NavSimOpenSceneE2E):
    r"""OpenScene E2E Dataset for closed-loop inference"""

    def load_annotations(self, ann_file):
        """
        Expect ann_file to contain only queue_length frames.
        """
        data_infos = mmcv.load(ann_file, file_format="pkl")['infos']
        self.index_map = [len(data_infos) - 1]   # only inference the last frame
        return data_infos

    def __getitem__(self, idx):
        # map nav filter idx to the data_infos idx.
        new_idx = self.index_map[idx]
        return self.prepare_test_data(new_idx)

    def init_dataset(self):
        return

    def prepare_test_data(self, index):
        """
        Do not check temporal sanity for closed loop testing
        """
        data_queue = []
        self.enbale_temporal_aug = False
        # ensure the first and final frame in same scene
        final_index = index
        first_index = index - self.queue_length + 1
        if first_index < 0:
            print('error, first index cannot less than zero')
            return None

        input_dict = self.get_data_info(final_index)
        if input_dict is None:
            return None

        prev_indexs_list = list(reversed(range(first_index, final_index)))
        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        data_queue.insert(0, example)

        ########## retrieve previous infos, frame by frame
        for i in prev_indexs_list:

            input_dict = self.get_data_info(i, prev_frame=True)
            if input_dict is None:
                return None

            self.pre_pipeline(input_dict)
            example = self.pipeline(input_dict)
            data_queue.insert(0, copy.deepcopy(example))

        # merge a sequence of data into one dictionary, only for temporal data
        data_queue = self.union2one_test(data_queue)
        return data_queue

    def init_mapping(self, canvas_size, patch_size, lane_ann_file):
        return
    
    def load_pdm_infos(self):
        return

    def get_zero_pdm(self, input_dict):
        pdm_list = [
            "no_at_fault_collisions",
            "drivable_area_compliance",
            "ego_progress",
            "time_to_collision_within_bound",
            "comfort",
            "score",
        ]
        for k in pdm_list:
            input_dict[k] = np.zeros((8192,))
        return input_dict

    def get_data_info(self, index, info=None, debug=False, prev_frame=False):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.
        Returns:
            dict: Data information that will be passed to the data \
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - sweeps (list[dict]): Infos of sweeps.
                - timestamp (float): Sample timestamp.
                - img_filename (str, optional): Image filename.
                - lidar2img (list[np.ndarray], optional): Transformations \
                    from lidar to different cameras.
                - ann_info (dict): Annotation info.
        """

        if info is None:
            info = self.data_infos[index]
        # standard protocal modified from SECOND.Pytorch
        input_dict = dict(
            sample_idx=info["token"],  # str: OpenScene unique sample token
            frame_idx=info["frame_idx"],  # int: 0-indexed frame IDs
            timestamp=info["timestamp"] / 1e6,  # int: OpenScene unique time index
            log_name=info["log_name"],  # str: OpenScene unique sample token
            log_token=info["log_token"],  # str: OpenScene unique sample token
            scene_name=info["scene_name"],  # str: OpenScene sequence name
            scene_token=info["scene_token"],  # str: OpenScene sequence name
            pts_filename=info["lidar_path"],  # str: relative path for the lidar data
            # sweeps=info["sweeps"],  # List[Dict]: list of infos for sweep data
            prev=info["sample_prev"],  # str: OpenScene unique sample token
            next=info["sample_next"],  # str: OpenScene unique sample token
            lidar2global_rotation=info["lidar2global"][:3, :3],
        )

        input_dict = self.update_transform(input_dict=input_dict, index=index)
        input_dict = self.update_sensor(input_dict=input_dict, index=index)
        input_dict = self.update_canbus(input_dict=input_dict, index=index)

        input_dict = self.get_zero_pdm(input_dict)
        input_dict['fail_mask'] = 0

        input_dict = self.update_ego_prediction(input_dict=input_dict, index=index)
        input_dict = self.update_ego_planning(input_dict=input_dict, index=index)

        return input_dict

    def evaluate(
        self,
        results,
        metric="bbox",
        logger=None,
        jsonfile_prefix=None,
        result_names=["pts_bbox"],
        show=False,
        out_dir=None,
        pipeline=None,
    ):
        return {"empty": 0.0}
