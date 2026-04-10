from typing import List, Dict, Any
import numpy as np
from nuscenes.eval.common.data_classes import MetricData

MAX_NUMBER_OF_MODES = 25

class Prediction(MetricData):
    """
    Stores predictions of Models.
    Metrics are calculated from Predictions.

    Attributes:
        instance: Instance token for prediction.
        sample: Sample token for prediction.
        prediction: Prediction of model [num_modes, n_timesteps, state_dim].
        probabilities: Probabilities of each mode [num_modes].
    """
    def __init__(self, sample: str, prediction: np.ndarray, scene_token: str, frame_idx: int):
        self.is_valid(sample, prediction, scene_token, frame_idx)

        self.sample = sample
        self.prediction = prediction
        self.scene_token = scene_token
        self.frame_idx = frame_idx

    @property
    def number_of_modes(self) -> int:
        return self.prediction.shape[0]

    def serialize(self):
        """ Serialize to json. """
        return {
            'sample': self.sample,
            'prediction': self.prediction.tolist(),
            'scene_token': self.scene_token,
            'frame_idx': self.frame_idx,
        }

    @classmethod
    def deserialize(cls, content: Dict[str, Any]):
        """ Initialize from serialized content. """
        return cls(
            sample=content['sample'],
            prediction=np.array(content['prediction']),
            scene_token=content['scene_token'],
            frame_idx=content['frame_idx'],
        )

    @staticmethod
    def is_valid(sample, prediction, scene_token, frame_idx):
        if not isinstance(prediction, np.ndarray):
            raise ValueError(f"Error: prediction must be of type np.ndarray. Received {str(type(prediction))}.")
        # if not isinstance(probabilities, np.ndarray):
        #     raise ValueError(f"Error: probabilities must be of type np.ndarray. Received {type(probabilities)}.")
        if not isinstance(scene_token, str):
            raise ValueError(f"Error: instance token must be of type string. Received {type(scene_token)}")
        if not isinstance(frame_idx, int):
            raise ValueError(f"Error: instance token must be of type int. Received {type(frame_idx)}")        
        if not isinstance(sample, str):
            raise ValueError(f"Error: sample token must be of type string. Received {type(sample)}.")
        if prediction.ndim != 3:
            raise ValueError("Error: prediction must have three dimensions (number of modes, number of timesteps, 2).\n"
                             f"Received {prediction.ndim}")
        # if probabilities.ndim != 1:
        #     raise ValueError(f"Error: probabilities must be a single dimension. Received {probabilities.ndim}.")
        # if len(probabilities) != prediction.shape[0]:
        #     raise ValueError("Error: there must be the same number of probabilities as predicted modes.\n"
        #                      f"Received {len(probabilities)} probabilities and {prediction.shape[0]} modes.")
        if prediction.shape[0] > MAX_NUMBER_OF_MODES:
            raise ValueError(f"Error: prediction contains more than {MAX_NUMBER_OF_MODES} modes.")

    def __repr__(self):
        return f"Prediction(sample={self.sample},"\
               f" prediction={self.prediction}, scene_token={self.scene_token}, frame_idx={self.frame_idx})"

