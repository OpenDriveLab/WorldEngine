import copy

from worldengine.manager.base_manager import BaseManager
from worldengine.components.maps.scenario_map import ScenarioMap
from worldengine.scenario.scenarios.scenario_description import ScenarioDescription as SD


class ScenarioMapManager(BaseManager):
    """ Initialize the map from the dataset pickle file. """
    PRIORITY = 0  
    DEFAULT_DATA_BUFFER_SIZE = 200

    def __init__(self):
        super(ScenarioMapManager, self).__init__()
        self._no_map = self.engine.global_config.get("no_map", False)
        self.store_map = self.engine.global_config.get("store_map", False)

        # register the map object.
        self._current_map = None
        self.num_scenarios = self.engine.num_scenarios
        self._stored_maps = [None for _ in range(self.num_scenarios)]

    def before_reset(self):
        # remove map from world before adding
        if self._current_map is not None:
            self.unload_map(self._current_map)

        self._current_map = None

    def reset(self):
        if not self._no_map:
            seed = self.engine.global_random_seed

            if self._stored_maps[seed] is None:
                m_data = self.engine.current_scene[SD.MAP_FEATURES]
                new_map = ScenarioMap(map_index=seed, map_data=m_data)
                if self.store_map:
                    self._stored_maps[seed] = new_map
            else:
                new_map = self._stored_maps[seed]
            self.load_map(new_map)

    def load_map(self, map):
        """ Attach curent map into the world node. """
        self._attach_map(map)
        self._current_map = map

    def unload_map(self, map):
        self._detach_map(map)

        self._current_map = None
        if not self.engine.global_config["store_map"]:
            map.destroy()

    def filter_path(self, start_lanes, end_lanes):
        """ Remove paths without feasible routes. """
        for start in start_lanes:
            for end in end_lanes:
                path = self._current_map.road_network.shortest_path(start[0].index, end[0].index)
                if len(path) > 0:
                    return (start[0].index, end[0].index)
        return None

    def destroy(self):
        self.clear_stored_maps()
        self._stored_maps = None
        self._current_map = None

        super(ScenarioMapManager, self).destroy()

    def clear_stored_maps(self):
        for m in self._stored_maps:
            if m is not None:
                self._detach_map(m)
                m.destroy()

        self._stored_maps = [None for _ in range(self.num_scenarios)]

    @property
    def num_stored_maps(self):
        return sum([1 if m is not None else 0 for m in self._stored_maps])

    @property
    def current_map(self):
        return self._current_map

    def _attach_map(self, map):
        pass

    def _detach_map(self, map):
        pass
