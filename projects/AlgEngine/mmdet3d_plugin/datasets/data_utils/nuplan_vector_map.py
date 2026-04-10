import os
import numpy as np
from pyquaternion import Quaternion
from shapely import affinity, ops
from shapely.geometry import LineString, box, MultiPolygon, MultiLineString

from nuplan.common.maps.nuplan_map.map_factory import get_maps_api
from nuplan.common.maps.maps_datatypes import SemanticMapLayer

class VectorizedLocalMap(object):
    CLASS2LABEL = {
        SemanticMapLayer.LANE: 0,
        SemanticMapLayer.LANE_CONNECTOR: 0,
        SemanticMapLayer.CROSSWALK: 1,
        SemanticMapLayer.ROADBLOCK: 2,
        SemanticMapLayer.INTERSECTION: 2,
        SemanticMapLayer.CARPARK_AREA: 2,
        SemanticMapLayer.ROADBLOCK_CONNECTOR: 2,
        SemanticMapLayer.WALKWAYS: 3
    }
    def __init__(
        self,
        map_root,
        map_version='nuplan-maps-v1.0',
        patch_size=(100, 100),     # h, w
        map_classes={
            'centerline': [SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR],
            'ped_crossing': [SemanticMapLayer.CROSSWALK],
            'road_boundary': [SemanticMapLayer.ROADBLOCK, SemanticMapLayer.INTERSECTION, SemanticMapLayer.CARPARK_AREA],
            # 'sidewalk': [SemanticMapLayer.WALKWAYS]
        },
        need_merged=['road_boundary'],
    ):
        super().__init__()
        self.map_classes = map_classes
        self.patch_size = patch_size
        self.need_merged = need_merged
        self.MAP_APIS_DICT = {
            "us-pa-pittsburgh-hazelwood" : get_maps_api(map_root, map_version, "us-pa-pittsburgh-hazelwood"),
            "sg-one-north" : get_maps_api(map_root, map_version, "sg-one-north"), 
            "us-ma-boston" : get_maps_api(map_root, map_version, "us-ma-boston"), 
            "us-nv-las-vegas-strip" : get_maps_api(map_root, map_version, "us-nv-las-vegas-strip")
        }

    def get_patch_coord(self, patch_box, patch_angle: float = 0.0):
        """
        Convert patch_box to shapely Polygon coordinates.
        :param patch_box: Patch box defined as [x_center, y_center, height, width].
        :param patch_angle: Patch orientation in degrees.
        :return: Box Polygon for patch_box.
        """
        patch_x, patch_y, patch_h, patch_w = patch_box

        x_min = patch_x - patch_w / 2.0
        y_min = patch_y - patch_h / 2.0
        x_max = patch_x + patch_w / 2.0
        y_max = patch_y + patch_h / 2.0

        patch = box(x_min, y_min, x_max, y_max)
        patch = affinity.rotate(patch, patch_angle, origin=(patch_x, patch_y), use_radians=False)

        return patch

    def gen_vectorized_samples(self, e2g_T, e2g_R, map_location):
        '''
        use lidar2global to get gt map layers
        '''
        x, y = (e2g_T[0], e2g_T[1])
        patch_angle = Quaternion(e2g_R).yaw_pitch_roll[0]
        patch_box = (x, y, self.patch_size[0], self.patch_size[1])
        patch_angle = patch_angle / np.pi * 180

        vectors = []
        for idx, class_name in enumerate(self.map_classes):
            geom = self.get_map_geom(patch_box, patch_angle, self.map_classes[class_name], map_location)
            if class_name in self.need_merged:
                line_list = self.merge_polys_to_lines(geom)
            else:
                line_list = self.geoms_to_lines(geom)

            for line in line_list:
                vectors.append({
                    'pts': line.astype(np.float32),
                    'pts_num': line.shape[0],
                    'type': idx
                })

        return vectors
    
    def gen_drivable_area(self, e2g_T, e2g_R, map_location,no_transform=False):
        x, y = (e2g_T[0], e2g_T[1])
        patch_angle = Quaternion(e2g_R).yaw_pitch_roll[0]
        patch_box = (x, y, self.patch_size[0], self.patch_size[1])
        patch_angle = patch_angle / np.pi * 180

        geom = self.get_map_geom(patch_box, patch_angle, self.map_classes['road_boundary'], map_location,no_transform)
        multi_polygon = self.merge_polys_to_one(geom)
        return multi_polygon

    def get_map_geom(self, patch_box, patch_angle, layer_names, map_location, no_transform=False):
        patch_x = patch_box[0]
        patch_y = patch_box[1]

        patch = self.get_patch_coord(patch_box, patch_angle)
        map_api = self.MAP_APIS_DICT[map_location]
        map_geom = []
        for layer_name in layer_names:
            records = map_api._get_proximity_map_object(patch, layer_name)
            for record in records:
                if layer_name in [SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR]:
                    line = record.baseline_path.linestring
                    new_line = line.intersection(patch)
                    if new_line.is_empty:
                        continue
                    if not no_transform:
                        new_line = affinity.rotate(new_line, -patch_angle, origin=(patch_x, patch_y), use_radians=False)
                        new_line = affinity.affine_transform(new_line,
                                                            [1.0, 0.0, 0.0, 1.0, -patch_x, -patch_y])
                    map_geom.append((layer_name, new_line))
                elif layer_name in [SemanticMapLayer.CROSSWALK, SemanticMapLayer.WALKWAYS, SemanticMapLayer.ROADBLOCK, SemanticMapLayer.INTERSECTION]:
                    polygon = record.polygon
                    if polygon.is_valid:
                        new_polygon = polygon.intersection(patch)      
                        if new_polygon.is_empty:
                            continue
                        new_polygon = affinity.rotate(new_polygon, -patch_angle,
                                                        origin=(patch_x, patch_y), use_radians=False)
                        new_polygon = affinity.affine_transform(new_polygon,
                                                                [1.0, 0.0, 0.0, 1.0, -patch_x, -patch_y])
                        if new_polygon.geom_type == 'Polygon':
                            new_polygon = MultiPolygon([new_polygon])
                    map_geom.append((layer_name, new_polygon))
        return map_geom

    def _one_type_line_geom_to_instances(self, line_geom):
        line_instances = []

        for line in line_geom:
            if not line.is_empty:
                if line.geom_type == 'MultiLineString':
                    for single_line in line.geoms:
                        line_instances.append(single_line)
                elif line.geom_type == 'LineString':
                    line_instances.append(line)
                else:
                    print(line.geom_type)
                    raise NotImplementedError
        
        line_instances = [np.asarray(line.coords) for line in line_instances if len(line.coords) > 1]
        return line_instances

    def geoms_to_lines(self, geoms):
        lines = []
        for layer_name, geom in geoms:
            if geom.geom_type == 'MultiPolygon':
                lines.extend(self.poly_to_lines(geom))
            else:
                lines.append(geom)

        return self._one_type_line_geom_to_instances(lines)

    def poly_to_lines(self, poly):

        max_x = self.patch_size[1] / 2
        max_y = self.patch_size[0] / 2
        local_patch = box(-max_x + 0.2, -max_y + 0.2, max_x - 0.2, max_y - 0.2)

        if poly.geom_type == 'Polygon':
            poly = MultiPolygon([poly])
        exteriors = []
        interiors = []
        for p in poly.geoms:
            exteriors.append(p.exterior)
            for inter in p.interiors:
                interiors.append(inter)
        
        results = []
        for ext in exteriors:
            if ext.is_ccw:
                ext.coords = list(ext.coords)[::-1]
            lines = ext.intersection(local_patch)
            if isinstance(lines, MultiLineString):
                lines = ops.linemerge(lines)
            results.append(lines)

        for inter in interiors:
            if not inter.is_ccw:
                inter.coords = list(inter.coords)[::-1]
            lines = inter.intersection(local_patch)
            if isinstance(lines, MultiLineString):
                lines = ops.linemerge(lines)
            results.append(lines)
        return results

    def merge_polys_to_lines(self, polygon_geom):
        roads = [poly[1] for poly in polygon_geom]

        exteriors = []
        interiors = []

        union_segments = ops.unary_union(roads)
        if union_segments.geom_type != 'MultiPolygon':
            union_segments = MultiPolygon([union_segments])
        for poly in union_segments.geoms:
            exteriors.append(poly.exterior)
            for inter in poly.interiors:
                interiors.append(inter)

        max_x = self.patch_size[1] / 2
        max_y = self.patch_size[0] / 2

        # cut polygon to lines
        local_patch = box(-max_x + 0.2, -max_y + 0.2, max_x - 0.2, max_y - 0.2)
        results = []
        for ext in exteriors:
            if ext.is_ccw:
                ext.coords = list(ext.coords)[::-1]
            lines = ext.intersection(local_patch)
            if isinstance(lines, MultiLineString):
                lines = ops.linemerge(lines)
            results.append(lines)

        for inter in interiors:
            if not inter.is_ccw:
                inter.coords = list(inter.coords)[::-1]
            lines = inter.intersection(local_patch)
            if isinstance(lines, MultiLineString):
                lines = ops.linemerge(lines)
            results.append(lines)

        return self._one_type_line_geom_to_instances(results)

    def merge_polys_to_one(self, polygon_geom):
        roads = [poly[1] for poly in polygon_geom]

        union_segments = ops.unary_union(roads)
        if union_segments.geom_type != 'MultiPolygon':
            union_segments = MultiPolygon([union_segments])
        
        return union_segments
