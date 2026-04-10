"""
Some constant parameters related to map objects.
"""

from worldengine.utils.type import WorldEngineObjectType


class MapTerrainSemanticColor:
    """ Color properties for map objects. """
    # YELLOW = 0.1
    # WHITE = 0.3

    # B,G,R
    GREY = (1, 1, 1, 1)
    YELLOW = (255 / 255, 200 / 255, 0 / 255, 1) # actually this is cyan

    START_POINT_COLOR = (1, 0, 0, 1)
    END_POINT_COLOR = (0, 0, 1, 1)
    WAYPOINT_COLOR = (0, 1, 0, 1)

    LANE_COLOR = (0.71, 0.7, 0., 1)
    ROUTE_COLOR = (121 / 255, 219 / 255, 101 / 255)

    CROSSWALK_COLOR = GREY
    LAND_COLOR = (0.4, 0.4, 0.4, 1)
    NAVI_COLOR = (0, 0.709, 0.709, 1)
    OTHER_AGENT_COLOR = (0, 0, 255, 1)

    @staticmethod
    def get_color(type):
        """
        Each channel represents a type. This should be aligned with shader terrain.frag.glsl
        Args:
            type: MetaDriveType

        Returns:

        """
        if WorldEngineObjectType.is_yellow_line(type):
            # return (255, 0, 0, 0)
            # return (1, 0, 0, 0)
            return MapTerrainSemanticColor.YELLOW
        elif WorldEngineObjectType.is_lane(type):
            # return 0.2
            return MapTerrainSemanticColor.LANE_COLOR
        elif type == WorldEngineObjectType.GROUND:
            # return 0.0
            return MapTerrainSemanticColor.LAND_COLOR
        elif (WorldEngineObjectType.is_white_line(type) or
              WorldEngineObjectType.is_road_boundary_line(type)):
            # return (0, 0, 0, 1)
            return MapTerrainSemanticColor.GREY
        elif type == WorldEngineObjectType.CROSSWALK:
            # return 0.4
            return MapTerrainSemanticColor.CROSSWALK_COLOR
        else:
            raise ValueError("Unsupported type: {}".format(type))


class DrivableAreaProperty:
    """ Bounding box or geometry properties for map objects.
    This is used for collision computation between foreground objects and map elements.
    """
    # road network property
    ID = None  # each block must have a unique ID
    SOCKET_NUM = None

    # visualization size property
    LANE_SEGMENT_LENGTH = 4
    STRIPE_LENGTH = 1.5
    LANE_LINE_WIDTH = 0.15
    LANE_LINE_THICKNESS = 0.016

    SIDEWALK_THICKNESS = 0.3
    SIDEWALK_LENGTH = 3
    SIDEWALK_WIDTH = 2
    SIDEWALK_LINE_DIST = 0.6

    GUARDRAIL_HEIGHT = 4.0

    # visualization color property
    LAND_COLOR = (0.4, 0.4, 0.4, 1)
    NAVI_COLOR = (0.709, 0.09, 0, 1)

    # for detection
    LANE_LINE_GHOST_HEIGHT = 1.0

    # TODO: lane line collision group
    # CONTINUOUS_COLLISION_MASK = 
    # BROKEN_COLLISION_MASK = 
    # SIDEWALK_COLLISION_MASK = 

    # for creating complex block, for example Intersection and roundabout consist of 4 part, which contain several road
    PART_IDX = 0
    ROAD_IDX = 0
    DASH = "_"

    #  when set to True, Vehicles will not generate on this block
    PROHIBIT_TRAFFIC_GENERATION = False



class PGLineType:
    """A lane side line type."""

    NONE = WorldEngineObjectType.LINE_UNKNOWN
    BROKEN = WorldEngineObjectType.LINE_BROKEN_SINGLE_WHITE
    CONTINUOUS = WorldEngineObjectType.LINE_SOLID_SINGLE_WHITE
    SIDE = WorldEngineObjectType.BOUNDARY_LINE
    GUARDRAIL = WorldEngineObjectType.GUARDRAIL

    @staticmethod
    def prohibit(line_type) -> bool:  # whether a lane can be passed.
        if line_type in [PGLineType.CONTINUOUS, PGLineType.SIDE]:
            return True
        else:
            return False


class PGLineColor:
    GREY = (1, 1, 1, 1)
    YELLOW = (255 / 255, 200 / 255, 0 / 255, 1)
