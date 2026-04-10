"""
Set of methods to convert specific dataset to the unified WorldEngine format:

The world engine format should be a dictionary as the following:
<scenario_id>:
    <id>: scenario_id
    <length>: the length of this scenario.
    <tracks>:
        states of different objects,
        <track_id> / <object_id>:
            <state:>
                <position>: position in the world coordinates.
                <heading>: heading in the world coordinates.
                <velocity>:
                <length>:
                <width>:
                <height>:
            <metadata>:
                <track_length>: length of the trajectory.
                <type>: class of the object.
                <object_id>: instance_id of the object.
    <map_features>:

"""