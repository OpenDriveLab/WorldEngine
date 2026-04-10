# DETECTION_NAMES_NUPLAN = ['vehicle', 'bicycle', 'pedestrian', 'traffic_cone', 'barrier', 'czone_sign', 'generic_object']

PRETTY_DETECTION_NAMES = {
    "Car": "Car",
    "Pedestrian": "Pedestrian",
    "Motorcycle": "Motorcycle",
    "Cyclist": "Cyclist",
}

DETECTION_COLORS = {
    "Car": "C0",
    "Pedestrian": "C5",
    "Motorcycle": "C6",
    "Cyclist": "C7",
}

ATTRIBUTE_NAMES = [
    "pedestrian.moving",
    "pedestrian.sitting_lying_down",
    "pedestrian.standing",
    "cycle.with_rider",
    "cycle.without_rider",
    "vehicle.moving",
    "vehicle.parked",
    "vehicle.stopped",
]

PRETTY_ATTRIBUTE_NAMES = {
    "pedestrian.moving": "Ped. Moving",
    "pedestrian.sitting_lying_down": "Ped. Sitting",
    "pedestrian.standing": "Ped. Standing",
    "cycle.with_rider": "Cycle w/ Rider",
    "cycle.without_rider": "Cycle w/o Rider",
    "vehicle.moving": "Veh. Moving",
    "vehicle.parked": "Veh. Parked",
    "vehicle.stopped": "Veh. Stopped",
}
