from dataclasses import dataclass


@dataclass
class FixIdentifier:
    fix_identifier: str
    path_terminator: str # e.g -> IF, TF etc, note: this is not IF, IAF, FAF, or MAP ! -> look for Description_Code -> 4th position describes the type of the waypoint -> A -> IAF, B -> IF, C -> FAF, M -> MAP, etc
    latitude: float
    longitude: float
    altitude: float
    sequence_number: str

    course: float | None
    speed_kts: float | None
    turn_direction: str | None
    altitude_restriction: str | None
