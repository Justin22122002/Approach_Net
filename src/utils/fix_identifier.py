from dataclasses import dataclass


@dataclass
class FixIdentifier:
    """
    Path terminator codes for ARINC 424 procedure legs.

    Code       Meaning
    ---------- ------------------------------------------------------------
    IF         Initial Fix: start of a route or transition.
    TF         Track to Fix: straight track from the previous fix to this fix.
    CF         Course to Fix: defined course to a fix.
    RF         Radius to Fix: constant-radius turn to a fix.
    DF         Direct to Fix: direct course from the previous position to fix.
    CA         Course to Altitude: fly a defined course until reaching altitude.
    VA         Heading to Altitude: fly a defined heading until reaching altitude.
    VI         Heading to Intercept: fly a heading to intercept a course.
    HM         Holding, manual termination.
    HF         Holding until the procedure's fix is reached again.
    HA         Holding until reaching a specified altitude.
    PI         Procedure turn / racetrack manoeuvre.
    """

    fix_identifier: str
    path_terminator: str # e.g -> IF, TF etc, note: this is not IF, IAF, FAF, or MAP ! -> look for Description_Code -> 4th position describes the type of the waypoint -> A -> IAF, B -> IF, C -> FAF, M -> MAP, etc
    latitude: float
    longitude: float
    altitude: float
    sequence_number: str
    description_code: str | None

    course: float | None
    speed_kts: float | None
    turn_direction: str | None
    altitude_restriction: str | None

    def get_fix_type(self) -> str | None:
        if self.description_code is None:
            return None

        code = self.description_code.ljust(4)

        fix_type = code[3]

        match fix_type:
            case "A":
                return "IAF"
            case "B":
                return "IF"
            case "C":
                return "IAF_HOLDING"
            case "D":
                return "IAF_WITH_FACF"
            case "E":
                return "FINAL_END_POINT"
            case "F":
                return "FAF"
            case "H":
                return "HOLDING_FIX"
            case "I":
                return "FACF"
            case "M":
                return "MAP"
            case _:
                return None
