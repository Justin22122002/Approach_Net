from dataclasses import dataclass

from utils.fix_identifier import FixIdentifier


@dataclass
class Approach:
    procedure_identifier: str
    route_type: str  # R, A, I
    fixes: list[FixIdentifier]

    def _split_into_segments(self) -> list[list[FixIdentifier]]:
        segments: list[list[FixIdentifier]] = []
        current: list[FixIdentifier] = []

        for fix in self.fixes:
            seq = float(fix.sequence_number) if str(fix.sequence_number).replace('.', '', 1).isdigit() else 0.0
            if current:
                prev_seq = float(current[-1].sequence_number) if str(current[-1].sequence_number).replace('.', '', 1).isdigit() else 0.0
                if seq <= prev_seq:
                    segments.append(current)
                    current = []
            current.append(fix)

        if current:
            segments.append(current)

        return segments

    def get_final_approach_fixes(
            self,
    ) -> tuple[
        list[FixIdentifier],
        FixIdentifier | None,
        FixIdentifier | None,
    ]:
        segments = self._split_into_segments()

        final_seg_idx: int | None = None
        rw_idx: int | None = None

        for i, segment in enumerate(segments):
            for j, fix in enumerate(segment):
                if fix.fix_identifier.startswith("RW"):
                    final_seg_idx = i
                    rw_idx = j
                    break

            if final_seg_idx is not None:
                break

        if final_seg_idx is None or rw_idx is None:
            return [], None, None

        final_segment = segments[final_seg_idx]
        rw_fix = final_segment[rw_idx]

        if_fix: FixIdentifier | None = None
        faf_fix: FixIdentifier | None = None

        for fix in final_segment:
            fix_type = fix.get_fix_type()

            if fix_type == "IF" and if_fix is None:
                if_fix = fix

            elif fix_type == "FAF" and faf_fix is None:
                faf_fix = fix

        if if_fix is not None:
            entry_fixes = [if_fix]

        else:
            entry_fixes = []

            if rw_idx >= 2:
                entry_fixes.append(final_segment[rw_idx - 2])

            fix_before = (
                final_segment[rw_idx - 1]
                if rw_idx >= 1
                else None
            )

            if fix_before:
                for i, segment in enumerate(segments):
                    if i == final_seg_idx:
                        continue

                    for j, fix in enumerate(segment):
                        if (
                                fix.fix_identifier == fix_before.fix_identifier
                                and j >= 1
                        ):
                            entry_fixes.append(segment[j - 1])

        if faf_fix is None:
            faf_fix = (
                final_segment[rw_idx - 1]
                if rw_idx >= 1
                else None
            )

        return entry_fixes, faf_fix, rw_fix