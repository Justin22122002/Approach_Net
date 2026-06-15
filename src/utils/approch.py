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

    def get_final_approach_fixes(self) -> tuple[list[FixIdentifier], FixIdentifier | None, FixIdentifier | None]:
        segments = self._split_into_segments()

        final_seg_idx, rw_idx = None, None
        for i, segment in enumerate(segments):
            indices = [j for j, f in enumerate(segment) if f.fix_identifier.startswith("RW")]
            if indices:
                final_seg_idx, rw_idx = i, indices[0]
                break

        if final_seg_idx is None:
            return [], None, None

        final_segment = segments[final_seg_idx]
        rw_fix = final_segment[rw_idx]
        fix_before = final_segment[rw_idx - 1] if rw_idx >= 1 else None

        entry_fixes: list[FixIdentifier] = []

        if rw_idx >= 2:
            entry_fixes.append(final_segment[rw_idx - 2])

        if fix_before:
            for i, segment in enumerate(segments):
                if i == final_seg_idx:
                    continue
                for j, f in enumerate(segment):
                    if f.fix_identifier == fix_before.fix_identifier and j >= 1:
                        entry_fixes.append(segment[j - 1])

        return entry_fixes, fix_before, rw_fix
