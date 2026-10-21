import numpy as np
from torch import nn, Tensor
from typing_extensions import override

from evaluation.utils.rollout_evaluator import DATASET_NAME, RolloutEvaluator
from ml.utils.data_loader.landing_action_dataset import LandingActionDataset
from ml.utils.run_cli import exit_if_test, parse_run_args
from engine.physics.landing_optimizer_rnav import LandingTrajectoryRnavOptimizer, state_to_casadi


class RnavOptimizerRollout(RolloutEvaluator):
    name: str = "rnav_optimizer"

    def __init__(
        self,
        dataset_name: str = DATASET_NAME,
        N: int = 200,
        T_min: float = 60.0,
        T_max: float = 1800.0,
        glideslope_deg: float = 3.0,
        resume: bool = True,
        mode: str = "bspline_state",
    ) -> None:
        self.constrained, self.resume, self.mode = False, resume, mode
        self.safety_layer = None
        self.solver = None
        self.run_id = dataset_name
        self.N, self.T_min, self.T_max = N, T_min, T_max
        self.tan_gs = float(np.tan(np.radians(glideslope_deg)))

    @override
    def create_model(self, input_dim: int, output_dim: int) -> nn.Module:
        raise NotImplementedError("RNAV optimizer has no model")

    @override
    def build_safety_layer(self) -> nn.Module:
        raise NotImplementedError("RNAV optimizer has no safety layer")

    @override
    def safe_predict(self, x_norm: Tensor, state_feat: np.ndarray) -> Tensor:
        raise NotImplementedError("RNAV optimizer has no safety layer")

    @override
    def step(self, state, pred: np.ndarray) -> dict:
        raise NotImplementedError("RNAV optimizer solves the whole trajectory at once")

    def approach_points(self) -> tuple[np.ndarray, np.ndarray, float]:
        fx, fy, _, bearing = self.rwy_feat.tolist()
        faf_x, faf_y = float(self.approach[0]), float(self.approach[1])

        b = np.radians(bearing)
        s_faf = -((faf_x - fx) * np.sin(b) + (faf_y - fy) * np.cos(b))
        faf_h = self.h_thr + s_faf * self.tan_gs

        goal = np.array([fx + self.cx, fy + self.cy, self.h_thr + self.airport_elev])
        via = np.array([faf_x + self.cx, faf_y + self.cy, faf_h + self.airport_elev])
        return goal, via, float(bearing)

    @override
    def rollout_flight(self, ds: LandingActionDataset, idx: int) -> tuple[list[dict], bool, float]:
        if idx not in ds._cache:
            raise ValueError("flight skipped by LandingActionDataset (no approach / no valid samples)")
        flights = ds.flight_dataset
        icao24, flight_id = flights.flights[idx]

        self.setup_flight(ds, idx)
        _, self.dynamics = flights.get_aircraft(icao24)
        start_point, _ = flights.get_start_and_end_points(flights[idx])
        state = self.build_start_state(start_point, self.dynamics, self.to_utm)

        goal, via, bearing = self.approach_points()
        x0 = state_to_casadi(state)
        k_via_frac_start, k_via_frac_end = self.via_window(x0, via, goal)
        print(f"via window: {k_via_frac_start:.2f} - {k_via_frac_end:.2f}")

        optimizer = LandingTrajectoryRnavOptimizer(
            state=state, N=self.N, T_min=self.T_min, T_max=self.T_max,
            k_via_frac_start=k_via_frac_start, k_via_frac_end=k_via_frac_end,
        )
        X_opt, U_opt, T_opt, success = optimizer.solve(
            x0=x0,
            goal=goal,
            heading_goal_deg=bearing,
            via_point=via,
        )
        solve_time_s = optimizer.last_solving_time
        if np.isnan(solve_time_s):
            print("[WARNING] solving time is NaN, check the solver settings")

        dt = T_opt / self.N
        rows: list[dict] = []
        for k in range(self.N + 1):
            rows.append({
                "icao24": icao24,
                "flight_id": flight_id,
                "step": k,
                "time_s": k * dt,
                "x": float(X_opt[0, k]),
                "y": float(X_opt[1, k]),
                "h": float(X_opt[2, k]),
                "v": float(X_opt[3, k]),
                "chi": float(X_opt[4, k]),
                "gamma": float(X_opt[5, k]),
                "thrust": float(U_opt[0, k]) if k < self.N else None,
                "load_factor": float(U_opt[1, k]) if k < self.N else None,
                "roll": float(U_opt[2, k]) if k < self.N else None,
            })
        return rows, bool(success), solve_time_s

    @staticmethod
    def via_window(
        x0: np.ndarray, via: np.ndarray, goal: np.ndarray,
        final_slowdown: float = 1.3, half_width: float = 0.15,
    ) -> tuple[float, float]:
        d_start_faf = float(np.linalg.norm(via[:2] - x0[:2]))
        d_faf_goal = float(np.linalg.norm(goal[:2] - via[:2]))
        frac = d_start_faf / max(d_start_faf + final_slowdown * d_faf_goal, 1.0)
        start = float(np.clip(frac - half_width, 0.10, 0.80))
        end = float(np.clip(frac + half_width, start + 0.05, 0.95))
        return start, end


if __name__ == "__main__":
    args = parse_run_args()
    exit_if_test(args, __file__, dataset_name=DATASET_NAME)

    rollout = RnavOptimizerRollout(dataset_name=DATASET_NAME, N=200, T_min=60.0, T_max=1800.0)
    rollout.run_airports(
        output_folder="rnav_optimized",
        dataset_name=rollout.run_id,
        data_root="./data",
    )
