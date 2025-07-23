"""Solver routines for the scheduling model."""

from typing import Optional
from ortools.sat.python import cp_model

from .model import ScheduleModel


def solve_model(sched: ScheduleModel, *, time_limit: Optional[int] = None) -> Optional[cp_model.CpSolver]:
    """Solve the provided model and return the solver instance."""
    solver = cp_model.CpSolver()
    if time_limit:
        solver.parameters.max_time_in_seconds = time_limit
    result = solver.Solve(sched.model)
    if result not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None
    return solver
