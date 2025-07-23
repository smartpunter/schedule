"""Model construction helpers for the scheduling problem."""

from typing import Any
from ortools.sat.python import cp_model

from .preprocessing import PreprocessedData


class ScheduleModel:
    """Wrapper around a CP-SAT model and its variables."""

    def __init__(self) -> None:
        self.model = cp_model.CpModel()
        # TODO: store intervals and helper variables


def build_feasible_model(data: PreprocessedData) -> ScheduleModel:
    """Build a basic model that only enforces hard constraints."""
    sched = ScheduleModel()
    # TODO: convert `data` into interval variables
    return sched


def build_optimized_model(base: ScheduleModel, data: PreprocessedData) -> ScheduleModel:
    """Extend the base model with soft constraints and objectives."""
    # TODO: add penalties and objective function
    return base
