from __future__ import annotations
from pydantic import BaseModel, Field, PositiveFloat, NonNegativeFloat, conint

class ConstraintConfig(BaseModel):
    gross_limit: PositiveFloat = 2.0
    net_target: float = 0.0
    position_cap_bps: PositiveFloat = 100.0
    leverage_target: PositiveFloat = 2.0
    adv_limit: PositiveFloat = 10.0
    enforce_beta_neutral: bool = True
    enforce_sector_neutral: bool = True
    beta_neutral: bool = True
    sector_neutral: bool = True
    beta_target: float = 0.0
    beta_tolerance: NonNegativeFloat = 0.001
    sector_tolerance: NonNegativeFloat = 0.0005
    turnover_gamma: NonNegativeFloat = 0.002
