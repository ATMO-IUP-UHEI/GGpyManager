from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

CONFIG_PATH = "config/"
"""Path to the configuration directory for a GRAMM or GRAL model run."""
SIMULATION_PATH = "simulations/"
"""Path to the simulations directory for a GRAMM or GRAL model run."""
CATALOG_ENTRY_PATH_FORMATTER = "sim_{sim_id:04}/"
"""Formatter for the path to a specific simulation entry in the catalog."""
WIND_FILE_EXTENSION = {"gramm": "00001.wnd", "gral": "00001.gff"}
"""File extension for wind field result files for GRAMM and GRAL models."""
STD_OUT_FILE_NAME = {"gramm": "gramm-log.txt", "gral": "gral-log.txt"}
STD_OUT_STRING_FOR_COMPLETED_SIMULATION = {
    "gramm": "0:  MMAIN : OUT 00001.wnd  00001.scl",
    "gral": "",
}
"""Standard output string indicating a completed simulation for GRAMM and GRAL."""
INPUT_FILES = {
    "gramm": {
        "required": [
            "meteopgt.all",
            "GRAMM.geb",
            "GRAMMin.dat",
            "IIN.dat",
            "Max_Proc.txt",
        ],
        "optional": [
            "ggeom.asc",
            "landuse.asc",
        ],
    },
    "gral": {
        "required": [
            "meteopgt.all",
            "GRAL.geb",
            "in.dat",
            "Max_Proc.txt",
        ],
        "optional": [
            # GRAMM files
            "GRAMM.geb",
            "landuse.asc",
            "ggeom.asc",
            # GRAL files
            "GRAL_FlowFields.txt",
            "Integrationtime.txt",
            "buildings.dat",
            "micro_vert_layers.txt",
            "cadastre.dat",
            "line.dat",
            "point.dat",
            "GRAL_topofile.txt",
        ],
    },
}

# GGPyManager configuration constants
STATUS_LOG_FILE_NAME = "catalog_status.nc"
MATCHING_LOSS_FILE_NAME = "matching_loss.nc"
CONCENTRATION_TIMESERIES_FILE_NAME = "concentration_timeseries.nc"
GRAMM_METEO_TIMESERIES_FILE_NAME = "gramm_meteo_timeseries.nc"
GRAL_METEO_TIMESERIES_FILE_NAME = "gral_meteo_timeseries.nc"


# Pydantic models for configuration validation
class BBox(BaseModel):
    """Bounding box coordinates in the specified CRS."""

    model_config = ConfigDict(extra="forbid")

    x0: float = Field(..., description="Western boundary coordinate")
    y0: float = Field(..., description="Southern boundary coordinate")
    x1: float = Field(..., description="Eastern boundary coordinate")
    y1: float = Field(..., description="Northern boundary coordinate")

    @field_validator("x1")
    @classmethod
    def x1_greater_than_x0(cls, v, info):
        if "x0" in info.data and v <= info.data["x0"]:
            raise ValueError("x1 must be greater than x0")
        return v


class GralConfig(BaseModel):
    """GRAL model configuration parameters."""

    model_config = ConfigDict(extra="forbid")

    conf_path: str = Field(..., description="Path to GRAL configuration directory")
    bbox: BBox = Field(..., description="GRAL domain bounding box")
    dx: float = Field(..., gt=0, description="Grid spacing in x-direction (meters)")
    dy: float = Field(..., gt=0, description="Grid spacing in y-direction (meters)")


class GrammConfig(BaseModel):
    """GRAMM model configuration parameters."""

    model_config = ConfigDict(extra="forbid")

    conf_path: str = Field(..., description="Path to GRAMM configuration directory")
    bbox: BBox = Field(..., description="GRAMM domain bounding box")
    dx: float = Field(..., gt=0, description="Grid spacing in x-direction (meters)")
    dy: float = Field(..., gt=0, description="Grid spacing in y-direction (meters)")
    nz: int = Field(..., gt=0, description="Number of vertical grid levels")
    z0: float = Field(..., gt=0, description="Height of first vertical level (meters)")
    vert_stretching: float = Field(
        ..., gt=1.0, description="Vertical stretching factor"
    )


class Domain(BaseModel):
    """Spatial domain configuration for GRAMM and GRAL models."""

    model_config = ConfigDict(extra="forbid")

    crs: str = Field(..., description="Coordinate reference system (e.g., EPSG:2154)")
    gral: GralConfig = Field(..., description="GRAL model domain configuration")
    gramm: GrammConfig = Field(..., description="GRAMM model domain configuration")


class Fluxes(BaseModel):
    """Emission flux grid configuration."""

    model_config = ConfigDict(extra="forbid")

    nx_areas: int = Field(
        ..., gt=0, description="Number of emission areas in x-direction"
    )
    ny_areas: int = Field(
        ..., gt=0, description="Number of emission areas in y-direction"
    )


class Matching(BaseModel):
    """Configuration for matching simulations with station measurements."""

    model_config = ConfigDict(extra="forbid")

    stations: dict[str, Literal["gral", "gramm"]] = Field(
        ..., description="Station names mapped to their model domain (gral or gramm)"
    )
    time_start: str = Field(
        ..., description="Start time for matching period (YYYY-MM-DD HH:MM)"
    )
    time_end: str = Field(
        ..., description="End time for matching period (YYYY-MM-DD HH:MM)"
    )
    n_best_simulations: int = Field(
        ..., gt=0, description="Number of best-matching simulations to select"
    )

    @field_validator("time_start", "time_end")
    @classmethod
    def validate_datetime_format(cls, v):
        try:
            datetime.strptime(v, "%Y-%m-%d %H:%M")
        except ValueError:
            raise ValueError(f"Time must be in format YYYY-MM-DD HH:MM, got: {v}")
        return v


class Config(BaseModel):
    """Main configuration for Paris 2025 simulation project."""

    model_config = ConfigDict(extra="forbid")

    main_path: str = Field(..., description="Main working directory path")
    data_path: str = Field(..., description="Path to input data directory")
    figures_path: str = Field(..., description="Output directory for figures")
    gramm_gral_conf_path: str = Field(
        ..., description="Path to GRAMM-GRAL configuration data"
    )
    output_path: str = Field(..., description="Main output directory")
    temporal_profiles_path: str = Field(
        ..., description="Path to temporal emission profiles (NetCDF)"
    )
    source_groups_path: str = Field(
        ..., description="Path to emission source groups (NetCDF)"
    )
    meteo_path: str = Field(..., description="Path to meteorological measurements")
    gramm_meteo_path: str = Field(
        ..., description="Path to GRAMM meteorological output at stations"
    )
    gral_meteo_path: str = Field(
        ..., description="Path to GRAL meteorological output at stations"
    )
    gral_co2_path: str = Field(..., description="Path to GRAL CO2 output at stations")
    domain: Domain = Field(..., description="Spatial domain configuration")
    fluxes: Fluxes = Field(..., description="Emission flux configuration")
    matching: Matching = Field(..., description="Station matching configuration")
