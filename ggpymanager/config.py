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


# NetCDF metadata

# Define metadata for each variable that are CF compliant
NETCDF_METADATA = {
    "sim_id": {
        "long_name": "Simulation ID",
        "description": "Unique identifier for each simulation",
    },
    "iteration": {
        "long_name": "Iteration step",
        "description": "Iteration step index for time-varying simulation parameters",
    },
    "version": {
        "long_name": "Model version",
        "description": "GRAMM/GRAL version string",
    },
    "plattform": {
        "long_name": "Platform",
        "description": "Operating system platform",
    },
    "dotnet_version": {
        "long_name": ".NET version",
        "description": ".NET runtime version",
    },
    "n_processors": {
        "long_name": "Number of processors",
        "units": "1",
        "description": "Number of CPU cores used",
    },
    "n_source_groups": {
        "long_name": "Number of source groups",
        "description": "Number of emission source groups in simulation",
    },
    "n_point_emissions": {
        "long_name": "Number of point emissions",
        "description": "Number of point emission sources",
    },
    "n_line_emissions": {
        "long_name": "Number of line emissions",
        "description": "Number of line emission sources",
    },
    "n_area_emissions": {
        "long_name": "Number of area emissions",
        "description": "Number of area emission sources",
    },
    "total_point_emissions": {
        "long_name": "Total point emissions",
        "units": "kg h-1",
        "description": "Total emission rate from point sources",
    },
    "total_line_emissions": {
        "long_name": "Total line emissions",
        "units": "kg h-1",
        "description": "Total emission rate from line sources",
    },
    "total_area_emissions": {
        "long_name": "Total area emissions",
        "units": "kg h-1",
        "description": "Total emission rate from area sources",
    },
    "n_horizontal_slices": {
        "long_name": "Number of horizontal slices",
        "description": "Number of horizontal slices in concentration grid",
    },
    "ggeom_file_read": {
        "long_name": "Geometry file read status",
        "description": "Whether geometry file was successfully read",
    },
    "gral_topofile_read": {
        "long_name": "GRAL topography file read status",
        "description": "Whether GRAL topography file was successfully read",
    },
    "building_file_read": {
        "long_name": "Building file read status",
        "description": "Whether building file was successfully read",
    },
    "point_emissions_read": {
        "long_name": "Point emissions read status",
        "description": "Whether point emission data was successfully read",
    },
    "line_emissions_read": {
        "long_name": "Line emissions read status",
        "description": "Whether line emission data was successfully read",
    },
    "area_emissions_read": {
        "long_name": "Area emissions read status",
        "description": "Whether area emission data was successfully read",
    },
    "advection_computated": {
        "long_name": "Advection computation status",
        "description": "Whether advection was computed in the simulation",
    },
    "numerical_stabilities": {
        "long_name": "Numerical stabilities",
        "description": "Numerical stability indicators during simulation",
    },
    "min_elevation": {
        "long_name": "Minimum elevation",
        "units": "m",
        "description": "Minimum terrain elevation in domain",
    },
    "max_elevation": {
        "long_name": "Maximum elevation",
        "units": "m",
        "description": "Maximum terrain elevation in domain",
    },
    "init_wind_speed": {
        "long_name": "Initial wind speed",
        "units": "m s-1",
        "description": "Initial wind speed for simulation",
    },
    "init_direction": {
        "long_name": "Initial wind direction",
        "units": "degree",
        "description": "Initial wind direction",
    },
    "u_component": {
        "long_name": "U wind component",
        "units": "m s-1",
        "description": "Zonal wind component",
    },
    "v_component": {
        "long_name": "V wind component",
        "units": "m s-1",
        "description": "Meridional wind component",
    },
    "init_stability_class": {
        "long_name": "Initial stability class",
        "description": "Atmospheric stability class",
    },
    "init_obukhov_length": {
        "long_name": "Initial Obukhov length",
        "units": "m",
        "description": "Initial Obukhov length for stability",
    },
    "obukhov_length": {
        "long_name": "Obukhov length",
        "units": "m",
        "description": "Obukhov length during simulation",
    },
    "boundary_layer_height": {
        "long_name": "Boundary layer height",
        "units": "m",
        "description": "Height of atmospheric boundary layer",
    },
    "roughness_length": {
        "long_name": "Surface roughness length",
        "units": "m",
        "description": "Aerodynamic roughness length",
    },
    "init_boundary_layer_height": {
        "long_name": "Initial boundary layer height",
        "units": "m",
        "description": "Initial height of atmospheric boundary layer",
    },
    "friction_velocity": {
        "long_name": "Friction velocity",
        "units": "m s-1",
        "description": "Surface friction velocity",
    },
    "simulation_attempt": {
        "long_name": "Simulation attempt number",
        "description": "If there were retries, this indicates the attempt number",
    },
    "simulation_time": {
        "long_name": "Simulation time",
        "units": "s",
        "description": "Total simulation time",
    },
    "total_simulation_time": {
        "long_name": "Total simulation time",
        "units": "s",
        "description": "Total time taken for the simulation",
    },
    "dispersion_time": {
        "long_name": "Dispersion time",
        "units": "s",
        "description": "Time taken for dispersion calculations",
    },
    "flow_field_time": {
        "long_name": "Flow field time",
        "units": "s",
        "description": "Time taken for flow field calculations",
    },
    "simulation_timestep": {
        "long_name": "Simulation timestep",
        "units": "s",
        "description": "Time step size",
    },
    "simulation_divergence": {
        "long_name": "Simulation divergence",
        "description": "Maximum divergence value",
    },
    "gramm_wind_speed": {
        "long_name": "GRAMM wind speed",
        "units": "m s-1",
        "description": "Wind speed output from GRAMM model",
    },
    "gramm_wind_direction": {
        "long_name": "GRAMM wind direction",
        "units": "degree",
        "description": "Wind direction output from GRAMM model",
    },
    "gramm_stability_class": {
        "long_name": "GRAMM stability class",
        "description": "Stability class output from GRAMM model",
    },
}
