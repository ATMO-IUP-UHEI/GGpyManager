"""Dataclasses for GRAMM/GRAL log metadata."""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class GRALLogMetadata:
    """Metadata extracted from GRAL log files.

    Attributes
    ----------
    version : Optional[str]
        GRAL version string.
    plattform : Optional[str]
        Platform information.
    dotnet_version : Optional[str]
        .NET version information.
    n_source_groups : Optional[int]
        Number of source groups.
    ggeom_file_read : bool
        Whether GRAMM geometry file was read.
    gral_topofile_read : bool
        Whether GRAL topography file was read.
    building_file_read : bool
        Whether building file was read.
    n_horizontal_slices : Optional[int]
        Number of horizontal concentration slices.
    point_emissions_read : bool
        Whether point emissions were read.
    n_point_emissions : Optional[int]
        Number of point emissions.
    total_point_emissions : Optional[float]
        Total point emissions.
    line_emissions_read : bool
        Whether line emissions were read.
    n_line_emissions : Optional[int]
        Number of line emissions.
    total_line_emissions : Optional[float]
        Total line emissions.
    area_emissions_read : bool
        Whether area emissions were read.
    n_area_emissions : Optional[int]
        Number of area emissions.
    total_area_emissions : Optional[float]
        Total area emissions.
    advection_computated : bool
        Whether advection was computed.
    numerical_stabilities : List[float]
        List of numerical stability values.
    obukhov_length : Optional[float]
        Obukhov length.
    boundary_layer_height : Optional[float]
        Boundary layer height.
    friction_velocity : Optional[float]
        Friction velocity.
    init_wind_speed : Optional[float]
        Initial wind speed.
    init_direction : Optional[float]
        Initial wind direction.
    init_stability_class : Optional[float]
        Initial stability class.
    gramm_wind_speed : Optional[float]
        GRAMM wind speed.
    gramm_direction : Optional[float]
        GRAMM wind direction.
    gramm_stability_class : Optional[float]
        GRAMM stability class.
    total_simulation_time : Optional[float]
        Total simulation time.
    dispersion_time : Optional[float]
        Dispersion calculation time.
    flow_field_time : Optional[float]
        Flow field calculation time.
    """

    version: Optional[str] = None
    plattform: Optional[str] = None
    dotnet_version: Optional[str] = None
    n_source_groups: Optional[int] = None
    ggeom_file_read: bool = False
    gral_topofile_read: bool = False
    building_file_read: bool = False
    n_horizontal_slices: Optional[int] = None
    point_emissions_read: bool = False
    n_point_emissions: Optional[int] = None
    total_point_emissions: Optional[float] = None
    line_emissions_read: bool = False
    n_line_emissions: Optional[int] = None
    total_line_emissions: Optional[float] = None
    area_emissions_read: bool = False
    n_area_emissions: Optional[int] = None
    total_area_emissions: Optional[float] = None
    advection_computated: bool = False
    numerical_stabilities: List[float] = field(default_factory=list)
    obukhov_length: Optional[float] = None
    boundary_layer_height: Optional[float] = None
    friction_velocity: Optional[float] = None
    init_wind_speed: Optional[float] = None
    init_direction: Optional[float] = None
    init_stability_class: Optional[float] = None
    gramm_wind_speed: Optional[float] = None
    gramm_direction: Optional[float] = None
    gramm_stability_class: Optional[float] = None
    total_simulation_time: Optional[float] = None
    dispersion_time: Optional[float] = None
    flow_field_time: Optional[float] = None


@dataclass
class GRAMMLogMetadata:
    """Metadata extracted from GRAMM log files.

    Attributes
    ----------
    version : Optional[str]
        GRAMM version string.
    plattform : Optional[str]
        Platform information.
    dotnet_version : Optional[str]
        .NET version information.
    n_processors : Optional[int]
        Number of processors used.
    ggeom_file_read : bool
        Whether geometry file was read.
    min_elevation : Optional[float]
        Minimum elevation in domain.
    max_elevation : Optional[float]
        Maximum elevation in domain.
    init_wind_speed : Optional[float]
        Initial wind speed.
    init_direction : Optional[float]
        Initial wind direction.
    u_component : Optional[float]
        U wind component.
    v_component : Optional[float]
        V wind component.
    init_stability_class : Optional[float]
        Initial stability class.
    init_obukhov_length : Optional[float]
        Initial Obukhov length.
    roughness_length : Optional[float]
        Surface roughness length.
    init_boundary_layer_height : Optional[float]
        Initial boundary layer height.
    friction_velocity : Optional[float]
        Friction velocity.
    simulation_attempt : List[int]
        List of simulation attempt numbers.
    simulation_time : List[float]
        List of simulation times.
    simulation_timestep : List[float]
        List of simulation timesteps.
    simulation_divergence : List[float]
        List of simulation divergence values.
    """

    version: Optional[str] = None
    plattform: Optional[str] = None
    dotnet_version: Optional[str] = None
    n_processors: Optional[int] = None
    ggeom_file_read: bool = False
    min_elevation: Optional[float] = None
    max_elevation: Optional[float] = None
    init_wind_speed: Optional[float] = None
    init_direction: Optional[float] = None
    u_component: Optional[float] = None
    v_component: Optional[float] = None
    init_stability_class: Optional[float] = None
    init_obukhov_length: Optional[float] = None
    roughness_length: Optional[float] = None
    init_boundary_layer_height: Optional[float] = None
    friction_velocity: Optional[float] = None
    simulation_attempt: List[int] = field(default_factory=list)
    simulation_time: List[float] = field(default_factory=list)
    simulation_timestep: List[float] = field(default_factory=list)
    simulation_divergence: List[float] = field(default_factory=list)
