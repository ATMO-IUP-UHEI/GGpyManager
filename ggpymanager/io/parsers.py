"""Log file parsers for GRAMM/GRAL simulation outputs."""

import logging
from pathlib import Path
from typing import Iterator

from ggpymanager.models.dataclasses import GRALLogMetadata, GRAMMLogMetadata


def parse_emission_data(
    l_iter: Iterator[str], has_parentheses: bool = False
) -> tuple[int, float]:
    """Parse emission data from log file lines.
    
    Parameters
    ----------
    l_iter : Iterator[str]
        Iterator over log file lines.
    has_parentheses : bool, optional
        Whether the emission count contains parentheses. Default is False.
        
    Returns
    -------
    tuple[int, float]
        Number of emissions and total emissions value.
    """
    n_line = next(l_iter).split(":")[1]
    if has_parentheses:
        n_emissions = int(n_line.split("(")[0])
    else:
        n_emissions = int(n_line)

    total_line = next(l_iter).split(":")[1]
    total_emissions = float(total_line.split("(")[0])

    return n_emissions, total_emissions


def parse_meteo_data(line: str) -> dict[str, float]:
    """Parse meteorological data from a log file line.
    
    Parameters
    ----------
    line : str
        Log file line containing meteorological data.
        
    Returns
    -------
    dict[str, float]
        Dictionary with wind_speed, direction, and stability_class.
    """
    parts = line.split(":")
    return {
        "wind_speed": float(parts[2].split()[0]),
        "direction": float(parts[3].split()[0]),
        "stability_class": float(parts[4].split()[0]),
    }


def filter_lines(raw_lines: list[str]) -> list[str]:
    """Filter and clean raw log file lines.
    
    Parameters
    ----------
    raw_lines : list[str]
        Raw lines from log file.
        
    Returns
    -------
    list[str]
        Cleaned and filtered lines.
    """
    lines = []
    for line in raw_lines:
        line = line.strip().lstrip("0: ")
        line = line.strip("|<>+- ")
        if line != "":
            lines.append(line)
    return lines


def read_gral_stdout(path: str) -> GRALLogMetadata:
    """Read and parse GRAL log file.
    
    Parameters
    ----------
    path : str
        Path to the GRAL log file.
        
    Returns
    -------
    GRALLogMetadata
        Parsed metadata from the GRAL log file.
    """
    lm = GRALLogMetadata()
    with Path(path).open() as f:
        raw_lines = f.readlines()

    lines = filter_lines(raw_lines)

    for i, l in enumerate(lines):
        l_iter = iter(lines[i + 1 :])

        try:
            match True:
                case _ if "VERSION" in l:
                    lm.version = l
                    lm.plattform = next(l_iter)
                    lm.dotnet_version = next(l_iter)

                case _ if "Source group count:" in l:
                    lm.n_source_groups = int(l.split(":")[1])

                case _ if "Reading GRAMM orography" in l:
                    lm.ggeom_file_read = True

                case _ if "Reading GRAL_topofile" in l:
                    lm.gral_topofile_read = True

                case _ if "Reading building file" in l:
                    lm.building_file_read = True

                case _ if (
                    "Total number of horizontal slices for concentration grid:" in l
                ):
                    n = int(l.split(":")[1])
                    lm.n_horizontal_slices = n
                    slice_height = []
                    for j in range(n):
                        slice_height.append(int(next(l_iter).split(":")[1]))

                case _ if "Reading file point.dat" in l:
                    lm.point_emissions_read = True
                    lm.n_point_emissions, lm.total_point_emissions = (
                        parse_emission_data(l_iter, has_parentheses=False)
                    )

                case _ if "Reading file line.dat" in l:
                    lm.line_emissions_read = True
                    lm.n_line_emissions, lm.total_line_emissions = (
                        parse_emission_data(l_iter, has_parentheses=True)
                    )

                case _ if "Reading file cadastre.dat" in l:
                    lm.area_emissions_read = True
                    lm.n_area_emissions, lm.total_area_emissions = (
                        parse_emission_data(l_iter, has_parentheses=False)
                    )

                case _ if "ADVECTION" in l:
                    lm.advection_computated = True
                    lm.numerical_stabilities = []
                    next_l = next(l_iter)
                    while next_l.startswith("ITERATION"):
                        lm.numerical_stabilities.append(float(next_l.split(":")[1]))
                        next_l = next(l_iter)

                case _ if "Obukhov length" in l:
                    lm.obukhov_length = float(l.split(":")[1])

                case _ if "Friction velocity" in l:
                    lm.friction_velocity = float(l.split(":")[1])

                case _ if "Boundary layer height" in l:
                    lm.boundary_layer_height = float(l.split(":")[1])

                case _ if "Init meteo:" in l:
                    meteo_data = parse_meteo_data(l)
                    lm.init_wind_speed = meteo_data["wind_speed"]
                    lm.init_direction = meteo_data["direction"]
                    lm.init_stability_class = meteo_data["stability_class"]

                case _ if "GRAMM meteo:" in l:
                    meteo_data = parse_meteo_data(l)
                    lm.gramm_wind_speed = meteo_data["wind_speed"]
                    lm.gramm_direction = meteo_data["direction"]
                    lm.gramm_stability_class = meteo_data["stability_class"]

                case _ if "Total simulation time" in l:
                    lm.total_simulation_time = float(l.split(":")[1])
                    lm.dispersion_time = float(next(l_iter).split(":")[1])
                    lm.flow_field_time = float(next(l_iter).split(":")[1])
        except Exception as e:
            logging.error(f"Error reading GRAL log file: {e}")
    return lm


def read_gramm_stdout(path: str) -> GRAMMLogMetadata:
    """Read and parse GRAMM log file.
    
    Parameters
    ----------
    path : str
        Path to the GRAMM log file.
        
    Returns
    -------
    GRAMMLogMetadata
        Parsed metadata from the GRAMM log file.
    """
    with Path(path).open() as f:
        raw_lines = f.readlines()

    lines = filter_lines(raw_lines)
    lm = GRAMMLogMetadata()

    for i, l in enumerate(lines):
        l_iter = iter(lines[i + 1 :])
        try:
            match True:
                case _ if "VERSION" in l:
                    lm.version = l
                    lm.plattform = next(l_iter)
                    lm.dotnet_version = next(l_iter)

                case _ if "maximum degree of parallelism" in l:
                    lm.n_processors = int(l.split(":")[1].split()[0])

                case _ if "Reading ggeom.asc" in l:
                    lm.ggeom_file_read = True
                    lm.min_elevation = float(l.split(":")[1].split()[0].rstrip("m"))
                    lm.max_elevation = float(l.split(":")[1].split()[1].rstrip("m"))

                case _ if "Wind direction" in l:
                    lm.init_direction = float(l.split(":")[1])

                case _ if "Wind speed" in l:
                    lm.init_wind_speed = float(l.split(":")[1].rstrip("m/s"))

                case _ if "U-component" in l:
                    lm.u_component = float(l.split(":")[1].rstrip("m/s"))

                case _ if "V-component" in l:
                    lm.v_component = float(l.split(":")[1].rstrip("m/s"))

                case _ if "Stability class" in l:
                    lm.init_stability_class = float(l.split(":")[1])

                case _ if "Obukhov length" in l:
                    lm.init_obukhov_length = float(l.split(":")[1].rstrip("m"))

                case _ if "Roughness length" in l:
                    lm.roughness_length = float(l.split(":")[1].rstrip("m"))

                case _ if "Boundary-Layer height" in l:
                    lm.init_boundary_layer_height = float(
                        l.split(":")[1].rstrip("m")
                    )

                case _ if "Friction velocity" in l:
                    lm.friction_velocity = float(l.split(":")[1].rstrip("m/s"))

                case _ if "WEATHER-SIT." in l:
                    next_l = next(l_iter)
                    lm.simulation_attempt.append(int(next_l.split()[0].split("/")[1]))
                    lm.simulation_time.append(float(next_l.split()[1]))
                    lm.simulation_timestep.append(float(next_l.split()[2]))
                    lm.simulation_divergence.append(float(next_l.split()[5]))
        except Exception as e:
            logging.error(f"Error reading GRAMM log file: {e}")
    return lm
