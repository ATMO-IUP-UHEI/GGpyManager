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