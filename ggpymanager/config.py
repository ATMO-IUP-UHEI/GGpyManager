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
    "gramm": [
        "meteopgt.all",
        "GRAMM.geb",
        "landuse.asc",
        "ggeom.asc",
        "DispNrGramm.txt",
        "GRAMMin.dat",
        "PercentGramm.txt",
        "IIN.dat",
        "Max_Proc.txt",
        "RoughnessUsed.txt",
    ],
    "gral": [
        "meteopgt.all",
        "GRAMM.geb",
        "landuse.asc",
        "ggeom.asc",
        "GRAL.geb",
        "buildings.dat",
        "in.dat",
        "micro_vert_layers.txt",
        "GRAL_FlowFields.txt",
        "Integrationtime.txt",
        "cadastre.dat",
        "point.dat",
        "GRAL_topofile.txt",
        "Max_Proc.txt",
    ],
}

# GGPyManager configuration constants
STATUS_LOG_FILE_NAME = "catalog_status.nc"
