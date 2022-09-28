from pathlib import Path
from multiprocessing import Pool
import numpy as np
import xarray as xr

from .simulation import Simulation, Status
from .catalog import Catalog
from . import utils

class Reader(Catalog):
    def __init__(self, catalog_path, sim_path, config_path):
        super().__init__(catalog_path, sim_path, config_path)
        self.init_simulations()

    def get_landuse(self):
        """
        Returns
        -------
        thermal_conductivity : array
        heat_conductivity : array
        surface_roughness : array
        moisture : array
        surface_emissions : array
        albedo : array
        """
        return utils.read_landuse(self.config_path / "landuse.asc")

    def get_topography(self):
        """
        Returns
        -------
        topography : np.array, 2-D  
            Topography of the GRAMM domain
        zgrid : np.array, 3-D
            Structure of the GRAMM grid cells.
        """
        return utils.read_topography(self.config_path / "ggeom.asc")

    def get_gramm_windfield(self, sim_id):
        """
        Parameters
        ----------
        sim_id : int

        Returns
        -------
        info
            Info about the GRAMM run.
        wind_u
            Wind in x-direction.
        wind_v
            Wind in y-direction.
        """
        return utils.read_gramm_windfield(self.simulations[sim_id] / "00001.wnd")

    def get_buildings(self):
        """
        Returns
        -------
        buildings
            Building height.
        """
        return utils.read_buildings(self.config_path / "buildings.dat")
    
    def get_gral_geometries(self):
        """
        Returns
        -------
        ahk
            Surface elevation.
        kkart
            Index of gral surface.
        buildings
            Building height.
        oro
            Surface elevation without buildings.
        """
        # Read geometries from one simulation which finished
        path = self.get_simulations(Status.finished).sim_sub_path / "GRAL_geometries.txt"
        return utils.read_gral_geometries(path)

    def get_gral_windfield(self, sim_id):
        """
        Returns
        -------
        wind_u
            Wind in x-direction.
        wind_v
            Wind in y-direction.
        wind_z
            Wind in z-direction.
        """
        path = self.simulations[sim_id].sim_sub_path / "00001.gff"
        return utils.read_gral_windfield(path)

    def get_concentration(self, sim_id):
        """
        Gets the concentration as stored in the "con.npz" file in the simulation.

        Returns
        -------
        con_dict : dict of np.array
            Contains all concentrations from one simulation. The keys have the structure
            "hxx" with the height "h" and the source group "xx". Not all source groups
            have written output.
        """
        con_dict = {}
        path = self.simulations[sim_id].sim_sub_path / "con.npz"
        with np.load(path, allow_pickle=True, mmap_mode="r") as conc_file:            
            for key in conc_file:
                con_matrix = conc_file[key].all().toarray()
                con_dict[key] = con_matrix
        return con_dict
