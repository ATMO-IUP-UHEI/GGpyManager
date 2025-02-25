import numpy as np
from ggpymanager import utils


def test_wind_conversion():
    # Test the conversion functions for wind speed and direction
    test_cases = {
        "N": (0, -1, 0),
        "NE": (-1, -1, 45),
        "E": (-1, 0, 90),
        "SE": (-1, 1, 135),
        "S": (0, 1, 180),
        "SW": (1, 1, 225),
        "W": (1, 0, 270),
        "NW": (1, -1, 315),
    }
    for key, data in test_cases.items():
        print(key)
        vector = data[:2]
        direction = data[2]
        computed_direction = utils.direction_from_vector(*vector)
        computed_speed = utils.wind_speed_from_vector(*vector)
        computed_vector = utils.vector_from_direction_and_speed(
            computed_direction, computed_speed
        )
        # Assert
        np.testing.assert_allclose(vector, computed_vector, atol=1e-6)
        np.testing.assert_allclose(direction, computed_direction, atol=1e-6)
