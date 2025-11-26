"""Unit tests for utils.projections module."""

import pytest

from ggpymanager.utils.projections import get_centered_custom_projection


class TestGetCenteredCustomProjection:
    """Tests for get_centered_custom_projection function."""

    def test_get_centered_custom_projection_returns_crs(self):
        """Test that function returns a CRS object."""
        result = get_centered_custom_projection(lat=49.0, lon=8.5)

        assert result is not None
        # Check if it's a valid CRS (has typical CRS attributes/methods)
        assert hasattr(result, "to_string") or hasattr(result, "to_proj4")

    @pytest.mark.parametrize(
        "lat,lon",
        [
            (0, 0),  # Equator, Prime Meridian
            (51.5, -0.1),  # London
            (49.0, 8.5),  # Heidelberg
            (-33.9, 18.4),  # Cape Town
            (35.7, 139.7),  # Tokyo
        ],
    )
    def test_get_centered_custom_projection_various_locations(self, lat, lon):
        """Test projection creation for various global locations."""
        result = get_centered_custom_projection(lat=lat, lon=lon)

        assert result is not None

    def test_get_centered_custom_projection_extreme_latitudes(self):
        """Test projection with extreme latitudes."""
        # Near north pole
        result_north = get_centered_custom_projection(lat=89, lon=0)
        assert result_north is not None

        # Near south pole
        result_south = get_centered_custom_projection(lat=-89, lon=0)
        assert result_south is not None

    def test_get_centered_custom_projection_extreme_longitudes(self):
        """Test projection with extreme longitudes."""
        result_east = get_centered_custom_projection(lat=0, lon=179)
        assert result_east is not None

        result_west = get_centered_custom_projection(lat=0, lon=-179)
        assert result_west is not None

    def test_get_centered_custom_projection_invalid_latitude(self):
        """Test that invalid latitude raises error."""
        with pytest.raises((ValueError, AssertionError)):
            get_centered_custom_projection(lat=91, lon=0)  # > 90°

        with pytest.raises((ValueError, AssertionError)):
            get_centered_custom_projection(lat=-91, lon=0)  # < -90°

    def test_get_centered_custom_projection_invalid_longitude(self):
        """Test that invalid longitude raises error or is handled."""
        # Depending on implementation, may wrap or raise error
        try:
            result = get_centered_custom_projection(lat=0, lon=181)
            # If no error, longitude was likely wrapped
            assert result is not None
        except (ValueError, AssertionError):
            # If error raised, that's also acceptable behavior
            pass

    def test_get_centered_custom_projection_consistency(self):
        """Test that same input gives consistent output."""
        result1 = get_centered_custom_projection(lat=49.0, lon=8.5)
        result2 = get_centered_custom_projection(lat=49.0, lon=8.5)

        # Should produce same projection
        assert str(result1) == str(result2)
