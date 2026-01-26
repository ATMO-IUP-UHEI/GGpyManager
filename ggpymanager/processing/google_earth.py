#!/usr/bin/env python3
"""
Script to add <altitudeMode>absolute</altitudeMode> tag to all Point elements in a KML
file.
"""

import xml.etree.ElementTree as ET

MODES = ["absolute", "relativeToGround", "clampToGround"]


def add_altitude_mode_to_points(
    kml_file: str,
    mode: str = "absolute",
    output_file: str | None = None,
):
    """
    Add altitudeMode tag to all Point elements in a KML file.

    Parameters
    ----------
    kml_file : str
        Path to input KML file
    mode : str
        Altitude mode to set. Must be one of 'absolute', 'relativeToGround', or
        'clampToGround'.
    output_file : str, optional
        Path to output KML file. If None, overwrites the input file.
    """
    assert mode in MODES, f"Invalid mode: {mode}. Must be one of {MODES}"
    # Define KML namespace
    namespace = {"kml": "http://www.opengis.net/kml/2.2"}

    # Register namespace to avoid ns0 prefix in output
    ET.register_namespace("", "http://www.opengis.net/kml/2.2")
    ET.register_namespace("gx", "http://www.google.com/kml/ext/2.2")
    ET.register_namespace("atom", "http://www.w3.org/2005/Atom")

    # Parse the KML file
    tree = ET.parse(kml_file)
    root = tree.getroot()

    # Find all Point elements
    points = root.findall(".//kml:Point", namespace)

    modified_count = 0

    for point in points:
        # Check if altitudeMode already exists
        altitude_mode = point.find("kml:altitudeMode", namespace)

        if altitude_mode is None:
            # Create and add altitudeMode element
            altitude_mode_elem = ET.Element(
                "{http://www.opengis.net/kml/2.2}altitudeMode"
            )
            altitude_mode_elem.text = mode

            # Insert before coordinates element if it exists, otherwise append
            coordinates = point.find("kml:coordinates", namespace)
            if coordinates is not None:
                coord_index = list(point).index(coordinates)
                point.insert(coord_index, altitude_mode_elem)
            else:
                point.append(altitude_mode_elem)

            modified_count += 1

    # Determine output file path
    if output_file is None:
        output_file = kml_file

    # Write the modified KML file
    tree.write(output_file, encoding="utf-8", xml_declaration=True)

    print(f"Processed {len(points)} Point elements")
    print(f"Added altitudeMode to {modified_count} Point elements")
    print(f"Output written to: {output_file}")
