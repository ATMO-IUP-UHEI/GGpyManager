#!/usr/bin/env python3
"""
Script to add <altitudeMode>absolute</altitudeMode> tag to all Point elements in a KML
file.
"""

import xml.etree.ElementTree as ET

MODES = ["absolute", "relativeToGround", "clampToGround"]
KML_NS = "http://www.opengis.net/kml/2.2"
NAMESPACE = {"kml": KML_NS}


def _register_kml_namespaces():
    """Register KML namespaces to avoid ns0 prefix in output."""
    ET.register_namespace("", KML_NS)
    ET.register_namespace("gx", "http://www.google.com/kml/ext/2.2")
    ET.register_namespace("atom", "http://www.w3.org/2005/Atom")


def _add_altitude_mode_to_point(point: ET.Element, mode: str) -> bool:
    """
    Add altitudeMode to a Point element if it doesn't exist.
    
    Returns True if altitude mode was added, False otherwise.
    """
    altitude_mode = point.find("kml:altitudeMode", NAMESPACE)
    
    if altitude_mode is not None:
        return False
    
    altitude_mode_elem = ET.Element(f"{{{KML_NS}}}altitudeMode")
    altitude_mode_elem.text = mode
    
    # Insert before coordinates element if it exists, otherwise append
    coordinates = point.find("kml:coordinates", NAMESPACE)
    if coordinates is not None:
        coord_index = list(point).index(coordinates)
        point.insert(coord_index, altitude_mode_elem)
    else:
        point.append(altitude_mode_elem)
    
    return True


def _add_name_to_placemark(placemark: ET.Element) -> bool:
    """
    Add name from first SimpleData to a Placemark if needed.
    
    Returns True if name was added/updated, False otherwise.
    """
    existing_name = placemark.find("kml:name", NAMESPACE)
    simple_data = placemark.find(".//SimpleData", NAMESPACE)
    
    if simple_data is None or not simple_data.text:
        return False
    
    if existing_name is None:
        name_elem = ET.Element(f"{{{KML_NS}}}name")
        name_elem.text = simple_data.text
        placemark.insert(0, name_elem)
        return True
    elif not existing_name.text or existing_name.text.strip() == "":
        existing_name.text = simple_data.text
        return True
    
    return False


def add_altitude_mode_to_points(
    kml_file: str,
    mode: str = "absolute",
    output_file: str | None = None,
    add_names: bool = False,
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
    add_names : bool, optional
        If True, adds the first SimpleData value as the placemark name.
    """
    assert mode in MODES, f"Invalid mode: {mode}. Must be one of {MODES}"
    
    _register_kml_namespaces()
    
    tree = ET.parse(kml_file)
    root = tree.getroot()
    
    # Process altitude modes
    points = root.findall(".//kml:Point", NAMESPACE)
    modified_count = sum(_add_altitude_mode_to_point(point, mode) for point in points)
    
    # Process placemark names
    names_added = 0
    if add_names:
        placemarks = root.findall(".//kml:Placemark", NAMESPACE)
        names_added = sum(_add_name_to_placemark(pm) for pm in placemarks)
    
    # Write output
    output_file = output_file or kml_file
    tree.write(output_file, encoding="utf-8", xml_declaration=True)
    
    print(f"Processed {len(points)} Point elements")
    print(f"Added altitudeMode to {modified_count} Point elements")
    if add_names:
        print(f"Added names to {names_added} Placemark elements")
    print(f"Output written to: {output_file}")
