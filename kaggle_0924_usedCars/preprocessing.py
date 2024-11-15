from idlelib.pyparse import trans

import pandas as pd
import numpy as np
import re
from collections import defaultdict


def extract_engine_info(engine_text: str) \
        -> pd.Series:
    # Use regular expressions to extract horsepower, liters, and cylinders
    hp_match = re.search(r"(\d+(\.\d+)?)HP", engine_text)
    liters_match = re.search(r"(\d+(\.\d+)?)(\s?L|\sLiter)", engine_text)
    cylinders_match = re.search(r"(\d+)\s*Cylinders?", engine_text, re.IGNORECASE)
    cylinders_v_match = re.search(r"(V|I|H|W)\-?(\d+)", engine_text)
    valve_match = re.search(r"(\d+)V", engine_text)
    gdi_match = re.search("GDI", engine_text)
    mpfi_match = re.search("MPFI", engine_text)
    pdi_match = re.search("PDI", engine_text)
    tfsi_match = re.search("TFSI", engine_text)
    dohc_match = re.search("DOHC", engine_text)
    sohc_match = re.search("SOHC", engine_text)
    turbo_match = re.search("Turbo", engine_text)
    ohv_match = re.search("OHV", engine_text)
    straight_match = re.search("Straight", engine_text)
    flat_match = re.search("Flat", engine_text)
    electric_match = re.search("Electric", engine_text)
    supercharged_match = re.search("Supercharged", engine_text)

    # Extract values or set them as None if not found
    horse_power = hp_match.group(1) if hp_match else None
    liters = liters_match.group(1) if liters_match else None
    cylinders = cylinders_match.group(1) if cylinders_match else None
    cylinders_v = cylinders_v_match.group(2) if cylinders_v_match else None
    cylinders_final = cylinders if cylinders is not None else cylinders_v
    valve = valve_match.group(1) if valve_match else None
    gdi = True if gdi_match else False
    mpfi = True if mpfi_match else False
    pdi = True if pdi_match else False
    tfsi = True if tfsi_match else False
    dohc = True if dohc_match else False
    sohc = True if sohc_match else False
    turbo = True if turbo_match else False
    ohv = True if ohv_match else False
    straight = True if straight_match else False
    flat = True if flat_match else False
    electric = True if electric_match else False
    supercharged = True if supercharged_match else False

    # other_info = re.sub(
    #     r"(\d+(\.\d+)?)HP|\d+(\.\d+)?(\s?L|\sLiter)|\d+\s*Cylinders?|(V|I|H|W)\-?(\d+)|(\d+)V|(\w+)\s*Fuel?|Engine|GDI|MPFI|PDI|TFSI|DOHC|SOHC|Turbo|OHV|Straight|Flat|Electric|Supercharged",
    #     '', engine_text).strip()

    return pd.Series([horse_power, liters, cylinders_final, valve,
                      gdi, mpfi, pdi, tfsi, dohc, sohc, turbo, ohv,
                      straight, flat, electric, supercharged])#, other_info])


def extract_transmission_info(transmission_text: str) \
        -> pd.Series:
    # Use regular expressions to extract automatic/manual, speed
    automatic_match = re.search(r"(A/T|AT| Automatic)", transmission_text, re.IGNORECASE)
    manual_match = re.search(r"(M/T|MT| Manual)", transmission_text, re.IGNORECASE)
    cvt_match = re.search(r"(CVT)", transmission_text, re.IGNORECASE)
    autoshift_match = re.search(r"Auto-Shift", transmission_text, re.IGNORECASE)
    dualshift_match = re.search(r"Dual Shift", transmission_text, re.IGNORECASE)
    overdrive_match = re.search(r"Overdrive", transmission_text, re.IGNORECASE)
    speeds_match = re.search(r"(\d+)(-|\s)Speed", transmission_text, re.IGNORECASE)

    # Extract values or set them as None if not found
    at_transmission = True if automatic_match else False
    manual_transmission = True if manual_match else False
    cvt_transmission = True if cvt_match else False
    autoshift_transmission = True if autoshift_match else False
    dualshift_transmission = True if dualshift_match else False
    overdrive_transmission = True if overdrive_match else False
    speeds_transmission = speeds_match.group(1) if speeds_match else None

    return pd.Series([at_transmission, manual_transmission, cvt_transmission,
                      autoshift_transmission, dualshift_transmission, overdrive_transmission,
                      pd.to_numeric(speeds_transmission)])


def transform_colors(color: str, color_mapping: dict) -> str:
    # Function to map a color to the closest common color
    color_lower = color.lower() if isinstance(color, str) else ''
    for specific_color, common_color in color_mapping.items():
        if specific_color in color_lower:
            return common_color
    return 'Other'  # Default if no match is found


def impute_values(row, colname: str):
    if row[colname] is not np.nan:
        return row[colname]
    elif row[colname+"_model"] is not np.nan:
        return row[colname+"_model"]
    return row[colname+"_brand"]


def preprocess_dataframe(df_input: pd.DataFrame) \
        -> pd.DataFrame:
    df = df_input.copy()
    # Change model year to model age
    df["model_age"] = df["model_year"].apply(lambda x: 2024-x)

    # Reduce number of fuel types
    idx = pd.isna(df["fuel_type"]) | (df["fuel_type"] == "–") | (df["fuel_type"] == "not supported")
    df.loc[idx, "fuel_type"] = "unknown"

    # Apply engine transformation
    df[["horse_power", "liters", "cylinders", "valve",
        "gdi", "mpfi", "pdi", "tfsi", "dohc", "sohc", "turbo", "ohv",
        "straight", "flat", "electric", "supercharged"]] = df["engine"].apply(extract_engine_info)
    df[["horse_power", "liters", "cylinders", "valve"]] = df[["horse_power", "liters", "cylinders", "valve"]].apply(pd.to_numeric)

    # Apply transmission transformation
    df[["at_transmission", "manual_transmission", "cvt_transmission",
        "autoshift_transmission", "dualshift_transmission", "overdrive_transmission",
        "speeds_transmission"]] = df["transmission"].apply(extract_transmission_info)

    # Apply color transformation to ext_col and int_col
    common_colors = {
        'Black': ['Black', 'Ebony', 'Onyx', 'Granite', 'Obsidian', 'Nero'],
        'White': ['White', 'Ivory', 'Cream', 'Pearl'],
        'Gray': ['Gray', 'Grey', 'Graphite', 'Charcoal'],
        'Silver': ['Silver', 'Aluminum', 'Metallic', 'Steel'],
        'Red': ['Red', 'Maroon', 'Burgundy'],
        'Blue': ['Blue', 'Navy', 'Azure', 'Blu', 'Sea'],
        'Green': ['Green', 'Olive', 'Emerald', 'Verde'],
        'Yellow': ['Yellow', 'Gold'],
        'Orange': ['Orange', 'Caviar', 'Arancio'],
        'Brown': ['Brown', 'Tan', 'Cocoa', 'Walnut'],
        'Purple': ['Purple', 'Violet', 'Plum'],
        'Beige': ['Beige', 'Sand', 'Champagne', 'Camel']
    }
    # Create a reverse mapping from specific colors to common colors
    color_mapping = defaultdict(str)
    for common_color, specific_colors in common_colors.items():
        for color in specific_colors:
            color_mapping[color.lower()] = common_color
    df['mapped_ext_color'] = df['ext_col'].apply(transform_colors, args=(color_mapping,))
    df['mapped_int_color'] = df['int_col'].apply(transform_colors, args=(color_mapping,))

    # Transform accidents into bool
    df["accident_bool"] = df["accident"] == "At least 1 accident or damage reported"

    # Transform clean_title into bool
    df["clean_title_bool"] = df["clean_title"] == "Yes"

    # Take only the first word of the model name
    df["fw_model"] = df["model"].apply(lambda x: x.split(' ')[0])

    # Fill NAs of numerical columns with median per brand-model
    imputation_df_model = (
        df[["brand", "fw_model", "horse_power", "liters", "cylinders", "valve", "speeds_transmission"]]
        .groupby(["brand", "fw_model"], as_index=False)).aggregate("median")
    imputation_df_model.columns = ["brand", "fw_model", "horse_power_model", "liters_model", "cylinders_model",
                                   "valve_model", "speeds_transmission_model"]
    imputation_df_brand = (
        df[["brand", "horse_power", "liters", "cylinders", "valve", "speeds_transmission"]]
        .groupby("brand", as_index=False)).aggregate("median")
    imputation_df_brand.columns = ["brand", "horse_power_brand", "liters_brand", "cylinders_brand", "valve_brand",
                                   "speeds_transmission_brand"]

    df = df.merge(imputation_df_model, on=["brand", "fw_model"])
    df = df.merge(imputation_df_brand, on=["brand"])

    df["horse_power"] = df.apply(lambda x: impute_values(x, "horse_power"), axis=1)
    df["liters"] = df.apply(lambda x: impute_values(x, "liters"), axis=1)
    df["cylinders"] = df.apply(lambda x: impute_values(x, "cylinders"), axis=1)
    df["valve"] = df.apply(lambda x: impute_values(x, "valve"), axis=1)
    df["speeds_transmission"] = df.apply(lambda x: impute_values(x, "speeds_transmission"), axis=1)

    df['horse_power'] = df['horse_power'].fillna(df['horse_power'].median())
    df['liters'] = df['liters'].fillna(df['liters'].median())
    df['cylinders'] = df['cylinders'].fillna(df['cylinders'].median())
    df['valve'] = df['valve'].fillna(df['valve'].median())
    df['speeds_transmission'] = df['speeds_transmission'].fillna(df['speeds_transmission'].median())

    # Log-transform the milage
    df["log_milage"] = np.log(df["milage"])

    # Return the transformed df with selected columns only
    return df[["id", "brand", "fw_model", "model_age", "log_milage", "fuel_type",
               "horse_power", "liters", "cylinders", "valve",
               "gdi", "mpfi", "pdi", "tfsi", "dohc", "sohc", "turbo", "ohv",
               "straight", "flat", "electric", "supercharged",
               "at_transmission", "manual_transmission", "cvt_transmission",
               "autoshift_transmission", "dualshift_transmission", "overdrive_transmission",
               "mapped_ext_color", "mapped_int_color",
               "speeds_transmission", "accident_bool", "clean_title_bool"]]
