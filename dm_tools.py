import pandas as pd
import numpy as np

def data_prep():
    df = pd.read_csv('data/hotel-booking.csv')
    # clean_df = df.copy()

    # Remove rows with total_guests == 0
    df = df[df["total_guests"] != 0].copy()
    df.reset_index(drop=True, inplace=True)

    # Month is categorical, so convert to string for one-hot encoding
    # Reasoning: the most appropriate grouping of data;
    #   day has no meaning without month
    #   weeks has no meaning without year, but we have no year
    #   weekends and weekdays values depend on week for context
    #   month could also provide seasonal context, hence a better model
    df["arrival_date_month"] = df["arrival_date_month"].astype(str)

    # is_canceled is a two-valued categorical (Y/N), so it should be binary
    df["is_canceled"] = df["is_canceled"].map({"N": 0, "Y": 1})

    # Convert invalid negative lead_time values to NaN, then replace with median. Also convert the data type back to int
    df.loc[df["lead_time"] < 0, "lead_time"] = np.nan
    df["lead_time"] = (df["lead_time"].fillna(df["lead_time"].median()).astype(int))

    # adr < 1: a hotel rate of 0 (free) is most definitely an error
    df.loc[df["adr"] < 1, "adr"] = np.nan

    # Replace missing child counts with 0, and convert to int
    df["children"] = df["children"].fillna(0).astype(int)

    # Replace missing categorical values with an Unknown category
    df["country"] = df["country"].fillna("Unknown")

    df["reserved_room_type"] = (
        df["reserved_room_type"]
        .replace("?", "Unknown")
        .fillna("Unknown")
    )

    # Replace missing ADR values with the median
    df["adr"] = df["adr"].fillna(
        df["adr"].median()
    )

    # Remove unnecessary/redundant variables
    columns_to_remove = [
        "booking_id",
        "company",
        "total_guests"
    ]

    df = df.drop(columns=columns_to_remove)

    # one-hot encode categoricals
    # Applies to country, reserved_room_type, and arrival_date_month
    df = pd.get_dummies(df)

    return df
