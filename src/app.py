import streamlit as st
import plotly.express as px
from datetime import datetime, timedelta
import pytz
from skyfield.api import load, Topos
import numpy as np
import plotly.graph_objects as go
import io
import os
import requests
from pytz import all_timezones
from tzlocal import get_localzone_name

st.set_page_config(layout="wide")
st.sidebar.title("Location and Time settings")
st.title("Satellite Visualization Dashboard")

# Sidebar form for user inputs
with st.sidebar.form("input_form"):
    if "latitude" not in st.session_state:
        st.session_state["latitude"] = "43.07154"
    if "longitude" not in st.session_state:
        st.session_state["longitude"] = "-89.40829"

    loc_row = st.columns(2)
    st.session_state["latitude"] = loc_row[0].text_input("Latitude (°)", st.session_state["latitude"])
    st.session_state["longitude"] = loc_row[1].text_input("Longitude (°)", st.session_state["longitude"])

    if "elevation" not in st.session_state:
        st.session_state["elevation"] = "40"
    st.session_state["elevation"] = st.text_input("Elevation (°)", st.session_state["elevation"])

    # Timezone setup
    all_tz_sorted = sorted(set(all_timezones))
    if "selected_timezone_label" not in st.session_state:
        st.session_state["selected_timezone_label"] = "UTC"

    st.session_state["selected_timezone_label"] = st.selectbox(
        "Choose timezone",
        all_tz_sorted,
        index=all_tz_sorted.index(st.session_state["selected_timezone_label"])
    )

    selected_timezone = st.session_state["selected_timezone_label"]

    # Start & End Time Inputs
    if "start_date" not in st.session_state:
        st.session_state["start_date"] = datetime.utcnow().date()
    if "start_time" not in st.session_state:
        st.session_state["start_time"] = datetime.utcnow().time()

    start_row = st.columns(2)
    st.session_state["start_date"] = start_row[0].date_input("Start Date", st.session_state["start_date"])
    st.session_state["start_time"] = start_row[1].time_input("Start Time", st.session_state["start_time"])

    if "end_date" not in st.session_state:
        st.session_state["end_date"] = datetime.utcnow().date() + timedelta(days=1)
    if "end_time" not in st.session_state:
        st.session_state["end_time"] = (datetime.utcnow() + timedelta(hours=1)).time()

    end_row = st.columns(2)
    st.session_state["end_date"] = end_row[0].date_input("End Date", st.session_state["end_date"])
    st.session_state["end_time"] = end_row[1].time_input("End Time", st.session_state["end_time"])

    # TLE input
    st.markdown("TLE Source")
    if "use_latest_tle" not in st.session_state:
        st.session_state["use_latest_tle"] = False
    st.session_state["use_latest_tle"] = st.checkbox("Use latest Celestrak Starlink TLE", value=st.session_state["use_latest_tle"])
    tle_file = st.file_uploader("Upload your own TLE file (.txt)", type=["txt"])
    refresh_button = st.form_submit_button("Run Visualization")


# Convert times to UTC
tz = pytz.timezone(selected_timezone)
start_time = datetime.combine(st.session_state["start_date"], st.session_state["start_time"])
end_time = datetime.combine(st.session_state["end_date"], st.session_state["end_time"])
start_time = tz.localize(start_time).astimezone(pytz.UTC)
end_time = tz.localize(end_time).astimezone(pytz.UTC)

# Load TLE data
def load_satellites(tle_file, use_latest):
    try:
        ts = load.timescale()
        if use_latest:
            url = "https://celestrak.com/NORAD/elements/starlink.txt"
            r = requests.get(url)
            tle_text = r.text
            tle_io = io.StringIO(tle_text)
            sats = load.tle_file(tle_io)
        else:
            if tle_file is not None:
                tle_text = tle_file.read().decode("utf-8")
                tle_io = io.StringIO(tle_text)
                sats = load.tle_file(tle_io)
            else:
                tle_path = "data/default.tle"
                sats = load.tle_file(tle_path)
        return {sat.name: sat for sat in sats}
    except Exception as e:
        st.error(f"Error loading TLEs: {e}")
        return {}

def check_field_of_view(ground_station, satellite_dict, start_time_sf, end_time_sf, min_elevation=40):
    visible_satellites = {}

    for sat_name, sat in satellite_dict.items():
        visible_satellites[sat_name] = {"obj": sat}
        rel_pos = sat - ground_station
        alt, _, _ = rel_pos.at(start_time_sf).altaz()

        if alt.degrees >= min_elevation:
            visible_satellites[sat_name][0] = [start_time_sf]  # Already visible at start

        t, events = sat.find_events(ground_station, start_time_sf, end_time_sf, altitude_degrees=min_elevation)

        for ti, event in zip(t, events):
            visible_satellites[sat_name].setdefault(event, [])

            if event == 0 or event == 1:
                visible_satellites[sat_name][event].append(ti)

            elif event == 2:
                alt2, _, _ = (sat - ground_station).at(ti).altaz()
                if alt2.degrees < min_elevation:
                    visible_satellites[sat_name][event].append(ti)

        if 1 not in visible_satellites[sat_name]:
            del visible_satellites[sat_name]

    return visible_satellites

# Calculate Doppler shifts
def doppler_calc(start_time_sf, end_time_sf, visible_sats, observer, time_step=1):
    doppler_shifts = {}
    all_range_rate = {}
    all_graph = {}
    ts = load.timescale()

    for sat, info in visible_sats.items():
        starts = info.get(0, [start_time_sf])
        ends = info.get(2, [end_time_sf])
        shifts = []
        these_ranges = []
        graph = []

        for t1, t2 in zip(starts, ends):
            current_time = t1
            while current_time.tt <= t2.tt:
                try:
                    pos = (info["obj"] - observer).at(current_time)
                    alt, az, dist, _, _, range_rate = pos.frame_latlon_and_rates(observer)
                    doppler_shift = 11.325e9 * (1 + (-range_rate.km_per_s) / 3e5) - 11.325e9
                    shifts.append((current_time.utc_iso(), doppler_shift))
                    these_ranges.append((current_time.utc_iso(), range_rate.km_per_s))
                    graph.append((current_time.utc_iso(), alt, az, dist, doppler_shift))
                except Exception as e:
                    print(f"Error at {current_time.utc_iso()} for {sat}: {e}")
                current_time = ts.utc(current_time.utc_datetime() + timedelta(seconds=time_step))

        doppler_shifts[sat] = shifts
        all_range_rate[sat] = these_ranges
        all_graph[sat] = graph

    return doppler_shifts, all_range_rate, all_graph

import pandas as pd

def build_polar_dataframe(all_graph):
    data = []

    for sat, records in all_graph.items():
        for record in records:
            time_iso, alt, az, dist, doppler = record

            data.append({
                "Satellite": sat,
                "Time": datetime.strptime(time_iso, "%Y-%m-%dT%H:%M:%SZ"),
                "Elevation": alt.degrees,
                "Azimuth": az.degrees,
                "Distance": dist.km,
                "Time_str": time_iso.replace("T", " ").replace("Z", "")  # For animation frame
            })

    df = pd.DataFrame(data)
    df.sort_values(by="Time", inplace=True)
    return df

def plot_polar_animation(polar_df):
    satellites = sorted(polar_df['Satellite'].unique())
    color_palette = px.colors.qualitative.Plotly
    sat_colors = {sat: color_palette[i % len(color_palette)] for i, sat in enumerate(satellites)}

    polar_df["DotSize"] = 10  

    fig = px.scatter_polar(
        polar_df,
        r="Elevation",
        theta="Azimuth",
        animation_frame="Time_str",
        animation_group="Satellite",
        color="Satellite",
        hover_name="Satellite",
        size="DotSize",  
        size_max=12,    
        range_r=[90, 0],
        title="Satellite Positions Over Time (Polar Plot)",
        color_discrete_sequence=[sat_colors[sat] for sat in satellites]
    )

    fig.update_layout(
        polar=dict(
            radialaxis=dict(showticklabels=True, ticks='',tickfont=dict(color='black')
            ),
            angularaxis=dict(direction='clockwise')
        ),
        height=700,
        margin=dict(l=40, r=20, t=50, b=40),
    )

    return fig



# Load satellites
ts = load.timescale()
satellites = load_satellites(tle_file, st.session_state["use_latest_tle"])


# Tabs for visualization
tab1, tab2, tab3 = st.tabs(["Doppler Shift", "Polar Plot", "Dome Plot"])
visible_sats = {}

with tab1:
    st.subheader("Doppler Shift")

    ground_station = Topos(
        latitude_degrees=float(st.session_state["latitude"]),
        longitude_degrees=float(st.session_state["longitude"]),
        elevation_m=float(st.session_state["elevation"])
    )
    start_sf = ts.utc(start_time)
    end_sf = ts.utc(end_time)

    visible_sats = check_field_of_view(ground_station, satellites, start_sf, end_sf)
    doppler_shifts, all_range_rate, all_graph = doppler_calc(start_sf, end_sf, visible_sats, ground_station)
    all_satellites = list(doppler_shifts.keys())



    graph_col, control_col = st.columns([6, 2])

    with control_col:
        with st.form("satellite_control_form"):
            st.markdown("##### Satellite Selection")
            select_all = st.checkbox("Select All", value=False)

            if select_all:
                selected_sats = st.multiselect(
                    "Satellites",
                    options=all_satellites,
                    default=all_satellites,
                    help="All satellites are selected.",
                    key="all_sat_selector"
                )
            else:
                selected_sats = st.multiselect(
                    "Satellites",
                    options=all_satellites,
                    default=all_satellites[:3],
                    help="Choose one or more satellites to display.",
                    key="manual_sat_selector"
                )

            run_graph = st.form_submit_button("Plot Selected")

    
    with graph_col:
        if selected_sats and run_graph:
            fig = go.Figure()
            utc = pytz.UTC

            for sat in selected_sats:
                shifts = doppler_shifts.get(sat, [])
                if not shifts:
                    continue

           
                if len(shifts) < 5:
                    continue

                values = [shift for _, shift in shifts]
                if max(values) - min(values) < 2000:
                    continue  
           
                times = [datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=utc) for ts, _ in shifts]

             
                fig.add_trace(go.Scatter(
                    x=times, y=values,
                    mode='lines',
                    name=sat,
                    line=dict(width=2)
                ))

            if not fig.data:
                st.warning("No satellites had meaningful Doppler data to display.")
            else:
                fig.update_layout(
                    title="Doppler Shifts Over Time",
                    xaxis_title="Timestamp (UTC)",
                    yaxis_title="Doppler Shift (Hz)",
                    hovermode="closest",
                    height=700,
                    margin=dict(l=40, r=20, t=50, b=40)
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Select satellites and click the button to view the graph.")


# 2D Tab
with tab2:
    st.subheader("Polar View")
    polar_df = build_polar_dataframe(all_graph)
    #st.write(polar_df.head()) 
    polar_fig = plot_polar_animation(polar_df)
    st.plotly_chart(polar_fig, use_container_width=True)



# 3D Tab
with tab3:
    st.subheader("Dome Plot")
    if refresh_button:
        st.info(f"Satellites in field of view: {len(visible_sats)}")
    else:
        st.info("Click the run button to see results.")
