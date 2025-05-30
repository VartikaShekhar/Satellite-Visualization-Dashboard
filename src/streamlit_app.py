import streamlit as st
from datetime import datetime, timedelta
import pytz
from skyfield.api import load, Topos
import numpy as np
import plotly.graph_objects as go
import io
import tempfile
import os
import requests

os.environ["STREAMLIT_HOME"] = os.getcwd()
st.set_page_config(layout="wide")
st.title("Satellite Visualization Dashboard")

#input
st.sidebar.header("Ground Station & Time Settings")
lat = float(st.sidebar.text_input("Latitude (°)", "43.07154"))
lon = float(st.sidebar.text_input("Longitude (°)", "-89.40829"))
start_date = st.sidebar.date_input("Start Date", datetime.utcnow().date())
start_time_input = st.sidebar.time_input("Start Time", datetime.utcnow().time())
start_time = datetime.combine(start_date, start_time_input).replace(tzinfo=pytz.UTC)

end_date = st.sidebar.date_input("End Date", datetime.utcnow().date() + timedelta(days=1))
end_time_input = st.sidebar.time_input("End Time", datetime.utcnow().time())
end_time = datetime.combine(end_date, end_time_input).replace(tzinfo=pytz.UTC)


st.sidebar.header("TLE Source")
use_latest_tle = st.sidebar.checkbox("Use latest Celestrak Starlink TLE", value=False)
tle_file = st.sidebar.file_uploader("Upload your own TLE file (.txt)", type=["txt"])


refresh_button = st.sidebar.button("Run Visualization")


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
                # fallback to default TLE file 
                tle_path = "src/data/default.tle"
                sats = load.tle_file(tle_path)
        return {sat.name: sat for sat in sats}
    except Exception as e:
        st.error(f"Error loading TLEs: {e}")
        return {}

#Load satellites now based on selected TLE
ts = load.timescale()
satellites = load_satellites(tle_file, use_latest_tle)
st.write(f"Loaded {len(satellites)} satellites from TLE")

#which satellites are visible
def check_field_of_view(ground_station, satellite_dict, start_time_sf, end_time_sf, min_elevation=40):
    visible_satellites = {}
    for sat_name, sat in satellite_dict.items():
        visible_satellites[sat_name] = {"obj": sat}
        rel_pos = sat - ground_station
        alt, _, _ = rel_pos.at(start_time_sf).altaz()
        if alt.degrees >= min_elevation:
            visible_satellites[sat_name][0] = [start_time_sf]
        t, events = sat.find_events(ground_station, start_time_sf, end_time_sf, altitude_degrees=min_elevation)
        for ti, event in zip(t, events):
            visible_satellites[sat_name].setdefault(event, []).append(ti)
        if 1 not in visible_satellites[sat_name]:
            del visible_satellites[sat_name]
    return visible_satellites

#Doppler shift over time
def doppler_calc(start_time_sf, end_time_sf, visible_sats, observer, time_step=10):
    doppler_shifts = {}
    ts = load.timescale()
    for sat, info in visible_sats.items():
        starts = info.get(0, [start_time_sf])
        ends = info.get(2, [end_time_sf])
        shifts = []
        for t1, t2 in zip(starts, ends):
            current_time = t1
            while current_time.tt <= t2.tt:
                try:
                    pos = (info["obj"] - observer).at(current_time)
                    alt, az, dist, _, _, range_rate = pos.frame_latlon_and_rates(observer)
                    doppler_shift = 11.325e9 * (1 + (-range_rate.km_per_s) / 3e5) - 11.325e9
                    shifts.append((current_time.utc_iso(), doppler_shift))
                except:
                    pass
                current_time = ts.utc(current_time.utc_datetime() + timedelta(seconds=time_step))
        doppler_shifts[sat] = shifts
    return doppler_shifts


    
#tabs - for 3 graphs

tab1, tab2, tab3 = st.tabs(["Doppler Shift", "Visibility", "Ground Track"])


with tab1:
    st.subheader("Doppler Shift")
    if refresh_button:
        ground_station = Topos(latitude_degrees=lat, longitude_degrees=lon)
        start_sf = ts.utc(start_time)
        end_sf = ts.utc(end_time)
        visible_sats = check_field_of_view(ground_station, satellites, start_sf, end_sf, min_elevation=40)
        st.info(f"Satellites in field of view: {len(visible_sats)}")
    else:
        st.info("Click the run button to see results.")


with tab2:
    st.subheader("2D View")
    if refresh_button:
        st.info(f"Satellites in field of view: {len(visible_sats)}")
    else:
        st.info("Click the run button to see results.")


with tab3:
    st.subheader("3D View")
    if refresh_button:
        st.info(f"Satellites in field of view: {len(visible_sats)}")
    else:
        st.info("Click the run button to see results.")
