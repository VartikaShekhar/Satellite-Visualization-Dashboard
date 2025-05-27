import streamlit as st
from skyfield.api import load, wgs84
from datetime import datetime, timedelta
import pytz
import os

# Constants
TLE_URL = "https://celestrak.org/NORAD/elements/gp.php?GROUP=starlink&FORMAT=tle"
ts = load.timescale()

st.set_page_config(layout="wide")
st.title("🛰️ Satellite Visibility Dashboard")

# Sidebar input
st.sidebar.header("Ground Station & Time Settings")
lat = float(st.sidebar.text_input("Latitude (°)", "43.07154"))
lon = float(st.sidebar.text_input("Longitude (°)", "-89.40829"))

start_date = st.sidebar.date_input("Start Date", datetime.utcnow().date())
start_time_input = st.sidebar.time_input("Start Time", datetime.utcnow().time())
start_time = datetime.combine(start_date, start_time_input).replace(tzinfo=pytz.UTC)

end_date = st.sidebar.date_input("End Date", datetime.utcnow().date() + timedelta(days=1))
end_time_input = st.sidebar.time_input("End Time", (datetime.utcnow() + timedelta(hours=1)).time())
end_time = datetime.combine(end_date, end_time_input).replace(tzinfo=pytz.UTC)

import tempfile
import shutil

@st.cache_data
def load_satellites(url):
    try:
        # Create a temp directory that Skyfield can write to
        with tempfile.TemporaryDirectory() as temp_dir:
            # Override Skyfield's default cache directory
            original_dir = os.getcwd()
            os.chdir(temp_dir)

            sats = load.tle_file(url)

            # Restore directory
            os.chdir(original_dir)

            return {sat.name: sat for sat in sats}
    except Exception as e:
        st.error(f"Error loading TLE: {e}")
        return {}


satellites = load_satellites(TLE_URL)

# Output satellite list
st.subheader("Loaded Satellites")
st.write(f"Found {len(satellites)} satellites.")
st.write(list(satellites.keys())[:10])  # show first 10 names
