import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import requests, os, tempfile, pytz
from datetime import datetime, timedelta
from skyfield.api import load, Topos

# Constants
f0 = 11.325e9
c = 3e5
ts = load.timescale()

# Safe writable directory for Hugging Face
data_dir = os.path.join(tempfile.gettempdir(), "satellite_data")
os.makedirs(data_dir, exist_ok=True)

st.set_page_config(layout="wide")
st.title("🛰️ Satellite Visualization Dashboard")

# Sidebar inputs
st.sidebar.header("Ground Station & Time Settings")
lat = st.sidebar.text_input("Latitude (°)", "43.07154")
lon = st.sidebar.text_input("Longitude (°)", "-89.40829")

start_date = st.sidebar.date_input("Start Date", datetime.utcnow().date())
start_time_input = st.sidebar.time_input("Start Time", datetime.utcnow().time())
start_time = datetime.combine(start_date, start_time_input).replace(tzinfo=pytz.UTC)

end_date = st.sidebar.date_input("End Date", datetime.utcnow().date() + timedelta(days=1))
end_time_input = st.sidebar.time_input("End Time", datetime.utcnow().time())
end_time = datetime.combine(end_date, end_time_input).replace(tzinfo=pytz.UTC)

tle_file_upload = st.sidebar.file_uploader("Upload Custom TLE (.txt)", type=["txt"])
use_default = st.sidebar.checkbox("Use Starlink TLE from Celestrak", value=True)
run_simulation = st.sidebar.button("▶️ Run Simulation")

def fetch_tle():
    url = "https://celestrak.org/NORAD/elements/gp.php?GROUP=starlink&FORMAT=tle"
    r = requests.get(url)
    tle_path = os.path.join(data_dir, "starlink.tle")
    with open(tle_path, "w") as f:
        f.write(r.text)
    return tle_path

def check_field_of_view(gs, tle_path, start_sf, end_sf):
    sats = load.tle_file(tle_path)
    vis = {}
    for sat in sats:
        vis[sat.name] = {"obj": sat}
        alt, _, _ = (sat - gs).at(start_sf).altaz()
        if alt.degrees >= 40:
            vis[sat.name][0] = [start_sf]
        t, events = sat.find_events(gs, start_sf, end_sf, altitude_degrees=40)
        for ti, ev in zip(t, events):
            vis[sat.name].setdefault(ev, []).append(ti)
        if 1 not in vis[sat.name]:
            del vis[sat.name]
    return vis

def doppler_calc(s, e, v_s, observer, step=1):
    doppler_shifts, all_graph = {}, {}
    for name, info in v_s.items():
        starts = info.get(0, [s])
        ends = info.get(2, [e])
        shifts, graph = [], []
        for t1, t2 in zip(starts, ends):
            curr = t1
            while curr.tt <= t2.tt:
                try:
                    pos = (info["obj"] - observer).at(curr)
                    alt, az, dist, _, _, rate = pos.frame_latlon_and_rates(observer)
                    doppler = f0 * (1 + (-rate.km_per_s) / c) - f0
                    shifts.append((curr.utc_iso(), doppler))
                    graph.append((curr.utc_iso(), alt, az, dist, doppler))
                except Exception as e:
                    st.warning(f"⚠️ Error at time {curr.utc_iso()}: {e}")
                curr = ts.utc(curr.utc_datetime() + timedelta(seconds=step))
        doppler_shifts[name] = shifts
        all_graph[name] = graph
    return doppler_shifts, all_graph

if run_simulation:
    st.info("⏳ Running simulation...")

    if start_time >= end_time:
        st.error("❌ Start time must be before end time.")
        st.stop()

    if not use_default and tle_file_upload is None:
        st.error("❌ Please upload a TLE file or enable 'Use Starlink TLE'")
        st.stop()

    try:
        observer = Topos(latitude_degrees=float(lat), longitude_degrees=float(lon))
    except Exception as e:
        st.error(f"❌ Invalid coordinates: {e}")
        st.stop()

    tle_path = fetch_tle() if use_default else os.path.join(data_dir, "uploaded.tle")
    if tle_file_upload and not use_default:
        with open(tle_path, "w") as f:
            f.write(tle_file_upload.getvalue().decode("utf-8"))

    try:
        start_sf, end_sf = ts.utc(start_time), ts.utc(end_time)
        v_sats = check_field_of_view(observer, tle_path, start_sf, end_sf)
        doppler_shifts, all_graph = doppler_calc(start_sf, end_sf, v_sats, observer)
    except Exception as e:
        st.error(f"❌ Simulation failed: {e}")
        st.stop()

    if not doppler_shifts:
        st.warning("⚠️ No satellites visible during this time window.")
        st.stop()

    data = []
    for sat, entries in all_graph.items():
        for t, alt, az, dist, doppler in entries:
            data.append({
                "Satellite": sat,
                "Time": pd.to_datetime(t),
                "Elevation": alt.degrees,
                "Azimuth": az.degrees,
                "Distance": dist.km,
                "Doppler Shift": doppler
            })
    df = pd.DataFrame(data).sort_values("Time")
    df["Time_str"] = df["Time"].dt.strftime('%Y-%m-%d %H:%M:%S')

    st.success("✅ Simulation complete!")

    # Doppler Plot
    st.subheader("📉 Doppler Shift Over Time")
    fig_doppler = go.Figure()
    for sat, values in doppler_shifts.items():
        times = [datetime.strptime(t, '%Y-%m-%dT%H:%M:%SZ') for t, _ in values]
        shifts = [s for _, s in values]
        fig_doppler.add_trace(go.Scatter(x=times, y=shifts, mode='lines', name=sat))
    fig_doppler.update_layout(xaxis_title="Time", yaxis_title="Doppler Shift (Hz)")
    st.plotly_chart(fig_doppler, use_container_width=True)

    # Polar Plot
    st.subheader("🧭 Polar Plot (Azimuth vs Elevation)")
    fig_polar = px.scatter_polar(
        df, r='Elevation', theta='Azimuth',
        color='Satellite', animation_frame='Time_str',
        range_r=[90, 0]
    )
    st.plotly_chart(fig_polar, use_container_width=True)

    # Dome Plot
    st.subheader("🌌 Dome Plot (3D Satellite Positions)")
    def polar_to_cartesian(az_deg, el_deg):
        az = np.radians(az_deg)
        el = np.radians(el_deg)
        r = 1
        x = r * np.cos(el) * np.cos(az)
        y = r * np.cos(el) * np.sin(az)
        z = r * np.sin(el)
        return x, y, z

    df[["x", "y", "z"]] = df.apply(lambda row: polar_to_cartesian(row["Azimuth"], row["Elevation"]), axis=1, result_type='expand')
    fig_dome = go.Figure()
    for sat in df["Satellite"].unique():
        subset = df[df["Satellite"] == sat]
        fig_dome.add_trace(go.Scatter3d(x=subset["x"], y=subset["y"], z=subset["z"],
                                        mode='markers', name=sat, marker=dict(size=3)))
    fig_dome.update_layout(scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False)))
    st.plotly_chart(fig_dome, use_container_width=True)

    # Data Table
    st.subheader("📋 Satellite Pass Data")
    st.dataframe(df)
