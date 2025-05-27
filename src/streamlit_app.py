# import altair as alt
# import numpy as np
# import pandas as pd
# import streamlit as st

# """
# # Welcome to Streamlit!

# Edit `/streamlit_app.py` to customize this app to your heart's desire :heart:.
# If you have any questions, checkout our [documentation](https://docs.streamlit.io) and [community
# forums](https://discuss.streamlit.io).

# In the meantime, below is an example of what you can do with just a few lines of code:
# """

# num_points = st.slider("Number of points in spiral", 1, 10000, 1100)
# num_turns = st.slider("Number of turns in spiral", 1, 300, 31)

# indices = np.linspace(0, 1, num_points)
# theta = 2 * np.pi * num_turns * indices
# radius = indices

# x = radius * np.cos(theta)
# y = radius * np.sin(theta)

# df = pd.DataFrame({
#     "x": x,
#     "y": y,
#     "idx": indices,
#     "rand": np.random.randn(num_points),
# })

# st.altair_chart(alt.Chart(df, height=700, width=700)
#     .mark_point(filled=True)
#     .encode(
#         x=alt.X("x", axis=None),
#         y=alt.Y("y", axis=None),
#         color=alt.Color("idx", legend=None, scale=alt.Scale()),
#         size=alt.Size("rand", legend=None, scale=alt.Scale(range=[1, 150])),
#     ))


import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import requests, os, tempfile
from datetime import datetime, timedelta
from skyfield.api import load, Topos, wgs84

# Constants
f0 = 11.325e9
c = 3e5
ts = load.timescale()

# ✅ Use temp dir (fix for Hugging Face Spaces)
data_dir = os.path.join(tempfile.gettempdir(), "satellite_data")
os.makedirs(data_dir, exist_ok=True)

st.set_page_config(layout="wide")
st.title("🛰️ Satellite Visualization Dashboard")

# Sidebar controls
st.sidebar.header("Ground Station & Time Settings")
lat = st.sidebar.text_input("Latitude (°)", "43.07154")
lon = st.sidebar.text_input("Longitude (°)", "-89.40829")
start_time = st.sidebar.datetime_input("Start Time (UTC)", datetime.utcnow())
end_time = st.sidebar.datetime_input("End Time (UTC)", datetime.utcnow() + timedelta(minutes=10))
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
                except:
                    pass
                curr = ts.utc(curr.utc_datetime() + timedelta(seconds=step))
        doppler_shifts[name] = shifts
        all_graph[name] = graph
    return doppler_shifts, all_graph

if run_simulation:
    observer = Topos(latitude_degrees=float(lat), longitude_degrees=float(lon))
    start_sf, end_sf = ts.utc(start_time), ts.utc(end_time)

    tle_path = fetch_tle() if use_default else os.path.join(data_dir, "uploaded.tle")
    if tle_file_upload and not use_default:
        with open(tle_path, "w") as f:
            f.write(tle_file_upload.getvalue().decode("utf-8"))

    v_sats = check_field_of_view(observer, tle_path, start_sf, end_sf)
    doppler_shifts, all_graph = doppler_calc(start_sf, end_sf, v_sats, observer)

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

    # Table
    st.subheader("📋 Satellite Pass Data")
    st.dataframe(df)
