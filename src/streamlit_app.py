import streamlit as st
import pytz
from pytz import all_timezones
from datetime import datetime, timedelta
from skyfield.api import load, Topos
from skyfield.api import wgs84
import io
import requests
import pandas as pd

import tempfile
import os
import plotly.graph_objects as go
import plotly.express as px
import numpy as np


st.set_page_config(layout="wide")
st.sidebar.title("Location and Time settings")
st.title("Satellite Visualization Dashboard")

f0 = 11.325e9
c = 3e5  # Speed of light in km/s
# Setup output directory
FULL_OUTPUT_DIR = tempfile.gettempdir()

# Sidebar form for inputs
with st.sidebar.form("input_form"):
    if "latitude" not in st.session_state:
        st.session_state["latitude"] = "43.07154"
    if "longitude" not in st.session_state:
        st.session_state["longitude"] = "-89.40829"

    loc_row = st.columns(2)
    st.session_state["latitude"] = loc_row[0].text_input("Latitude (°)", st.session_state["latitude"])
    st.session_state["longitude"] = loc_row[1].text_input("Longitude (°)", st.session_state["longitude"])

    if "min_elevation" not in st.session_state:
        st.session_state["min_elevation"] = "40"
    st.session_state["min_elevation"] = st.text_input("Min Elevation (°)", st.session_state["min_elevation"])

    # Timezone setup
    all_tz_sorted = sorted(set(all_timezones))
    if "user_tmz_label" not in st.session_state:
        st.session_state["user_tmz_label"] = "UTC"

    st.session_state["user_tmz_label"] = st.selectbox(
        "Choose timezone",
        all_tz_sorted,
        index=all_tz_sorted.index(st.session_state["user_tmz_label"])
    )

    selected_timezone = st.session_state["user_tmz_label"]

    # Start & End Time Inputs
    if "start_date" not in st.session_state:
        st.session_state["start_date"] = datetime.utcnow().date()
    if "start_time" not in st.session_state:
        st.session_state["start_time"] = datetime.utcnow().time()

    start_row = st.columns(2)
    st.session_state["start_date"] = start_row[0].date_input("Start Date", st.session_state["start_date"])
    st.session_state["start_time"] = start_row[1].time_input("Start Time", st.session_state["start_time"])

    if "end_date" not in st.session_state:
        st.session_state["end_date"] = datetime.utcnow().date() 
    if "end_time" not in st.session_state:
        st.session_state["end_time"] = (datetime.utcnow() + timedelta(minutes=10)).time()

    end_row = st.columns(2)
    st.session_state["end_date"] = end_row[0].date_input("End Date", st.session_state["end_date"])
    st.session_state["end_time"] = end_row[1].time_input("End Time", st.session_state["end_time"])

    # TLE input
    st.markdown("TLE Source")
    if "use_latest_tle" not in st.session_state:
        st.session_state["use_latest_tle"] = False
    st.session_state["use_latest_tle"] = st.checkbox("Use latest Celestrak Starlink TLE", value=st.session_state["use_latest_tle"])
    tle_file = st.file_uploader("Upload your own TLE file (.txt)", type=["txt"])
    st.session_state["tle_file"] = tle_file
    refresh_button = st.form_submit_button("Run Visualization")

tz = pytz.timezone(st.session_state["user_tmz_label"])
start_time = datetime.combine(st.session_state["start_date"], st.session_state["start_time"])
end_time = datetime.combine(st.session_state["end_date"], st.session_state["end_time"])
start_time = tz.localize(start_time).astimezone(pytz.UTC)
end_time = tz.localize(end_time).astimezone(pytz.UTC)

# Convert times to UTC after form is submitted
tz = pytz.timezone(st.session_state["user_tmz_label"])
start_time = datetime.combine(st.session_state["start_date"], st.session_state["start_time"])
end_time = datetime.combine(st.session_state["end_date"], st.session_state["end_time"])
start_time = tz.localize(start_time).astimezone(pytz.UTC)
end_time = tz.localize(end_time).astimezone(pytz.UTC)

def get_celestrak_tle(local_path="data/starlink.tle"):
    url = "https://celestrak.org/NORAD/elements/gp.php?GROUP=starlink&FORMAT=tle"
    # Only create directory if local_path is a string (not a file-like object)
    if isinstance(local_path, str):
        dir_name = os.path.dirname(local_path)
        if dir_name and not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)
        if not os.path.exists(local_path):
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            with open(local_path, "w") as f:
                f.write(response.text)
    return local_path


def load_tle_data(tle_file, use_latest):
    try:
        if use_latest:
            tle_path = get_celestrak_tle()
            satellites = load.tle_file(tle_path)
            print("Loaded latest Starlink TLE from Celestrak (cached locally).")

        elif tle_file is not None:
            # Handle Streamlit file uploader
            if hasattr(tle_file, "read"):
                tle_bytes = tle_file.read()
                tle_text = tle_bytes.decode("utf-8")

                # Save to a temporary local file (because Skyfield requires a real file path)
                temp_dir = tempfile.gettempdir()
                local_tle_path = os.path.join(temp_dir, "uploaded.tle")
                with open(local_tle_path, "w") as f:
                    f.write(tle_text)

                satellites = load.tle_file(local_tle_path)
                print("Loaded TLE from uploaded file (saved temporarily).")

            elif isinstance(tle_file, str) and os.path.exists(tle_file):
                satellites = load.tle_file(tle_file)
                print("Loaded TLE from provided file path.")

            else:
                raise ValueError("Invalid TLE input: must be a file upload or valid file path.")

        else:
            fallback_path = "data/starlink.tle"
            satellites = load.tle_file(fallback_path)
            print("No TLE file uploaded - Using default fallback file.")

        # Optional: log the names of first few satellites
        print("First 3 satellites:")
        for sat in satellites[:3]:
            print(sat.name)

        return satellites

    except Exception as e:
        st.error(f"Failed to load TLE data: {e}")
        return []

def check_field_of_view(ground_station, tle_file, start_time_sf, end_time_sf, min_elevation=0):
  '''
  Check which satellites are visible from the ground station within the specified time range.
  Args:
    ground_station: Skyfield wgs84.Topos object representing the ground station location.
    tle_file: Full Path to the TLE file, make sure it is fresh
    start_time_sf: Skyfield time object start time, UTC time
    end_time_sf: Skyfield time object end time, UTC time.
  Returns:
    Dictionary -> {SAT: {event: [timestamp]}}.
    timestamps are in Skyfield UTC format.
    events are:
      0 -> rise,
      1 -> peak,
      2 -> set.
    SAT is the name of the satellite. as a string
  '''
  satellites = load_tle_data(tle_file, st.session_state["use_latest_tle"])

 
  satellite_dict = {sat.name: sat for sat in satellites}
  #print(len(satellite_dict))
  visible_satellites = {} # {SAT: {event: [t]}}
  print("Start Time (Skyfield UTC):", start_time_sf)
  print("End Time (Skyfield UTC):", end_time_sf)
  print(start_time_sf.utc_iso())
  print(end_time_sf.utc_iso())
  for sat_name, sat in satellite_dict.items():
 
    visible_satellites[sat_name] = {"obj": sat}
    relative_pos_sat = sat - ground_station
    alt, az, dist = relative_pos_sat.at(start_time_sf).altaz()
    # print(f"SAT: {sat_name}, alt: {alt}, az: {az}, dist: {dist}")
 
    # events: 0->rises, 1->peaks, 2->sets
    t, events = sat.find_events(ground_station, start_time_sf, end_time_sf, altitude_degrees=min_elevation)
    #if events.size != 0:
        
        #print(f"SAT: {sat_name}, t: {t.utc_iso()}, events: {events}")
    for ti, event in zip(t, events):
      curr_val = visible_satellites[sat_name].get(event, [])
      if curr_val == []:
        visible_satellites[sat_name][event] = []
 
      if event == 0 or event == 1:
        visible_satellites[sat_name][event].append(ti)
      if event == 2:
        alt, az, dist = (visible_satellites[sat_name]['obj'] - ground_station).at(ti).altaz()
        if alt.degrees < min_elevation:
          visible_satellites[sat_name][event].append(ti)
    # if 1 not in visible_satellites[sat_name]:
    #     del visible_satellites[sat_name]
    if not any(k in visible_satellites[sat_name] for k in [0, 1 ,2]):
        # print(f"SAT: {visible_satellites[sat_name]} {str(visible_satellites[sat_name])}, no events")
        del visible_satellites[sat_name]
  st.write(len(visible_satellites), "visible satellites found")      
  return visible_satellites

def doppler_calc(s, e, v_s, observer, time_step=1, f0=f0):
    '''
    Calculate the Doppler shift for each satellite in the visible satellites dictionary.
    Args:
        s: Start time as a datetime object, UTC timezone.
        e: End time as a datetime object, UTC timezone.
        v_s: Dictionary of visible satellites from check_field_of_view function, assumes that same format.
        observer: Skyfield wgs84.Topos object representing the ground station location.
        time_step: Time step in seconds for the Doppler shift calculation, minimum can be 1 second for smooth graph.
    Returns:
    '''
    doppler_shifts = {}
    all_range_rate = {}
    all_graph = {}
    # Convert input start and end times to Skyfield <Time> objects
    # start_time = ts.utc(s)
    # end_time = ts.utc(e)
    # start_time = ts.utc(datetime.utcfromtimestamp(s_unix))
    # end_time = ts.utc(datetime.utcfromtimestamp(e_unix))
 
    for sat, info in v_s.items():  
        starts = info.get(0, [s])  
        ends = info.get(2, [e])  # If no end times, default to end_time
 
        shifts = []
        these_ranges = []
        graph = []
        for t1, t2 in zip(starts, ends):  # t1 and t2 are <Time> objects
 
            current_time = t1
            while current_time.tt <= t2.tt:  # Compare using `.tt` (Terrestrial Time)
                try:
                    # Calculate Doppler shift at the current time
                    pos = (info["obj"] - observer).at(current_time)
                    alt, az, dist, _, _, range_rate = pos.frame_latlon_and_rates(observer)

                    doppler_shift = f0 * ((-1 * range_rate.km_per_s) / c)
                     
                    # Append the current time and Doppler shift
                    shifts.append((current_time.utc_iso(), doppler_shift))
                    these_ranges.append((current_time.utc_iso(), range_rate.km_per_s))
                    graph.append((current_time.utc_iso(), alt, az, dist))
                except Exception as exc:
                    print(f"Error processing satellite {sat} at {current_time.utc_iso()}: {exc}")
                    # break
 
                # Increment current_time by time step
                current_time = ts.utc(current_time.utc_datetime() + timedelta(seconds=time_step))  # Add time step
 
            doppler_shifts[sat] = shifts
            all_range_rate[sat] = these_ranges
            all_graph[sat] = graph
 
    return doppler_shifts, all_range_rate, all_graph

@st.cache_data(show_spinner="Computing Doppler shifts...")
def cached_doppler(start_time, end_time, sats, observer, step):
    return doppler_calc(start_time, end_time, sats, observer, step)

 
def doppler_dict(doppler_shifts, f0=f0):
    '''
    Convert the doppler shifts data structure to a more usable format i.e. a dictionary where each satellite has a dictionary of timestamps and their corresponding Doppler shifts.
    also subtracts the frequency f0 from each Doppler shift value.
    Args:
        doppler_shifts: Dictionary of Doppler shifts from doppler_calc function.
        f0: Frequency of the satellite in Hz, default is SAT_FREQ[THIS_SAT].
    Returns:
        {SAT: {timestamp: doppler_shift}}
    '''
    doppler_shifts_dict = {}
    for sat, info in doppler_shifts.items():
        this_ds = {}
        for ts, ds in info:
            # timestamp, ds = info[0], info[1]
            this_ds[ts] = ds
        doppler_shifts_dict[sat] = this_ds
    return doppler_shifts_dict

 
def get_dopplers_timestamps(doppler_df, sat):
    """
    Convert the doppler_df DataFrame to a lists of timestamps and Doppler shifts.
    Args:
        doppler_df: DataFrame containing Doppler shifts.
    Returns:
        list of timestamps and list of Doppler shifts.
    """
    # Filter by satellite name
    # Extract and convert timestamps
    filtered_df = doppler_df[doppler_df["Satellite"] == sat]
    timestamps = filtered_df["Timestamp (UTC)"].apply(
        lambda t: pd.to_datetime(t, utc=True).to_pydatetime()
    ).tolist()
 
    # Extract Doppler shift values
    doppler_shifts = filtered_df["Doppler Shift (Hz)"].tolist()
 
    return timestamps, doppler_shifts

import plotly.graph_objects as go

def plot_polar_positions(visible_sats, observer):
    fig = go.Figure()

    for sat_name, sat_info in visible_sats.items():
        sat = sat_info["obj"]

        az_list = []
        el_list = []
        labels = []

        # Try to include rise, peak, set if they exist
        for event_type in [0, 1, 2]:
            for t in sat_info.get(event_type, []):
                alt, az, _ = (sat - observer).at(t).altaz()
                az_list.append(az.degrees)
                el_list.append(90 - alt.degrees)  # Invert for polar radius
                labels.append(f"{sat_name} {['Rise', 'Peak', 'Set'][event_type]}")

        if az_list:
            fig.add_trace(go.Scatterpolar(
                r=el_list,
                theta=az_list,
                mode='markers+lines',
                name=sat_name,
                text=labels,
                marker=dict(size=6),
            ))

    fig.update_layout(
        title="Satellite Polar Positions (Azimuth vs Elevation)",
        polar=dict(
            radialaxis=dict(range=[0, 90], angle=90, tickangle=90),
            angularaxis=dict(direction='clockwise', rotation=90),
        ),
        height=700,
        showlegend=True
    )

    return fig

def plot_dome_static(sat_data):
    """
    sat_data: List of dicts with keys: ['Azimuth', 'Elevation', 'Satellite']
    """
    fig = go.Figure()

    for sat_name in set(d["Satellite"] for d in sat_data):
        points = [d for d in sat_data if d["Satellite"] == sat_name]
        az = np.radians([p["Azimuth"] for p in points])
        el = [p["Elevation"] for p in points]
        r = [90 - e for e in el]  # closer to center = higher in sky

        # Convert polar (az, r) to 3D cartesian
        x = np.cos(az) * r
        y = np.sin(az) * r
        z = el  # Elevation as Z height

        # 1. Hemisphere surface
        u = np.linspace(0, np.pi / 2, 30)  # Only half sphere (dome)
        v = np.linspace(0, 2 * np.pi, 30)
        xh = np.outer(np.sin(u), np.cos(v))
        yh = np.outer(np.sin(u), np.sin(v))
        zh = np.outer(np.cos(u), np.ones_like(v))

        fig.add_trace(go.Surface(
            x=xh, y=yh, z=zh,
            colorscale='Blues',
            opacity=0.3,
            showscale=False,
            hoverinfo='skip',
            name='Earth Dome'
        ))

        # 2. Now separately calculate each satellite's 3D path
        for sat_name in set(d["Satellite"] for d in sat_data):
            points = [d for d in sat_data if d["Satellite"] == sat_name]
            az = np.radians([p["Azimuth"] for p in points])
            el = [p["Elevation"] for p in points]
            r = [1] * len(el)

            x = [np.cos(az[i]) * np.cos(np.radians(90 - el[i])) for i in range(len(el))]
            y = [np.sin(az[i]) * np.cos(np.radians(90 - el[i])) for i in range(len(el))]
            z = [np.sin(np.radians(90 - el[i])) for i in range(len(el))]

            fig.add_trace(go.Scatter3d(
                x=x, y=y, z=z,
                mode='lines+markers',
                name=sat_name,
                marker=dict(size=3),
                line=dict(width=2)
            ))


        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z,
            mode='markers+lines',
            name=sat_name,
            marker=dict(size=4),
            line=dict(width=1)
        ))

    fig.update_layout(
        scene=dict(
            xaxis=dict(title='X (Azimuth)', visible=False),
            yaxis=dict(title='Y (Azimuth)', visible=False),
            zaxis=dict(title='Elevation (°)', range=[0, 90])
        ),
        title='Dome Plot: Satellite Sky View',
        height=700
    )

    return fig

import numpy as np
import plotly.graph_objects as go

def plot_dome_with_distance(sat_data, scale=1.0):
    """
    sat_data: List of dicts with keys: ['Azimuth', 'Elevation', 'Distance_km', 'Satellite']
    scale: Divide all distances by this value (e.g., 1000 for 1000 km = 1 unit)
    """
    fig = go.Figure()

    # Plot the ground (origin)
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[0],
        mode='markers',
        marker=dict(size=8, color='blue'),
        name='Ground Station'
    ))

    for sat_name in set(d["Satellite"] for d in sat_data):
        points = [d for d in sat_data if d["Satellite"] == sat_name]
        az = np.radians([p["Azimuth"] for p in points])
        el = np.radians([p["Elevation"] for p in points])
        r = [p["Distance_km"] / scale for p in points]  # scale distance

        # Convert to 3D Cartesian
        phi = np.radians(90) - el
        x = r * np.sin(phi) * np.cos(az)
        y = r * np.sin(phi) * np.sin(az)
        z = r * np.cos(phi)

        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z,
            mode='lines+markers',
            name=sat_name,
            marker=dict(size=3),
            line=dict(width=2)
        ))

    fig.update_layout(
        scene=dict(
            xaxis_title='East (scaled)',
            yaxis_title='North (scaled)',
            zaxis_title='Up (scaled)',
            aspectmode='data'
        ),
        title='3D Dome Plot with Distance',
        height=700
    )

    return fig


tab1, tab2, tab3 = st.tabs(["Doppler Shift", "Polar Plot", "Dome Plot"])
visible_sats = {}
ts = load.timescale()

if "graph_shown" not in st.session_state:
    st.session_state["graph_shown"] = True
    st.session_state["run_graph"] = True 

if "polar_shown" not in st.session_state:
    st.session_state["polar_shown"] = True
    st.session_state["run_polar"] = True  

if "dome_shown" not in st.session_state:
    st.session_state["dome_shown"] = True
    st.session_state["run_dome"] = True  



with tab1:
    st.subheader("Doppler Shift")
    st.markdown(
        "This graph shows how the Doppler shift (Hz) changes over time for each selected satellite. "
        "The x-axis is time (UTC), the y-axis is Doppler shift."
    )

    ground_station = wgs84.latlon(
        float(st.session_state["latitude"]),
        float(st.session_state["longitude"]),
        elevation_m=float(st.session_state["min_elevation"])
    )
    start_sf = ts.utc(start_time)
    end_sf = ts.utc(end_time)
    print("starting simu graph")

    tle_file_used = st.session_state.get("tle_file", None)
    visible_sats = check_field_of_view(ground_station, tle_file_used, start_sf, end_sf, float(st.session_state["min_elevation"]))
    print("length of sats", len(visible_sats))
    doppler_shifts, all_range_rate, all_graph = doppler_calc(start_sf, end_sf, visible_sats, ground_station)
    all_satellites = list(doppler_shifts.keys())
    print("ldone simu")
    print("ldone simu")



    graph_col, control_col = st.columns([7, 2])

    with control_col:
        with st.form("satellite_control_form"):
            st.markdown("##### Satellite Selection")

            
            col1, col2 = st.columns([2, 1])
            select_all = col1.checkbox("Select All", value=False)

            # Red style for "Done"
            st.markdown("""
                <style>
                    div.stButton > button:first-child {
                        background-color: #d9534f;
                        color: white;
                    }
                </style>
            """, unsafe_allow_html=True)

            if "run_graph" not in st.session_state:
                st.session_state["run_graph"] = False
            if "selected_sats" not in st.session_state:
                st.session_state["selected_sats"] = all_satellites[:100]

            # Multiselect logic
            if select_all:
                selected_sats = st.multiselect(
                    "Satellites", options=all_satellites, default=all_satellites,
                    help="All satellites are selected.", key="all_sat_selector"
                )
            else:
                default_sats = all_satellites[:100] if len(all_satellites) >= 100 else all_satellites
                selected_sats = st.multiselect(
                    "Satellites", options=all_satellites, default=default_sats,
                    help="Showing first 100 satellites by default.", key="manual_sat_selector"
                )

            run_graph = col2.form_submit_button("Done")

            if run_graph:
                st.session_state["run_graph"] = True
                st.session_state["selected_sats"] = (
                    all_satellites if select_all else selected_sats
                )


    # Force default graph on initial render
    if "graph_shown" not in st.session_state:
        st.session_state["graph_shown"] = True
        run_graph = True  

        
    with graph_col:
        if st.session_state["selected_sats"] and st.session_state["run_graph"]:
            fig = go.Figure()
            utc = pytz.UTC

            for sat in selected_sats:
                shifts = doppler_shifts.get(sat, [])
                if not shifts:
                    continue

                # Extract and center shifts (just like matplotlib version)
                timestamps, raw_vals = zip(*shifts)
                try:
                    times = [datetime.strptime(t, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=utc) for t in timestamps]
                except Exception as e:
                    st.error(f"Time parsing error for {sat}: {e}")
                    continue

                centered_vals = [float(val) for val in raw_vals]

                fig.add_trace(go.Scatter(
                    x=times,
                    y=centered_vals,
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
                    hovermode="x unified",
                    height=700,
                    margin=dict(l=40, r=20, t=50, b=40)
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Select satellites and click the button to view the graph.")

with tab2:
    st.subheader("Polar Plot")
    st.markdown(
        "This animated polar plot shows the position of each selected satellite in the sky over time. "
        "The ground station is at the center (zenith), the edge is the horizon. "
        "Azimuth is the angle around the plot (0°=N, 90°=E, 180°=S, 270°=W)."
    )

    graph_col2, control_col2 = st.columns([6, 2])

    with control_col2:
        with st.form("polar_sat_control"):
            st.markdown("##### Satellite Selection")
            col1, col2 = st.columns([2, 1])
            select_all_polar = col1.checkbox("Select All", value=False)

            if select_all_polar:
                selected_sats_polar = st.multiselect(
                    "Satellites",
                    options=all_satellites,
                    default=all_satellites,
                    key="all_sat_selector_polar"
                )
            else:
                default_polar = all_satellites[:20]
                selected_sats_polar = st.multiselect(
                    "Satellites",
                    options=all_satellites,
                    default=default_polar,
                    key="manual_sat_selector_polar"
                )

            run_polar = col2.form_submit_button("Done")

            if run_polar:
                st.session_state["run_polar"] = True
                st.session_state["selected_sats_polar"] = (
                    all_satellites if select_all_polar else selected_sats_polar
                )

        selected_sats_polar = st.session_state.get("selected_sats_polar", all_satellites[:100])

    with graph_col2:
        if st.session_state.get("run_polar", False) and selected_sats_polar:
            # Gather all unique time points for animation
            all_times = set()
            sat_positions = {}
            for sat_name in selected_sats_polar:
                data = all_graph.get(sat_name, [])
                positions = []
                for t, alt, az, _ in data:
                    all_times.add(t)
                    positions.append((t, az.degrees, alt.degrees))
                sat_positions[sat_name] = positions

            sorted_times = sorted(all_times)
            if not sorted_times:
                st.warning("No polar data available for selected satellites.")
            else:
                color_map = px.colors.qualitative.Plotly
                sat_colors = {sat: color_map[i % len(color_map)] for i, sat in enumerate(selected_sats_polar)}

                # Add trails and animated dots
                frames = []
                for idx, current_time in enumerate(sorted_times):
                    frame_data = []
                    for sat_name in selected_sats_polar:
                        positions = sat_positions[sat_name]
                        past_positions = [(az, el) for t, az, el in positions if t <= current_time]
                        if not past_positions:
                            continue
                        az_trail, el_trail = zip(*past_positions)
                        r_trail = [90 - el for el in el_trail]

                        # Trail line
                        frame_data.append(go.Scatterpolar(
                            r=r_trail,
                            theta=az_trail,
                            mode="lines",
                            line=dict(width=1.5, color=sat_colors[sat_name]),
                            name=f"{sat_name} Trail",
                            showlegend=False,
                            hoverinfo="skip"
                        ))

                        # Latest position
                        az, el = az_trail[-1], el_trail[-1]
                        frame_data.append(go.Scatterpolar(
                            r=[90 - el],
                            theta=[az],
                            mode="markers",
                            marker=dict(size=12, color=sat_colors[sat_name]),
                            name=sat_name,
                            showlegend=False,
                            hovertemplate=f"{sat_name}<br>Azimuth: {az:.1f}°<br>Elevation: {el:.1f}°"
                        ))

                    frames.append(go.Frame(data=frame_data, name=str(idx)))

                # Initial data: all satellites at their first available time (show just the first dot)
                initial_data = []
                for idx, sat_name in enumerate(selected_sats_polar):
                    positions = sat_positions[sat_name]
                    if positions:
                        az, el = positions[0][1], positions[0][2]
                        r = 90 - el
                        initial_data.append(go.Scatterpolar(
                            r=[r],
                            theta=[az],
                            mode="markers",
                            marker=dict(size=12, color=sat_colors[sat_name]),
                            name=sat_name,
                           # showlegend=True,  # Always show legend for all satellites
                            hovertemplate=f"{sat_name}<br>Azimuth: {az:.1f}°<br>Elevation: {el:.1f}°"
                        ))

                az_ticks = list(range(0, 360, 30))
                az_labels = [
                    "N" if deg == 0 else
                    "E" if deg == 90 else
                    "S" if deg == 180 else
                    "W" if deg == 270 else
                    f"{deg}°"
                    for deg in az_ticks
                ]

                def format_time_label(t):
                    try:
                        dt = pd.to_datetime(t)
                        user_tz = pytz.timezone(st.session_state["user_tmz_label"])
                        dt = dt.tz_localize(pytz.UTC).astimezone(user_tz)
                        return dt.strftime("%Y-%m-%d %H:%M:%S")
                    except Exception:
                        return str(t)

                slider_steps = [{
                    "label": format_time_label(sorted_times[i]),
                    "method": "animate",
                    "args": [[str(i)], {
                        "frame": {"duration": 0, "redraw": True},
                        "mode": "immediate",
                        "transition": {"duration": 0}
                    }]
                } for i in range(len(sorted_times))]

                fig = go.Figure(
                    data=initial_data,
                    frames=frames
                )
                # Optional dashed ring to visualize min elevation threshold
                min_el = float(st.session_state["min_elevation"])
                fig.add_trace(go.Scatterpolar(
                    r=[90 - min_el] * 360,
                    theta=list(range(360)),
                    mode="lines",
                    line=dict(color="gray", dash="dot"),
                    name=f"Min Elevation = {min_el}°",
                    showlegend=True
                ))

                fig.update_layout(
                    title="Animated Satellite Sky Positions (Polar View)",
                    margin=dict(l=80, r=80, t=80, b=80),
                    polar=dict(
                        radialaxis=dict(
                            range=[0, 90],
                            angle=90,
                            tickangle=90,
                            tickvals=list(range(0, 91, 10)),
                            ticktext=["Zenith"] + [f"{d}°" for d in range(10, 91, 10)],
                            tickfont=dict(color="black")
                        ),
                        angularaxis=dict(
                            direction='clockwise',
                            rotation=90,
                            tickvals=az_ticks,
                            ticktext=az_labels,
                            tickfont=dict(color="white", size=12)
                        )
                    ),
                    height=950,
                    legend=dict(itemsizing='constant'),
                    updatemenus=[{
                        "type": "buttons",
                        "buttons": [{
                            "label": "Play",
                            "method": "animate",
                            "args": [None, {
                                "frame": {"duration": 100, "redraw": True},
                                "fromcurrent": True,
                                "transition": {"duration": 0}
                            }]
                        }, {
                            "label": "Pause",
                            "method": "animate",
                            "args": [[None], {
                                "frame": {"duration": 0, "redraw": False},
                                "mode": "immediate",
                                "transition": {"duration": 0}
                            }]
                        }],
                        "direction": "left",
                        "pad": {"r": 10, "t": 0, "b": 0},
                        "x": 0.15,
                        "y": -0.18,
                        "xanchor": "left",
                        "yanchor": "top",
                        "showactive": False,
                    }],
                    sliders=[{
                        "active": 0,
                        "yanchor": "top",
                        "xanchor": "left",
                        "currentvalue": {
                            "prefix": "Time: ",
                            "visible": True,
                            "xanchor": "right"
                        },
                        "transition": {"duration": 0, "easing": "cubic-in-out"},
                        "pad": {"b": 10, "t": 0},
                        "len": 0.8,
                        "x": 0.15,
                        "y": -0.25,
                        "steps": slider_steps
                    }]
                )

                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Select and click Done to view the polar plot.")

with tab3:
    st.subheader("3D Dome Plot (with Distance)")
    st.markdown(
        "This 3D dome plot shows where satellites move in the sky above you, with distance from the ground station. "
        "X = East, Y = North, Z = Up. The ground station is at the center. Distances are scaled (1000 km = 1 unit)."
    )

    all_names = list(all_graph.keys())
    default_dome = all_names[:5]
    with st.form("dome_selector_form"):
        col1, col2 = st.columns([2, 1])
        select_all_dome = col1.checkbox("Select All", value=False, key="select_all_dome")
        if select_all_dome:
            selected_dome_sats = st.multiselect(
                "Satellites for Dome Plot",
                options=all_names,
                default=all_names,
                key="all_sat_selector_dome"
            )
        else:
            selected_dome_sats = st.multiselect(
                "Satellites for Dome Plot",
                options=all_names,
                default=default_dome,
                key="manual_sat_selector_dome"
            )
        dome_plot_btn = col2.form_submit_button("Done")

    sat_data = []
    for sat_name in selected_dome_sats:
        data = all_graph.get(sat_name, [])
        for t, alt, az, dist in data:
            sat_data.append({
                "Azimuth": az.degrees,
                "Elevation": alt.degrees,
                "Distance_km": dist.km,
                "Satellite": sat_name
            })
    if sat_data:
        fig = plot_dome_with_distance(sat_data, scale=1000)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No data for selected satellites.")
