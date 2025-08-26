import os
import tempfile
os.environ["HOME"] = tempfile.gettempdir()
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
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import plotly.io as pio
import csv
from skyfield.toposlib import GeographicPosition
from skyfield.timelib import Time
from typing import Dict, Any


st.set_page_config(layout="wide")
st.sidebar.title("Location and Time settings")
st.title("Satellite Visualization Dashboard")

st.markdown(""" <style> .stTabs [data-baseweb="tab-list"] { gap: 12px; } .stTabs [data-baseweb="tab"] { height: 40px; padding: 16px 24px; background-color: black; border-radius: 8px 8px 0 0; border: 1px solid #e9ecef; font-weight: 500; transition: all 0.2s ease; } .stTabs [aria-selected="true"] { background-color: #ffffff; border-bottom: 3px solid #ff4b4b; color: #ff4b4b; font-weight: 600; } .stTabs [aria-selected="false"]:hover { background-color: #e9ecef; transform: translateY(-2px); } </style> """, unsafe_allow_html=True)

f0 = 11.325e9
c = 3e5 
curr_dir = os.path.join(tempfile.gettempdir(), "satellite_data")  # Safe temp directory
os.makedirs(curr_dir, exist_ok=True)
FULL_OUTPUT_DIR = tempfile.gettempdir()


# Collects and describes user input for satellite visualization.
#     Inputs:
#         latitude (str): Latitude in degrees, as a string (e.g., "43.07154").
#         longitude (str): Longitude in degrees, as a string (e.g., "-89.40829").
#         min_elevation (str): Minimum elevation angle in degrees, as a string (e.g., "40").
#         user_tmz_label (str): Timezone label (e.g., "UTC", "America/Chicago").
#         start_date (datetime.date): Start date for the visualization.
#         start_time (datetime.time): Start time for the visualization.
#         end_date (datetime.date): End date for the visualization.
#         end_time (datetime.time): End time for the visualization.
#         use_latest_tle (bool): Whether to use the latest Celestrak Starlink TLE (True/False).
#         tle_file (UploadedFile or None): Uploaded TLE file object from Streamlit, or None if not provided.

with st.sidebar.form("input_form"):
    if "latitude" not in st.session_state:
        st.session_state["latitude"] = "43.07154"
    if "longitude" not in st.session_state:
        st.session_state["longitude"] = "-89.40829"

    loc_row = st.columns(2)
    st.session_state["latitude"] = loc_row[0].text_input("Latitude (°)", st.session_state["latitude"])
    st.session_state["longitude"] = loc_row[1].text_input("Longitude (°)", st.session_state["longitude"])

    elev_row = st.columns(2)
    if "ground_elevation" not in st.session_state:
        st.session_state["ground_elevation"] = "266.3"
    st.session_state["ground_elevation"] = elev_row[0].text_input("Ground Station Elevation (m)", st.session_state["ground_elevation"])

    if "min_elevation" not in st.session_state:
        st.session_state["min_elevation"] = "40"
    st.session_state["min_elevation"] = elev_row[1].text_input("Min Satellite Elevation (°)", st.session_state["min_elevation"])

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
        st.session_state["start_time"] = datetime.utcnow().time().strftime("%H:%M:%S")

    start_row = st.columns(2)
    st.session_state["start_date"] = start_row[0].date_input("Start Date & Time", st.session_state["start_date"])
    st.session_state["start_time"] = start_row[1].text_input("HH:MM:SS", value=st.session_state["start_time"])

    if "end_date" not in st.session_state:
        st.session_state["end_date"] = datetime.utcnow().date() 
    if "end_time" not in st.session_state:
        st.session_state["end_time"] = (datetime.utcnow() + timedelta(minutes=10)).time().strftime("%H:%M:%S")

    end_row = st.columns(2)
    st.session_state["end_date"] = end_row[0].date_input("End Date & Time", st.session_state["end_date"])
    st.session_state["end_time"] = end_row[1].text_input("HH:MM:SS", value=st.session_state["end_time"])

    # Convert times to UTC after form is submitted
    tz = pytz.timezone(st.session_state["user_tmz_label"])
    
    # Parse start time from text input
    start_time_str = st.session_state["start_time"]
    try:
        start_time = datetime.strptime(start_time_str, "%H:%M:%S").time()
    except ValueError:
        st.error("Invalid start time format. Please use HH:MM:SS")
        st.stop()
    
    # Parse end time from text input
    end_time_str = st.session_state["end_time"]
    try:
        end_time = datetime.strptime(end_time_str, "%H:%M:%S").time()
    except ValueError:
        st.error("Invalid end time format. Please use HH:MM:SS")
        st.stop()
    
    start_datetime = datetime.combine(st.session_state["start_date"], start_time)
    end_datetime = datetime.combine(st.session_state["end_date"], end_time)
    start_datetime = tz.localize(start_datetime).astimezone(pytz.UTC)
    end_datetime = tz.localize(end_datetime).astimezone(pytz.UTC)


    # TLE input
    st.markdown("TLE Source")
    if "use_latest_tle" not in st.session_state:
        st.session_state["use_latest_tle"] = False
    st.session_state["use_latest_tle"] = st.checkbox("Use latest Celestrak Starlink TLE", value=st.session_state["use_latest_tle"])
    
    # TLE file uploader with size validation
    tle_file_obj = st.file_uploader(
        "Upload your own TLE file (.txt)", 
        type=["txt"],
        help="Maximum file size: 2MB. For larger files, use the Celestrak option above."
    )
    
    # Validate file size
    if tle_file_obj is not None:
        try:
            file_size = len(tle_file_obj.getvalue())
            if file_size > 2 * 1024 * 1024:  # 2MB limit
                st.error(f"File too large ({file_size/1024:.1f}KB). Maximum size is 2MB. Please use a smaller file or the Celestrak option.")
                tle_file_obj = None
            else:
                print(f"File uploaded successfully ({file_size/1024:.1f}KB)")
        except Exception as e:
            st.error(f"Error reading file: {str(e)}. Please try uploading again.")
            tle_file_obj = None
    
    st.session_state["tle_file_obj"] = tle_file_obj
    refresh_button = st.form_submit_button("Run Visualization")


def fetch_latest_tle_from_celestark():
    url = 'https://celestrak.org/NORAD/elements/gp.php?GROUP=starlink&FORMAT=tle'
    print(f"Downloading updated Starlink TLE data from {url}")
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()  
    except requests.RequestException as e:
        print(f"Error downloading Starlink TLE data: {e}")
        return None
    os.makedirs(curr_dir, exist_ok=True)
    tle_path = os.path.join(curr_dir, "starlink.tle")
    with open(tle_path, 'w') as f:
        f.write(r.text)
    return tle_path


def check_field_of_view(ground_station : GeographicPosition , tle_file : str, start_time_sf : Time, end_time_sf : Time, min_elevation : int = 10) -> Dict[str, Dict[int, Any]]:
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

  satellites = load.tle_file(tle_file)

  satellite_dict = {sat.name: sat for sat in satellites}
  print(len(satellite_dict))
  visible_satellites = {} # {SAT: {event: [t]}}

  print("Start Time (Skyfield UTC):", start_time_sf)
  print("End Time (Skyfield UTC):", end_time_sf)
  print(start_time_sf.utc_iso())
  print(end_time_sf.utc_iso())
  for sat_name, sat in satellite_dict.items():

    visible_satellites[sat_name] = {"obj": sat}
    relative_pos_sat = sat - ground_station
    alt0, az, dist = relative_pos_sat.at(start_time_sf).altaz()
    # print(f"SAT: {sat_name}, alt: {alt}, az: {az}, dist: {dist}")

    # events: 0->rises, 1->peaks, 2->sets
    t, events = sat.find_events(ground_station, start_time_sf, end_time_sf, altitude_degrees=min_elevation)
    # if events.size != 0:
    #     print(f"SAT: {sat_name}, t: {t.utc_iso()}, events: {events}")
    for ti, event in zip(t, events):
      curr_val = visible_satellites[sat_name].get(event, [])
      if curr_val == []:
        visible_satellites[sat_name][event] = []

      if event == 0 or event == 1:
        visible_satellites[sat_name][event].append(ti)
      if event == 2:
        alt, az, dist = (visible_satellites[sat_name]['obj'] - ground_station).at(ti).altaz()
        if alt.degrees <= min_elevation:
          visible_satellites[sat_name][event].append(ti)


    # if 1 not in visible_satellites[sat_name]:
    #     del visible_satellites[sat_name]
    if not any(k in visible_satellites[sat_name] for k in (0, 1, 2)):
        if alt0.degrees >= min_elevation:
            visible_satellites[sat_name].setdefault(0, []).append(start_time_sf)
        else:
            del visible_satellites[sat_name]
  st.write(f"Number of visible satellites: {len(visible_satellites)}")
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

def write_satellites_to_csv(doppler_shifts, filename="satellite_data.csv"):
    """
    Write all satellites in field of view to a CSV file with columns:
    1) Satellite name
    2) Timestamp
    3) Doppler shift
    
    Args:
        doppler_shifts: Dictionary of Doppler shifts from doppler_calc function
        filename: Output CSV filename
    """
    rows = []
    
    for sat, shifts in doppler_shifts.items():
        for timestamp, doppler_shift in shifts:
            rows.append({
                'Satellite': sat,
                'Timestamp (UTC)': timestamp,
                'Doppler Shift (Hz)': float(doppler_shift)
            })
    
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(filename, index=False)
        st.success(f"Saved {len(df):,} rows → {filename}")
        return filename
    else:
        st.warning("No satellite data available to export")
        return None

def plot_doppler(doppler_shifts, timezone='UTC', selected_sats=None):
    fig = go.Figure()

    # If a selection is provided, filter satellites
    sats_to_plot = doppler_shifts.keys() if selected_sats is None else selected_sats

    for sat in sats_to_plot:
        shifts = doppler_shifts.get(sat, [])
        if not shifts:
            continue
        times, vals = zip(*shifts)
        # Convert string times to datetime objects
        times = [datetime.strptime(t, '%Y-%m-%dT%H:%M:%SZ') for t in times]
        fig.add_trace(go.Scatter(
            x=times,
            y=vals,
            mode='lines',
            name=sat
        ))

    fig.update_layout(
        xaxis_title='Time (UTC)',
        yaxis_title='Doppler Shift (Hz)',
        legend_title='Satellite',
        hovermode='x unified',
        template='plotly_white',
        margin=dict(l=40, r=20, t=30, b=40),
        height=700
    )

    st.plotly_chart(fig)
 
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

        # Create hover data with real values
        hover_data = []
        for p in points:
            hover_data.append([p["Distance_km"], p["Elevation"], p["Azimuth"]])

        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z,
            mode='lines',
            name=sat_name,
            line=dict(width=3),
            hovertemplate=f"{sat_name}<br>Distance: %{{customdata[0]:.1f}} km<br>Elevation: %{{customdata[1]:.1f}}°<br>Azimuth: %{{customdata[2]:.1f}}°<extra></extra>",
            customdata=hover_data
        ))

    fig.update_layout(
        scene=dict(
            xaxis_title='East (km)',
            yaxis_title='North (km)',
            zaxis_title='Altitude',
            aspectmode='data'
        ),
        #title='3D Dome Plot with Distance',
        height=700
    )

    return fig

def plot_dome_animated(all_graph, scale=1.0, target_time=None):
    """
    Create an animated 3D dome plot with time slider.
    Shows full satellite tracks with moving dots along the tracks.
    
    Args:
        all_graph: Dictionary from doppler_calc with satellite trajectory data
        scale: Distance scaling factor
        target_time: Optional timestamp to jump to (format: "YYYY-MM-DDTHH:MM:SSZ")
    """
    # Collect all unique timestamps and create full tracks
    all_times = set()
    sat_positions = {}
    sat_tracks = {}
    
    # Color palette for satellites
    colors = px.colors.qualitative.Plotly
    
    for sat_name, data in all_graph.items():
        positions = []
        track_x, track_y, track_z = [], [], []
        
        for timestamp, alt, az, dist in data:
            all_times.add(timestamp)
            positions.append((timestamp, az.degrees, alt.degrees, dist.km))
            
            # Create full track using same method as static plot
            r = dist.km / scale
            az_rad = np.radians(az.degrees)
            el_rad = np.radians(alt.degrees)
            
            # Convert to 3D Cartesian (same as static plot)
            phi = np.radians(90) - el_rad
            x = r * np.sin(phi) * np.cos(az_rad)
            y = r * np.sin(phi) * np.sin(az_rad)
            z = r * np.cos(phi)
            
            track_x.append(x)
            track_y.append(y)
            track_z.append(z)
        
        sat_positions[sat_name] = positions
        sat_tracks[sat_name] = (track_x, track_y, track_z)
    
    # Sort timestamps
    sorted_times = sorted(all_times)
    
    if not sorted_times:
        st.warning("No data available for animation")
        st.stop()
    
    # Find target frame index if target_time is provided
    target_frame_idx = 0
    if target_time:
        try:
            # Parse flexible time formats
            target_dt = None
            
            # Parse the time string (now always in YYYY-MM-DD HH:MM:SS format)
            try:
                target_dt = pd.to_datetime(target_time, format="%Y-%m-%d %H:%M:%S")
            except:
                # Fallback to pandas automatic parsing
                target_dt = pd.to_datetime(target_time)
            
            # Find the closest timestamp
            min_diff = float('inf')
            for i, timestamp in enumerate(sorted_times):
                timestamp_dt = pd.to_datetime(timestamp)
                diff = abs((target_dt - timestamp_dt).total_seconds())
                if diff < min_diff:
                    min_diff = diff
                    target_frame_idx = i
                    
        except Exception as e:
            st.warning(f"Invalid time format. Please use the time picker above.")
            target_frame_idx = 0
    
    # Create initial data with tracks and first positions
    initial_data = []
    
    # Add ground station
    initial_data.append(go.Scatter3d(
        x=[0], y=[0], z=[0],
        mode='markers',
        marker=dict(size=8, color='blue'),
        name='Ground Station'
    ))
    
    # Add satellite tracks and target positions
    target_time = sorted_times[target_frame_idx]
    for i, (sat_name, positions) in enumerate(sat_positions.items()):
        # Get track coordinates
        track_x, track_y, track_z = sat_tracks[sat_name]
        sat_color = colors[i % len(colors)]
        
        # Add full track (static)
        initial_data.append(go.Scatter3d(
            x=track_x, y=track_y, z=track_z,
            mode='lines',
            name=f"{sat_name} Track",
            line=dict(width=2, color=sat_color),
            showlegend=False,  # Hide tracks from legend
            hovertemplate=f"{sat_name} Track<br>Distance: %{{customdata[0]:.1f}} km<br>Elevation: %{{customdata[1]:.1f}}°<extra></extra>",
            customdata=[[dist.km, alt.degrees] for timestamp, alt, az, dist in data]
        ))
        
        # Find target position
        current_pos = None
        for timestamp, az, el, dist in positions:
            if timestamp <= target_time:
                current_pos = (az, el, dist)
            else:
                # Found a timestamp beyond target_time, use the last valid position
                break
        
        # If no position found, try to use the first available position
        if current_pos is None and positions:
            current_pos = positions[0][1:]  # Use first position as fallback
        
        if current_pos:
            az, el, dist = current_pos
            r = dist / scale
            
            # Convert to 3D Cartesian (same as static plot)
            az_rad = np.radians(az)
            el_rad = np.radians(el)
            phi = np.radians(90) - el_rad
            x = r * np.sin(phi) * np.cos(az_rad)
            y = r * np.sin(phi) * np.sin(az_rad)
            z = r * np.cos(phi)
            
            # Add satellite marker (same color as track)
            initial_data.append(go.Scatter3d(
                x=[x], y=[y], z=[z],
                mode='markers',
                name=sat_name,
                marker=dict(size=8, color=sat_color),
                showlegend=True,  # Change from False to True
                hovertemplate=f"{sat_name}<br>Distance: {dist:.1f} km<br>Elevation: {el:.1f}°<br>Azimuth: {az:.1f}°<extra></extra>"
            ))
    
    # Create figure with initial data
    fig = go.Figure(data=initial_data)
    
    # Create frames for animation (optimized to reduce data size)
    frames = []
    
    # Sample frames to reduce data size (every 5th frame)
    frame_step = max(1, len(sorted_times) // 100)  # Max 100 frames
    frame_indices = range(0, len(sorted_times), frame_step)
    
    for idx in frame_indices:
        current_time = sorted_times[idx]
        frame_data = []
        
        # Add ground station to each frame
        frame_data.append(go.Scatter3d(
            x=[0], y=[0], z=[0],
            mode='markers',
            marker=dict(size=8, color='blue'),
            name='Ground Station',
            showlegend=False
        ))
        
        # Add tracks and current satellite positions
        for i, (sat_name, positions) in enumerate(sat_positions.items()):
            track_x, track_y, track_z = sat_tracks[sat_name]
            sat_color = colors[i % len(colors)]
            
            # Add track (same in all frames)
            frame_data.append(go.Scatter3d(
                x=track_x, y=track_y, z=track_z,
                mode='lines',
                name=f"{sat_name} Track",
                line=dict(width=2, color=sat_color),
                showlegend=False,  # Hide tracks from legend
                hovertemplate=f"{sat_name} Track<br>Distance: %{{customdata[0]:.1f}} km<br>Elevation: %{{customdata[1]:.1f}}°<extra></extra>",
                customdata=[[dist.km, alt.degrees] for timestamp, alt, az, dist in data]
            ))
            
            # Find current position
            current_pos = None
            for timestamp, az, el, dist in positions:
                if timestamp <= current_time:
                    current_pos = (az, el, dist)
                else:
                    # Found a timestamp beyond current_time, use the last valid position
                    break
            
            # If no position found, try to use the first available position
            if current_pos is None and positions:
                current_pos = positions[0][1:]  # Use first position as fallback
            
            if current_pos:
                az, el, dist = current_pos
                r = dist / scale
                
                # Convert to 3D Cartesian (same as static plot)
                az_rad = np.radians(az)
                el_rad = np.radians(el)
                phi = np.radians(90) - el_rad
                x = r * np.sin(phi) * np.cos(az_rad)
                y = r * np.sin(phi) * np.sin(az_rad)
                z = r * np.cos(phi)
                
                # Add satellite marker (same color as track)
                frame_data.append(go.Scatter3d(
                    x=[x], y=[y], z=[z],
                    mode='markers',
                    name=sat_name,
                    marker=dict(size=8, color=sat_color),
                    showlegend=True,  # Change from False to True
                    hovertemplate=f"{sat_name}<br>Distance: {dist:.1f} km<br>Elevation: {el:.1f}°<br>Azimuth: {az:.1f}°<extra></extra>"
                ))
        
        frames.append(go.Frame(data=frame_data, name=str(idx)))
    
    # Add frames
    fig.frames = frames
    
    # Update layout with animation controls
    fig.update_layout(
        scene=dict(
            xaxis_title='East (km)',
            yaxis_title='North (km)',
            zaxis_title='Altitude (km)',
            aspectmode='data'
        ),
        title='Animated 3D Dome Plot with Tracks',
        height=700,
        updatemenus=[{
            "type": "buttons",
            "buttons": [
                {
                    "label": "Play",
                    "method": "animate",
                    "args": [None, {
                        "frame": {"duration": 100, "redraw": True},
                        "fromcurrent": True,
                        "transition": {"duration": 0}
                    }]
                },
                {
                    "label": "Pause",
                    "method": "animate",
                    "args": [[None], {
                        "frame": {"duration": 0, "redraw": False},
                        "mode": "immediate",
                        "transition": {"duration": 0}
                    }]
                }
            ],
            "direction": "left",
            "pad": {"r": 10, "t": 0, "b": 0},
            "x": 0.1,
            "y": 0,
            "xanchor": "right",
            "yanchor": "top",
            "showactive": False,
        }],
        sliders=[{
            "active": target_frame_idx,
            "yanchor": "top",
            "xanchor": "left",
            "currentvalue": {
                "prefix": "Time: ",
                "visible": True,
                "xanchor": "right"
            },
            "transition": {"duration": 0, "easing": "cubic-in-out"},
            "pad": {"b": 10, "t": 0},
            "len": 0.9,
            "x": 0.1,
            "y": 0,
            "steps": [{
                "label": sorted_times[i],
                "method": "animate",
                "args": [[str(i)], {
                    "frame": {"duration": 0, "redraw": True},
                    "mode": "immediate",
                    "transition": {"duration": 0}
                }]
            } for i in frame_indices]
        }]
    )
    
    return fig

def write_satellites_to_csv(doppler_shifts, filename="satellite_data.csv"):
    """
    Write all satellites in field of view to a CSV file with columns:
    1) Satellite name
    2) Timestamp
    3) Doppler shift
    
    Args:
        doppler_shifts: Dictionary of Doppler shifts from doppler_calc function
        filename: Output CSV filename
    """
    rows = []
    
    for sat, shifts in doppler_shifts.items():
        for timestamp, doppler_shift in shifts:
            rows.append({
                'Satellite': sat,
                'Timestamp (UTC)': timestamp,
                'Doppler Shift (Hz)': float(doppler_shift)
            })
    
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(filename, index=False)
        st.toast(f"Saved {len(df):,} rows → {filename}")
        return filename
    else:
        st.warning("No satellite data available to export")
        return None


visible_sats = {}
ts = load.timescale()
# Decide on the TLE file to use:
if (st.session_state["tle_file_obj"] is not None):
    try:
        if st.session_state["tle_file_obj"] is not None:
            try:
                tle_bytes = st.session_state["tle_file_obj"].read()
                tle_text = tle_bytes.decode('utf-8')
            except Exception as upload_error:
                st.error(f"Error reading uploaded file: {str(upload_error)}. This might be a temporary issue. Please try uploading again or use the Celestrak option.")
                st.stop()
        
        # Validate TLE content
        if len(tle_text.strip()) == 0:
            st.error("TLE data is empty. Please provide valid TLE data.")
            st.stop()
        
        # Check if it looks like a TLE file
        lines = tle_text.strip().split('\n')
        if len(lines) < 2:
            st.error("Data doesn't appear to be valid TLE format. TLE data should contain satellite data in two-line format.")
            st.stop()
        
        tle_file = os.path.join(curr_dir, "uploaded.tle")
        with open(tle_file, 'w') as f:
            f.write(tle_text)
        st.toast("TLE data processed successfully!")
        
    except UnicodeDecodeError:
        st.error("File encoding error. Please upload a text file (.txt) with UTF-8 encoding.")
        st.stop()
    except Exception as e:
        st.error(f"Error processing TLE data: {str(e)}. Please try uploading again or use the Celestrak option.")
        st.stop()

elif st.session_state["use_latest_tle"]:
    try:
        tle_file = fetch_latest_tle_from_celestark()
        if tle_file is None:
            st.error("Failed to fetch latest TLE data from Celestrak!")
        else:
            st.toast("Downloaded latest TLE file from Celestrak!")
    except Exception as e:
        st.error(f"Error fetching latest TLE data: {e}")
else:
    st.error("No TLE file provided! Use teh side bar to upload a TLE or click option to use the latest TLE from Celestrack")
    st.stop()
        
ground_station = wgs84.latlon(
        float(st.session_state["latitude"]),
        float(st.session_state["longitude"]),
        elevation_m=float(st.session_state["ground_elevation"])
    )
observer = ground_station
start_sf = ts.utc(start_datetime)
end_sf = ts.utc(end_datetime)

visible_satellites = check_field_of_view(ground_station, tle_file, start_sf, end_sf, float(st.session_state["min_elevation"]))
doppler_shifts, all_range_rate, all_graph = doppler_calc(start_sf, end_sf, visible_satellites, observer, time_step=1)
#print(all_graph['STARLINK-3587'])

tab1, tab2, tab3 = st.tabs(["Doppler Shift", "Polar Plot", "Dome Plot"])

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
    
    # Add CSV export button
    if st.button("Export Satellite Data to CSV"):
        csv_filename = write_satellites_to_csv(doppler_shifts)
        if csv_filename:
            with open(csv_filename, 'r') as f:
                csv_data = f.read()
            st.download_button(
                label="Download CSV",
                data=csv_data,
                file_name=csv_filename,
                mime="text/csv"
            )
    
    graph_col, control_col = st.columns([7, 2])
    all_satellites = list(doppler_shifts.keys())

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

    with graph_col:
        plot_doppler(doppler_shifts, selected_sats=st.session_state["selected_sats"])


with tab2:
    st.subheader("Polar Plot")
    st.markdown(
        "This polar plot shows satellite positions in the sky over time. "
        "The ground station is at the center (zenith), and the edge is the horizon. "
        "Azimuth is the angle around the circle (0°=N, 90°=E, 180°=S, 270°=W). "
        "Elevation is the distance from the center (90° at center, 0° at edge)."
    )

    # Create two columns for satellite selector and time picker
    selector_col, time_col = st.columns([1, 1])
    
    with selector_col:
        # Satellite selection form (like tab3)
        all_names_polar = list(all_graph.keys())  # Only satellites in field of view
        default_polar = all_names_polar[:10]
        
        with st.form("polar_selector_form"):
            col1, col2 = st.columns([2, 1])
            select_all_polar = col1.checkbox("Select All", value=False, key="select_all_polar")
            
            if select_all_polar:
                selected_polar_sats = st.multiselect(
                    "Satellites for Polar Plot",
                    options=all_names_polar,
                    default=all_names_polar,
                    help="All satellites are selected.",
                    key="all_sat_selector_polar"
                )
            else:
                selected_polar_sats = st.multiselect(
                    "Satellites for Polar Plot",
                    options=all_names_polar,
                    default=default_polar,
                    help="Choose one or more satellites to display.",
                    key="manual_sat_selector_polar"
                )
            polar_plot_btn = col2.form_submit_button("Done")

    with time_col:
        # Time picker interface
        if selected_polar_sats:
            with st.form("polar_time_picker_form"):
                st.markdown("**Jump to time:**")
                
                # Create time picker columns with jump button
                time_col1, time_col2, time_col3, time_col4, jump_col = st.columns([1, 1, 1, 1, 1])
                
                with time_col1:
                    hour = st.selectbox("Hour", range(0, 24), index=10, key="polar_hour")
                
                with time_col2:
                    minute = st.selectbox("Minute", range(0, 60), index=30, key="polar_minute")
                
                with time_col3:
                    second = st.selectbox("Second", range(0, 60), index=0, key="polar_second")
                
                with time_col4:
                    jump_date = st.date_input("Date", value=datetime.now().date(), key="polar_date")
                
                with jump_col:
                    st.write("") 
                    jump_btn = st.form_submit_button("Jump", type="primary")
                    if jump_btn:
                        time_input = f"{jump_date} {hour:02d}:{minute:02d}:{second:02d}"
                        if time_input:
                            st.session_state["polar_jump_to_time"] = time_input
                            st.toast(f"Jumping to {time_input}")
                        else:
                            st.warning("Please enter a timestamp")

    # Polar plot implementation
    if selected_polar_sats:
        # Filter all_graph to only include selected satellites
        filtered_graph = {sat: all_graph[sat] for sat in selected_polar_sats if sat in all_graph}
        
        if filtered_graph:
            # Collect all unique timestamps and create full tracks
            all_times = set()
            sat_positions = {}
            sat_tracks = {}
            
            # Color palette for satellites
            colors = px.colors.qualitative.Plotly
            
            # Function to create lighter version of a color
            def lighten_color(color, factor=0.6):
                """Create a lighter version of a color by mixing with white"""
                import colorsys
                # Convert hex to RGB
                color = color.lstrip('#')
                r, g, b = tuple(int(color[i:i+2], 16) for i in (0, 2, 4))
                # Convert to HSV
                h, s, v = colorsys.rgb_to_hsv(r/255, g/255, b/255)
                # Increase value (brightness) and decrease saturation
                v = min(1.0, v + (1-v) * factor)
                s = s * (1 - factor * 0.5)
                # Convert back to RGB
                r, g, b = colorsys.hsv_to_rgb(h, s, v)
                # Convert to hex
                return f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}'
            
            for sat_name, data in filtered_graph.items():
                positions = []
                track_az, track_el = [], []
                
                for timestamp, alt, az, dist in data:
                    all_times.add(timestamp)
                    positions.append((timestamp, az.degrees, alt.degrees))
                    
                    # Create full track for polar plot
                    track_az.append(az.degrees)
                    track_el.append(alt.degrees)
                
                sat_positions[sat_name] = positions
                sat_tracks[sat_name] = (track_az, track_el)
            
            # Sort timestamps
            sorted_times = sorted(all_times)
            
            if not sorted_times:
                st.warning("No data available for animation")
                st.stop()
            
            # Create initial data with tracks and first positions
            initial_data = []
            
            # Add satellite tracks and target positions
            target_time = sorted_times[0]
            for i, (sat_name, positions) in enumerate(sat_positions.items()):
                # Get track coordinates
                track_az, track_el = sat_tracks[sat_name]
                sat_color = colors[i % len(colors)]
                
                # Add full track (static)
                initial_data.append(go.Scatterpolar(
                    r=track_el,
                    theta=track_az,
                    mode='lines',
                    name=f"{sat_name} Track",
                    line=dict(width=2, color=lighten_color(sat_color)),  # Lighter track color
                    showlegend=False
                ))
                
                # Find target position
                current_pos = None
                for timestamp, az, el in positions:
                    if timestamp <= target_time:
                        current_pos = (az, el)
                    else:
                        # Found a timestamp beyond target_time, use the last valid position
                        break
                
                # If no position found, try to use the first available position
                if current_pos is None and positions:
                    current_pos = positions[0][1:]  # Use first position as fallback
                
                if current_pos:
                    az, el = current_pos
                    if el >= float(st.session_state["min_elevation"]):
                        # Add satellite marker (same color as track)
                        initial_data.append(go.Scatterpolar(
                            r=[el],
                            theta=[az],
                            mode='markers',
                            name=sat_name,
                            marker=dict(size=12, color=sat_color),  # Original bright color for dots
                            showlegend=True
                        ))
            
            # Create figure with initial data
            fig = go.Figure(data=initial_data)
            
            # Create frames for animation (optimized to reduce data size)
            frames = []
            
            # Sample frames to reduce data size (every 5th frame)
            frame_step = max(1, len(sorted_times) // 100)  # Max 100 frames
            frame_indices = range(0, len(sorted_times), frame_step)
            
            for idx in frame_indices:
                current_time = sorted_times[idx]
                frame_data = []
                
                # Add tracks and current satellite positions
                for i, (sat_name, positions) in enumerate(sat_positions.items()):
                    track_az, track_el = sat_tracks[sat_name]
                    sat_color = colors[i % len(colors)]
                    
                    # Add track (same in all frames)
                    frame_data.append(go.Scatterpolar(
                        r=track_el,
                        theta=track_az,
                        mode='lines',
                        name=f"{sat_name} Track",
                        line=dict(width=1, color=lighten_color(sat_color)),  # Lighter track color
                        showlegend=False
                    ))
                    
                    # Find current position
                    current_pos = None
                    for timestamp, az, el in positions:
                        if timestamp <= current_time:
                            current_pos = (az, el)
                        else:
                            # Found a timestamp beyond current_time, use the last valid position
                            break
                    
                    # If no position found, try to use the first available position
                    if current_pos is None and positions:
                        current_pos = positions[0][1:]  # Use first position as fallback
                    
                    if current_pos:
                        az, el = current_pos
                        if el >= float(st.session_state["min_elevation"]):
                            # Add satellite marker (same color as track)
                            frame_data.append(go.Scatterpolar(
                                r=[el],
                                theta=[az],
                                mode='markers',
                                name=sat_name,
                                marker=dict(size=12, color=sat_color),  # Original bright color for dots
                                showlegend=True
                            ))
                
                frames.append(go.Frame(data=frame_data, name=str(idx)))
            
            # Add frames
            fig.frames = frames
            
            # Update layout with animation controls
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        range=[90, 0],
                        angle=90,
                        tickangle=90,
                        tickvals=list(range(90, -1, -10)),
                        ticktext=["Zenith"] + [f"{d}°" for d in range(80, -1, -10)],
                        tickfont=dict(color="black")
                    ),
                    angularaxis=dict(
                        direction='clockwise',
                        rotation=90,
                        tickvals=[0, 45, 90, 135, 180, 225, 270, 315],
                        ticktext=["N", "NE", "E", "SE", "S", "SW", "W", "NW"],
                        tickfont=dict(color="white", size=12)
                    )
                ),
                title='Animated Polar Plot with Tracks',
                height=800,
                updatemenus=[{
                    "type": "buttons",
                    "buttons": [
                        {
                            "label": "Play",
                            "method": "animate",
                            "args": [None, {
                                "frame": {"duration": 100, "redraw": True},
                                "fromcurrent": True,
                                "transition": {"duration": 0}
                            }]
                        },
                        {
                            "label": "Pause",
                            "method": "animate",
                            "args": [[None], {
                                "frame": {"duration": 0, "redraw": False},
                                "mode": "immediate",
                                "transition": {"duration": 0}
                            }]
                        }
                    ],
                    "direction": "left",
                    "pad": {"r": 10, "t": 0, "b": 0},
                    "x": 0.1,
                    "y": 0,
                    "xanchor": "right",
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
                    "len": 0.9,
                    "x": 0.1,
                    "y": 0,
                    "steps": [{
                        "label": sorted_times[i],
                        "method": "animate",
                        "args": [[str(i)], {
                            "frame": {"duration": 0, "redraw": True},
                            "mode": "immediate",
                            "transition": {"duration": 0}
                        }]
                    } for i in frame_indices]
                }]
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No data available for selected satellites.")
    else:
        st.info("Please select satellites and click Done to view the polar plot.")


with tab3:
    st.subheader("3D Dome Plot")
    st.markdown(
        "This animated 3D dome plot shows satellite positions over time. "
        "Use the time slider to see satellite movement, or click Play to animate. "
        "X = East, Y = North, Z = Altitude. The ground station is at the center. "
        "Negative values indicate: East = negative values are West, North = negative values are South."
    )

    # Add plot type selector
   
    
    all_names = list(all_graph.keys())
    default_dome = all_names[:10]
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
    
    # Plot type selector and time input in same row
    plot_col1, plot_col2 = st.columns([2, 4])
    
    with plot_col1:
        plot_type = st.radio(
            "Select Plot Type:",
            ["Static Plot", "Animated Plot"],
            index=1  # Default to animated
        )
    
    with plot_col2:
        if plot_type == "Animated Plot":
            # Time picker interface in a form
            with st.form("time_picker_form"):
                st.markdown("**Jump to time:**")
                
                # Create time picker columns with jump button
                time_col1, time_col2, time_col3, time_col4, jump_col = st.columns([1, 1, 1, 1, 1])
                
                with time_col1:
                    hour = st.selectbox("Hour", range(0, 24), index=10, key="jump_hour")
                
                with time_col2:
                    minute = st.selectbox("Minute", range(0, 60), index=30, key="jump_minute")
                
                with time_col3:
                    second = st.selectbox("Second", range(0, 60), index=0, key="jump_second")
                
                with time_col4:
                    # Date picker
                    jump_date = st.date_input(
                        "Date",
                        value=datetime.now().date(),
                        key="jump_date"
                    )
                
                with jump_col:
                    # Jump button next to the pickers
                    st.write("") 
                    jump_btn = st.form_submit_button("Jump", type="primary")
                    # Make the button red using custom CSS
                    st.markdown(
                        """
                        <style>
                        button[kind="primary"] {
                            background-color: #d9534f !important;
                            color: white !important;
                        }
                        </style>
                        """,
                        unsafe_allow_html=True,
                    )
                    if jump_btn:
                        time_input = f"{jump_date} {hour:02d}:{minute:02d}:{second:02d}"
                        if time_input:
                            st.session_state["jump_to_time"] = time_input
                            st.toast(f"Jumping to {time_input}")
                        else:
                            st.warning("Please enter a timestamp")

    if plot_type == "Static Plot":
        # Original static plot
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
            fig = plot_dome_with_distance(sat_data, scale=1.0)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data for selected satellites.")
    
    else:
        # Animated plot
        if selected_dome_sats:
            # Filter all_graph to only include selected satellites
            filtered_graph = {sat: all_graph[sat] for sat in selected_dome_sats if sat in all_graph}
            
            if filtered_graph:
                # Get target time from session state if available
                target_time = st.session_state.get("jump_to_time", None)
                fig = plot_dome_animated(filtered_graph, scale=1.0, target_time=target_time)
                st.plotly_chart(fig, use_container_width=True)
                
                # Clear the target time after using it
                if target_time:
                    st.session_state.pop("jump_to_time", None)
            else:
                st.info("No data for selected satellites.")
        else:
            st.info("Please select satellites and click Done to view the animated plot.")
