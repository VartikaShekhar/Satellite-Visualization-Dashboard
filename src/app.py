import os
import streamlit as st

# Set environment variables to avoid permission issues in Hugging Face deployment
os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'
os.environ['STREAMLIT_BROWSER_GATHER_USAGE_STATS'] = 'false'
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib'
os.environ['STREAMLIT_CONFIG_DIR'] = '/tmp/.streamlit'

# Create necessary directories
os.makedirs('/tmp/matplotlib', exist_ok=True)
os.makedirs('/tmp/.streamlit', exist_ok=True)

import pytz