import sys

sys.path.append(
    "/Users/benpfeffer/Library/Mobile Documents/com~apple~CloudDocs/projects-portfolio-main/5-octobre"
)

import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from src.metrics.customer_analytics import *
from src.metrics.revenue_analytics import *
from src.metrics.time_analytics import *
from src.metrics.basic_metrics import *
from src.analysis import *

from src.visualization import *
from src.plot import *


def main():
    st.set_page_config(
        page_title="5-Octobre",
        layout="wide",
        initial_sidebar_state="expanded",
        page_icon=":octopus:",
    )
    # st.sidebar.image(
    tabs = st.tabs(["Home", "Basic", "Customer", "Revenue", "Time", "Products", "Web"])

    with tabs[0]:
        with st.container():
            st.title("5 Octobre")
            st.write(
                "Welcome to the 5 Octobre dashboard. This dashboard is designed to help you understand the performance of your business."
            )

    with tabs[1]:
        st.title("Quick Insights")

        with st.container():
            st.write("Here are some quick insights about your business.")
            cols = st.columns(3)

            cols[0].pyplot(plt)


if __name__ == "__main__":
    main()
