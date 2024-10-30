import streamlit as st

# from model import *
from model import Model

TITLE = "Vaccination Goal Visualizer"
VERSION = 4.2

plotly_chart_config = {
    "displaylogo": False,
    "modeBarButtonsToRemove": [
        "resetScale",
        "toggleHover",
        "toggleSpikelines",
        "autoScale2d",
        "hoverClosestCartesian",
        "hoverCompareCartesian",
    ],
}


# Defines the application
def app():
    st.set_page_config(page_title=TITLE, page_icon=None)

    st.title(f"{TITLE}, v{VERSION}")

    st.warning("The project ended in June 2023")

    st.markdown(
        """
        This app approximates a date when our global vaccination goals will be achieved given the current rate of vaccination; data are updated every day and automatically reflected in charts.
        * **Data sources:** [Our World In Data](https://ourworldindata.org/covid-vaccinations), [United Nations](https://population.un.org/wpp)
        * **Disclaimer:** *vaccination data in certain regions is reported inconsistently; plot figures are estimated and can not be fully accurate*
        * Another excellent project using the same data from OWID: [Covidvax.live](https://covidvax.live/)
    """
    )

    with st.spinner(text="Preparing vaccination and population data..."):
        model = Model()

    locations = st.multiselect(
        "Select location(s)", model.sorted_unique_locations, default="World"
    )

    with st.spinner(text="Loading..."):
        st.warning(
            "Tip: you can zoom in on and pan the chart, select areas, drag the axes and more..."
        )
        for location in locations:
            df, location_info = model.process_location_vaccination_data(location)

            fig = model.make_plot(df, location_info, location)
            st.plotly_chart(fig, use_container_width=False, config=plotly_chart_config)

            with st.expander(label="Show/Hide Dataset", expanded=False):
                st.markdown(
                    """Missing data are indicated with <span style='font-weight:bold;'>nan</span>, <span style='background-color:#ffff00'>duplicate values</span> in <b>daily_vaccinations</b> are the result of missing data, last valid observations were used to fill the gap (forward fill)""",
                    unsafe_allow_html=True,
                )

                df_for_display = model.process_df_for_display(df)
                st.dataframe(df_for_display)

                download_button = model.make_download_button(df, location)
                st.markdown(download_button, unsafe_allow_html=True)


if __name__ == "__main__":
    app()
