import base64
import copy
from datetime import timedelta
from io import StringIO

import humanize
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pycountry
import streamlit as st

population_data = "data/WPP2019_POP_F01_1_TOTAL_POPULATION_BOTH_SEXES.xlsx"
year = "2020"

location_codes = {
    "Asia": 935,
    "Cape Verde": 132,
    "Europe": 908,
    "High income": 1503,
    "Low income": 1500,
    "North America": 905,
    "Oceania": 909,
    "South America": 931,
    "South Korea": 410,
    "Upper middle income": 1502,
    "World": 900,
}

# TODO: needs a better algorithm
# Commented out values have been fixed
problematic_locations = [
    "Bonaire Sint Eustatius and Saba",
    "Burkina Faso",
    "Cook Islands",
    "Democratic Republic of Congo",
    "European Union",
    "Faeroe Islands",
    "Guernsey",
    "Niger",
    "Nigeria",
    "Northern Cyprus",
    "Pitcairn",
    "Sao Tome and Principe",
    "South Sudan",
    "Sudan",
    "Tanzania",
    "Turkmenistan",
    "Uganda",
    # "Bhutan",
    # "Falkland Islands",
]


class Model:
    def __init__(self):
        self.vaccination_data = self.get_vaccination_data()
        self.sorted_unique_locations = sorted(self.vaccination_data.location.unique())
        self.population_data = self.get_population_data()

    @st.cache(show_spinner=False, suppress_st_warning=True)
    def get_vaccination_data(self) -> pd.DataFrame:
        try:
            df = pd.read_csv(
                "data/vaccinations.csv",
                usecols=["date", "location", "daily_vaccinations"],
                parse_dates=["date"],
            )
        except Exception as e:
            st.error(e)
            st.stop()

        return df.query("date < '2023-06-01'")

    @st.cache(show_spinner=False, suppress_st_warning=True)
    def get_population_data(self) -> pd.DataFrame:
        try:
            df = pd.read_excel(population_data)
        except Exception as e:
            st.warning(e)
            st.stop()

        # Remove first 14 rows containing unrelated information
        df = df.iloc[15:]

        # Set the first row as column names
        df.columns = df.iloc[0]

        # Remove the first row containing column names as it is no longer needed
        df = df.iloc[1:]

        return df

    def make_download_button(self, df, location):
        output = StringIO()
        df.to_csv(output)
        output_value = output.getvalue()
        b64 = base64.b64encode(
            output_value.encode()
        ).decode()  # strings <-> bytes conversions is necessary here
        output.close()
        return f'<a href="data:file/csv;base64,{b64}" download="Vaccination Info - {location}.csv">Download CSV</a>'

    def process_df_for_display(self, df):
        df.reset_index(drop=True, inplace=True)

        daily_vaccinations_duplicates = df["daily_vaccinations"].duplicated(keep=False)
        daily_vaccinations_duplicate_rows = daily_vaccinations_duplicates[
            daily_vaccinations_duplicates
        ].index.values

        return df.style.apply(
            lambda x: [
                "background: yellow"
                if x.name in daily_vaccinations_duplicate_rows
                else ""
            ],
            axis=1,
            subset="daily_vaccinations",
        )

    def set_predicted_or_factual(self, df, location_info, location, perc: int) -> None:
        def closest(lst, K):
            return lst[min(range(len(lst)), key=lambda i: abs(lst[i] - K))]

        df = df.astype({"percent_fully_vaccinated": int})

        percent_fully_vaccinated_list = df["percent_fully_vaccinated"].tolist()

        perc_approx = closest(percent_fully_vaccinated_list, perc)

        if (
            perc - perc_approx > np.diff(percent_fully_vaccinated_list).max()
            or perc_approx - perc < -np.diff(percent_fully_vaccinated_list).max()
        ):
            perc_approx = perc

        try:
            location_info[location][f"date_vacc_{perc}_perc"] = df.loc[
                df["percent_fully_vaccinated"] == perc_approx, "date"
            ][-1]
            location_info[location][f"daily_vaccinations_cumsum_{perc}_perc"] = df.loc[
                df["percent_fully_vaccinated"] == perc_approx,
                "daily_vaccinations_cumsum",
            ][-1]
        except IndexError:
            location_info[location][f"date_vacc_{perc}_perc"] = location_info[location][
                f"goal_date_{perc}_perc"
            ]
            location_info[location][
                f"daily_vaccinations_cumsum_{perc}_perc"
            ] = location_info[location][f"goal_vaccinations_{perc}_perc"]

    @st.cache(show_spinner=False, allow_output_mutation=True)
    def make_plot(self, df_, location_info_, location):
        df = copy.deepcopy(df_)
        location_info = copy.deepcopy(location_info_)

        self.set_predicted_or_factual(df, location_info, location, 10)
        self.set_predicted_or_factual(df, location_info, location, 30)
        self.set_predicted_or_factual(df, location_info, location, 50)
        self.set_predicted_or_factual(df, location_info, location, 70)
        self.set_predicted_or_factual(df, location_info, location, 80)
        self.set_predicted_or_factual(df, location_info, location, 100)

        fig = go.Figure()

        fig.add_trace(
            go.Line(
                fill="tozeroy",
                hoverinfo="y",
                hovertemplate="%{y:,}",
                marker_color="rgba(115, 65, 225, 0.5)",
                marker_line_width=0,
                name="total",
                x=df["date"],
                y=df["daily_vaccinations_cumsum"],
                yaxis="y1",
            )
        )

        fig.add_trace(
            go.Line(
                hoverinfo="skip",
                hovertemplate=None,
                line=dict(color="#ffffff", width=6),
                marker_color="#ffffff",
                marker_line_width=0,
                name="daily",
                x=df["date"],
                y=df["daily_vaccinations"],
                yaxis="y2",
            )
        )
        fig.add_trace(
            go.Line(
                hoverinfo="y",
                hovertemplate="%{y:,}",
                line=dict(color="Red", width=2),
                marker_color="Red",
                marker_line_width=0,
                name="daily",
                x=df["date"],
                y=df["daily_vaccinations"],
                yaxis="y2",
            )
        )

        fig.update_layout(
            xaxis=dict(
                showgrid=False,
                title="<b>Days</b>",
            ),
            yaxis=dict(
                anchor="x",
                rangemode="tozero",
                showgrid=False,
                title="<b>Total vaccinations,</b> shots (cumulative daily sum)",
            ),
            yaxis2=dict(
                anchor="x",
                overlaying="y1",
                rangemode="nonnegative",
                scaleanchor="y1",
                scaleratio=50,
                showgrid=False,
                side="right",
                title="<b>Daily vaccinations,</b> shots",
            ),
        )

        _daily_vaccinations = location_info[location]["daily_vaccinations"]
        _date = location_info[location]["date"]
        _population = int(location_info[location]["population"])
        _vacc_start_date = location_info[location]["vacc_start_date"]
        _vaccinated_people = int(location_info[location]["vaccinated_people"])

        fig.add_annotation(
            align="left",
            bgcolor="rgba(245, 247, 243, 0.85)",
            showarrow=False,
            text=f"<b>Population:</b> {humanize.intword(_population)} ({year})<br><b>Vaccination started:</b> {_vacc_start_date:%B %d, %Y}<br><b>{_date:%B %d, %Y}:</b> {_vaccinated_people:,} ({int(_vaccinated_people*100/int(_population))}%) people<br>have received at least 2 doses of vaccine,<br>{int(_daily_vaccinations):,} shots were administered",
            x=0.05,
            xref="paper",
            y=0.95,
            yref="paper",
        )

        # ---------------------------------------------------------------------------------------------------------------------------

        def add_goal_marker(
            perc: int, xanchor: str, xshift: int, text: str = None
        ) -> None:
            xanchor = xanchor
            xshift = xshift
            if int(location_info[location][f"days_to_goal_{perc}_perc"]) <= 0:
                text = f"<b>{perc}%</b>"
                font_color = "Green"
                color = "Green"
                xanchor = "right"
                xshift = 5
            else:
                text = text or f"<b>{perc}%</b>"
                font_color = "Orange"
                color = "Orange"
            fig.add_annotation(
                align="center",
                bgcolor="rgba(255,255,255,0.85)",
                font_color=font_color,
                showarrow=False,
                text=text,
                x=location_info[location][f"date_vacc_{perc}_perc"],
                xanchor=xanchor,
                xshift=xshift,
                y=location_info[location][f"daily_vaccinations_cumsum_{perc}_perc"],
                yanchor="bottom",
            )
            fig.add_shape(
                line=dict(color="White", width=2),
                type="line",
                x0=location_info[location][f"date_vacc_{perc}_perc"],
                x1=location_info[location][f"date_vacc_{perc}_perc"],
                y0=0,
                y1=location_info[location][f"daily_vaccinations_cumsum_{perc}_perc"],
            )
            fig.add_shape(
                line=dict(color=color, width=2, dash="dot"),
                type="line",
                x0=location_info[location][f"date_vacc_{perc}_perc"],
                x1=location_info[location][f"date_vacc_{perc}_perc"],
                y0=0,
                y1=location_info[location][f"daily_vaccinations_cumsum_{perc}_perc"],
            )

        def get_annotation_for_goal_marker(perc: int) -> str:
            return "<b>-- {_perc}% --</b><br>in <b>{_days_to_goal} day(s)</b><br><b>({_goal_date:%B %Y})</b><br>~ {_goal_vaccinations} doses".format(
                _perc=perc,
                _days_to_goal=int(location_info[location][f"days_to_goal_{perc}_perc"]),
                _goal_date=location_info[location][f"goal_date_{perc}_perc"],
                _goal_vaccinations=humanize.intword(
                    int(location_info[location][f"goal_vaccinations_{perc}_perc"])
                ),
            )

        add_goal_marker(10, "right", 5)
        add_goal_marker(30, "right", 5)
        add_goal_marker(50, "right", 5)
        add_goal_marker(70, "right", 5)  # , get_annotation_for_goal_marker(70)
        add_goal_marker(80, "left", -5, get_annotation_for_goal_marker(80))
        add_goal_marker(100, "left", -5, get_annotation_for_goal_marker(100))

        fig.update_layout(
            {"plot_bgcolor": "#f5f7f3", "paper_bgcolor": "#f5f7f3"},
            hoverlabel=dict(font_color="white"),
            hovermode="x",
            showlegend=False,
            title_text=f"{location}",
            width=800,
        )

        return fig

    def process_location_vaccination_data(self, location):
        df = self.vaccination_data[self.vaccination_data["location"] == location]

        df["daily_vaccinations_cumsum"] = df["daily_vaccinations"].transform("cumsum")

        df.set_index("location", inplace=True, drop=True)

        location_info_series = df.iloc[-1]

        if location in problematic_locations:
            st.error(
                f"Sorry, cannot make a chart for {location}. Here is the raw data instead."
            )
            st.dataframe(df)
            st.stop()

        location_info = {location_info_series.name: dict(location_info_series)}

        location_info[location].update(**self.get_location_population(location, year))

        location_info[location]["vacc_start_date"] = df.iloc[0, 0]

        location_info[location]["vaccinated_people"] = (
            location_info[location]["daily_vaccinations_cumsum"] / 2
        )

        df["percent_fully_vaccinated"] = df["daily_vaccinations_cumsum"].apply(
            lambda x: (x * 100) / (location_info[location]["population"] * 2)
        )

        def set_predict_goal_date(perc: int) -> None:
            location_info[location][f"goal_vaccinations_{perc}_perc"] = (
                location_info[location]["population"] * perc / 100
            ) * 2
            location_info[location][f"days_to_goal_{perc}_perc"] = int(
                (
                    (location_info[location]["population"] * perc / 100)
                    - location_info[location]["vaccinated_people"]
                )
                / location_info[location]["daily_vaccinations"]
                * 2
            )
            location_info[location][f"goal_date_{perc}_perc"] = location_info[location][
                "date"
            ] + timedelta(days=location_info[location][f"days_to_goal_{perc}_perc"])

        set_predict_goal_date(10)
        set_predict_goal_date(100)
        set_predict_goal_date(80)
        set_predict_goal_date(70)
        set_predict_goal_date(50)
        set_predict_goal_date(30)

        df["percent_fully_vaccinated"].fillna(0, inplace=True)

        return df, location_info

    # @st.cache(show_spinner=False)
    def get_location_population(self, location: str, year: str) -> dict:
        location_population = {}

        try:
            if location in list(location_codes.keys()):
                code = location_codes[location]
            else:
                code_obj = pycountry.countries.search_fuzzy(location)[0]
                code = int(code_obj.numeric)
            location_population["population"] = (
                self.population_data.loc[
                    self.population_data[self.population_data.columns[4]] == code, year
                ].values[0]
                * 1000
            )
        except Exception:
            st.error(
                f"Something went wrong when looking up data for {location}. Please select a different location. I am working on a solution."
            )
            st.stop()

        return location_population
