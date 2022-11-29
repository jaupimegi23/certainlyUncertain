import altair as alt


def get_chart(data):
    hover = alt.selection_single(
        fields=["date"],
        nearest=True,
        on="mouseover",
        empty="none",
    )

    line = (
        alt.Chart(data, height=500)
        .mark_line()
        .encode(
            x=alt.X("date", title="Date"),
            y=alt.Y("weather", title="Weather variable")
        )
    )

    band1 = alt.Chart(data).mark_area(
        opacity=0.5
        ).encode(
            x='date',
            y='lower',
            y2='upper'
        )

    band2 = alt.Chart(data).mark_area(
    opacity=0.5
    ).encode(
        x='date',
        y='lower_pred',
        y2='upper_pred'
    )

    # Draw points on the line, and highlight based on selection
    points = line.transform_filter(hover).mark_circle(size=65)

    # Draw a rule at the location of the selection
    tooltips = (
        alt.Chart(data)
        .mark_rule()
        .encode(
            x="date",
            y="weather",
            opacity=alt.condition(hover, alt.value(0.3), alt.value(0)),
            tooltip=[
                alt.Tooltip("date", title="Date"),
                alt.Tooltip("weather", title="Weather variable value"),
            ],
        )
        .add_selection(hover)
    )

    return (line + band1 + band2 + points + tooltips).interactive()