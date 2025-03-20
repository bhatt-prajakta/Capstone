import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# Sample data generation (replace with your actual data)
np.random.seed(42)
n_loans = 1000

data = {
    "loan_id": range(n_loans),
    "amount": np.random.normal(50000, 20000, n_loans),
    "interest_rate": np.random.normal(0.05, 0.02, n_loans),
    "term_months": np.random.choice([36, 60, 84], n_loans),
    "credit_score": np.random.normal(700, 50, n_loans),
    "debt_to_income": np.random.normal(0.3, 0.1, n_loans),
    "default_risk": np.random.normal(0.1, 0.05, n_loans),
    "loan_status": np.random.choice(
        ["Current", "Late", "Default", "Paid"], n_loans, p=[0.8, 0.1, 0.05, 0.05]
    ),
}

df = pd.DataFrame(data)

# Initialize the Dash app
app = dash.Dash(__name__)

# Define the layout
app.layout = html.Div(
    [
        html.H1("Loan Analytics Dashboard"),
        # Filters
        html.Div(
            [
                html.Label("Filter by Loan Status:"),
                dcc.Dropdown(
                    id="status-filter",
                    options=[
                        {"label": status, "value": status}
                        for status in df["loan_status"].unique()
                    ],
                    value=list(df["loan_status"].unique()),
                    multi=True,
                ),
            ],
            style={"width": "30%", "margin": "20px"},
        ),
        # First row of visualizations
        html.Div(
            [
                # Loan Amount Distribution
                html.Div(
                    [dcc.Graph(id="amount-dist")],
                    style={"width": "48%", "display": "inline-block"},
                ),
                # Risk vs Credit Score Scatter
                html.Div(
                    [dcc.Graph(id="risk-score-scatter")],
                    style={"width": "48%", "display": "inline-block"},
                ),
            ]
        ),
        # Second row of visualizations
        html.Div(
            [
                # Loan Status Breakdown
                html.Div(
                    [dcc.Graph(id="status-pie")],
                    style={"width": "48%", "display": "inline-block"},
                ),
                # Interest Rate vs Debt-to-Income
                html.Div(
                    [dcc.Graph(id="rate-dti-scatter")],
                    style={"width": "48%", "display": "inline-block"},
                ),
            ]
        ),
    ]
)


# Callbacks for interactive filtering
@app.callback(
    [
        Output("amount-dist", "figure"),
        Output("risk-score-scatter", "figure"),
        Output("status-pie", "figure"),
        Output("rate-dti-scatter", "figure"),
    ],
    [Input("status-filter", "value")],
)
def update_graphs(selected_statuses):
    # Filter data based on selection
    filtered_df = df[df["loan_status"].isin(selected_statuses)]

    # Amount Distribution
    amount_dist = px.histogram(
        filtered_df,
        x="amount",
        title="Loan Amount Distribution",
        labels={"amount": "Loan Amount ($)", "count": "Number of Loans"},
        color="loan_status",
    )

    # Risk vs Credit Score
    risk_score = px.scatter(
        filtered_df,
        x="credit_score",
        y="default_risk",
        title="Default Risk vs Credit Score",
        labels={"credit_score": "Credit Score", "default_risk": "Default Risk"},
        color="loan_status",
        size="amount",
    )

    # Loan Status Breakdown
    status_pie = px.pie(
        filtered_df, names="loan_status", title="Loan Status Distribution"
    )

    # Interest Rate vs DTI
    rate_dti = px.scatter(
        filtered_df,
        x="debt_to_income",
        y="interest_rate",
        title="Interest Rate vs Debt-to-Income Ratio",
        labels={
            "debt_to_income": "Debt-to-Income Ratio",
            "interest_rate": "Interest Rate",
        },
        color="loan_status",
        trendline="ols",
    )

    return amount_dist, risk_score, status_pie, rate_dti


if __name__ == "__main__":
    app.run_server(debug=True)
