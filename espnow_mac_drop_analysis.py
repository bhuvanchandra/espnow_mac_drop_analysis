import dash
from dash import dcc, html, Output, Input
import numpy as np
from scipy.optimize import fsolve
import plotly.graph_objects as go

# -------------------------------------------------
# Fixed MAC parameters for the Bianchi model
# -------------------------------------------------
# 1 Mbps (classical Bianchi model saturates at a fixed rate)
bitrate_default = 1e6
ACK = 0                 # No ACK (ESP-NOW)
SIFS = 28e-6            # seconds
slot = 50e-6            # seconds
DIFS = 128e-6           # seconds
W = 128                 # Minimum contention window
m = 3                   # Maximum backoff stage
H = 400                 # PHY header bits (not used for collisions)
prop_delay = 0          # negligible
overhead = 400          # overhead bits for unsaturated model

# -------------------------------------------------
# Node range: 2 to 200
# -------------------------------------------------
n_array = np.arange(2, 201)

# Default user-selected values
default_bitrate = 54    # Mbps
default_payload = 250   # bytes
default_pubfreq = 10    # Hz
default_rfber = 1e-6    #
activity_factor = 0.5

# -------------------------------------------------
# BianchiRF: Classical Saturated Model
# -------------------------------------------------


class BianchiRF:
    """
    Classical saturated Bianchi model for 802.11.

    Collision probability equations:
      τ = 2(1-2p) / [(W+1)(1-2p) + pW(1-(2p)^m)]
      p = 1 - (1-τ)^(n-1)

    MAC success:
      P_success_MAC = [n × τ × (1-τ)^(n-1)] / [1 - (1-τ)^n]

    MAC drop (%) = 100 × [1 - P_success_MAC]
    """

    def __init__(self, bitrate, n, ACK, SIFS, slot, DIFS, E_P, W, m, H, prop_delay, RF_BER=0):
        self.bitrate = bitrate
        self.n = n
        self.ACK = ACK
        self.SIFS = SIFS
        self.slot = slot
        self.DIFS = DIFS
        self.E_P = E_P
        self.W = W
        self.m = m
        self.H = H
        self.prop_delay = prop_delay
        self.RF_BER = RF_BER  # not used in the MAC collision calculation
        self.p = 0
        self.t = 0
        self.calculate_p_t()
        self.calculate_MAC_success()
        self.Overall_P_success = self.P_success_MAC
        self.Overall_P_drop = 1 - self.Overall_P_success

    def calculate_p_t(self):
        def equations(x):
            p, tau = x
            backoff_sum = sum((2.0 * p) ** i for i in range(self.m))
            eq1 = p - 1.0 + (1.0 - tau) ** (self.n - 1.0)
            eq2 = 2.0 / ((W + 1) + (W * p * (1 - (2.0 * p) ** m)
                                    ) / (1 - 2.0 * p)) - tau
            return (eq1, eq2)
        self.p, self.t = fsolve(equations, (0.1, 0.1))

    def calculate_MAC_success(self):
        # Probability at least one node transmits.
        self.P_tr = 1 - (1 - self.t) ** self.n
        if self.P_tr > 0:
            self.P_success_MAC = (
                self.n * self.t * (1 - self.t) ** (self.n - 1)) / self.P_tr
        else:
            self.P_success_MAC = 0

# -------------------------------------------------
# Compute overall drop for one of three models.
# -------------------------------------------------


def compute_overall_drop(n_array, payload, pub_freq, selected_bitrate, selected_RF_BER, traffic_model):
    # Convert payload to bits for the RF success exponent
    E_P_val = payload * 8
    if traffic_model == "Saturated (Bianchi)":
        mac_success = []
        for n in n_array:
            model = BianchiRF(selected_bitrate, n, ACK, SIFS,
                              slot, DIFS, E_P_val, W, m, H, prop_delay)
            mac_success.append(model.P_success_MAC)
        mac_success = np.array(mac_success)
    elif traffic_model == "Unsaturated (Bianchi)":
        L_total = payload * 8 + overhead
        T_packet = L_total / selected_bitrate
        q = min(1, pub_freq * T_packet)
        mac_success = []
        for n in n_array:
            n_eff = max(q * n, 1)
            model = BianchiRF(selected_bitrate, n_eff, ACK,
                              SIFS, slot, DIFS, E_P_val, W, m, H, prop_delay)
            mac_success.append(model.P_success_MAC)
        mac_success = np.array(mac_success)
    else:  # "Unsaturated (Aloha)"
        L_total = payload * 8 + overhead
        T_packet = L_total / selected_bitrate
        G = n_array * pub_freq * T_packet
        mac_success = np.exp(-2 * G)

    # RF success portion
    RF_success = (1 - selected_RF_BER) ** (payload * 8)
    # Overall success = MAC_success * RF_success
    Overall_success = mac_success * RF_success
    # Overall drop
    return (1 - Overall_success) * 100


# -------------------------------------------------
# Dash App Layout
# -------------------------------------------------
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H2("ESP-NOW MAC Drop Analysis",
            style={'textAlign': 'center', 'marginTop': '10px'}),

    html.Div([
        html.Label("Traffic Model:"),
        dcc.Dropdown(
            id="traffic-model-dropdown",
            options=[
                {"label": "Saturated (Bianchi)",
                 "value": "Saturated (Bianchi)"},
                {"label": "Unsaturated (Bianchi)",
                 "value": "Unsaturated (Bianchi)"},
                {"label": "Unsaturated (Aloha)",
                 "value": "Unsaturated (Aloha)"}
            ],
            value="Saturated (Bianchi)",
            clearable=False
        )
    ], style={'width': '18%', 'display': 'inline-block', 'marginRight': '2%'}),

    html.Div([
        html.Label("Bitrate (Mbps):"),
        dcc.Dropdown(
            id="bitrate-dropdown",
            options=[{"label": f"{br} Mbps", "value": br}
                     for br in [1, 2, 5.5, 6, 9, 11, 12, 18, 24, 36, 48, 54]],
            value=default_bitrate,  # default
            clearable=False
        )
    ], style={'width': '18%', 'display': 'inline-block', 'marginRight': '2%'}),

    html.Div([
        html.Label("Payload (bytes):"),
        dcc.Dropdown(
            id="payload-dropdown",
            options=[{"label": f"{pl} bytes", "value": pl}
                     for pl in [1, 5, 10, 50, 125, 225, 250]],
            value=default_payload,
            clearable=False
        )
    ], style={'width': '18%', 'display': 'inline-block', 'marginRight': '2%'}),

    html.Div([
        html.Label("Publish Frequency (Hz):"),
        dcc.Dropdown(
            id="pubfreq-dropdown",
            options=[{"label": f"{pf} Hz", "value": pf}
                     for pf in [1, 10, 5, 50, 100, 200]],
            value=default_pubfreq,
            clearable=False
        )
    ], style={'width': '18%', 'display': 'inline-block', 'marginRight': '2%'}),

    html.Div([
        html.Label("RF BER:"),
        dcc.Dropdown(
            id="rfber-dropdown",
            options=[{"label": f"{rf:.0e}", "value": rf}
                     for rf in [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]],
            value=default_rfber,
            clearable=False
        )
    ], style={'width': '18%', 'display': 'inline-block'}),

    dcc.Graph(id="drop-graph", style={'marginTop': '20px'}),

    dcc.Markdown(id="formula-summary",
                 style={"padding": "20px", "border": "1px solid #ddd", "marginTop": "20px"})
])

# -------------------------------------------------
# Disable the Publish Frequency dropdown for Saturated model
# -------------------------------------------------


@app.callback(
    Output("pubfreq-dropdown", "disabled"),
    Input("traffic-model-dropdown", "value")
)
def disable_pubfreq(traffic_model):
    return traffic_model == "Saturated (Bianchi)"

# -------------------------------------------------
# Update Graph and Formula Summary
# -------------------------------------------------


@app.callback(
    [Output("drop-graph", "figure"),
     Output("formula-summary", "children")],
    [Input("traffic-model-dropdown", "value"),
     Input("bitrate-dropdown", "value"),
     Input("payload-dropdown", "value"),
     Input("pubfreq-dropdown", "value"),
     Input("rfber-dropdown", "value")]
)
def update_graph(traffic_model, selected_bitrate, selected_payload, selected_pubfreq, selected_rfber):
    if selected_bitrate is None:
        selected_bitrate = default_bitrate
    if selected_payload is None:
        selected_payload = default_payload
    if selected_pubfreq is None:
        selected_pubfreq = default_pubfreq
    if selected_rfber is None:
        selected_rfber = default_rfber

    selected_bitrate_bps = selected_bitrate * 1e6

    # Compute drop
    drop_vals = compute_overall_drop(
        n_array, selected_payload, selected_pubfreq,
        selected_bitrate_bps, selected_rfber, traffic_model
    )

    # Build figure
    fig = go.Figure(
        data=go.Scatter(
            x=n_array.tolist(),
            y=drop_vals.tolist(),
            mode="lines+markers",
            hovertemplate="Nodes: %{x}<br>Overall Drop: %{y:.2f}%",
            line=dict(color="blue")
        )
    )
    fig.update_layout(
        title=f"Overall Packet Drop vs. Number of Devices<br>"
              f"Model: {traffic_model} | Bitrate: {selected_bitrate} Mbps | Payload: {selected_payload} bytes | "
              f"Publish Freq: {selected_pubfreq} Hz | RF BER: {selected_rfber:.0e}",
        xaxis_title="Number of Devices",
        yaxis_title="Overall Drop (%)"
    )

    # Summaries of the formulas in a friendlier style
    formula_text = f"""
### Key Formulas

**Overall Packet Drop (%)** = 100 × \\(1 - [P\\_success,MAC × P\\_success,RF]\\)

1. **MAC Success (Saturated Bianchi)**
   - Solve for \\(p\\) and \\(\\tau\\) with:
     \\(\\tau = \\frac{{2(1-2p)}}{{(W+1)(1-2p) + pW[1-(2p)^m]}}\\)
     \\(p = 1 - (1-\\tau)^{{(n-1)}}\\)
   - Then:
     \\(P\\_{{success,MAC}} = \\frac{{n\\tau (1-\\tau)^{{n-1}}}}{{1 - (1-\\tau)^n}}\\)

2. **MAC Success (Unsaturated Bianchi)**
   - Define an activity factor \\(q = \\min[1, (\\text{{Publish Freq}}) × T\\_packet]\\).
   - Effective nodes: \\(n\\_eff = \\max(q × n, 1)\\).
   - Solve Bianchi equations with \\(n\\_eff\\) in place of \\(n\\).

3. **MAC Success (Unsaturated Aloha)**
   - \\(P\\_{{success,MAC}} = \\exp(-2G)\\),
     where \\(G = n × (\\text{{Publish Freq}}) × T\\_{{packet}}\\).

4. **RF Success**
   \\(P\\_{{success,RF}} = (1 - \\text{{RF BER}})^{{(\\text{{Payload}} × 8)}}\\)

5. **Overall**
   \\(P\\_{{success,total}} = P\\_{{success,MAC}} × P\\_{{success,RF}}\\).
   \\(\\text{{Overall Drop (%)}} = 100 × [1 - P\\_{{success,total}}]\\).

> **Note:** When “Saturated (Bianchi)” is selected, Publish Frequency is irrelevant, so that dropdown is disabled.
"""

    return fig, formula_text


if __name__ == "__main__":
    app.run(debug=False)
