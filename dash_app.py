"""
DAiSEE Feature Visualization Dashboard — Plotly Dash
Reads features.pkl with keys: X (N,150,78), y, feature_names,
video_paths, failed_videos, sequence_length, num_features

Usage:
  DAISEE_PKL=/path/to/features.pkl python app.py
  python app.py          # looks for features.pkl next to this file
  python app.py --debug  # hot-reload
"""

import pickle, sys, os
import numpy as np
from pathlib import Path
from scipy.stats import skew, kurtosis

import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Palette ────────────────────────────────────────────────────────────────────
BG        = "#F5F0E8"
CARD      = "#FDFAF5"
BORDER    = "#E8DFC8"
BORDER_MD = "#D4C9A8"
TEXT      = "#1A1510"
TEXT_MID  = "#5C4F3A"
TEXT_MUTE = "#9C8E78"
CORAL     = "#D4622A"
CORAL_S   = "#F2886A"
CORAL_P   = "#FAE5DC"
SAND      = "#C4A882"
OLIVE     = "#7A8B5A"
TEAL      = "#3A7A74"
GOLD      = "#B8860B"
MAUVE     = "#8B5E7A"
NAVY      = "#2C3E6B"
WHITE     = "#FFFEF9"
DANGER    = "#C0392B"
SUCCESS   = "#27AE60"

def rgba(hex6, alpha):
    """Convert '#RRGGBB' + alpha float → 'rgba(r,g,b,a)'."""
    r = int(hex6[1:3], 16)
    g = int(hex6[3:5], 16)
    b = int(hex6[5:7], 16)
    return f"rgba({r},{g},{b},{alpha})"

GROUP_META = [
    dict(id="au",       label="Action Units",    short="AU", color=CORAL,  count=17,
         desc="Facial muscle activations via FACS coding"),
    dict(id="head",     label="Head Pose",        short="HP", color=TEAL,   count=6,
         desc="3D head orientation + translation"),
    dict(id="gaze",     label="Gaze Direction",   short="GZ", color=GOLD,   count=4,
         desc="Eye gaze vectors for both eyes"),
    dict(id="eye",      label="Eye Landmarks",    short="EL", color=MAUVE,  count=10,
         desc="Eye openness, blink & pupil positions"),
    dict(id="landmark", label="Facial Landmarks", short="FL", color=OLIVE,  count=28,
         desc="Normalised facial geometry coordinates"),
    dict(id="emotion",  label="Emotion Probs",    short="EP", color="#C05A3A", count=7,
         desc="Softmax scores per basic emotion"),
    dict(id="hog",      label="HOG / LBP",        short="TX", color=NAVY,   count=6,
         desc="Histogram of Oriented Gradients + LBP"),
]

STAT_COLORS = dict(
    mean=CORAL, std=TEAL,  min=NAVY,
    max=OLIVE,  range=GOLD, skew=MAUVE, kurt="#C05A3A",
)
STATS = ["mean", "std", "min", "max", "range", "skew", "kurt"]

# ── Layout helper ──────────────────────────────────────────────────────────────
_AX = dict(gridcolor=BORDER, linecolor=BORDER_MD, zeroline=False)

def L(**kw):
    """Build a complete Plotly layout dict, safely merging xaxis/yaxis overrides."""
    base = dict(
        paper_bgcolor=CARD, plot_bgcolor=BG,
        font=dict(family="DM Sans, sans-serif", color=TEXT, size=11),
        margin=dict(l=52, r=18, t=36, b=42),
        xaxis=dict(**_AX),
        yaxis=dict(**_AX),
        hoverlabel=dict(bgcolor=WHITE, bordercolor=BORDER,
                        font=dict(family="DM Sans", color=TEXT, size=11)),
    )
    for k, v in kw.items():
        if k in ("xaxis", "yaxis") and isinstance(v, dict):
            base[k].update(v)
        else:
            base[k] = v
    return base

def radar_layout(title_text):
    return dict(
        paper_bgcolor=CARD, plot_bgcolor=BG,
        font=dict(family="DM Sans,sans-serif", color=TEXT, size=11),
        margin=dict(l=30, r=30, t=40, b=30), height=220,
        polar=dict(
            bgcolor=BG,
            radialaxis=dict(visible=True, gridcolor=BORDER, tickfont=dict(size=8)),
            angularaxis=dict(gridcolor=BORDER, tickfont=dict(size=10)),
        ),
        title=dict(text=title_text, font=dict(size=12, family="Fraunces,serif"), x=0),
        hoverlabel=dict(bgcolor=WHITE, bordercolor=BORDER,
                        font=dict(family="DM Sans", color=TEXT, size=11)),
    )

def spark_layout(nrows):
    return dict(
        paper_bgcolor=CARD, plot_bgcolor=BG,
        font=dict(family="DM Sans,sans-serif", color=TEXT, size=9),
        margin=dict(l=8, r=8, t=40, b=8),
        height=max(200, nrows * 70), showlegend=False,
        title=dict(text="Temporal Sparklines (mean signal per feature)",
                   font=dict(size=12, family="Fraunces,serif"), x=0),
        hoverlabel=dict(bgcolor=WHITE, bordercolor=BORDER,
                        font=dict(family="DM Sans", color=TEXT, size=11)),
    )

# ── Data ───────────────────────────────────────────────────────────────────────
PKL_PATH = os.environ.get("DAISEE_PKL", "daisee_features/complete_dataset.pkl")

def load_data(path):
    p = Path(path)
    if not p.exists():
        return None, f"File not found: {path}"
    try:
        with open(p, "rb") as f:
            raw = pickle.load(f)
        X             = np.array(raw["X"], dtype=np.float32)
        return dict(
            X             = X,
            y             = raw.get("y", {}),
            feature_names = list(raw["feature_names"]),
            video_paths   = list(raw.get("video_paths", [])),
            failed_videos = list(raw.get("failed_videos", [])),
            seq_len       = int(raw.get("sequence_length", X.shape[1])),
            num_features  = int(raw.get("num_features", X.shape[2])),
        ), None
    except Exception as e:
        return None, str(e)

def build_feature_index(names):
    index, cursor = [], 0
    for g in GROUP_META:
        for _ in range(g["count"]):
            name = names[cursor] if cursor < len(names) else f"feat_{cursor}"
            index.append(dict(name=name, group=g["id"], group_label=g["label"],
                              color=g["color"], short=g["short"], idx=cursor))
            cursor += 1
    while cursor < len(names):
        index.append(dict(name=names[cursor], group="other", group_label="Other",
                          color=SAND, short="OT", idx=cursor))
        cursor += 1
    return index

def compute_statistics(X):
    N, T, D = X.shape
    mc, sc = X.mean(axis=1), X.std(axis=1)
    mn, mx = X.min(axis=1),  X.max(axis=1)
    sk = np.zeros((N, D), dtype=np.float32)
    kt = np.zeros((N, D), dtype=np.float32)
    for i in range(N):
        sk[i] = skew(X[i], axis=0)
        kt[i] = kurtosis(X[i], axis=0)
    return dict(
        mean  = mc.mean(0), std   = sc.mean(0),
        min   = mn.mean(0), max   = mx.mean(0),
        range = (mx - mn).mean(0),
        skew  = sk.mean(0), kurt  = kt.mean(0),
    )

DATA, LOAD_ERROR = load_data(PKL_PATH)
FEAT_INDEX   = build_feature_index(DATA["feature_names"]) if DATA else []
GLOBAL_STATS = compute_statistics(DATA["X"])               if DATA else {}

# ── UI helpers ─────────────────────────────────────────────────────────────────
CARD_STYLE = dict(background=CARD, border=f"1px solid {BORDER}",
                  borderRadius="14px", padding="18px", marginBottom="12px")

def heading(text, size=15, italic=None):
    parts = ([html.Span(text+" "), html.Span(italic, style={"color": CORAL, "fontStyle": "italic"})]
             if italic else [html.Span(text)])
    return html.Div(parts, style={"fontFamily": "'Fraunces',serif", "fontSize": size,
                                  "fontWeight": 700, "color": TEXT, "marginBottom": 4})

def sub(text):
    return html.Div(text, style={"fontSize": 10, "color": TEXT_MUTE,
                                 "marginBottom": 12, "fontFamily": "'DM Sans',sans-serif"})

def kpi_card(title, value, note, color):
    return html.Div([
        html.Div(title, style={"fontSize": 9, "color": color, "fontWeight": 700,
                               "letterSpacing": "0.1em", "marginBottom": 6}),
        html.Div(value, style={"fontFamily": "'Fraunces',serif", "fontSize": 24,
                               "fontWeight": 700, "color": TEXT, "lineHeight": 1}),
        html.Div(note,  style={"fontSize": 9, "color": TEXT_MUTE, "marginTop": 4}),
    ], style={"flex": "1 1 120px", "background": CARD,
              "border": f"1px solid {rgba(color, 0.25)}", "borderRadius": "12px",
              "padding": "14px 16px"})

def make_group_opts():
    return [{"label": "All Features (78)", "value": "all"}] + [
        {"label": f"{g['short']} — {g['label']} ({g['count']})", "value": g["id"]}
        for g in GROUP_META
    ]

def make_feat_opts(group="all"):
    feats = FEAT_INDEX if group == "all" else [f for f in FEAT_INDEX if f["group"] == group]
    return [{"label": f["name"], "value": f["idx"]} for f in feats]

# ── App ────────────────────────────────────────────────────────────────────────
app = dash.Dash(__name__, title="DAiSEE Feature Dashboard",
                suppress_callback_exceptions=True)

app.layout = html.Div([
    html.Link(rel="stylesheet",
              href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600"
                   "&family=Fraunces:ital,wght@0,600;0,700;1,500&display=swap"),

    # HEADER
    html.Div([
        html.Div([
            html.Div("◈", style={
                "width": 40, "height": 40, "borderRadius": 10, "flexShrink": 0,
                "background": f"linear-gradient(135deg,{CORAL},{CORAL_S})",
                "display": "flex", "alignItems": "center", "justifyContent": "center",
                "fontSize": 20, "color": WHITE,
                "boxShadow": f"0 2px 10px {rgba(CORAL, 0.25)}",
            }),
            html.Div([
                html.Div("DAiSEE · Feature Analysis Dashboard",
                         style={"fontSize": 10, "color": TEXT_MUTE,
                                "letterSpacing": "0.12em", "textTransform": "uppercase",
                                "marginBottom": 3, "fontFamily": "'DM Sans',sans-serif"}),
                html.Div([
                    html.Span("78-Dimensional Feature ", style={
                        "fontFamily": "'Fraunces',serif", "fontSize": 22,
                        "fontWeight": 700, "color": TEXT}),
                    html.Span("Statistical Dashboard", style={
                        "fontFamily": "'Fraunces',serif", "fontSize": 22,
                        "fontWeight": 700, "color": CORAL, "fontStyle": "italic"}),
                ]),
            ]),
        ], style={"display": "flex", "alignItems": "center", "gap": 14}),

        html.Div([
            dcc.Input(id="pkl-path", type="text", value=PKL_PATH,
                      placeholder="Path to features.pkl",
                      style={"padding": "7px 12px", "borderRadius": 8, "width": 280,
                             "border": f"1px solid {BORDER_MD}",
                             "background": WHITE, "color": TEXT,
                             "fontSize": 12, "fontFamily": "'DM Sans',sans-serif"}),
            html.Button("Load File", id="load-btn", n_clicks=0, style={
                "padding": "7px 18px", "borderRadius": 8, "border": "none",
                "background": CORAL, "color": WHITE, "fontSize": 12,
                "fontWeight": 600, "cursor": "pointer",
                "fontFamily": "'DM Sans',sans-serif",
                "boxShadow": f"0 2px 8px {rgba(CORAL, 0.25)}"}),
            html.Div(id="load-status"),
        ], style={"display": "flex", "gap": 8, "alignItems": "center", "flexWrap": "wrap"}),
    ], style={"background": WHITE, "borderBottom": f"1px solid {BORDER}",
              "padding": "20px 32px", "display": "flex",
              "justifyContent": "space-between", "alignItems": "center",
              "flexWrap": "wrap", "gap": 12}),

    # CONTROLS
    html.Div([
        html.Div([
            html.Span("STATISTIC", style={"fontSize": 10, "color": TEXT_MUTE,
                                          "fontWeight": 600, "marginRight": 8}),
            dcc.Dropdown(id="stat-dd", clearable=False, value="mean",
                         options=[{"label": s.upper(), "value": s} for s in STATS],
                         style={"width": 130, "fontSize": 12}),
        ], style={"display": "flex", "alignItems": "center", "gap": 6}),

        html.Div([
            html.Span("GROUP", style={"fontSize": 10, "color": TEXT_MUTE,
                                      "fontWeight": 600, "marginRight": 8}),
            dcc.Dropdown(id="group-dd", clearable=False, value="all",
                         options=make_group_opts(),
                         style={"width": 240, "fontSize": 12}),
        ], style={"display": "flex", "alignItems": "center", "gap": 6}),

        html.Div([
            html.Span("FEATURE", style={"fontSize": 10, "color": TEXT_MUTE,
                                        "fontWeight": 600, "marginRight": 8}),
            dcc.Dropdown(id="feat-dd", clearable=False, value=0,
                         options=make_feat_opts(),
                         style={"width": 220, "fontSize": 12}),
        ], style={"display": "flex", "alignItems": "center", "gap": 6}),

        dcc.RadioItems(id="tab-radio", value="overview", inline=True,
            options=[
                {"label": "Overview",     "value": "overview"},
                {"label": "Temporal",     "value": "temporal"},
                {"label": "Distribution", "value": "distribution"},
                {"label": "Heatmap",      "value": "heatmap"},
                {"label": "Label Stats",  "value": "labels"},
            ],
            inputStyle={"marginRight": 4},
            labelStyle={"marginRight": 16, "fontSize": 12, "cursor": "pointer",
                        "color": TEXT_MID, "fontFamily": "'DM Sans',sans-serif"}),
    ], style={"background": WHITE, "borderBottom": f"1px solid {BORDER}",
              "padding": "10px 32px", "display": "flex",
              "gap": 24, "flexWrap": "wrap", "alignItems": "center"}),

    html.Div(id="kpi-strip",
             style={"display": "flex", "gap": 10,
                    "padding": "14px 32px 6px", "flexWrap": "wrap"}),

    html.Div(id="main-content", style={"padding": "8px 32px 32px"}),

    html.Div([
        html.Span("78 features · 150 frames · 7 statistics → ",
                  style={"color": TEXT_MUTE}),
        html.Span("546-D XGBoost input",
                  style={"color": TEXT, "fontWeight": 600}),
        html.Span("  ·  DAiSEE Affective Computing",
                  style={"color": TEXT_MUTE}),
    ], style={"background": WHITE, "borderTop": f"1px solid {BORDER}",
              "padding": "10px 32px", "fontSize": 10,
              "fontFamily": "'DM Sans',sans-serif"}),

    dcc.Store(id="data-store", data={"loaded": DATA is not None}),

], style={"background": BG, "minHeight": "100vh",
          "fontFamily": "'DM Sans',sans-serif"})


# ── Callbacks ──────────────────────────────────────────────────────────────────

@app.callback(
    Output("data-store",  "data"),
    Output("load-status", "children"),
    Output("load-status", "style"),
    Output("feat-dd",     "options"),
    Output("feat-dd",     "value"),
    Input("load-btn",     "n_clicks"),
    State("pkl-path",     "value"),
    prevent_initial_call=True,
)
def cb_reload(_, path):
    global DATA, LOAD_ERROR, FEAT_INDEX, GLOBAL_STATS
    DATA, LOAD_ERROR = load_data(path)
    base = {"fontSize": 11, "alignSelf": "center", "fontFamily": "'DM Sans',sans-serif"}
    if DATA is None:
        return {"loaded": False}, f"✗ {LOAD_ERROR}", {**base, "color": DANGER}, [], None
    FEAT_INDEX   = build_feature_index(DATA["feature_names"])
    GLOBAL_STATS = compute_statistics(DATA["X"])
    msg = (f"✓ Loaded {DATA['X'].shape[0]} clips × "
           f"{DATA['seq_len']} frames × {DATA['num_features']} features")
    return {"loaded": True}, msg, {**base, "color": SUCCESS}, make_feat_opts(), 0


@app.callback(
    Output("feat-dd", "options", allow_duplicate=True),
    Input("group-dd", "value"),
    prevent_initial_call=True,
)
def cb_feat_opts(group):
    return make_feat_opts(group)


@app.callback(
    Output("kpi-strip", "children"),
    Input("data-store", "data"),
    Input("stat-dd",    "value"),
    Input("group-dd",   "value"),
)
def cb_kpis(store, stat, active_group):
    if not store or not store.get("loaded") or not GLOBAL_STATS:
        return html.Div("Load a .pkl file to begin.",
                        style={"color": TEXT_MUTE, "fontSize": 13, "padding": "8px 0"})
    cards = []
    for g in GROUP_META:
        idxs   = [f["idx"] for f in FEAT_INDEX if f["group"] == g["id"]]
        if not idxs:
            continue
        avg    = float(np.abs(GLOBAL_STATS[stat][idxs]).mean())
        active = active_group == g["id"]
        cards.append(html.Div([
            html.Div([
                html.Div(style={"width": 8, "height": 8, "borderRadius": "50%",
                                "background": g["color"]}),
                html.Span(g["short"], style={"fontSize": 10, "color": g["color"],
                                             "fontWeight": 700, "letterSpacing": "0.08em"}),
            ], style={"display": "flex", "alignItems": "center", "gap": 6, "marginBottom": 6}),
            html.Div(f"{avg:.4f}", style={"fontFamily": "'Fraunces',serif",
                                          "fontSize": 22, "fontWeight": 700, "color": TEXT}),
            html.Div(f"{g['label']} · {stat}",
                     style={"fontSize": 9, "color": TEXT_MUTE, "marginTop": 4}),
        ], style={
            "flex": "1 1 110px", "borderRadius": "12px", "padding": "13px 15px",
            "background":  rgba(g["color"], 0.05) if active else CARD,
            "border": f"1px solid {g['color']}" if active else f"1px solid {BORDER}",
        }))
    return cards


@app.callback(
    Output("main-content", "children"),
    Input("data-store", "data"),
    Input("tab-radio",  "value"),
    Input("stat-dd",    "value"),
    Input("group-dd",   "value"),
    Input("feat-dd",    "value"),
)
def cb_render(store, tab, stat, group, feat_idx):
    if not store or not store.get("loaded"):
        return html.Div("⬆  Enter the path to your features.pkl and click Load File.",
                        style={"color": TEXT_MUTE, "fontSize": 14,
                               "textAlign": "center", "marginTop": 80})
    feat_idx = int(feat_idx or 0)
    fo = FEAT_INDEX[feat_idx] if feat_idx < len(FEAT_INDEX) else FEAT_INDEX[0]
    fg = next((g for g in GROUP_META if g["id"] == fo["group"]), GROUP_META[0])
    flt = FEAT_INDEX if group == "all" else [f for f in FEAT_INDEX if f["group"] == group]

    if tab == "overview":     return tab_overview(stat, feat_idx, fo, fg, flt)
    if tab == "temporal":     return tab_temporal(feat_idx, fo, flt)
    if tab == "distribution": return tab_distribution(feat_idx, fo)
    if tab == "heatmap":      return tab_heatmap()
    if tab == "labels":       return tab_labels()
    return html.Div("Unknown tab")


# ── Tab renderers ──────────────────────────────────────────────────────────────

def tab_overview(stat, feat_idx, fo, fg, flt):
    # Bar
    bar = go.Figure(go.Bar(
        x=[f["name"] for f in flt],
        y=[float(GLOBAL_STATS[stat][f["idx"]]) for f in flt],
        marker=dict(color=[f["color"] for f in flt], line=dict(width=0)),
        hovertemplate="<b>%{x}</b><br>" + stat + ": %{y:.4f}<extra></extra>",
    ))
    bar.update_layout(**L(
        height=220, bargap=0.25,
        title=dict(text=f"{stat.upper()} — {len(flt)} features",
                   font=dict(size=12, family="Fraunces,serif"), x=0),
        xaxis=dict(tickangle=-40, tickfont=dict(size=8)),
    ))

    # Radar
    rv, rl = [], []
    for g in GROUP_META:
        idxs = [f["idx"] for f in FEAT_INDEX if f["group"] == g["id"]]
        if idxs:
            rv.append(float(np.abs(GLOBAL_STATS[stat][idxs]).mean()))
            rl.append(g["short"])
    rv.append(rv[0]); rl.append(rl[0])
    radar = go.Figure(go.Scatterpolar(
        r=rv, theta=rl, fill="toself",
        fillcolor=rgba(STAT_COLORS[stat], 0.14),
        line=dict(color=STAT_COLORS[stat], width=2),
        hovertemplate="<b>%{theta}</b><br>avg|" + stat + "|: %{r:.4f}<extra></extra>",
    ))
    radar.update_layout(**radar_layout("Group Radar"))

    # All-stats
    sf = go.Figure(go.Bar(
        x=STATS,
        y=[float(GLOBAL_STATS[s][feat_idx]) for s in STATS],
        marker=dict(color=[STAT_COLORS[s] for s in STATS], line=dict(width=0)),
        hovertemplate="<b>%{x}</b>: %{y:.4f}<extra></extra>",
    ))
    sf.update_layout(**L(
        height=190, bargap=0.3,
        title=dict(text=f"All Stats — {fo['name']}",
                   font=dict(size=12, family="Fraunces,serif"), x=0),
    ))

    # Group × stat heatmap
    z_hm, y_hm = [], []
    for g in GROUP_META:
        idxs = [f["idx"] for f in FEAT_INDEX if f["group"] == g["id"]]
        if not idxs:
            continue
        y_hm.append(g["label"])
        z_hm.append([float(np.abs(GLOBAL_STATS[s][idxs]).mean()) for s in STATS])
    hm = go.Figure(go.Heatmap(
        z=z_hm, x=STATS, y=y_hm,
        colorscale=[[0, BG], [0.5, rgba(CORAL_S, 0.5)], [1, CORAL]],
        hovertemplate="<b>%{y}</b> · %{x}<br>avg: %{z:.4f}<extra></extra>",
        showscale=True, colorbar=dict(thickness=10, tickfont=dict(size=9)),
    ))
    hm.update_layout(**L(
        height=260, margin=dict(l=130, r=80, t=40, b=30),
        title=dict(text="Group × Statistic Matrix",
                   font=dict(size=12, family="Fraunces,serif"), x=0),
        yaxis=dict(autorange="reversed"),
    ))

    return html.Div([
        html.Div([
            html.Div([dcc.Graph(figure=bar,   config={"displayModeBar": False})],
                     style={**CARD_STYLE, "flex": "2 1 340px"}),
            html.Div([dcc.Graph(figure=radar, config={"displayModeBar": False})],
                     style={**CARD_STYLE, "flex": "1 1 220px"}),
        ], style={"display": "flex", "gap": 12, "flexWrap": "wrap"}),
        html.Div([
            heading(fo["name"], size=15),
            sub(f"{fo['group_label']} · feat[{fo['idx']}] · {fg['desc']}"),
            dcc.Graph(figure=sf, config={"displayModeBar": False}),
        ], style=CARD_STYLE),
        html.Div([dcc.Graph(figure=hm, config={"displayModeBar": False})], style=CARD_STYLE),
    ])


def tab_temporal(feat_idx, fo, flt):
    X = DATA["X"]
    frames = np.arange(150)
    msig   = X[:, :, feat_idx].mean(axis=0)
    ssig   = X[:, :, feat_idx].std(axis=0)
    color  = fo["color"]

    area = go.Figure()
    area.add_trace(go.Scatter(
        x=np.concatenate([frames, frames[::-1]]),
        y=np.concatenate([msig + ssig, (msig - ssig)[::-1]]),
        fill="toself", fillcolor=rgba(color, 0.12),
        line=dict(color="rgba(0,0,0,0)"), hoverinfo="skip", name="±1 std",
    ))
    area.add_trace(go.Scatter(
        x=frames, y=msig, mode="lines",
        line=dict(color=color, width=2), name="mean",
        hovertemplate="Frame %{x}<br>Mean: %{y:.4f}<extra></extra>",
    ))
    area.update_layout(**L(
        height=230, xaxis_title="Frame", yaxis_title="Feature Value",
        legend=dict(x=0.01, y=0.99, font=dict(size=10)),
        title=dict(
            text=f"Temporal Signal — {fo['name']}  (mean ± std, {X.shape[0]} clips)",
            font=dict(size=12, family="Fraunces,serif"), x=0),
    ))

    palette  = [CORAL, TEAL, GOLD, MAUVE, NAVY]
    s_idxs   = np.linspace(0, X.shape[0] - 1, min(5, X.shape[0]), dtype=int)
    clips    = go.Figure()
    for ci, idx in enumerate(s_idxs):
        clips.add_trace(go.Scatter(
            x=frames, y=X[idx, :, feat_idx], mode="lines",
            line=dict(color=palette[ci % len(palette)], width=1.5),
            name=f"clip {idx}",
            hovertemplate=f"Clip {idx} · Frame %{{x}}: %{{y:.4f}}<extra></extra>",
        ))
    clips.update_layout(**L(
        height=200, xaxis_title="Frame",
        legend=dict(x=1.0, y=1.0, font=dict(size=9)),
        title=dict(text=f"Individual Clips — {fo['name']}",
                   font=dict(size=12, family="Fraunces,serif"), x=0),
    ))

    show  = flt[:24]
    ncols = 4
    nrows = max(1, (len(show) + ncols - 1) // ncols)
    spark = make_subplots(rows=nrows, cols=ncols,
                          vertical_spacing=0.04, horizontal_spacing=0.03)
    for i, f in enumerate(show):
        r, c = divmod(i, ncols)
        sig  = X[:, :, f["idx"]].mean(axis=0)
        spark.add_trace(
            go.Scatter(x=frames, y=sig, mode="lines",
                       line=dict(color=f["color"], width=1.2), name=f["name"],
                       hovertemplate=f"<b>{f['name']}</b><br>Frame %{{x}}: %{{y:.3f}}<extra></extra>"),
            row=r + 1, col=c + 1,
        )
        spark.update_xaxes(showticklabels=False, gridcolor=BORDER, row=r+1, col=c+1)
        spark.update_yaxes(showticklabels=False, gridcolor=BORDER, row=r+1, col=c+1)
    spark.update_layout(**spark_layout(nrows))

    return html.Div([
        html.Div([dcc.Graph(figure=area,  config={"displayModeBar": False})], style=CARD_STYLE),
        html.Div([dcc.Graph(figure=clips, config={"displayModeBar": False})], style=CARD_STYLE),
        html.Div([dcc.Graph(figure=spark, config={"displayModeBar": False})], style=CARD_STYLE),
    ])


def tab_distribution(feat_idx, fo):
    X     = DATA["X"]
    color = fo["color"]
    vals  = X[:, :, feat_idx].flatten()

    hist = go.Figure(go.Histogram(
        x=vals, nbinsx=40,
        marker=dict(color=color, line=dict(color=rgba(color, 0.5), width=0.5)),
        hovertemplate="Value: %{x:.3f}<br>Count: %{y}<extra></extra>",
    ))
    hist.update_layout(**L(
        height=220, xaxis_title="Value", yaxis_title="Count",
        title=dict(text=f"Value Distribution — {fo['name']} (all clips × frames)",
                   font=dict(size=12, family="Fraunces,serif"), x=0),
    ))

    box = go.Figure()
    for g in GROUP_META:
        idxs = [f["idx"] for f in FEAT_INDEX if f["group"] == g["id"]]
        if not idxs:
            continue
        box.add_trace(go.Box(
            y=X[:, :, idxs].flatten(), name=g["short"],
            marker=dict(color=g["color"]), line=dict(color=g["color"]),
            boxmean=True,
            hovertemplate=f"<b>{g['label']}</b><br>%{{y:.4f}}<extra></extra>",
        ))
    box.update_layout(**L(
        height=240, yaxis_title="Feature Value", showlegend=False,
        title=dict(text="Feature Value Distribution by Group",
                   font=dict(size=12, family="Fraunces,serif"), x=0),
    ))

    scatter = go.Figure()
    for g in GROUP_META:
        fg = [f for f in FEAT_INDEX if f["group"] == g["id"]]
        if not fg:
            continue
        scatter.add_trace(go.Scatter(
            x=[float(GLOBAL_STATS["mean"][f["idx"]]) for f in fg],
            y=[float(GLOBAL_STATS["std"][f["idx"]])  for f in fg],
            mode="markers", name=g["label"],
            text=[f["name"] for f in fg],
            marker=dict(color=g["color"], size=8, opacity=0.85,
                        line=dict(color=WHITE, width=0.5)),
            hovertemplate="<b>%{text}</b><br>mean: %{x:.4f}<br>std: %{y:.4f}<extra></extra>",
        ))
    scatter.update_layout(**L(
        height=260, xaxis_title="Mean", yaxis_title="Std",
        legend=dict(x=1.0, y=1.0, font=dict(size=10)),
        title=dict(text="Mean vs Std — All 78 Features",
                   font=dict(size=12, family="Fraunces,serif"), x=0),
    ))

    sf = go.Figure(go.Bar(
        y=STATS,
        x=[float(GLOBAL_STATS[s][feat_idx]) for s in STATS],
        orientation="h",
        marker=dict(color=[STAT_COLORS[s] for s in STATS], line=dict(width=0)),
        hovertemplate="<b>%{y}</b>: %{x:.4f}<extra></extra>",
    ))
    sf.update_layout(**L(
        height=220, xaxis_title="Value",
        title=dict(text=f"Stat Profile — {fo['name']}",
                   font=dict(size=12, family="Fraunces,serif"), x=0),
    ))

    return html.Div([
        html.Div([
            html.Div([dcc.Graph(figure=hist, config={"displayModeBar": False})],
                     style={**CARD_STYLE, "flex": "1 1 300px"}),
            html.Div([dcc.Graph(figure=sf,   config={"displayModeBar": False})],
                     style={**CARD_STYLE, "flex": "1 1 240px"}),
        ], style={"display": "flex", "gap": 12, "flexWrap": "wrap"}),
        html.Div([dcc.Graph(figure=box,     config={"displayModeBar": False})], style=CARD_STYLE),
        html.Div([dcc.Graph(figure=scatter, config={"displayModeBar": False})], style=CARD_STYLE),
    ])


def tab_heatmap():
    labels = [f["name"] for f in FEAT_INDEX]
    z_full = np.array([[float(GLOBAL_STATS[s][f["idx"]]) for s in STATS]
                       for f in FEAT_INDEX])

    full = go.Figure(go.Heatmap(
        z=z_full, x=STATS, y=labels, zmid=0,
        colorscale=[[0, rgba(NAVY, 0.38)], [0.5, BG], [1, CORAL]],
        hovertemplate="<b>%{y}</b> · %{x}: %{z:.4f}<extra></extra>",
        showscale=True,
        colorbar=dict(thickness=10, tickfont=dict(size=9), len=0.9),
    ))
    full.update_layout(**L(
        height=max(500, len(FEAT_INDEX) * 14 + 80),
        margin=dict(l=130, r=80, t=60, b=20),
        xaxis=dict(side="top", tickfont=dict(size=11)),
        yaxis=dict(tickfont=dict(size=8), autorange="reversed"),
        title=dict(text="Full Feature × Statistic Matrix (78 × 7)",
                   font=dict(size=12, family="Fraunces,serif"), x=0),
    ))

    corr      = np.corrcoef(z_full.T)
    corr_text = [[f"{corr[i][j]:.2f}" for j in range(7)] for i in range(7)]
    corr_fig  = go.Figure(go.Heatmap(
        z=corr, x=STATS, y=STATS,
        colorscale=[[0, NAVY], [0.5, BG], [1, CORAL]],
        zmid=0, zmin=-1, zmax=1,
        text=corr_text, texttemplate="%{text}",
        hovertemplate="%{y} × %{x}: %{z:.3f}<extra></extra>",
        showscale=True, colorbar=dict(thickness=10, tickfont=dict(size=9)),
    ))
    corr_fig.update_layout(**L(
        height=300, margin=dict(l=60, r=80, t=40, b=40),
        title=dict(text="Statistic Correlation Matrix",
                   font=dict(size=12, family="Fraunces,serif"), x=0),
    ))

    return html.Div([
        html.Div([dcc.Graph(figure=full, config={
            "displayModeBar": True,
            "modeBarButtonsToRemove": ["select2d", "lasso2d"],
        })], style=CARD_STYLE),
        html.Div([dcc.Graph(figure=corr_fig, config={"displayModeBar": False})],
                 style=CARD_STYLE),
    ])


def tab_labels():
    y_dict = DATA.get("y", {})
    if not y_dict:
        return html.Div("No label data found in the .pkl file.",
                        style={"color": TEXT_MUTE, "padding": 40, "textAlign": "center"})
    X       = DATA["X"]
    palette = [CORAL, TEAL, GOLD, NAVY]
    panels  = []

    for lname, larr in y_dict.items():
        arr            = np.array(larr)
        unique, counts = np.unique(arr, return_counts=True)

        bar = go.Figure(go.Bar(
            x=[str(u) for u in unique], y=counts.tolist(),
            marker=dict(color=palette[:len(unique)], line=dict(width=0)),
            hovertemplate="Class %{x}<br>Count: %{y}<extra></extra>",
        ))
        bar.update_layout(**L(
            height=200, bargap=0.3, xaxis_title="Class", yaxis_title="Count",
            title=dict(text=f"Label Distribution — {lname}",
                       font=dict(size=12, family="Fraunces,serif"), x=0),
        ))

        top10       = list(range(min(10, 78)))
        top10_names = [FEAT_INDEX[i]["name"] for i in top10]
        cls_means   = []
        for cls in unique:
            mask = arr == cls
            if not mask.any():
                cls_means.append([0.0] * len(top10))
                continue
            cm = X[mask].mean(axis=(0, 1))
            cls_means.append([float(cm[i]) for i in top10])

        hm = go.Figure(go.Heatmap(
            z=cls_means, x=top10_names,
            y=[f"Class {u}" for u in unique],
            colorscale=[[0, BG], [1, CORAL]],
            hovertemplate="<b>%{y}</b><br>%{x}: %{z:.4f}<extra></extra>",
            showscale=True, colorbar=dict(thickness=8, tickfont=dict(size=9)),
        ))
        hm.update_layout(**L(
            height=180, margin=dict(l=70, r=70, t=40, b=60),
            xaxis=dict(tickfont=dict(size=8), tickangle=-30),
            title=dict(text=f"Avg Feature by Class — {lname} (top 10 features)",
                       font=dict(size=11, family="Fraunces,serif"), x=0),
        ))

        panels.append(html.Div([
            html.Div([
                html.Div([dcc.Graph(figure=bar, config={"displayModeBar": False})],
                         style={"flex": "1 1 250px"}),
                html.Div([dcc.Graph(figure=hm,  config={"displayModeBar": False})],
                         style={"flex": "2 1 340px"}),
            ], style={"display": "flex", "gap": 12, "flexWrap": "wrap"}),
        ], style=CARD_STYLE))

    n      = X.shape[0]
    n_fail = len(DATA.get("failed_videos", []))
    summary = html.Div([
        heading("Dataset Summary", size=14),
        html.Div(style={"height": 10}),
        html.Div([
            kpi_card("TOTAL CLIPS",   str(n),               "processed successfully",    CORAL),
            kpi_card("FAILED VIDEOS", str(n_fail),           "skipped during extraction", GOLD),
            kpi_card("SEQUENCE LEN",  str(DATA["seq_len"]),  "frames per clip",           TEAL),
            kpi_card("FEATURE DIM",   str(DATA["num_features"]), "features per frame",   MAUVE),
        ], style={"display": "flex", "gap": 10, "flexWrap": "wrap"}),
    ], style=CARD_STYLE)

    return html.Div([summary] + panels)


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port  = int(os.environ.get("PORT", 8050))
    debug = "--debug" in sys.argv
    print(f"\n  DAiSEE Feature Dashboard  →  http://localhost:{port}")
    if DATA is not None:
        print(f"  Pre-loaded: {DATA['X'].shape[0]} clips × "
              f"{DATA['seq_len']} frames × {DATA['num_features']} features")
    else:
        print("  No data pre-loaded — use the file picker in the UI")
    print()
    app.run(debug=debug, port=port, host="0.0.0.0")
