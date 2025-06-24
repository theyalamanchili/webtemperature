import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
from io import BytesIO
from scipy.optimize import fsolve
import plotly.graph_objects as go

# -----------------------------------------------
# Background Styling and Footer
# -----------------------------------------------
def add_styles():
    st.markdown(
        """
        <style>
        .stApp { background-color: #f9f9f9; color: #111; }
        .footer { position: fixed; bottom: 0; width: 100%; text-align: center;
                  font-size: 0.8rem; color: #555; }
        </style>
        """, unsafe_allow_html=True)

def footer():
    st.markdown(
        """
        <div class='footer'>Version α-1 | © 2025 Texas A&amp;M University</div>
        """, unsafe_allow_html=True)

# -----------------------------------------------
# Login Screen
# -----------------------------------------------
def login():
    logo = Image.open("MEEN_logo.png")
    st.image(logo, width=300)
    st.subheader("Please log in to continue")
    with st.form("login_form"):
        user = st.text_input("Username")
        pwd  = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")
    if submitted:
        creds = {"Guest": "GuestPass",
                 "Aditya": "Yalamanchili",
                 "Prabhakar": "Pagilla",
                 "admin": "adminpass"}
        if creds.get(user) == pwd:
            st.session_state.logged_in = True
            st.rerun()
        else:
            st.error("Invalid username or password.")

# -----------------------------------------------
# Solvers
# -----------------------------------------------
def solve_2d(k, rho, c, v, T0, Tinf, h, t, L, N):
    Yh = t/2
    dT = T0 - Tinf
    beta = rho * c * v / (2 * k)
    Bi = h * Yh / k
    def fz(z): return np.tan(z) - Bi / z
    eps = 1e-6
    z = np.zeros(N)
    z[0] = fsolve(fz, [eps, np.pi/2 - eps])[0]
    odds = np.arange(1, 2*N, 2)
    for i in range(1, N):
        lo = odds[i-1]*np.pi/2 + eps
        hi = lo + np.pi - 2*eps
        z[i] = fsolve(fz, [lo, hi])[0]
    lam = z / Yh
    a = np.array([
        (2*dT*np.sin(z[i])) / (z[i] + np.sin(z[i]) * np.cos(z[i]))
        for i in range(N)
    ])
    x = np.linspace(0, L, 600)
    y = np.linspace(-Yh, Yh, 300)
    X, Yg = np.meshgrid(x, y)
    Theta = sum(
        a[i] * np.exp((beta - np.sqrt(beta**2 + lam[i]**2)) * X) * np.cos(lam[i] * Yg)
        for i in range(N)
    )
    return x, y, X, Yg, Tinf + Theta

def solve_1d(k, rho, c, v, T0, Tinf, h, t, W, x):
    Yh = t/2
    dT = T0 - Tinf
    beta = rho * c * v / (2 * k)
    A = 2 * W * Yh
    P = 2 * W + 2 * Yh
    m2 = h * P / (k * A)
    mu = beta - np.sqrt(beta**2 + m2)
    return Tinf + dT * np.exp(mu * x)

# -----------------------------------------------
# Main Application
# -----------------------------------------------
add_styles()

# Initialize session flags
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "readme_expanded" not in st.session_state:
    st.session_state.readme_expanded = True

if not st.session_state.logged_in:
    login()
    st.stop()

st.title("Web Temperature Distribution Simulator")

# — show solved‐for banner if present —
if "solved_param" in st.session_state:
    st.success(
        f"🔄 Predicted {st.session_state.solved_param}: "
        f"{st.session_state.solved_value:.3f}"
    )

# Read-Me / User Guide
with st.expander("📖 Read Me / User Guide", expanded=st.session_state.readme_expanded):
    st.markdown(r"""
    <!-- Insert the full read-me markdown content here, as in your original app -->
    """, unsafe_allow_html=False)

# Sidebar: Scenario
st.sidebar.header("1. Web Transport Scenario")
scenario = st.sidebar.selectbox(
    "Scenario",
    ["Free span convective cooling", "Web in heating/cooling zone", "Web on heated/cooled roller"]
)
if scenario == "Free span convective cooling":
    st.sidebar.image("BC1.png", use_container_width=True)
elif scenario == "Web in heating/cooling zone":
    st.sidebar.image("BC2.png", use_container_width=True)
else:
    st.sidebar.image("BC3.png", use_container_width=True)

scenario_desc = {
    "Free span convective cooling":
        "Unsupported span between rollers; both faces cool (or heat) by convection with the surrounding air.",
    "Web in heating/cooling zone":
        "Section of free span that traverses a finite-length oven / furnace / IR panel held at set temperature.",
    "Web on heated/cooled roller":
        "Region where the moving web wraps around a temperature-controlled roller."
}
st.sidebar.info(scenario_desc[scenario])

# Sidebar: Material
st.sidebar.header("2. Material Properties")
matlib = {
    'PET':      {'k':0.2,   'rho':1390, 'c':1400},
    'Aluminum': {'k':237,   'rho':2700, 'c':897},
    'Copper':   {'k':401,   'rho':8960, 'c':385}
}
materials = list(matlib.keys()) + ['Custom']
mat = st.sidebar.selectbox("Material", materials, index=0)
if mat != 'Custom':
    p = matlib[mat]
    st.sidebar.markdown(
        f"Thermal conductivity, k: **{p['k']}**  \n"
        f"Density, ρ: **{p['rho']}**  \n"
        f"Specific heat, c: **{p['c']}**"
    )
    k, rho, c = p['k'], p['rho'], p['c']
else:
    k   = st.sidebar.number_input("Thermal conductivity, k (W/m·K)", 0.1, 500.0, 0.2)
    rho = st.sidebar.number_input("Density, ρ (kg/m³)", 100, 20000, 1390)
    c   = st.sidebar.number_input("Specific heat, c (J/kg·K)", 100, 5000, 1400)

# Sidebar: Temperatures & Convection (invertible 1D)
st.sidebar.header("3. Temperatures & Convection")
missing = st.sidebar.selectbox(
    "Solve for…",
    ["– none –", "Inlet temperature (T₀)", "Web speed (v)", "h (convective coeff)"]
)
if missing == "– none –":
    T0  = st.sidebar.number_input("Inlet temperature, T₀ (°C)", -50.0, 500.0, 200.0)
    v   = st.sidebar.number_input("Web speed, v (m/s)", 0.01, 10.0, 1.6)
    h   = st.sidebar.number_input("Convective coeff, h (W/m²·K)", 1.0, 1e4, 100.0)
elif missing == "Inlet temperature (T₀)":
    Tout = st.sidebar.number_input("Outlet temp, T_out @ x=L (°C)", -50.0, 500.0, 100.0)
    v    = st.sidebar.number_input("Web speed, v (m/s)", 0.01, 10.0, 1.6)
    h    = st.sidebar.number_input("Convective coeff, h (W/m²·K)", 1.0, 1e4, 100.0)
    T0   = None
elif missing == "Web speed (v)":
    Tout = st.sidebar.number_input("Outlet temp, T_out @ x=L (°C)", -50.0, 500.0, 100.0)
    T0   = st.sidebar.number_input("Inlet temperature, T₀ (°C)", -50.0, 500.0, 200.0)
    h    = st.sidebar.number_input("Convective coeff, h (W/m²·K)", 1.0, 1e4, 100.0)
    v    = None
else:  # missing == "h (convective coeff)"
    Tout = st.sidebar.number_input("Outlet temp, T_out @ x=L (°C)", -50.0, 500.0, 100.0)
    T0   = st.sidebar.number_input("Inlet temperature, T₀ (°C)", -50.0, 500.0, 200.0)
    v    = st.sidebar.number_input("Web speed, v (m/s)", 0.01, 10.0, 1.6)
    h    = None

Tinf = st.sidebar.number_input("Ambient temperature, T∞ (°C)", -50.0, 200.0, 25.0)

# Sidebar: Transport & Process
st.sidebar.header("4. Transport & Process Params")
t = st.sidebar.number_input("Thickness, t (m)", 1e-6, 1e-2, 0.001)
W = st.sidebar.number_input("Width, W (m)", 0.01, 5.0, 1.0)
L = st.sidebar.number_input("Span length, L (m)", 0.1, 50.0, 10.0)
with st.sidebar.expander("5. Default Parameters", expanded=False):
    N = st.slider("Number of eigenmodes, N", 5, 50, 20)

# -----------------------------------------------
# Compute button logic
# -----------------------------------------------
if st.sidebar.button("Compute"):
    st.session_state.readme_expanded = False

    # invert 1-D for the missing parameter
    if scenario == "Free span convective cooling" and missing != "– none –":
        Yh = t/2
        A  = 2 * W * Yh
        P  = 2 * W + 2 * Yh

        def mu_of(v_val, h_val):
            beta = rho * c * v_val / (2 * k)
            m2   = h_val * P / (k * A)
            return beta - np.sqrt(beta**2 + m2)

        if missing == "Inlet temperature (T₀)":
            μ = mu_of(v, h)
            T0 = Tinf + (Tout - Tinf) * np.exp(-μ * L)
            st.session_state.solved_param = "Inlet temperature (T₀)"
            st.session_state.solved_value = T0

        elif missing == "Web speed (v)":
            def fv(v_val):
                return Tinf + (T0 - Tinf) * np.exp(mu_of(v_val, h) * L) - Tout
            v = float(fsolve(fv, x0=0.1)[0])
            st.session_state.solved_param = "Web speed (v)"
            st.session_state.solved_value = v

        else:  # solving for h
            def fh(h_val):
                return Tinf + (T0 - Tinf) * np.exp(mu_of(v, h_val) * L) - Tout
            h = float(fsolve(fh, x0=100)[0])
            st.session_state.solved_param = "Convective coeff (h)"
            st.session_state.solved_value = h

    # now run the normal 1D + 2D solvers
    if scenario == "Free span convective cooling":
        x, y, X, Yg, T2 = solve_2d(k, rho, c, v, T0, Tinf, h, t, L, N)
        T1 = solve_1d(k, rho, c, v, T0, Tinf, h, t, W, x)
        st.session_state.update(x=x, y=y, X=X, Yg=Yg, T2=T2, T1=T1, ready=True)
        st.rerun()
    else:
        st.warning("Analytical solution for this scenario will be added soon. Please check back.")
        st.session_state.ready = False

# -----------------------------------------------
# Display results (unchanged)
# -----------------------------------------------
if scenario == "Free span convective cooling" and st.session_state.get('ready'):
    x, y, X, Yg, T2, T1 = (
        st.session_state[var] for var in ['x','y','X','Yg','T2','T1']
    )
    Yh = t/2
    Bi = h * Yh / k
    Pe = v * L / (k/(rho*c))

    st.subheader("2D Temperature Contour")
    show = st.checkbox("Show contour lines & labels", True)
    fig = go.Figure(go.Contour(
        z=T2, x=x, y=y,
        colorscale="Turbo", ncontours=60,
        contours=dict(showlines=show, showlabels=show, labelfont=dict(size=12)),
        colorbar=dict(title="Temperature (°C)")
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="Span location (m)",
        yaxis_title="Transverse location within web (m)"
    )
    st.plotly_chart(fig, use_container_width=True)
    df_cont = pd.DataFrame({'x': X.flatten(), 'y': Yg.flatten(), 'T': T2.flatten()})
    buf1 = BytesIO(); df_cont.to_csv(buf1, index=False); buf1.seek(0)
    st.download_button("Download Contour CSV", buf1, "contour.csv", "text/csv")
    st.markdown(f"**Biot (Bi):** {Bi:.2f} — **Péclet (Pe):** {Pe:.1f}")

    st.subheader("Temperature Profiles vs Span")
    sel = {
        "Centerline (T_c)":        st.checkbox("Centerline (T_c)", True),
        "Top surface (T_top)":     st.checkbox("Top surface (T_top)", True),
        "Bottom surface (T_bot)":  st.checkbox("Bottom surface (T_bot)", True),
        "Thickness-average (T_avg)": st.checkbox("Thickness-average (T_avg)", True),
        "1-D Lumped (T_1D)":       st.checkbox("1-D Lumped (T_1D)", True),
    }
    figp = go.Figure()
    if sel["Centerline (T_c)"]:
        figp.add_trace(go.Scatter(x=x, y=T2[np.argmin(np.abs(y))],
                                  mode="lines", name="Centerline (T_c)"))
    if sel["Top surface (T_top)"]:
        figp.add_trace(go.Scatter(x=x, y=T2[np.argmin(np.abs(y - Yh))],
                                  mode="lines", name="Top surface (T_top)"))
    if sel["Bottom surface (T_bot)"]:
        figp.add_trace(go.Scatter(x=x, y=T2[np.argmin(np.abs(y + Yh))],
                                  mode="lines", name="Bottom surface (T_bot)"))
    if sel["Thickness-average (T_avg)"]:
        figp.add_trace(go.Scatter(x=x, y=T2.mean(axis=0),
                                  mode="lines", name="Thickness-average (T_avg)"))
    if sel["1-D Lumped (T_1D)"]:
        figp.add_trace(go.Scatter(x=x, y=T1, mode="lines",
                                  name="1-D Lumped (T_1D)", line=dict(dash="dash")))
    figp.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        xaxis_title='Span location (m)', yaxis_title='Temperature (°C)',
        legend_title='Profiles'
    )
    st.plotly_chart(figp, use_container_width=True)

    df_prof = pd.DataFrame({"x": x})
    series_map = {
        "Centerline (T_c)":         ("T_c",   T2[np.argmin(np.abs(y))]),
        "Top surface (T_top)":      ("T_top", T2[np.argmin(np.abs(y - Yh))]),
        "Bottom surface (T_bot)":   ("T_bot", T2[np.argmin(np.abs(y + Yh))]),
        "Thickness-average (T_avg)":("T_avg", T2.mean(axis=0)),
        "1-D Lumped (T_1D)":        ("T_1D",  T1)
    }
    for label, (col, data) in series_map.items():
        if sel[label]:
            df_prof[col] = data
    buf2 = BytesIO(); df_prof.to_csv(buf2, index=False); buf2.seek(0)
    st.download_button("Download Profiles CSV", buf2, "profiles.csv", "text/csv")

    st.subheader("Temperature Differences vs Span")
    ds = {
        "ΔT_c-top":  st.checkbox("ΔT_c-top", True),
        "ΔT_avg-1D": st.checkbox("ΔT_avg-1D", True)
    }
    figd = go.Figure()
    if ds["ΔT_c-top"]:
        figd.add_trace(go.Scatter(
            x=x,
            y=T2[np.argmin(np.abs(y))] - T2[np.argmin(np.abs(y - Yh))],
            mode="lines", name="ΔT_c-top"
        ))
    if ds["ΔT_avg-1D"]:
        figd.add_trace(go.Scatter(
            x=x,
            y=T2.mean(axis=0) - T1,
            mode="lines", name="ΔT_avg-1D"
        ))
    figd.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        xaxis_title='Span location (m)', yaxis_title='ΔTemperature (°C)',
        legend_title='Differences'
    )
    st.plotly_chart(figd, use_container_width=True)

    df_diff = pd.DataFrame({"x": x})
    diff_map = {
        "ΔT_c-top":  ("dT_c_top",  T2[np.argmin(np.abs(y))] - T2[np.argmin(np.abs(y - Yh))]),
        "ΔT_avg-1D": ("dT_avg_1D", T2.mean(axis=0) - T1)
    }
    for label, (col, data) in diff_map.items():
        if ds[label]:
            df_diff[col] = data
    buf3 = BytesIO(); df_diff.to_csv(buf3, index=False); buf3.seek(0)
    st.download_button("Download Differences CSV", buf3, "differences.csv", "text/csv")

footer()
