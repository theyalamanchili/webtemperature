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

    # ----- one-shot form -----
    with st.form("login_form"):
        user = st.text_input("Username")
        pwd  = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")   # ↵ also submits

    if submitted:
        creds = {"Guest": "GuestPass",
                 "Aditya": "Yalamanchili",
                 "Prabhakar": "Pagilla",
                 "admin": "adminpass"}
        if creds.get(user) == pwd:
            st.session_state.logged_in = True
            st.rerun()     # works on all recent Streamlit versions
        else:
            st.error("Invalid username or password.")


# -----------------------------------------------
# Solvers
# -----------------------------------------------
def solve_2d(k, rho, c, v, T0, Tinf, h, t, L, N):
    Yh = t/2
    dT = T0 - Tinf
    beta = rho*c*v/(2*k)
    Bi = h*Yh/k
    def fz(z): return np.tan(z) - Bi/z
    eps=1e-6
    z=np.zeros(N)
    z[0]=fsolve(fz,[eps,np.pi/2-eps])[0]
    odds=np.arange(1,2*N,2)
    for i in range(1,N):
        lo=odds[i-1]*np.pi/2+eps
        hi=lo+np.pi-2*eps
        z[i]=fsolve(fz,[lo,hi])[0]
    lam=z/Yh
    a=np.array([(2*dT*np.sin(z[i]))/(z[i]+np.sin(z[i])*np.cos(z[i])) for i in range(N)])
    x=np.linspace(0,L,600)
    y=np.linspace(-Yh,Yh,300)
    X,Yg=np.meshgrid(x,y)
    Theta=sum(a[i]*np.exp((beta-np.sqrt(beta**2+lam[i]**2))*X)*np.cos(lam[i]*Yg) for i in range(N))
    return x,y,X,Yg, Tinf+Theta

def solve_1d(k, rho, c, v, T0, Tinf, h, t, W, x):
    Yh=t/2; dT=T0-Tinf; beta=rho*c*v/(2*k)
    A=2*W*Yh; P=2*W+2*Yh; m2=h*P/(k*A)
    mu=beta-np.sqrt(beta**2+m2)
    return Tinf + dT*np.exp(mu*x)

# -----------------------------------------------
# Main Application
# -----------------------------------------------
add_styles()
# --------------------------- session flags ---------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "readme_expanded" not in st.session_state:
    st.session_state.readme_expanded = True        # open at launch
if 'logged_in' not in st.session_state:
    st.session_state.logged_in=False
if not st.session_state.logged_in:
    login()
    st.stop()
st.title("Web Temperature Distribution Simulator")

# Read-Me / User Guide
with st.expander("📖 Read Me / User Guide", expanded=st.session_state.readme_expanded):

    # ─────────────────────────────
    #  Overview
    # ─────────────────────────────
    st.markdown(
        r"""
This tool was developed by Aditya Yalamanchili under the supervision of Prof. Prabhakar Pagilla at the Department of Mechanical Engineering, Texas A&M University. The development of this tool is part of an ongoing research project aimed at advancing the understanding of temperature distribution in moving webs within roll-to-roll (R2R) manufacturing systems. This work builds upon a series of research efforts led by the team, including conference presentations, proceedings and academic manuscripts, as listed in the citations below. These efforts reflect the continuous commitment to refining the theoretical models, validating the solutions through experimental data, and expanding the tool’s capabilities. The tool will be continuously updated as research progresses, incorporating newer solutions, improved algorithms, and additional features to enhance accuracy and usability.

---
**Overview & Background**  
In roll‑to‑roll (R2R) manufacturing, a continuous web of material travels through processing zones  
(e.g. ovens, furnaces, cooling spans, heated rollers). Predicting its *steady‑state* temperature field is crucial for  
avoiding thermal defects (curl, wrinkles, web‑elongation), ensuring process uniformity,  
and improving web‑tension regulation.

This app solves two analytical models under steady‑state conditions:

**2‑D convection–diffusion model** (across *y* and along *x*):
""",
        unsafe_allow_html=False,
    )

    st.latex(
        r"""
\frac{\partial^{2} T}{\partial x^{2}}
  + \frac{\partial^{2} T}{\partial y^{2}}
  - 2\beta \frac{\partial T}{\partial x}=0,
\qquad
\beta = \frac{\rho\,c\,v}{2\,k}
"""
    )

    st.markdown(
        r"""
**1‑D lumped‑capacitance model** (uniform through‑thickness):
""",
        unsafe_allow_html=False,
    )

    st.latex(
        r"""
\frac{d^{2} T}{d x^{2}}
  - 2\beta \frac{d T}{d x}
  - m^{2}\!\left(T - T_{\infty}\right)=0,
\qquad
m^{2} = \frac{h\,P}{k\,A}
"""
    )

    # ─────────────────────────────
    #  Assumptions
    # ─────────────────────────────
    st.markdown(
        r"""
**Assumptions**

* Material properties ($k$, $\rho$, $c$) constant  
* Steady‑state (no time dependence)  
* Idealised boundary conditions (convective cooling / roller contact)  
* Series truncation error $\propto 1/N$ (increase $N$ for higher accuracy)

---

**How to Use**

1. **Select Web Transport Scenario** (span cooling, roller contact, or heating zone).  
2. **Set Material Properties** (choose library entry or custom $k$, $\rho$, $c$).  
3. **Enter Temperatures & Convection** ($T_{0}$, $T_{\infty}$, $h$).  
4. **Fill Transport & Process Parameters** ($v$, $t$, $W$, $L$).  
5. **Adjust Default Parameters** (number of eigen‑modes $N$).  
6. Click **Compute**.  
7. Use the **checkboxes** beneath each plot to toggle curves.  
8. Download CSVs of the contour, profiles, or difference data.

**Input Variables** — see sidebar labels (italic notation, SI units).

**Computed Outputs**

1. **2‑D Temperature Contour**  
   * *x*‑axis: span position $x$ (m)  
   * *y*‑axis: through‑thickness $y$ (m)  
   * Colour: local temperature $T(x,y)$ (Turbo scale)

2. **Temperature Profiles vs. Span**  
   * Centre‑line: $T_{\mathrm{c}}(x)=T(x,0)$  
   * Top surface: $T_{\mathrm{top}}(x)=T\!\bigl(x,+t/2\bigr)$  
   * Bottom surface: $T_{\mathrm{bot}}(x)=T\!\bigl(x,-t/2\bigr)$  
   * Thickness‑average: $\displaystyle T_{\mathrm{avg}}(x)=\frac{1}{t}\int_{-t/2}^{t/2}T(x,y)\,dy$  
   * 1‑D lumped model: $T_{1\mathrm{D}}(x)$  

3. **Temperature Differences vs. Span**  
   * $\Delta T_{c-\mathrm{top}}(x)=T_{\mathrm{c}}(x)-T_{\mathrm{top}}(x)$  
   * $\Delta T_{\mathrm{avg}-1\mathrm{D}}(x)=T_{\mathrm{avg}}(x)-T_{1\mathrm{D}}(x)$  

4. **CSV Downloads**  
   * Contour CSV — full $(x,y,T)$ field  
   * Profiles CSV — span vs. selected profiles  
   * Differences CSV — span vs. selected differences  

---
***Research Background***
The model, methods, and code used in this application are part of ongoing research efforts aimed at understanding the steady-state temperature distribution in moving webs, specifically in roll-to-roll (R2R) manufacturing systems. The research has been presented and discussed at the following conferences:

**Publications and Proceedings**  
   * Lu, Y., & Pagilla, P. R. (2014).*Modeling of temperature distribution in moving webs in roll-to-roll manufacturing. *Journal of Thermal Science and Engineering Applications*, 6(4), 041012.  
   * Cobos Torres, E. O., & Pagilla, P. R. (2017). Temperature distribution in moving webs heated by radiation panels: model development and experimental validation. *Journal of Dynamic Systems, Measurement, and Control*, 139(5), 051003.
   * Jabbar, K. A., & Pagilla, P. R. (2018). Modeling and analysis of web span tension dynamics considering thermal and viscoelastic effects in roll-to-roll manufacturing. *Journal of Manufacturing Science and Engineering*, 140(5), 051005.
   * Lu, Y., & Pagilla, P. R. (2012,). Modeling of temperature distribution in a moving web transported over a heat transfer roller. *Dynamic Systems and Control Conference* (Vol. 45301, pp. 405-414).
   * Torres, E. O. C., & Pagilla, P. R. (2016). A governing equation for moving web temperature heated by radiative panels. *American Control Conference (ACC)* (pp. 858-863).

**Ongoing Pre-Print**  
   * Yalamanchili, A.\,V.; Pagilla, P.\,R. (2025). Steady‑State Temperature Distribution in Moving Webs in Roll‑to‑Roll Manufacturing (In preparation).

**Conference Presentations**  
   * Yalamanchili, A. V. & Pagilla, P.R. (2024). Closd-form analytical solutions for steady‑state temperature distribution in moving webs. *Roll‑to‑Roll (R2R) USA Conference & Expo*, Charlotte, NC, USA.  
   * Yalamanchili, A. V. & Pagilla, P.R. (2024). Modeling of steady‑state temperature distribution in moving webs in roll‑to‑roll manufacturing. *ASME Summer Heat Transfer Conference 2024*, Anaheim, CA, USA.



---

**Disclaimer**  
The analytical solutions provided by this application are based on theoretical models with assumptions including constant material properties, steady-state conditions, and idealized boundary conditions. Series truncation is determined by the chosen parameter number of modes. Higher number of modes improves accuracy but may significantly increase computation time.

While every effort has been made to ensure accuracy, the results generated by this tool are approximate and may not fully represent real-world scenarios. Therefore, users are strongly encouraged to validate the outputs through experimental testing specific to their application before making any engineering or operational decisions.

**Liability and Use of Results**  
The creators and developers of this application assume no responsibility for any inaccuracies, errors, or omissions within the tool, nor for any outcomes, damages, or liabilities resulting from the use of this application. The user assumes full responsibility for the use of the tool and any results derived from it.

**Ongoing Development**  
This application is a work in progress, and the methods, code, and analytical models are actively being tested and refined. We are continuously working to identify and fix potential bugs, errors, and limitations in the calculations. Users are encouraged to report any issues they encounter for further improvement.

By using this application, you acknowledge that you have read, understood, and agreed to the terms of this disclaimer.

---

**Citation**

Yalamanchili, A.\,V.; Pagilla, P.\,R. (2025). *Steady‑State Temperature Distribution in Moving Webs in Roll‑to‑Roll Manufacturing*.
""",
        unsafe_allow_html=False,
    )


# Sidebar Sections
st.sidebar.header("1. Web Transport Scenario")
scenario=st.sidebar.selectbox("Scenario",["Free span convective cooling","Web in heating/cooling zone","Web on heated/cooled roller"])
if scenario=="Free span convective cooling": st.sidebar.image("BC1.png",use_container_width=True)
elif scenario=="Web in heating/cooling zone": st.sidebar.image("BC2.png",use_container_width=True)
else: st.sidebar.image("BC3.png",use_container_width=True)
# ───────── brief scenario descriptions ─────────
scenario_desc = {
    "Free span convective cooling":
        "Unsupported span between rollers; both faces cool (or heat) by convection with the surrounding air.",
    "Web in heating/cooling zone":
        "Section of free span that traverses a finite‑length oven / furnace / IR panel held at set temperature.",
    "Web on heated/cooled roller":
        "Region where the moving web wraps around a temperature‑controlled roller providing a constant heat flux through conduction on one surface while the other surface is subject to covection."
}
st.sidebar.info(scenario_desc[scenario])
st.sidebar.header("2. Material Properties")
matlib={'PET':{'k':0.2,'rho':1390,'c':1400},'Aluminum':{'k':237,'rho':2700,'c':897},'Copper':{'k':401,'rho':8960,'c':385}}
materials=list(matlib.keys())+['Custom']
mat=st.sidebar.selectbox("Material",materials,index=0)
if mat!='Custom':
    p=matlib[mat]
    st.sidebar.markdown(
        f"Thermal conductivity, *k* (W·m⁻¹·K⁻¹): **{p['k']}**  \n"
        f"Density, *ρ* (kg·m⁻³): **{p['rho']}**  \n"
        f"Specific heat, *c* (J·kg⁻¹·K⁻¹): **{p['c']}**"
    )
    k,rho,c=p['k'],p['rho'],p['c']
else:
    k=st.sidebar.number_input("Thermal conductivity, *k* (W·m⁻¹·K⁻¹)",0.1,500.0,0.2)
    rho=st.sidebar.number_input("Density, *ρ* (kg·m⁻³)",100,20000,1390)
    c=st.sidebar.number_input("Specific heat, *c* (J·kg⁻¹·K⁻¹)",100,5000,1400)

st.sidebar.header("3. Temperatures & Convection")
T0=st.sidebar.number_input("Inlet temperature, *T₀* (°C)",-50.0,500.0,200.0)
Tinf=st.sidebar.number_input("Ambient temperature, *T∞* (°C)",-50.0,200.0,25.0)
h=st.sidebar.number_input("Convective coef, *h* (W·m⁻²·K⁻¹)",1.0,1e4,100.0)

st.sidebar.header("4. Transport & Process Params")
v=st.sidebar.number_input("Web speed, *v* (m·s⁻¹)",0.01,10.0,1.6)
t=st.sidebar.number_input("Thickness, *t* (m)",1e-6,1e-2,0.001)
W=st.sidebar.number_input("Width, *W* (m)",0.01,5.0,1.0)
L=st.sidebar.number_input("Span length, *L* (m)",0.1,50.0,10.0)

with st.sidebar.expander("5. Default Parameters",expanded=False):
    st.markdown("Number of eigenmodes, *N* (–): increase for accuracy vs. compute time.")
    N=st.slider("Series terms, *N* (–)",5,50,20)

# Compute
if st.sidebar.button("Compute"):
    # collapse the README for all scenarios
    st.session_state.readme_expanded = False

    if scenario == "Free span convective cooling":
        x, y, X, Yg, T2 = solve_2d(k, rho, c, v, T0, Tinf, h, t, L, N)
        T1 = solve_1d(k, rho, c, v, T0, Tinf, h, t, W, x)
        st.session_state.update(x=x, y=y, X=X, Yg=Yg, T2=T2, T1=T1,
                                ready=True)
    else:
        st.warning("Analytical solution for this scenario will be added soon. Please check back.")
        st.session_state.ready = False

    st.rerun()          # ← force a fresh run with the new flag


# Display
# Display only for scenarios that have results
if scenario == "Free span convective cooling" and st.session_state.get('ready'):
    x,y,X,Yg,T2,T1=(st.session_state[var] for var in ['x','y','X','Yg','T2','T1'])
    Yh=t/2; Bi=h*Yh/k; Pe=v*L/(k/(rho*c))

    st.subheader("2D Temperature Contour")
    st.markdown("**X-axis:** Span location (m) — **Y-axis:** Transverse (through-thickness) location within web (m)")
# ───────── contour figure ─────────
    show = st.checkbox("Show contour lines & labels", value=True)
    
    fig = go.Figure(
        go.Contour(
            z=T2, x=x, y=y,
            colorscale="Turbo", ncontours=60,
            contours=dict(showlines=show, showlabels=show,
                          labelfont=dict(size=12)),
            colorbar=dict(title="Temperature (°C)")
        )
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="Span location,  x  (m)",
        yaxis_title="Transverse location within web,  y  (m)"
    )
    st.plotly_chart(fig, use_container_width=True)
    df_cont=pd.DataFrame({'x':X.flatten(),'y':Yg.flatten(),'T':T2.flatten()})
    buf1=BytesIO();df_cont.to_csv(buf1,index=False);buf1.seek(0)
    st.download_button("Download Contour CSV",buf1,"contour.csv","text/csv")
    st.markdown(f"**Biot, *Bi*:** {Bi:.2f} — **Péclet, *Pe*:** {Pe:.1f}")

    # Profiles
    st.subheader("Temperature Profiles vs Span")
    st.markdown("Curves show *T(x)* at key thickness locations along the span.")
    # ───────── profile checkboxes ─────────
    sel = {
        "Centerline (T_c)"        : st.checkbox("Centerline (T_c)",        True),
        "Top surface (T_top)"     : st.checkbox("Top surface (T_top)",     True),
        "Bottom surface (T_bot)"  : st.checkbox("Bottom surface (T_bot)",  True),
        "Thickness‑average (T_avg)": st.checkbox("Thickness‑average (T_avg)", True),
        "1‑D Lumped (T_1D)"       : st.checkbox("1‑D Lumped (T_1D)",       True),
    }
    figp = go.Figure()
    if sel["Centerline (T_c)"]:
        figp.add_trace(go.Scatter(x=x, y=T2[np.argmin(np.abs(y))],
                                  mode="lines", name="Centerline (T_c)"))
    if sel["Top surface (T_top)"]:
        figp.add_trace(go.Scatter(x=x, y=T2[np.argmin(np.abs(y-Yh))],
                                  mode="lines", name="Top surface (T_top)"))
    if sel["Bottom surface (T_bot)"]:
        figp.add_trace(go.Scatter(x=x, y=T2[np.argmin(np.abs(y+Yh))],
                                  mode="lines", name="Bottom surface (T_bot)"))
    if sel["Thickness‑average (T_avg)"]:
        figp.add_trace(go.Scatter(x=x, y=T2.mean(axis=0),
                                  mode="lines", name="Thickness‑average (T_avg)"))
    if sel["1‑D Lumped (T_1D)"]:
        figp.add_trace(go.Scatter(x=x, y=T1, mode="lines",
                                  name="1‑D Lumped (T_1D)", line=dict(dash="dash")))
    figp.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', xaxis_title='Span location (m)', yaxis_title='Temperature (°C)', legend_title='Profiles')
    st.plotly_chart(figp, use_container_width=True)
    # -----------  Build DataFrame for download -----------
    df_prof = pd.DataFrame({"x": x})
    
    # map “menu label ↦ CSV column name ↦ data series”
    series_map = {
        "Centerline (T_c)"        : ("T_c",   T2[np.argmin(np.abs(y))]          ),
        "Top surface (T_top)"     : ("T_top", T2[np.argmin(np.abs(y - Yh))]     ),
        "Bottom surface (T_bot)"  : ("T_bot", T2[np.argmin(np.abs(y + Yh))]     ),
        "Thickness‑average (T_avg)": ("T_avg", T2.mean(axis=0)                  ),
        "1‑D Lumped (T_1D)"       : ("T_1D",  T1                                ),
    }
    for label, (col, data) in series_map.items():
        if sel[label]:                     # only add columns that were plotted
            df_prof[col] = data
    # -----------  Offer CSV download -----------
    buf2 = BytesIO()
    df_prof.to_csv(buf2, index=False)
    buf2.seek(0)
    st.download_button("Download Profiles CSV", buf2,
                       file_name="profiles.csv", mime="text/csv")

    # Differences
    st.subheader("Temperature Differences vs Span")
    st.markdown("Difference curves to highlight deviations between profiles.")
    # ───────── difference checkboxes ─────────
    ds = {
        "ΔT_c-top"   : st.checkbox("ΔT_c-top",   True),
        "ΔT_avg-1D"  : st.checkbox("ΔT_avg-1D",  True),
    }
    
    figd = go.Figure()
    if ds["ΔT_c-top"]:
        figd.add_trace(go.Scatter(
            x=x,
            y=T2[np.argmin(np.abs(y))] - T2[np.argmin(np.abs(y-Yh))],
            mode="lines", name="ΔT_c-top"
        ))
    if ds["ΔT_avg-1D"]:
        figd.add_trace(go.Scatter(
            x=x,
            y=T2.mean(axis=0) - T1,
            mode="lines", name="ΔT_avg-1D"
        ))
    figd.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', xaxis_title='Span location (m)', yaxis_title='ΔTemperature (°C)', legend_title='Differences')
    st.plotly_chart(figd, use_container_width=True)
    # -----------  Build DataFrame for download -----------
    df_diff = pd.DataFrame({"x": x}) 
    diff_map = {
        "ΔT_c-top":  ("dT_c_top",  T2[np.argmin(np.abs(y))] - T2[np.argmin(np.abs(y - Yh))]),
        "ΔT_avg-1D": ("dT_avg_1D", T2.mean(axis=0)          - T1),
    }
    for label, (col, data) in diff_map.items():
        if ds[label]:                    # only add the columns that were displayed
            df_diff[col] = data
    
    # -----------  Offer CSV download -----------
    buf3 = BytesIO()
    df_diff.to_csv(buf3, index=False)
    buf3.seek(0)
    st.download_button("Download Differences CSV", buf3,
                       file_name="differences.csv", mime="text/csv")


footer()
