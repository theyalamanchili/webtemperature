import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
from io import BytesIO
from scipy.optimize import fsolve
import plotly.graph_objects as go

# -----------------------------------------------
# Background Styling
# -----------------------------------------------
def add_styles():
    st.markdown(
        """
        <style>
        .stApp { background-color: #f9f9f9; color: #111; }
        .footer { position: fixed; bottom: 0; width: 100%; text-align: center; font-size: 0.8rem; color: #555; }
        </style>
        """, unsafe_allow_html=True)

# -----------------------------------------------
# Login
# -----------------------------------------------
def login():
    logo = Image.open("MEEN_logo.png")
    st.image(logo, width=300)
    st.header("Please log in to continue")
    user = st.text_input("Username")
    pwd = st.text_input("Password", type="password")
    if st.button("Login"):
        creds = {"Guest":"GuestPass","Aditya":"Yalamanchili","Prabhakar":"Pagilla","admin":"adminpass"}
        if creds.get(user)==pwd:
            st.session_state.logged_in = True
            st.experimental_rerun()
        else:
            st.error("Invalid username or password.")

    if creds.get(user) == pwd:
        st.session_state.logged_in = True
        # try to re‐run the script immediately; if not available, just return
        if hasattr(st, "experimental_rerun"):
            st.experimental_rerun()
        else:
            return


# -----------------------------------------------
# Footer
# -----------------------------------------------
def footer():
    st.markdown(
        """
        <div class='footer'>Version α 0.1 | © 2025 Texas A&amp;M University</div>
        """, unsafe_allow_html=True)

# -----------------------------------------------
# 2D / 1D Solvers
# -----------------------------------------------
def solve_2d(k, rho, c, v, T0, Tinf, h, t, L, N):
    Y = t/2
    dT = T0 - Tinf
    beta = rho*c*v/(2*k)
    Bi   = h*Y/k
    def fz(z): return np.tan(z) - Bi/z
    eps = 1e-6
    z = np.zeros(N)
    z[0] = fsolve(fz,[eps, np.pi/2-eps])[0]
    odds = np.arange(1,2*N,2)
    for i in range(1, N):
        lo = odds[i-1]*np.pi/2 + eps
        hi = lo + np.pi - 2*eps
        z[i] = fsolve(fz,[lo,hi])[0]
    lam = z/Y
    a   = np.array([(2*dT*np.sin(z[i]))/(z[i]+np.sin(z[i])*np.cos(z[i])) for i in range(N)])
    # finer grid
    x = np.linspace(0, L, 600)
    y = np.linspace(-Y, Y, 300)
    X, Yg = np.meshgrid(x,y)
    Theta = sum(
        a[i]*np.exp((beta - np.sqrt(beta**2+lam[i]**2))*X)*np.cos(lam[i]*Yg)
        for i in range(N)
    )
    return x,y,X,Yg,(Tinf + Theta)

def solve_1d(k, rho, c, v, T0, Tinf, h, t, W, x):
    Y = t/2
    dT = T0 - Tinf
    beta = rho*c*v/(2*k)
    A = 2*W*Y; P = 2*W + 2*Y
    m2 = h*P/(k*A)
    mu = beta - np.sqrt(beta**2 + m2)
    return Tinf + dT*np.exp(mu*x)

# -----------------------------------------------
# App
# -----------------------------------------------
add_styles()
if 'logged_in' not in st.session_state:
    st.session_state.logged_in=False
if not st.session_state.logged_in:
    login()
else:
    st.title("Web Temperature Distribution Simulator")
    # BC selector & image
    st.sidebar.header("Boundary Condition")
    bc = st.sidebar.selectbox("Choose a case:",
        ["Free span convective cooling","Web over heated/cooled roller","Web in heating/cooling zone"]
    )
    if bc=="Free span convective cooling":
        st.image("BC1.png", caption="Free span convective cooling", use_column_width=True)
        st.sidebar.markdown("_Cooling on both surfaces by convection._")
    elif bc=="Web over heated/cooled roller":
        st.image("BC3.png", caption="Web over roller", use_column_width=True)
        st.sidebar.markdown("_Contact with roller at fixed T._")
    else:
        st.image("BC2.png", caption="Web in zone", use_column_width=True)
        st.sidebar.markdown("_Heating/cooling zone model._")
    # materials
    st.sidebar.header("Material Props")
    matlib={ 'Al':{'k':237,'rho':2700,'c':897}, 'Cu':{'k':401,'rho':8960,'c':385}, 'PET':{'k':0.2,'rho':1390,'c':1400} }
    mat=st.sidebar.selectbox("Material", list(matlib.keys())+['Custom'])
    if mat!='Custom': k,rho,c=matlib[mat].values(); st.sidebar.write(f"k={k}, ρ={rho}, c={c}")
    else:
        k   = st.sidebar.number_input("k [W/m·K]", 0.1,500.,0.2)
        rho = st.sidebar.number_input("ρ [kg/m³]", 100,20000,1400)
        c   = st.sidebar.number_input("c [J/kg·K]", 100,5000,1400)
    st.sidebar.markdown("---")
    # process
    st.sidebar.header("Process Params")
    v    = st.sidebar.number_input("v [m/s]", 0.01,10.,1.6)
    T0   = st.sidebar.number_input("T₀ [°C]", -50,500,200)
    Tinf = st.sidebar.number_input("T∞ [°C]", -50,200,25)
    h    = st.sidebar.number_input("h [W/m²·K]", 1,10000,100)
    t    = st.sidebar.number_input("thickness t [m]",1e-6,1e-2,0.001,step=1e-6,format="%.6f")
    W    = st.sidebar.number_input("Width W [m]",0.01,5.,1.0)
    L    = st.sidebar.number_input("Span L [m]",0.1,50.,10.0)
    N    = st.sidebar.slider("Terms N",5,50,20)
    if st.button("Compute"):
        if bc=="Free span convective cooling":
            x,y,X,Yg,T2=solve_2d(k,rho,c,v,T0,Tinf,h,t,L,N)
            T1=solve_1d(k,rho,c,v,T0,Tinf,h,t,W,x)
            st.session_state.update(x=x,y=y,X=X,Yg=Yg,T2=T2,T1=T1)
            st.session_state.ready=True
        else:
            st.warning("Solver coming soon.")
    if st.session_state.get('ready',False):
        x,y,X,Yg,T2,T1=(st.session_state[k] for k in ['x','y','X','Yg','T2','T1'])
        # dimless
        Bi_num= h*(t/2)/k
        Pe_num= v*L/(k/(rho*c))
        # contour
        st.subheader("2D Contour")
        show=st.checkbox("Show lines&labels")
        fig=go.Figure(go.Contour(
            z=T2,x=x,y=y,colorscale='Turbo',ncontours=60,
            contours=dict(showlines=show,showlabels=show,labelfont=dict(size=12,color='black'))
        ))
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',margin=dict(l=40,r=40,t=40,b=40))
        fig.update_xaxes(showgrid=False); fig.update_yaxes(showgrid=False)
        st.plotly_chart(fig,use_container_width=True)
        # Bi,Pe
        st.markdown(f"**Bi:** {Bi_num:.2f} &nbsp;&nbsp; **Pe:** {Pe_num:.1f}")
        st.markdown(
            """Bi: surface vs conduction. Low Bi→conduction-dominated.  
Pé: advection vs conduction. High Pe→advection-dominated."""
        )
        # profiles
        st.subheader("Profiles")
        idx_mid=np.argmin(np.abs(y))
        idx_top=np.argmin(np.abs(y-(t/2)))
        idx_bot=np.argmin(np.abs(y+(t/2)))
        Tavg=T2.mean(axis=0); Tmid=T2[idx_mid]; Ttop=T2[idx_top]; Tbot=T2[idx_bot]
        # styles
        line_styles={
            'avg':{'color':'blue','dash':'solid'},
            'mid':{'color':'green','dash':'dash'},
            'top':{'color':'red','dash':'dot'},
            'bot':{'color':'purple','dash':'dashdot'},
            '1d':{'color':'black','dash':'longdash'}
        }
        mark_sym={'avg':'circle','mid':'square','top':'triangle-up','bot':'triangle-down','1d':'x'}
        show_avg=st.checkbox("2D ⟨T⟩")
        show_mid=st.checkbox("Mid-plane")
        show_top=st.checkbox("Top surface")
        show_bot=st.checkbox("Bot surface")
        show_1d=st.checkbox("1D soln")
        if any([show_avg,show_mid,show_top,show_bot,show_1d]):
            fig2=go.Figure()
            if show_avg:
                fig2.add_trace(go.Scatter(x=x,y=Tavg,mode='lines+markers',
                    name='2D avg',line=line_styles['avg'],
                    marker=dict(symbol=mark_sym['avg'],color=line_styles['avg']['color'])
                ))
            if show_mid:
                fig2.add_trace(go.Scatter(x=x,y=Tmid,mode='lines+markers',
                    name='Mid-plane',line=line_styles['mid'],
                    marker=dict(symbol=mark_sym['mid'],color=line_styles['mid']['color'])
                ))
            if show_top:
                fig2.add_trace(go.Scatter(x=x,y=Ttop,mode='lines+markers',
                    name='Top surface',line=line_styles['top'],
                    marker=dict(symbol=mark_sym['top'],color=line_styles['top']['color'])
                ))
            if show_bot:
                fig2.add_trace(go.Scatter(x=x,y=Tbot,mode='lines+markers',
                    name='Bot surface',line=line_styles['bot'],
                    marker=dict(symbol=mark_sym['bot'],color=line_styles['bot']['color'])
                ))
            if show_1d:
                fig2.add_trace(go.Scatter(x=x,y=T1,mode='lines+markers',
                    name='1D soln',line=line_styles['1d'],
                    marker=dict(symbol=mark_sym['1d'],color=line_styles['1d']['color'])
                ))
            fig2.update_layout(paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',
                xaxis_title='x (m)',yaxis_title='T (°C)',legend=dict(title='Profile'))
            fig2.update_xaxes(showgrid=False); fig2.update_yaxes(showgrid=False)
            st.plotly_chart(fig2,use_container_width=True)
        # differences
        st.subheader("Differences")
        dm=st.checkbox("Mid−Top")
        da=st.checkbox("avg−1D")
        if any([dm,da]):
            fig3=go.Figure()
            if dm:
                fig3.add_trace(go.Scatter(x=x,y=Tmid-Ttop,mode='lines+markers',
                    name='Mid−Top',line=dict(color='orange',dash='dash'),marker=dict(symbol='circle',color='orange')
                ))
            if da:
                fig3.add_trace(go.Scatter(x=x,y=Tavg-T1,mode='lines+markers',
                    name='avg−1D',line=dict(color='brown',dash='dot'),marker=dict(symbol='square',color='brown')
                ))
            fig3.update_layout(paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',
                xaxis_title='x (m)',yaxis_title='ΔT (°C)',legend=dict(title='Δ Profiles'))
            fig3.update_xaxes(showgrid=False); fig3.update_yaxes(showgrid=False)
            st.plotly_chart(fig3,use_container_width=True)
        # download
        df=pd.DataFrame({'x':X.flatten(),'y':Yg.flatten(),'T':T2.flatten()})
        buf=BytesIO();df.to_csv(buf,index=False);buf.seek(0)
        st.download_button("Download CSV",buf,"temp_contour.csv","text/csv")
    footer()
