from pathlib import Path
import textwrap
from matplotlib import image as mpimg
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch
from matplotlib.colors import TwoSlopeNorm
from PIL import Image

from style import COL, CYCLE, clean_axes, panel_label
from cif_render import render_cif

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "outputs_review"

TARGET_LABELS = {
    "ch4_5p8": r"CH$_4$ 5.8 bar",
    "ch4_65": r"CH$_4$ 65 bar",
    "co2_0p015": r"CO$_2$ 0.015 bar",
    "co2_0p15": r"CO$_2$ 0.15 bar",
}
SPLIT_LABELS = {
    "chemistry_cluster_grouped": "Chemistry-cluster grouped",
    "random": "Random",
    "topology_grouped": "Topology grouped",
    "geo_cluster_grouped": "Geometry-cluster grouped",
}


def read_csv(folder, name):
    return pd.read_csv(DATA / folder / name)


def save_figure(fig, outdir, stem, vector=True, dpi=500, raster_pdf=False):
    outdir = OUT / outdir
    outdir.mkdir(parents=True, exist_ok=True)
    png_path = outdir / f"{stem}.png"
    pdf_path = outdir / f"{stem}.pdf"
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight", pad_inches=.06)
    if raster_pdf:
        # Appropriate for image-dominated galleries: fast, compact, and preserves the exact PNG appearance.
        with Image.open(png_path) as im:
            im.convert("RGB").save(pdf_path, "PDF", resolution=float(dpi))
    else:
        fig.savefig(pdf_path, bbox_inches="tight", pad_inches=.06)
    if vector:
        fig.savefig(outdir / f"{stem}.svg", bbox_inches="tight", pad_inches=.06)
    plt.close(fig)


def figure1():
    qdf = read_csv("main_figures", "Figure_5b_quintile_rates.csv")
    edf = read_csv("main_figures", "Figure_5a_enrichment_top30.csv")
    q = np.arange(1,6)
    def rates(name):
        x=qdf[qdf.descriptor==name].sort_values("quintile")
        return x.positive_surprise_rate.to_numpy()*100
    avaf, asa, density = rates("AVAf"), rates("ASA"), rates("Density")
    wanted=[("AVAf","low AVAf"),("AVAg","low AVAg"),("AVA","low AVA"),("gASA","low gASA"),
            ("POAVA","low POAVA"),("ASA","low ASA"),("Density","high density"),("vASA","low VASA")]
    or_vals=[]
    for col,label in wanted:
        row=edf[edf.group_col==col].sort_values("odds_ratio",ascending=False).iloc[0]
        or_vals.append((label,float(row.odds_ratio)))

    fig=plt.figure(figsize=(13.8,8.35),dpi=170)
    gs=fig.add_gridspec(2,2,width_ratios=[1.05,1.62],height_ratios=[1.0,1.0],wspace=.34,hspace=.52)

    # A — deliberately simple vector schematic: no pseudo-molecular structure.
    ax=fig.add_subplot(gs[:,0]); ax.axis("off"); panel_label(ax,"A",x=-.03,y=.99)
    ax.text(.08,.95,"Global pore descriptors can miss\nlocal adsorption environments",transform=ax.transAxes,
            fontsize=14.8,fontweight="bold",va="top",color=COL["ink"],linespacing=1.08)

    # Open-pore card.
    card1=FancyBboxPatch((.07,.59),.38,.25,boxstyle="round,pad=.018,rounding_size=.025",
                         ec="#6F98B8",fc="#F3F8FB",lw=1.45,transform=ax.transAxes)
    ax.add_patch(card1)
    # two smooth pore walls and sparse guest markers
    ax.add_patch(FancyBboxPatch((.115,.755),.29,.035,boxstyle="round,pad=.006,rounding_size=.018",ec="none",fc="#9AB5CA",transform=ax.transAxes))
    ax.add_patch(FancyBboxPatch((.115,.64),.29,.035,boxstyle="round,pad=.006,rounding_size=.018",ec="none",fc="#9AB5CA",transform=ax.transAxes))
    for xy in [(.16,.713),(.27,.700),(.36,.722)]:
        ax.add_patch(Circle(xy,.012,transform=ax.transAxes,fc="#5F6B76",ec="white",lw=.5))
    ax.text(.26,.615,"open pore",transform=ax.transAxes,ha="center",va="center",fontsize=11.2,fontweight="bold",color=COL["navy"])

    # Compact local-pocket card.
    card2=FancyBboxPatch((.55,.59),.38,.25,boxstyle="round,pad=.018,rounding_size=.025",
                         ec="#C77B70",fc="#FFF5F2",lw=1.45,transform=ax.transAxes)
    ax.add_patch(card2)
    # converging walls indicate confinement without pretending to be an atomistic structure
    ax.add_patch(FancyBboxPatch((.595,.725),.105,.045,boxstyle="round,pad=.006,rounding_size=.020",ec="none",fc="#DEA49A",transform=ax.transAxes))
    ax.add_patch(FancyBboxPatch((.78,.725),.105,.045,boxstyle="round,pad=.006,rounding_size=.020",ec="none",fc="#DEA49A",transform=ax.transAxes))
    ax.add_patch(FancyBboxPatch((.64,.635),.20,.038,boxstyle="round,pad=.006,rounding_size=.018",ec="none",fc="#DEA49A",transform=ax.transAxes))
    ax.add_patch(Circle((.74,.704),.026,transform=ax.transAxes,fc=COL["amber"],ec="white",lw=.8))
    for xy in [(.66,.712),(.82,.712),(.74,.660)]:
        ax.add_patch(Circle(xy,.009,transform=ax.transAxes,fc=COL["salmon"],ec="white",lw=.45))
        ax.add_patch(FancyArrowPatch(xy,(.735,.700),transform=ax.transAxes,arrowstyle="->",mutation_scale=8,lw=.9,color=COL["salmon"],alpha=.8))
    ax.text(.74,.615,"compact pocket",transform=ax.transAxes,ha="center",va="center",fontsize=11.2,fontweight="bold",color=COL["salmon"])

    ax.text(.26,.545,"larger accessible region\nweaker local field",transform=ax.transAxes,ha="center",va="top",fontsize=10.3,color=COL["muted"])
    ax.text(.74,.545,"stronger confinement\nand local interactions",transform=ax.transAxes,ha="center",va="top",fontsize=10.3,color=COL["muted"])
    ax.add_patch(FancyArrowPatch((.28,.45),(.72,.45),transform=ax.transAxes,arrowstyle="-|>",mutation_scale=15,lw=1.5,color=COL["salmon"]))
    ax.text(.50,.405,"similar global descriptors can hide different local physics",transform=ax.transAxes,
            ha="center",fontsize=10.5,fontweight="bold",color=COL["salmon"])
    box=FancyBboxPatch((.075,.12),.85,.20,boxstyle="round,pad=.022,rounding_size=.02",ec=COL["teal"],fc=COL["pale_teal"],lw=1.3,transform=ax.transAxes)
    ax.add_patch(box)
    ax.text(.50,.275,"Working hypothesis",transform=ax.transAxes,ha="center",fontsize=11.8,fontweight="bold",color=COL["teal"])
    ax.text(.50,.205,"Large positive residuals may flag local adsorption physics\nnot fully represented by global pore descriptors; they do not\nby themselves prove a unique microscopic mechanism.",transform=ax.transAxes,
            ha="center",va="center",fontsize=10.0,color=COL["ink"],linespacing=1.18)

    # B
    ax1=fig.add_subplot(gs[0,1]); panel_label(ax1,"B",x=-.09,y=1.06)
    ax1.plot(q,avaf,marker="o",ms=7,lw=2.4,color=COL["teal"],label="Accessible-volume fraction")
    ax1.plot(q,asa,marker="s",ms=6.5,lw=2.4,color=COL["blue"],label="Accessible surface area")
    ax1.plot(q,density,marker="^",ms=7.5,lw=2.4,color=COL["amber"],label="Density")
    ax1.set_xticks(q,[f"Q{i}" for i in q]); ax1.set_ylim(0,34)
    ax1.set_xlabel("Descriptor quintile (low → high)"); ax1.set_ylabel("Positive-surprise rate (%)")
    ax1.set_title("A monotonic chemical-regime signal across pore descriptors",loc="left",pad=10)
    ax1.legend(loc="upper center",ncol=3,bbox_to_anchor=(.54,1.01))
    # Explicitly separate the two Q1 labels so they never collide.
    ax1.annotate(f"{avaf[0]:.1f}%",(1,avaf[0]),xytext=(-14,15),textcoords="offset points",ha="center",fontweight="bold",fontsize=10.5,color=COL["teal"])
    ax1.annotate(f"{asa[0]:.1f}%",(1,asa[0]),xytext=(18,-22),textcoords="offset points",ha="center",fontweight="bold",fontsize=10.5,color=COL["blue"])
    ax1.annotate(f"{density[-1]:.1f}%",(5,density[-1]),xytext=(-3,12),textcoords="offset points",ha="center",fontweight="bold",fontsize=10.5,color=COL["amber"])
    clean_axes(ax1)

    # C
    ax2=fig.add_subplot(gs[1,1]); panel_label(ax2,"C",x=-.09,y=1.06)
    labels=[x[0] for x in or_vals]; vals=np.array([x[1] for x in or_vals]); y=np.arange(len(labels))[::-1]
    colors=[COL["salmon"] if "density" in l else COL["ocean"] for l in labels]
    ax2.barh(y,vals,color=colors,height=.54,edgecolor="white",linewidth=.6)
    ax2.axvline(1,ls="--",lw=1,color=COL["muted"])
    ax2.set_yticks(y,labels); ax2.set_xlim(0,6.25)
    ax2.set_xlabel("Odds ratio for positive surprise")
    ax2.set_title("Exceptional underprediction is enriched in compact regimes",loc="left",pad=10)
    for yy,val in zip(y,vals): ax2.text(val+.08,yy,f"{val:.2f}",va="center",fontsize=10.3)
    clean_axes(ax2)

    # No overall figure title: manuscript caption provides the figure-level statement.
    fig.subplots_adjust(left=.075,right=.985,top=.965,bottom=.105)
    save_figure(fig,"main","Figure_1_low_accessibility_regime")


FROZEN=DATA/"frozen_structures"


def _show_structure_png(ax, png_path, title, subtitle="", title_fontsize=12.0, subtitle_fontsize=9.3):
    img=mpimg.imread(png_path)
    ax.imshow(img, interpolation="lanczos", resample=True)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_facecolor("white")
    ax.text(0.00, 1.03, title, transform=ax.transAxes, ha="left", va="bottom", fontsize=title_fontsize, fontweight="bold", color=COL["ink"])
    if subtitle:
        ax.text(0.00, -0.05, subtitle, transform=ax.transAxes, ha="left", va="top", fontsize=subtitle_fontsize, color=COL["muted"])


def _frozen_main_png(stem):
    mapping={
        "OSAVUA": FROZEN/"figure_ready"/"main"/"Figure2_OSAVUA.png",
        "VAGTAA01": FROZEN/"figure_ready"/"main"/"Figure2_VAGTAA01.png",
        "VAGTAA": FROZEN/"figure_ready"/"main"/"Figure2_VAGTAA.png",
    }
    return mapping[stem]


def _frozen_si_png(index1):
    return FROZEN/"figure_ready"/"si"/f"S8_{index1:02d}.png"


def _find_cif(token):
    token=token.lower().replace("-","_").replace(".","_")
    for p in (DATA/"cifs").glob("*.cif"):
        n=p.stem.lower().replace("-","_").replace(".","_")
        if token in n: return p
    raise FileNotFoundError(token)


def figure2():
    cb=pd.read_csv(DATA/"casebook"/"paper_b_candidate_casebook_complete.csv")
    ids=["DB12-OSAVUA_clean","DB12-VAGTAA01_clean","DB12-VAGTAA_clean","DB1-Zn2O8-BTC_A-irmof8_A_No11"]
    rows=[]
    for ident in ids:
        r=cb[cb.display_id==ident].iloc[0]; rows.append(r)

    fig=plt.figure(figsize=(13.9,9.0),dpi=170)
    gs=fig.add_gridspec(2,3,height_ratios=[1.02,1.0],width_ratios=[1,1,1],wspace=.23,hspace=.43)
    display=["OSAVUA","VAGTAA01","VAGTAA"]
    for i,name in enumerate(display):
        ax=fig.add_subplot(gs[0,i])
        r=rows[i]
        sub=f"reference {r.y_true:.2f}  |  prediction {r.y_pred:.2f}  |  residual {r.residual:+.2f} mmol g$^{{-1}}$"
        png=_frozen_main_png(name)
        if png.exists():
            _show_structure_png(ax,png,name,sub,title_fontsize=12.7,subtitle_fontsize=9.8)
        else:
            ax.remove(); ax=fig.add_subplot(gs[0,i],projection="3d")
            token=name.lower()+"_clean" if name!="VAGTAA01" else "vagtaa01_clean"
            render_cif(ax,_find_cif(token),name,sub,view=(18,-63))
        # Keep panel letter separated from the structure name.
        panel_label(ax,chr(65+i),x=-.075,y=1.095)

    # D
    ax=fig.add_subplot(gs[1,:2]); panel_label(ax,"D",x=-.075,y=1.07)
    names=["OSAVUA","VAGTAA01","VAGTAA",r"Zn$_2$O$_8$–BTC case"]
    y=np.arange(4)[::-1]
    for yy,r in zip(y,rows):
        ax.plot([r.y_pred,r.y_true],[yy,yy],lw=5,color="#DDE4E8",solid_capstyle="round",zorder=1)
        ax.scatter(r.y_pred,yy,s=76,color=COL["blue"],edgecolor="white",linewidth=.7,zorder=3,label="Prediction" if yy==y[0] else None)
        ax.scatter(r.y_true,yy,s=76,color=COL["coral"],edgecolor="white",linewidth=.7,zorder=3,label="Reference" if yy==y[0] else None)
        ax.text(r.y_true+.12,yy,f"{r.residual:+.2f}",va="center",fontsize=10.6,fontweight="bold",color=COL["coral"])
    ax.set_yticks(y,names); ax.set_xlim(0,9.6)
    ax.set_xlabel(r"CO$_2$ uptake at 0.15 bar (mmol g$^{-1}$)")
    ax.set_title("The strongest calibrated failures are chemically large",loc="left",pad=10)
    # Legend is inside the otherwise empty upper-right region, away from the x-axis caption.
    ax.legend(loc="upper right",ncol=2,bbox_to_anchor=(.99,.99)); clean_axes(ax)

    # E — equal-height, symmetric stacked context boxes.
    ax=fig.add_subplot(gs[1,2]); ax.axis("off"); panel_label(ax,"E",x=-.075,y=1.07)
    ax.text(.06,.98,"Independent structural context",transform=ax.transAxes,va="top",fontsize=13.4,fontweight="bold")
    def textbox(y0,edge,face,title,body):
        h=.245
        p=FancyBboxPatch((.045,y0),.91,h,boxstyle="round,pad=.018,rounding_size=.018",ec=edge,fc=face,lw=1.25,transform=ax.transAxes)
        ax.add_patch(p)
        ax.text(.085,y0+h-.045,title,transform=ax.transAxes,va="top",fontsize=10.4,fontweight="bold",color=edge)
        ax.text(.085,y0+h-.108,body,transform=ax.transAxes,va="top",fontsize=8.95,color=COL["ink"],linespacing=1.16)
    textbox(.665,COL["teal"],COL["pale_teal"],"Verified literature annotation",
            "Villajos et al., SI Table S4:\nVAGTAA01 = Mg / OMS Yes;\nOSAVUA = Ca / OMS Yes.")
    textbox(.365,COL["blue"],COL["pale_blue"],"VAGTAA numerical context",
            "Exact independent source for the previously quoted\nLCD, void fraction and density remains pending.\nThose numbers are not shown here.")
    textbox(.065,COL["coral"],COL["pale_coral"],"Interpretation boundary",
            "Local-site / confinement remains a hypothesis.\nThese annotations do not independently validate\nthe present CO$_2$ labels or prove a unique mechanism.")

    # No figure-level title; leave that work to the manuscript caption.
    fig.subplots_adjust(left=.06,right=.985,top=.955,bottom=.10)
    save_figure(fig,"main","Figure_2_structure_resolved_chemistry",vector=False)


def figure3():
    counts=read_csv("main_figures","Figure_3a_candidate_counts.csv")
    risk=read_csv("main_figures","Figure_3b_risk_gate_tradeoff.csv")
    stab=read_csv("main_figures","Figure_3c_stability.csv")
    dm=read_csv("main_figures","Figure_3d_decision_map_sample.csv")
    fig,axes=plt.subplots(2,2,figsize=(13.6,9.0),dpi=170)
    ax=axes[0,0]; panel_label(ax,"A")
    vals=counts["count"].astype(float).to_numpy(); labels=counts["class"].tolist(); y=np.arange(len(vals))[::-1]
    ax.barh(y,vals,height=.52,color=[COL["blue"],COL["sky"],COL["teal"],COL["purple"],COL["amber"],COL["red"]],edgecolor="white")
    ax.set_xscale("log"); ax.set_xlim(.6,max(vals)*1.8); ax.set_yticks(y,labels)
    for yy,v in zip(y,vals):
        if v>0: ax.text(v*1.07,yy,f"{int(v):,}",va="center",fontsize=10.2)
        else: ax.text(.68,yy,"0",va="center",fontsize=10.2,fontweight="bold")
    ax.set_xlabel("Candidate count (log scale)"); ax.set_title("Representative held-out decision classes",loc="left",pad=9); clean_axes(ax)

    ax=axes[0,1]; panel_label(ax,"B")
    short=["Predicted","Confident","Adaptive confident"]; x=np.arange(3); w=.24
    false=risk.false_elite_rate.to_numpy(float)*100; recall=risk.elite_recall.to_numpy(float)*100
    false_plot=false.copy(); false_plot[2]=np.nan
    ax.bar(x-w/2,false_plot,w,color=COL["red"],edgecolor="white",label="False-promotion rate")
    ax.bar(x+w/2,recall,w,color=COL["teal"],edgecolor="white",label="Elite recall")
    for i,v in enumerate(false_plot):
        if np.isfinite(v): ax.text(i-w/2,v+1.1,f"{v:.1f}%",ha="center",fontsize=10.0)
    ax.text(2-w/2,1.2,"N/A\n(no selected\ncandidates)",ha="center",va="bottom",fontsize=9.0,color=COL["red"],fontweight="bold")
    for i,v in enumerate(recall): ax.text(i+w/2,v+1.1,f"{v:.1f}%",ha="center",fontsize=10.0)
    ax.set_xticks(x,short); ax.set_ylabel("Percent"); ax.set_ylim(0,58); ax.set_title("Risk–recall operating points",loc="left",pad=9); ax.legend(loc="upper right"); clean_axes(ax)

    ax=axes[1,0]; panel_label(ax,"C")
    class_colors={"positive_conformal_anomaly":COL["coral"],"negative_conformal_anomaly":COL["blue"],"possible_elite":COL["sky"],"uncertain_or_nonelite":"#D6DCE1"}
    for cls,sub in dm.groupby("decision_class_main"):
        label={"positive_conformal_anomaly":"positive surprise","negative_conformal_anomaly":"negative surprise","possible_elite":"possible elite","uncertain_or_nonelite":"ordinary / other"}.get(cls,cls)
        ax.scatter(sub.y_pred,sub.y_true,s=27,c=class_colors.get(cls,COL["grey"]),alpha=.76,edgecolor="white",linewidth=.25,label=label)
    mn=min(dm.y_pred.min(),dm.y_true.min()); mx=max(dm.y_pred.max(),dm.y_true.max())
    ax.plot([mn,mx],[mn,mx],"--",lw=1.2,color=COL["muted"])
    ax.set_xlabel(r"Predicted CO$_2$ uptake at 0.15 bar (mmol g$^{-1}$)")
    ax.set_ylabel(r"Reference CO$_2$ uptake at 0.15 bar (mmol g$^{-1}$)")
    ax.set_title("Uncertainty-aware decision map",loc="left",pad=9)
    ax.text(.98,.04,"display sample: n = 900\nadaptive 90% bounds",transform=ax.transAxes,ha="right",va="bottom",fontsize=9.5,color=COL["muted"])
    ax.legend(loc="upper left",fontsize=9.0); clean_axes(ax)

    ax=axes[1,1]; panel_label(ax,"D")
    st=stab.copy(); x=np.arange(len(st)); w=.26
    ax.bar(x-w/2,st.jaccard_pred_mean,w,color=COL["blue"],edgecolor="white",label="Predicted top-5%")
    ax.bar(x+w/2,st.jaccard_lcb_mean,w,color=COL["teal"],edgecolor="white",label="Adaptive 90% LCB top-5%")
    ax.set_xticks(x,[SPLIT_LABELS.get(s,s) for s in st.split_type],rotation=20,ha="right")
    ax.set_ylabel("Mean Jaccard overlap"); ax.set_title("Shortlist overlap across split realizations",loc="left",pad=9)
    ax.legend(loc="upper right"); clean_axes(ax)
    fig.subplots_adjust(left=.085,right=.985,top=.95,bottom=.105,wspace=.34,hspace=.38)
    save_figure(fig,"main","Figure_3_calibrated_screening_decisions")


def _matrix(df):
    if "split_type" in df.columns:
        df=df.set_index("split_type")
    return df


def figure4():
    cov=_matrix(read_csv("main_figures","Figure_2a_coverage_deviation_by_split_target.csv"))
    widths=_matrix(read_csv("main_figures","Figure_2b_interval_width_by_split_target.csv"))
    trade=read_csv("main_figures","Figure_2c_rmse_coverage_screening_tradeoff.csv")
    adapt=read_csv("main_figures","Figure_2d_standard_vs_adaptive_conformal.csv")
    cov95=cov.astype(float)-0.05
    fig,axes=plt.subplots(2,2,figsize=(14.8,9.4),dpi=170)

    # A
    ax=axes[0,0]; panel_label(ax,"A",x=-.12,y=1.07)
    maxabs=max(.005,float(np.abs(cov95.to_numpy()).max()))
    im=ax.imshow(cov95.to_numpy(),aspect="auto",cmap="RdBu_r",norm=TwoSlopeNorm(vcenter=0,vmin=-maxabs,vmax=maxabs))
    ax.set_xticks(range(cov95.shape[1]),[TARGET_LABELS.get(c,c) for c in cov95.columns],rotation=20,ha="right")
    ax.set_yticks(range(cov95.shape[0]),[SPLIT_LABELS.get(s,s) for s in cov95.index])
    for i in range(cov95.shape[0]):
        for j in range(cov95.shape[1]): ax.text(j,i,f"{cov95.iloc[i,j]:+.3f}",ha="center",va="center",fontsize=10.4,fontweight="medium")
    cb=fig.colorbar(im,ax=ax,fraction=.046,pad=.04); cb.ax.set_title("Coverage − 0.95",fontsize=9.8,pad=7); cb.ax.tick_params(labelsize=9.7)
    ax.set_title("Coverage deviation from the 95% target",loc="left",pad=10); ax.set_xlabel("Adsorption target"); ax.set_ylabel("Split"); ax.grid(False)

    # B
    ax=axes[0,1]; panel_label(ax,"B",x=-.12,y=1.07)
    im=ax.imshow(widths.to_numpy(float),aspect="auto",cmap="YlGnBu")
    ax.set_xticks(range(widths.shape[1]),[TARGET_LABELS.get(c,c) for c in widths.columns],rotation=20,ha="right")
    ax.set_yticks(range(widths.shape[0]),[SPLIT_LABELS.get(s,s) for s in widths.index])
    for i in range(widths.shape[0]):
        for j in range(widths.shape[1]): ax.text(j,i,f"{widths.iloc[i,j]:.2f}",ha="center",va="center",fontsize=10.4,color=COL["ink"],fontweight="medium")
    cb=fig.colorbar(im,ax=ax,fraction=.046,pad=.04); cb.ax.set_title("95% width",fontsize=9.8,pad=7); cb.ax.tick_params(labelsize=9.7)
    ax.set_title("Interval-width cost of extrapolation",loc="left",pad=10); ax.set_xlabel("Adsorption target"); ax.set_ylabel("Split"); ax.grid(False)

    # C
    ax=axes[1,0]; panel_label(ax,"C",x=-.12,y=1.07)
    for k,(split,sub) in enumerate(trade.groupby("split_type")):
        ax.scatter(sub.rmse,sub.coverage,s=70+80*(sub.width/sub.width.max()),color=CYCLE[k],alpha=.87,edgecolor="white",linewidth=.7,label=SPLIT_LABELS.get(split,split))
    ax.axhline(.95,ls="--",lw=1.15,color=COL["muted"])
    ax.text(.98,.06,"nominal 0.95",transform=ax.transAxes,ha="right",fontsize=9.7,color=COL["muted"])
    ax.set_xlabel(r"Mean RMSE (mmol g$^{-1}$)"); ax.set_ylabel("Empirical coverage")
    ax.set_title("Accuracy alone does not certify decisions",loc="left",pad=10)
    ax.legend(loc="upper left",fontsize=9.2)
    ax.text(.98,.91,r"marker size $\propto$ interval width",transform=ax.transAxes,ha="right",fontsize=9.5,color=COL["muted"],
            bbox=dict(boxstyle="round,pad=.25",fc="white",ec=COL["grid"],lw=.8))
    clean_axes(ax)

    # D
    ax=axes[1,1]; panel_label(ax,"D",x=-.12,y=1.07)
    x=np.arange(len(adapt)); w=.27
    ax.bar(x-w/2,adapt.standard_width,w,color=COL["blue"],edgecolor="white",label="Standard width")
    ax.bar(x+w/2,adapt.adaptive_width,w,color=COL["teal"],edgecolor="white",label="Adaptive width")
    for i,(a,b) in enumerate(zip(adapt.standard_width,adapt.adaptive_width)):
        ax.text(i-w/2,a+.05,f"{a:.2f}",ha="center",fontsize=9.6); ax.text(i+w/2,b+.05,f"{b:.2f}",ha="center",fontsize=9.6)
    ax.set_xticks(x,[SPLIT_LABELS.get(s,s) for s in adapt.split_type],rotation=18,ha="right")
    ax.set_ylabel(r"Mean 95% interval width (mmol g$^{-1}$)")
    ax2=ax.twinx(); ax2.plot(x,adapt.standard_coverage,marker="o",ms=6.5,ls="--",lw=1.7,color=COL["navy"],label="Standard coverage")
    ax2.plot(x,adapt.adaptive_coverage,marker="s",ms=6.5,ls="-.",lw=1.8,color=COL["salmon"],label="Adaptive coverage")
    ax2.axhline(.95,ls=":",lw=1.1,color=COL["muted"],alpha=.85); ax2.set_ylabel("Empirical coverage")
    h1,l1=ax.get_legend_handles_labels(); h2,l2=ax2.get_legend_handles_labels()
    ax.legend(h1+h2,l1+l2,loc="lower center",bbox_to_anchor=(.50,1.015),ncol=2,fontsize=9.0,columnspacing=1.2,handlelength=1.8)
    ax.set_title("Adaptive intervals reduce width",loc="left",pad=35); clean_axes(ax)
    fig.subplots_adjust(left=.09,right=.95,top=.95,bottom=.115,wspace=.56,hspace=.52)
    save_figure(fig,"main","Figure_4_transfer_uncertainty")


def si1():
    df=read_csv("si_figures","Figure_S1_target_detection_summary.csv")
    labs=[TARGET_LABELS.get(x,x) for x in df.target_key]
    fig,axes=plt.subplots(1,2,figsize=(12.8,5.1),dpi=170)
    ax=axes[0]; panel_label(ax,"A")
    x=np.arange(len(df)); ax.bar(x,df["mean"],width=.55,yerr=df["std"],color=CYCLE[:len(df)],edgecolor="white",capsize=4)
    ax.set_xticks(x,labs,rotation=20,ha="right"); ax.set_ylabel(r"Uptake (mmol g$^{-1}$)"); ax.set_title("Target means and spread",loc="left",pad=9); clean_axes(ax)
    ax=axes[1]; panel_label(ax,"B")
    ax.scatter(df["median"],df["max"],s=115,c=CYCLE[:len(df)],edgecolor="white",linewidth=.7)
    for i,r in df.iterrows(): ax.annotate(labs[i],(r["median"],r["max"]),xytext=(6,6),textcoords="offset points",fontsize=9.7)
    ax.set_xlabel(r"Median uptake (mmol g$^{-1}$)"); ax.set_ylabel(r"Maximum uptake (mmol g$^{-1}$)"); ax.set_title("Target dynamic range",loc="left",pad=9); clean_axes(ax)
    fig.subplots_adjust(left=.085,right=.98,top=.92,bottom=.21,wspace=.33); save_figure(fig,"si","Figure_S1_target_detection_summary")


def si2():
    fam=read_csv("si_figures","Figure_S2_feature_redundancy.csv"); sp=read_csv("si_figures","Figure_S2_split_diagnostics.csv")
    fig,axes=plt.subplots(1,2,figsize=(13.2,5.15),dpi=170)
    ax=axes[0]; panel_label(ax,"A")
    y=np.arange(len(fam))[::-1]; colors=[COL["blue"] if x else COL["grey"] for x in fam.included_in_model_jobs]
    ax.barh(y,fam.n_columns,height=.53,color=colors,edgecolor="white"); ax.set_yticks(y,[x.replace("_"," ") for x in fam.feature_family])
    for yy,(_,r) in zip(y,fam.iterrows()): ax.text(r.n_columns+.5,yy,"included" if r.included_in_model_jobs else "excluded duplicate",va="center",fontsize=9.4,color=COL["ink"])
    ax.set_xlabel("Number of columns"); ax.set_title("Feature-family redundancy / execution status",loc="left",pad=9); clean_axes(ax)
    ax=axes[1]; panel_label(ax,"B")
    x=np.arange(len(sp)); colors=[COL["teal"] if s=="available" else COL["coral"] for s in sp.status]
    ax.bar(x,np.ones(len(sp)),width=.58,color=colors,edgecolor="white")
    ax.set_xticks(x,[SPLIT_LABELS.get(s,s) for s in sp.split_type],rotation=20,ha="right"); ax.set_ylim(0,1.1); ax.set_yticks([])
    for i,r in sp.iterrows(): ax.text(i,.50,"available" if r.status=="available" else "unavailable",ha="center",va="center",fontweight="bold",color="white",fontsize=9.7)
    ax.set_title("Split availability diagnostics",loc="left",pad=9); clean_axes(ax,grid=False)
    fig.subplots_adjust(left=.11,right=.98,top=.92,bottom=.21,wspace=.36); save_figure(fig,"si","Figure_S2_feature_split_diagnostics")


def _heatmap_si(fname,stem,title,cbar_label):
    df=read_csv("si_figures",fname).set_index("target_key")
    fig,ax=plt.subplots(figsize=(8.0,5.9),dpi=170)
    im=ax.imshow(df.to_numpy(float),aspect="auto",cmap="Blues")
    ax.set_xticks(range(df.shape[1]),[SPLIT_LABELS.get(c,c) for c in df.columns],rotation=20,ha="right")
    ax.set_yticks(range(df.shape[0]),[TARGET_LABELS.get(i,i) for i in df.index])
    for i in range(df.shape[0]):
        for j in range(df.shape[1]): ax.text(j,i,f"{df.iloc[i,j]:.2f}",ha="center",va="center",fontsize=11.2,fontweight="medium")
    cb=fig.colorbar(im,ax=ax,fraction=.05,pad=.04); cb.set_label(cbar_label); cb.ax.tick_params(labelsize=10.2)
    ax.set_xlabel("Split"); ax.set_ylabel("Adsorption target"); ax.grid(False)
    # Per request, no figure title for S3/S4.
    fig.subplots_adjust(left=.20,right=.91,top=.96,bottom=.19); save_figure(fig,"si",stem)


def si3(): _heatmap_si("Figure_S3_full_rmse_by_target_split.csv","Figure_S3_full_rmse_by_target_split","",r"RMSE (mmol g$^{-1}$)")
def si4(): _heatmap_si("Figure_S4_full_interval_width_by_target_split.csv","Figure_S4_full_interval_width_by_target_split","",r"Mean 95% interval width (mmol g$^{-1}$)")


def si5():
    df=read_csv("si_figures","Figure_S5_coverage_distribution.csv"); x=df.test_empirical_coverage.dropna()
    fig,ax=plt.subplots(figsize=(7.7,5.3),dpi=170); ax.hist(x,bins=30,color=COL["teal"],edgecolor="white",linewidth=.55)
    ax.axvline(.95,ls="--",lw=1.4,color=COL["ink"],label="Nominal 95%")
    ax.axvline(x.mean(),ls=":",lw=1.7,color=COL["amber"],label=f"Mean = {x.mean():.3f}")
    ax.set_xlabel("Saved empirical coverage"); ax.set_ylabel("Completed jobs"); ax.legend(); clean_axes(ax)
    fig.subplots_adjust(left=.13,right=.97,top=.96,bottom=.15); save_figure(fig,"si","Figure_S5_coverage_distribution")


def si6():
    df=read_csv("si_figures","Figure_S6_filter_sensitivity.csv"); labels=[x.replace("_"," ") for x in df["filter"]]; y=np.arange(len(df))[::-1]
    fig,axes=plt.subplots(1,2,figsize=(13.4,5.8),dpi=170)
    ax=axes[0]; panel_label(ax,"A"); ax.barh(y,df.n_rows,height=.52,color=COL["blue"],edgecolor="white"); ax.set_yticks(y,labels)
    for yy,v in zip(y,df.n_rows):
        if v>0: ax.text(v-max(df.n_rows)*.012,yy,f"{int(v):,}",va="center",ha="right",fontsize=9.4,color="white",fontweight="bold")
        else: ax.text(.08,yy,"0",va="center",fontsize=9.4,fontweight="bold")
    ax.set_xlabel("Rows retained"); ax.set_title("Rows retained by QC / filter",loc="left",pad=9); clean_axes(ax)

    ax=axes[1]; panel_label(ax,"B"); rates=df.rate_positive_anomaly_90.astype(float)*100
    plot_rates=rates.copy(); plot_rates[df.n_rows.eq(0)]=np.nan
    ax.barh(y,plot_rates,height=.50,color=COL["amber"],edgecolor="white"); ax.set_yticks(y,labels); ax.set_xlabel("90% positive-surprise rate (%)"); ax.set_title("Positive-surprise rate after filtering",loc="left",pad=9)
    vmax=max(1.0,float(np.nanmax(plot_rates))); ax.set_xlim(0,vmax*1.18)
    for yy,v,n in zip(y,plot_rates,df.n_rows):
        if np.isfinite(v): ax.text(v+.12,yy,f"{v:.1f}%",va="center",fontsize=9.4)
        elif n==0: ax.text(vmax*.60,yy,"N/A — empty subset",va="center",ha="center",fontsize=9.4,fontweight="bold",color=COL["coral"],
                             bbox=dict(boxstyle="round,pad=.22",fc=COL["pale_coral"],ec=COL["coral"],lw=.8))
    clean_axes(ax); ax.grid(False, axis="y")
    fig.subplots_adjust(left=.19,right=.98,top=.92,bottom=.13,wspace=.54); save_figure(fig,"si","Figure_S6_filter_sensitivity")


def si7():
    proc=read_csv("si_figures","Figure_S7_process_triage_source.csv")
    gal=read_csv("si_figures","Figure_S8_casebook_gallery_source.csv")
    # The former external-support status panel is intentionally removed.
    fig,axes=plt.subplots(1,2,figsize=(11.6,5.0),dpi=170)
    ax=axes[0]; panel_label(ax,"A")
    pc=proc.process_proxy_status.value_counts(); y=np.arange(len(pc))[::-1]
    ax.barh(y,pc.values,height=.46,color=[COL["amber"] if "warning" in str(s).lower() else COL["teal"] for s in pc.index],edgecolor="white")
    ax.set_yticks(y,[s.replace("-"," ") for s in pc.index]); ax.set_xlabel("Curated casebook count"); ax.set_title("Inherited process-proxy labels",loc="left",pad=9)
    ax.text(.02,.04,"Heuristic labels; not external validation.",transform=ax.transAxes,fontsize=9.4,color=COL["muted"]); clean_axes(ax)

    ax=axes[1]; panel_label(ax,"B")
    cc=gal.paper_b_case_class.value_counts(); y=np.arange(len(cc))[::-1]
    ax.barh(y,cc.values,height=.46,color=COL["purple"],edgecolor="white"); ax.set_yticks(y,[textwrap.fill(s,22) for s in cc.index]); ax.set_xlabel("Curated casebook count"); ax.set_title("Casebook class balance",loc="left",pad=9)
    ax.text(.02,.04,"Selected 14-case gallery; not population prevalence.",transform=ax.transAxes,fontsize=9.4,color=COL["muted"]); clean_axes(ax)
    fig.subplots_adjust(left=.13,right=.985,top=.92,bottom=.16,wspace=.58); save_figure(fig,"si","Figure_S7_external_process_casebook_diagnostics")


def si8():
    gal=read_csv("si_figures","Figure_S8_casebook_gallery_source.csv")
    cifs=sorted((DATA/"cifs").glob("*.cif"))
    fig=plt.figure(figsize=(14.4,14.8),dpi=170); gs=fig.add_gridspec(4,4,wspace=.12,hspace=.32)
    used_fallback=False
    for i in range(16):
        if i>=len(gal):
            ax=fig.add_subplot(gs[i//4,i%4]); ax.axis("off"); continue
        r=gal.iloc[i]
        case=str(r.paper_b_case_class).replace("possible but not confident elite","possible elite")
        title=f"{i+1}. {textwrap.fill(str(r.display_id),18)}"
        sub=f"{case}\nresidual {r.residual:+.2f} mmol g$^{{-1}}$"
        png=_frozen_si_png(i+1)
        if png.exists():
            ax=fig.add_subplot(gs[i//4,i%4])
            _show_structure_png(ax,png,title,sub,title_fontsize=8.7,subtitle_fontsize=7.8)
        else:
            used_fallback=True
            ax=fig.add_subplot(gs[i//4,i%4],projection="3d")
            if i >= len(cifs):
                ax.set_axis_off(); ax.text2D(.5,.5,"CIF missing",transform=ax.transAxes,ha="center")
            else:
                render_cif(ax,cifs[i],title,sub,view=(18,-63),title_fontsize=8.7,subtitle_fontsize=7.4)
    fig.suptitle("Full structural casebook gallery — review regeneration",fontsize=16.2,fontweight="bold",y=.985)
    footer=("Portable CIF fallback used because the original frozen OVITO/Blender PNG masters were not supplied. "
            "This review output is not the locked publication artwork." if used_fallback else
            "Frozen OVITO/Blender structure renders used; this remains a review regeneration, not the locked publication PDF.")
    fig.text(.5,.012,footer,ha="center",fontsize=8.8,color=COL["muted"])
    fig.subplots_adjust(left=.025,right=.985,top=.945,bottom=.055)
    save_figure(fig,"si","Figure_S8_full_structural_casebook_gallery",vector=False,dpi=400,raster_pdf=True)


def toc():
    qdf=read_csv("main_figures","Figure_5b_quintile_rates.csv"); vals=qdf[qdf.descriptor=="AVAf"].sort_values("quintile").positive_surprise_rate.to_numpy()*100
    fig,ax=plt.subplots(figsize=(11.5,2.75),dpi=180); ax.axis("off")
    b1=FancyBboxPatch((.02,.20),.22,.62,boxstyle="round,pad=.02",ec=COL["blue"],fc=COL["pale_blue"],lw=1.4,transform=ax.transAxes); ax.add_patch(b1)
    for xy in [(.055,.35),(.09,.58),(.19,.38),(.18,.65),(.075,.68)]: ax.add_patch(Circle(xy,.011,transform=ax.transAxes,fc=COL["muted"],ec="none"))
    ax.text(.13,.10,"global porosity",transform=ax.transAxes,ha="center",fontsize=9,color=COL["muted"])
    b2=FancyBboxPatch((.31,.20),.22,.62,boxstyle="round,pad=.02",ec=COL["coral"],fc=COL["pale_coral"],lw=1.4,transform=ax.transAxes); ax.add_patch(b2)
    for xy in [(.345,.34),(.37,.62),(.49,.36),(.47,.64),(.35,.72),(.50,.72)]: ax.add_patch(Circle(xy,.013,transform=ax.transAxes,fc="#D98279",ec="none"))
    ax.add_patch(Circle((.42,.50),.021,transform=ax.transAxes,fc=COL["orange"],ec="white",lw=.6)); ax.text(.42,.10,"local high-affinity pocket",transform=ax.transAxes,ha="center",fontsize=9,color=COL["muted"])
    ax.add_patch(FancyArrowPatch((.25,.50),(.30,.50),transform=ax.transAxes,arrowstyle="->",mutation_scale=14,lw=1.5,color=COL["coral"]))
    ax.text(.64,.67,"CALIBRATED FAILURE",transform=ax.transAxes,ha="center",fontsize=12,fontweight="bold",color=COL["coral"])
    ax.text(.64,.45,r"highlights compact MOFs whose CO$_2$ uptake\nis underpredicted by global descriptors",transform=ax.transAxes,ha="center",fontsize=10.4,color=COL["ink"])
    ax.text(.64,.27,"mechanistic interpretation remains a hypothesis",transform=ax.transAxes,ha="center",fontsize=8.7,color=COL["muted"])
    x0=.82
    for i,v in enumerate(vals): ax.add_patch(plt.Rectangle((x0+i*.032,.20),.026,.48*v/max(vals),transform=ax.transAxes,fc=COL["teal"],ec="none"))
    ax.text(.88,.10,"AVAf quintiles",transform=ax.transAxes,ha="center",fontsize=9,color=COL["muted"])
    save_figure(fig,"toc","toc_graphic")


ALL = [figure1, figure2, figure3, figure4, si1, si2, si3, si4, si5, si6, si7, si8]
