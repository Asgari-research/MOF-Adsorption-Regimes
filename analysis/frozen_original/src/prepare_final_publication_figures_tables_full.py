#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prepare final publication figures, tables, source data, and reproducibility code
for the conformal MOF anomaly-screening manuscript.

This script consumes a completed v5 pipeline result directory and produces a
self-contained high-impact-journal package. It does not rerun ML.
"""
from __future__ import annotations
import os, re, json, shutil, zipfile, math, textwrap, glob, warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import patheffects as pe
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

warnings.filterwarnings('ignore')

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
BASE = Path('/mnt/data/conformal_final_work/input')
OUT = Path('/mnt/data/conformal_mof_final_publication_package')
ZIP_OUT = Path('/mnt/data/conformal_mof_final_publication_package.zip')
PIPELINE_CODE = Path('/mnt/data/conformal_mof_anomaly_screening_pipeline_paperB_v5_highimpact_qc.py')
MANIFEST_CSV = Path('/mnt/data/PUBLICATION_OUTPUT_MANIFEST.csv')

for p in [OUT, ZIP_OUT]:
    if p.exists():
        if p.is_dir(): shutil.rmtree(p)
        else: p.unlink()

DIRS = {
    'code': OUT/'code',
    'fig_main': OUT/'figures'/'main',
    'fig_si': OUT/'figures'/'si',
    'fig_struct': OUT/'figures'/'structure_panels',
    'src_main': OUT/'source_data'/'main_figures',
    'src_si': OUT/'source_data'/'si_figures',
    'src_tables': OUT/'source_data'/'tables',
    'tables_main': OUT/'tables'/'main',
    'tables_si': OUT/'tables'/'si',
    'tex_main': OUT/'latex_tables'/'main',
    'tex_si': OUT/'latex_tables'/'si',
    'cifs': OUT/'structural_case_studies'/'selected_cifs',
    'reports': OUT/'reports',
    'manifests': OUT/'manifests',
    'original': OUT/'original_results_key_files',
}
for d in DIRS.values(): d.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# Style
# -----------------------------------------------------------------------------
COL = {
    'ink':'#1F2933','muted':'#64748B','grid':'#DDE6EF','blue':'#2F6B9A','sky':'#8DC1DD',
    'teal':'#2A9D8F','green':'#5A9C6E','gold':'#E3A12E','orange':'#E76F51','red':'#B23A48',
    'purple':'#6D5A9C','grey':'#A6AFB8','pale_blue':'#EAF4FA','pale_teal':'#E8F5F2',
    'pale_orange':'#FFF1E8','pale_purple':'#F3F0FA'
}
CYCLE = [COL['blue'], COL['orange'], COL['teal'], COL['purple'], COL['gold'], COL['green'], COL['red'], COL['grey']]

plt.rcParams.update({
    'figure.facecolor':'white','axes.facecolor':'white','savefig.facecolor':'white','font.family':'DejaVu Sans',
    'font.size':8.5,'axes.titlesize':10.5,'axes.titleweight':'bold','axes.labelsize':9,'xtick.labelsize':7.5,
    'ytick.labelsize':7.5,'axes.edgecolor':COL['ink'],'axes.linewidth':0.8,'grid.color':COL['grid'],
    'grid.linewidth':0.55,'grid.alpha':0.65,'legend.frameon':False,'legend.fontsize':7,
    'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none'
})


def rd(rel: str, **kw) -> pd.DataFrame:
    p = BASE / rel
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p, **kw)


def save_all(fig, base: Path):
    base.parent.mkdir(parents=True, exist_ok=True)
    outs=[]
    base_s = str(base).lower()
    # SVG output for dense 3D atomistic panels can be extremely slow/large.
    # Keep SVG for clean 2D figures, and use PNG/PDF for structure-heavy figures.
    if ('structure' in base_s) or ('figure_6' in base_s) or ('figure_s8' in base_s):
        formats = ['png','pdf']
    else:
        formats = ['png','pdf','svg']
    for ext in formats:
        fp=base.with_suffix('.'+ext)
        fig.savefig(fp, dpi=360 if ext=='png' else 240, bbox_inches='tight', facecolor='white')
        outs.append(fp)
    plt.close(fig)
    return outs


def panel(ax, lab):
    t = ax.text(-0.12, 1.10, lab, transform=ax.transAxes, ha='left', va='top', fontsize=13, fontweight='bold', color=COL['ink'],
                bbox=dict(boxstyle='round,pad=0.18', facecolor='white', edgecolor=COL['grid'], linewidth=0.6), zorder=20)
    t.set_path_effects([pe.withStroke(linewidth=2.5, foreground='white')])


def style(ax, title='', xlabel='', ylabel='', grid=True):
    ax.set_title(title, pad=8, color=COL['ink'])
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    if grid: ax.grid(True, zorder=0)
    ax.set_axisbelow(True)
    for sp in ['top','right']:
        ax.spines[sp].set_visible(False)
    for sp in ['left','bottom']:
        ax.spines[sp].set_color(COL['ink'])
    ax.tick_params(colors=COL['muted'])


def slug(x: Any) -> str:
    s = str(x).lower()
    s = re.sub(r'[^a-z0-9]+','_',s).strip('_')
    return s or 'item'


def fmt_num(x, nd=3):
    try:
        if pd.isna(x): return ''
        x=float(x)
        if x != 0 and abs(x) < 1e-3: return f'{x:.2e}'
        if abs(x) >= 1000: return f'{x:,.0f}'
        return f'{x:.{nd}f}'
    except Exception:
        return str(x)


def tex_escape(s):
    s=str(s)
    return (s.replace('\\','\\textbackslash{}').replace('&','\\&').replace('%','\\%').replace('$','\\$')
             .replace('#','\\#').replace('_','\\_').replace('{','\\{').replace('}','\\}'))


def save_table(df: pd.DataFrame, csv_path: Path, tex_path: Optional[Path]=None, caption='', label='', max_rows_tex=80):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    if tex_path is not None:
        tex_path.parent.mkdir(parents=True, exist_ok=True)
        d = df.head(max_rows_tex).copy()
        try:
            tex = d.to_latex(index=False, escape=True, caption=caption, label=label, longtable=False)
        except Exception:
            cols=list(d.columns)
            lines=['\\begin{table}','\\centering']
            if caption: lines.append('\\caption{'+tex_escape(caption)+'}')
            if label: lines.append('\\label{'+tex_escape(label)+'}')
            lines.append('\\begin{tabular}{'+'l'*len(cols)+'}')
            lines.append('\\hline')
            lines.append(' & '.join(tex_escape(c) for c in cols)+' \\\\')
            lines.append('\\hline')
            for _,r in d.iterrows():
                lines.append(' & '.join(tex_escape(r[c]) for c in cols)+' \\\\')
            lines+=['\\hline','\\end{tabular}','\\end{table}']
            tex='\n'.join(lines)
        tex_path.write_text(tex, encoding='utf-8')

# -----------------------------------------------------------------------------
# Load data
# -----------------------------------------------------------------------------
metrics = rd('tables/main/table_1_model_calibration_screening_summary.csv')
enrich = rd('tables/main/table_2_chemistry_enrichment_positive_anomalies.csv')
targets = rd('tables/si/table_si_detected_targets_and_availability.csv')
risk_gate = rd('figure_data/main/Figure_6c_risk_gate_tradeoff.csv')
class_counts = rd('figure_data/main/Figure_5_6_fixed_candidate_class_counts.csv')
if not class_counts.empty and set(class_counts.columns) >= {'class','count'}:
    pass
elif not class_counts.empty and class_counts.shape[1] >= 2:
    class_counts = class_counts.iloc[:, :2]; class_counts.columns = ['class','count']
casebook = rd('paper_b_impact_additions/tables/paper_b_candidate_casebook_compact.csv')
external = rd('paper_b_impact_additions/tables/paper_b_external_match_strict_validation.csv')
process = rd('paper_b_impact_additions/tables/paper_b_process_stability_triage.csv')
split_diag = rd('qc_reports/split_availability_diagnostics.csv')
filter_sens = rd('qc_reports/posthoc_filter_sensitivity_report.csv')
redundancy = rd('qc_reports/feature_family_redundancy_report.csv')
qrate = rd('figure_data/main/Figure_5b_positive_surprise_rate_by_pore_quintile.csv')
stability = rd('figure_data/main/Figure_3c_shortlist_stability_pred_vs_adaptive_lcb.csv')
decision_map = rd('figure_data/main/Figure_6b_uncertainty_decision_map_sample.csv')
pvals = rd('figure_data/main/Figure_4c_anomaly_pvalues.csv')
pos50 = rd('figure_data/main/Figure_4a_positive_surprise_top50.csv')
neg50 = rd('figure_data/main/Figure_4b_negative_surprise_top50.csv')
anom_rates = rd('figure_data/main/Figure_4d_target_anomaly_rates.csv', header=None)
if not anom_rates.empty and anom_rates.shape[1]>=2:
    anom_rates.columns=['target','rate']
mech_counts = rd('paper_b_impact_additions/figure_data/PaperB_Figure_mechanistic_family_counts.csv')
if not mech_counts.empty and 'positive_surprise_mechanistic_family' in mech_counts.columns:
    mech_counts = mech_counts.rename(columns={'positive_surprise_mechanistic_family':'mechanistic_family'})
elif not mech_counts.empty and mech_counts.shape[1]>=2:
    mech_counts = mech_counts.iloc[:, :2]; mech_counts.columns=['mechanistic_family','count']
ext_counts = rd('paper_b_impact_additions/figure_data/PaperB_Figure_external_match_confidence_counts.csv')
elif_ext_dummy = None
if not ext_counts.empty and set(ext_counts.columns) >= {'external_match_confidence','count'}:
    pass
elif not ext_counts.empty and ext_counts.shape[1]>=2:
    ext_counts = ext_counts.iloc[:, :2]; ext_counts.columns=['external_match_confidence','count']
proc_counts = rd('paper_b_impact_additions/figure_data/PaperB_Figure_process_proxy_counts.csv')
if not proc_counts.empty and set(proc_counts.columns) >= {'process_proxy_status','count'}:
    pass
elif not proc_counts.empty and proc_counts.shape[1]>=2:
    proc_counts = proc_counts.iloc[:, :2]; proc_counts.columns=['process_proxy_status','count']

# Copy all source figure data and original key files
if (BASE/'figure_data').exists():
    shutil.copytree(BASE/'figure_data', OUT/'source_data'/'original_figure_data', dirs_exist_ok=True)
for folder in ['tables','qc_reports','paper_b_impact_additions','structural_case_studies']:
    src=BASE/folder
    if src.exists(): shutil.copytree(src, OUT/'original_results_key_files'/folder, dirs_exist_ok=True)
if PIPELINE_CODE.exists(): shutil.copy2(PIPELINE_CODE, DIRS['code']/PIPELINE_CODE.name)
if MANIFEST_CSV.exists(): shutil.copy2(MANIFEST_CSV, DIRS['manifests']/MANIFEST_CSV.name)

# Copy CIFs
cif_src = BASE/'structural_case_studies'/'selected_cifs'
if cif_src.exists():
    for p in cif_src.glob('*.cif'):
        shutil.copy2(p, DIRS['cifs']/p.name)

# -----------------------------------------------------------------------------
# Table preparation
# -----------------------------------------------------------------------------
# Main Table 1: one best random setup per target plus strongest grouped stress test per target
mt1 = pd.DataFrame()
if not metrics.empty:
    m = metrics.copy()
    for c in m.columns:
        if c.endswith('_mean') or c in ['test_rmse_mean','test_empirical_coverage_mean','test_mean_interval_width_mean','test_top5_jaccard_lcb_vs_true_mean','test_adaptive_empirical_coverage_mean']:
            m[c] = pd.to_numeric(m[c], errors='coerce')
    # best by RMSE per target for random and best grouped by RMSE
    rows=[]
    for t, sub in m.groupby('target_key'):
        rnd=sub[sub['split_type'].eq('random')].sort_values(['test_rmse_mean','test_mean_interval_width_mean']).head(1)
        grp=sub[sub['split_type'].ne('random')].sort_values(['test_rmse_mean','test_mean_interval_width_mean']).head(1)
        if not rnd.empty: rows.append(rnd.assign(summary_role='Best random-split setup'))
        if not grp.empty: rows.append(grp.assign(summary_role='Best grouped-split stress test'))
    mt1 = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    cols=['summary_role','target_key','feature_family','model_name','split_type','n_seeds','test_rmse_mean','test_empirical_coverage_mean','test_mean_interval_width_mean','test_top5_jaccard_lcb_vs_true_mean','test_adaptive_empirical_coverage_mean']
    cols=[c for c in cols if c in mt1.columns]
    mt1 = mt1[cols]
    mt1 = mt1.rename(columns={
        'target_key':'Target','feature_family':'Feature family','model_name':'Model','split_type':'Split','n_seeds':'Seeds',
        'test_rmse_mean':'RMSE','test_empirical_coverage_mean':'Empirical coverage','test_mean_interval_width_mean':'Mean interval width',
        'test_top5_jaccard_lcb_vs_true_mean':'Top-5% LCB recovery','test_adaptive_empirical_coverage_mean':'Adaptive coverage','summary_role':'Role'})
    save_table(mt1, DIRS['tables_main']/'Table_1_calibration_screening_summary.csv', DIRS['tex_main']/'Table_1_calibration_screening_summary.tex',
               'Condensed calibration, efficiency, and screening summary for the most informative random and grouped split settings.', 'tab:main_calibration_summary')
    shutil.copy2(DIRS['tables_main']/'Table_1_calibration_screening_summary.csv', DIRS['src_tables']/'Table_1_calibration_screening_summary_source.csv')

# Main Table 2: decision classes and risk gates
mt2_parts=[]
if not class_counts.empty:
    cc=class_counts.copy()
    cc.columns=['Decision class','Count']
    cc['Type']='Candidate count'
    mt2_parts.append(cc[['Type','Decision class','Count']])
if not risk_gate.empty:
    rg=risk_gate.copy().rename(columns={'gate':'Decision class','false_elite_rate':'False-elite rate','elite_recall':'Elite recall'})
    rg['Type']='Risk gate'
    rg['Count']=''
    mt2_parts.append(rg[['Type','Decision class','Count','False-elite rate','Elite recall']])
mt2 = pd.concat(mt2_parts, ignore_index=True, sort=False) if mt2_parts else pd.DataFrame()
save_table(mt2, DIRS['tables_main']/'Table_2_certificate_decision_classes_and_risk.csv', DIRS['tex_main']/'Table_2_certificate_decision_classes_and_risk.tex',
           'Certificate decision classes and representative false-elite risk/recall trade-off for the main screening target.', 'tab:main_decision_classes')
shutil.copy2(DIRS['tables_main']/'Table_2_certificate_decision_classes_and_risk.csv', DIRS['src_tables']/'Table_2_certificate_decision_classes_and_risk_source.csv')

# Main Table 3: top enrichment regimes
mt3 = pd.DataFrame()
if not enrich.empty:
    mt3 = enrich.head(12).copy()
    keep=['group_col','group','n_group','n_label_in_group','label_rate_in_group','label_rate_out_group','odds_ratio','fisher_p']
    mt3=mt3[[c for c in keep if c in mt3.columns]]
    mt3=mt3.rename(columns={'group_col':'Descriptor/family','group':'Regime','n_group':'n in regime','n_label_in_group':'Positive surprises','label_rate_in_group':'Rate in regime','label_rate_out_group':'Rate outside','odds_ratio':'Odds ratio','fisher_p':'Fisher p'})
    save_table(mt3, DIRS['tables_main']/'Table_3_positive_surprise_enrichment.csv', DIRS['tex_main']/'Table_3_positive_surprise_enrichment.tex',
               'Most enriched descriptor regimes among CO2 0.15 bar positive-surprise conformal anomalies.', 'tab:main_positive_surprise_enrichment')
    shutil.copy2(DIRS['tables_main']/'Table_3_positive_surprise_enrichment.csv', DIRS['src_tables']/'Table_3_positive_surprise_enrichment_source.csv')

# Main Table 4: selected casebook shortlist
mt4 = pd.DataFrame()
if not casebook.empty:
    priority_ids = ['DB12-OSAVUA_clean','DB12-VAGTAA01_clean','DB12-VAGTAA_clean','DB1-Zn2O8-BTC_A-irmof8_A_No11','DB12-TIXVON_clean','DB1-Zn2O8-AZO_A-irmof8_A_No357','DB15-cds_N29_E173_opt','DB15-cds_N87_E131_opt','DB13-cds-Syn022177','DB0-m5_o1_o11_f0_bcu.sym.34']
    cb=casebook.copy()
    cb['_order']=cb['display_id'].apply(lambda x: priority_ids.index(x) if x in priority_ids else 999)
    cb=cb.sort_values(['_order','structure_rendering_priority_score'], ascending=[True,False]).head(10)
    keep=['paper_b_case_class','display_id','positive_surprise_mechanistic_family','y_true','y_pred','residual','PLD','LCD','Density','AVAf','ASA','external_match_confidence','process_proxy_status','recommended_rendering_priority']
    keep=[c for c in keep if c in cb.columns]
    mt4=cb[keep].rename(columns={'paper_b_case_class':'Case class','display_id':'MOF ID','positive_surprise_mechanistic_family':'Mechanistic family','y_true':'True uptake','y_pred':'Predicted uptake','residual':'Residual','external_match_confidence':'External tier','process_proxy_status':'Process proxy','recommended_rendering_priority':'Rendering priority'})
    save_table(mt4, DIRS['tables_main']/'Table_4_structural_casebook_shortlist.csv', DIRS['tex_main']/'Table_4_structural_casebook_shortlist.tex',
               'Main-text structural casebook shortlist for CIF rendering and mechanism interpretation.', 'tab:main_structural_casebook')
    shutil.copy2(DIRS['tables_main']/'Table_4_structural_casebook_shortlist.csv', DIRS['src_tables']/'Table_4_structural_casebook_shortlist_source.csv')

# Copy all existing SI tables and some transformed ones
for p in (BASE/'tables'/'si').glob('*'):
    if p.is_file(): shutil.copy2(p, DIRS['tables_si']/p.name)
for p in (BASE/'qc_reports').glob('*.csv'):
    shutil.copy2(p, DIRS['tables_si']/('qc_'+p.name))
for p in (BASE/'paper_b_impact_additions'/'tables').glob('*.csv'):
    shutil.copy2(p, DIRS['tables_si']/p.name)

# -----------------------------------------------------------------------------
# CIF rendering helpers
# -----------------------------------------------------------------------------
COV_RAD = {'H':0.31,'C':0.76,'N':0.71,'O':0.66,'F':0.57,'P':1.07,'S':1.05,'Cl':1.02,'Br':1.20,'I':1.39,
           'Zn':1.22,'Cu':1.32,'Ca':1.76,'Zr':1.54,'Fe':1.24,'Mg':1.41,'Al':1.21,'Ni':1.24,'Co':1.26,'Mn':1.39}
E_COL={'H':'#C9CDD2','C':'#4B5563','N':'#2F6B9A','O':'#E76F51','F':'#2A9D8F','S':'#E3A12E','P':'#A16207','Cl':'#5A9C6E','Br':'#8B5CF6','I':'#7C3AED','Zn':'#6D5A9C','Cu':'#B87333','Ca':'#2A9D8F','Zr':'#3B82F6','Fe':'#B23A48','Mg':'#84CC16','Al':'#94A3B8','Ni':'#16A34A','Co':'#2563EB','Mn':'#F97316'}


def cell_matrix(a,b,c,alpha,beta,gamma):
    al,be,ga=np.deg2rad([alpha,beta,gamma])
    va=np.array([a,0,0.])
    vb=np.array([b*np.cos(ga), b*np.sin(ga), 0.])
    cx=c*np.cos(be)
    cy=c*(np.cos(al)-np.cos(be)*np.cos(ga))/max(np.sin(ga),1e-9)
    cz=math.sqrt(max(c*c-cx*cx-cy*cy,0))
    vc=np.array([cx,cy,cz])
    return np.vstack([va,vb,vc]).T


def parse_cif(path: Path):
    txt=path.read_text(errors='ignore').splitlines()
    vals={}
    for line in txt:
        for key in ['_cell_length_a','_cell_length_b','_cell_length_c','_cell_angle_alpha','_cell_angle_beta','_cell_angle_gamma']:
            if line.strip().startswith(key):
                try: vals[key]=float(line.split()[1].strip().strip("'\""))
                except: pass
    if len(vals)<6: return pd.DataFrame(), np.eye(3)
    M=cell_matrix(vals['_cell_length_a'],vals['_cell_length_b'],vals['_cell_length_c'],vals['_cell_angle_alpha'],vals['_cell_angle_beta'],vals['_cell_angle_gamma'])
    rows=[]; in_loop=False; headers=[]
    i=0
    while i<len(txt):
        line=txt[i].strip()
        if line=='loop_':
            j=i+1; headers=[]
            while j<len(txt) and txt[j].strip().startswith('_'):
                headers.append(txt[j].strip()); j+=1
            if '_atom_site_fract_x' in headers and '_atom_site_fract_y' in headers and '_atom_site_fract_z' in headers:
                while j<len(txt):
                    l=txt[j].strip()
                    if not l or l.startswith('_') or l=='loop_' or l.startswith('data_'): break
                    parts=re.findall(r"'(?:[^']*)'|\S+", l)
                    if len(parts)>=len(headers):
                        rec=dict(zip(headers, parts))
                        try:
                            el=rec.get('_atom_site_type_symbol', rec.get('_atom_site_label','X'))
                            el=re.sub(r'[^A-Za-z]','',el).capitalize()
                            fx=float(rec['_atom_site_fract_x'].strip('()'))
                            fy=float(rec['_atom_site_fract_y'].strip('()'))
                            fz=float(rec['_atom_site_fract_z'].strip('()'))
                            cart=M@np.array([fx,fy,fz])
                            rows.append({'element':el,'x':cart[0],'y':cart[1],'z':cart[2],'fx':fx,'fy':fy,'fz':fz})
                        except Exception:
                            pass
                    j+=1
                break
        i+=1
    return pd.DataFrame(rows), M


def render_cif_panel(ax, cif: Optional[Path], title='', metrics_text='', view=(25,35)):
    ax.set_axis_off()
    if cif is None or not cif.exists():
        ax.add_patch(Rectangle((0,0),1,1,transform=ax.transAxes,facecolor=COL['pale_blue'],edgecolor=COL['blue']))
        ax.text(.5,.58,'CIF not available',ha='center',va='center',fontweight='bold')
        ax.text(.5,.42,title,ha='center',va='center',fontsize=6)
        return
    atoms, M = parse_cif(cif)
    if atoms.empty:
        ax.text(.5,.5,'CIF parse failed',ha='center',va='center')
        return
    # Basic bonds by distances for non-H atoms, capped for speed/clarity
    # Downsample extremely large CIFs for fast drafting-quality previews.
    if len(atoms) > 450:
        atoms = atoms.sample(450, random_state=1).sort_index()
    coords=atoms[['x','y','z']].values
    els=atoms['element'].tolist()
    # Center and normalize view
    ax.view_init(*view)
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.pane.set_edgecolor('white')
        axis.pane.set_facecolor((1,1,1,0))
    ax.grid(False)
    # bonds
    n=len(atoms)
    if n <= 120:
        for i in range(n):
            if els[i]=='H': continue
            for j in range(i+1,n):
                if els[j]=='H': continue
                d=np.linalg.norm(coords[i]-coords[j])
                cutoff=1.25*(COV_RAD.get(els[i],0.8)+COV_RAD.get(els[j],0.8))
                if d<cutoff and d>0.4:
                    ax.plot([coords[i,0],coords[j,0]],[coords[i,1],coords[j,1]],[coords[i,2],coords[j,2]], color='#9AA6B2', lw=0.55, alpha=0.55, zorder=1)
    # atoms by element
    for el in sorted(set(els)):
        sub=atoms[atoms['element']==el]
        size=10 if el=='H' else (45 if el not in ['C','N','O'] else 22)
        alpha=0.25 if el=='H' else 0.95
        ax.scatter(sub['x'],sub['y'],sub['z'], s=size, color=E_COL.get(el,'#64748B'), edgecolor='white', linewidth=0.25, alpha=alpha, depthshade=True)
    # cell edges
    O=np.zeros(3); a=M[:,0]; b=M[:,1]; c=M[:,2]
    vertices=[O,a,b,c,a+b,a+c,b+c,a+b+c]
    edges=[(0,1),(0,2),(0,3),(1,4),(1,5),(2,4),(2,6),(3,5),(3,6),(4,7),(5,7),(6,7)]
    for u,v in edges:
        pu,pv=vertices[u],vertices[v]
        ax.plot([pu[0],pv[0]],[pu[1],pv[1]],[pu[2],pv[2]], color=COL['ink'], lw=0.45, alpha=0.35)
    ax.set_title(title, fontsize=7.2, fontweight='bold', pad=2)
    ax.text2D(0.02,0.02,metrics_text, transform=ax.transAxes, fontsize=5.6, color=COL['ink'], bbox=dict(boxstyle='round,pad=0.2',facecolor='white',edgecolor=COL['grid'],alpha=.85))

# -----------------------------------------------------------------------------
# Figures
# -----------------------------------------------------------------------------
# Figure 1
fig=plt.figure(figsize=(13,7.6), dpi=160)
gs=fig.add_gridspec(2,3, height_ratios=[1.05,1], width_ratios=[1.25,1,1], wspace=.34, hspace=.42)
ax=fig.add_subplot(gs[0,:]); ax.axis('off')
boxes=[('Ranked list', 'Predicted uptake\nwithout certified risk', COL['pale_blue'], COL['blue']),
       ('Split conformal calibration', 'Calibration residuals\nproduce [L,U] intervals', COL['pale_teal'], COL['teal']),
       ('Certified triage', 'possible elite\nconfident elite\nfragile elite', COL['pale_purple'], COL['purple']),
       ('Anomaly casebook', 'positive surprises\nnegative controls\nstructural examples', COL['pale_orange'], COL['orange'])]
for i,(t,b,fc,ec) in enumerate(boxes):
    x=0.04+i*0.24
    patch=FancyBboxPatch((x,0.24),0.18,0.50,boxstyle='round,pad=0.025,rounding_size=0.025',facecolor=fc,edgecolor=ec,linewidth=1.4,transform=ax.transAxes)
    ax.add_patch(patch)
    ax.text(x+0.09,0.59,t,ha='center',va='center',fontweight='bold',fontsize=11,color=COL['ink'],transform=ax.transAxes)
    ax.text(x+0.09,0.42,b,ha='center',va='center',fontsize=8.5,color=COL['muted'],transform=ax.transAxes)
    ax.text(x+0.015,0.78,chr(97+i),fontsize=13,fontweight='bold',color=ec,transform=ax.transAxes)
    if i<3:
        ax.add_patch(FancyArrowPatch((x+0.185,0.49),(x+0.235,0.49),transform=ax.transAxes,arrowstyle='-|>',mutation_scale=15,lw=1.2,color=COL['muted']))
ax.text(.5,.06,'Central message: high-throughput MOF screening should report calibrated discovery decisions, not only top-k rankings.',ha='center',va='center',fontsize=12,fontweight='bold',color=COL['ink'],transform=ax.transAxes)

ax=fig.add_subplot(gs[1,0])
if not targets.empty:
    tgt=targets.copy(); tgt['target_description']=tgt['target_description'].str.replace('CO2','CO$_2$').str.replace('CH4','CH$_4$')
    ax.barh(np.arange(len(tgt)), tgt['n_non_missing'], color=[COL['blue'],COL['sky'],COL['teal'],COL['purple']], edgecolor='white')
    ax.set_yticks(np.arange(len(tgt))); ax.set_yticklabels(tgt['target_description'])
    for i,v in enumerate(tgt['n_non_missing']): ax.text(v*1.005,i,f'{int(v):,}',va='center',fontsize=7)
    targets.to_csv(DIRS['src_main']/'Figure_1d_target_availability.csv',index=False)
style(ax,'Four adsorption targets detected cleanly','non-missing MOFs',''); panel(ax,'e')

ax=fig.add_subplot(gs[1,1])
if not split_diag.empty:
    sd=split_diag.copy(); sd.to_csv(DIRS['src_main']/'Figure_1e_split_availability.csv',index=False)
    colors=[COL['teal'] if s=='available' else COL['red'] for s in sd['status']]
    ax.bar(np.arange(len(sd)), [1]*len(sd), color=colors, edgecolor='white')
    ax.set_xticks(np.arange(len(sd))); ax.set_xticklabels(sd['split_type'],rotation=35,ha='right')
    ax.set_yticks([])
    for i,r in sd.iterrows(): ax.text(i,.5,r['status'],ha='center',va='center',fontsize=7,fontweight='bold',color='white')
style(ax,'Split design and availability','',''); panel(ax,'f')

ax=fig.add_subplot(gs[1,2])
if not class_counts.empty:
    cc=class_counts.copy(); cc.to_csv(DIRS['src_main']/'Figure_1f_output_classes.csv',index=False)
    ax.barh(np.arange(len(cc)), cc['count'], color=[COL['blue'],COL['sky'],COL['teal'],COL['purple'],COL['orange'],COL['red']][:len(cc)], edgecolor='white')
    ax.set_yticks(np.arange(len(cc))); ax.set_yticklabels(cc['class'])
    for i,v in enumerate(cc['count']): ax.text(float(v)*1.01 if float(v)>0 else .5,i,str(int(v)),va='center',fontsize=7)
style(ax,'Representative output classes','candidate count',''); panel(ax,'g')
fig.suptitle('Figure 1. From ranked MOF screening to conformal discovery certificates', fontsize=15, fontweight='bold')
fig.tight_layout(rect=[0,0,1,.95])
save_all(fig, DIRS['fig_main']/'Figure_1_rankings_to_certificates')

# Figure 2
covdev=rd('figure_data/main/Figure_2a_coverage_deviation_by_split_target.csv', index_col=0)
widths=rd('figure_data/main/Figure_2b_interval_width_by_split_target.csv', index_col=0)
trade=rd('figure_data/main/Figure_2c_rmse_coverage_screening_tradeoff.csv')
adapt=rd('figure_data/main/Figure_2d_standard_vs_adaptive_conformal.csv')
for name,df in [('Figure_2a_coverage_deviation_by_split_target',covdev),('Figure_2b_interval_width_by_split_target',widths),('Figure_2c_rmse_coverage_screening_tradeoff',trade),('Figure_2d_standard_vs_adaptive_conformal',adapt)]:
    if not df.empty: df.to_csv(DIRS['src_main']/(name+'.csv'))
fig,axes=plt.subplots(2,2,figsize=(12,8.5),dpi=160); axes=axes.ravel()
ax=axes[0]
if not covdev.empty:
    max_abs = max(abs(float(covdev.min().min())), abs(float(covdev.max().max())), 0.08)
    norm=TwoSlopeNorm(vcenter=0, vmin=-max_abs, vmax=max_abs)
    im=ax.imshow(covdev.values, cmap=LinearSegmentedColormap.from_list('cov',[COL['blue'],'white',COL['orange']]), norm=norm, aspect='auto')
    ax.set_xticks(range(covdev.shape[1])); ax.set_xticklabels(covdev.columns, rotation=30, ha='right')
    ax.set_yticks(range(covdev.shape[0])); ax.set_yticklabels(covdev.index)
    for i in range(covdev.shape[0]):
        for j in range(covdev.shape[1]): ax.text(j,i,f'{covdev.values[i,j]:+.3f}',ha='center',va='center',fontsize=7)
    fig.colorbar(im, ax=ax, fraction=.046,pad=.04,label='coverage − 0.90')
style(ax,'Conservative empirical 90% coverage','Target','Split',grid=False); panel(ax,'a')
ax=axes[1]
if not widths.empty:
    im=ax.imshow(np.log1p(widths.values), cmap=LinearSegmentedColormap.from_list('eff',['#F7FBFF',COL['sky'],COL['blue'],COL['ink']]), aspect='auto')
    ax.set_xticks(range(widths.shape[1])); ax.set_xticklabels(widths.columns, rotation=30, ha='right')
    ax.set_yticks(range(widths.shape[0])); ax.set_yticklabels(widths.index)
    for i in range(widths.shape[0]):
        for j in range(widths.shape[1]): ax.text(j,i,f'{widths.values[i,j]:.2g}',ha='center',va='center',fontsize=7)
    fig.colorbar(im, ax=ax, fraction=.046,pad=.04,label='log(1+width)')
style(ax,'Interval width reveals extrapolation cost','Target','Split',grid=False); panel(ax,'b')
ax=axes[2]
if not trade.empty:
    for k,(split,sub) in enumerate(trade.groupby('split_type')):
        size=50+110*(sub['width']/max(trade['width'].max(),1e-9)) if 'width' in trade else 80
        ax.scatter(sub['rmse'],sub['coverage'],s=size,color=CYCLE[k],alpha=.82,edgecolor='white',linewidth=.6,label=split)
        for _,r in sub.iterrows(): ax.text(r['rmse'],r['coverage'],str(r['model_name']).upper(),fontsize=6.5,color=COL['ink'])
    ax.axhline(.90,ls='--',lw=1,color=COL['muted']); ax.legend(loc='lower right')
style(ax,'Accuracy alone does not certify screening risk','Mean RMSE','Empirical coverage'); panel(ax,'c')
ax=axes[3]
if not adapt.empty:
    x=np.arange(len(adapt)); w=.34
    ax.bar(x-w/2,adapt['standard_width'],w,label='standard width',color=COL['blue'],edgecolor='white')
    ax.bar(x+w/2,adapt['adaptive_width'],w,label='adaptive width',color=COL['teal'],edgecolor='white')
    ax.set_xticks(x); ax.set_xticklabels(adapt['split_type'],rotation=30,ha='right')
    ax.legend(loc='upper left')
    ax2=ax.twinx()
    ax2.plot(x,adapt['standard_coverage'],marker='o',ls='--',color=COL['blue'],label='std. cov.')
    ax2.plot(x,adapt['adaptive_coverage'],marker='s',ls=':',color=COL['teal'],label='adapt. cov.')
    ax2.axhline(.90,ls='--',lw=.9,color=COL['muted']); ax2.set_ylabel('Coverage')
style(ax,'Adaptive certificates reduce width with coverage retained','Split','Interval width'); panel(ax,'d')
fig.suptitle('Figure 2. Calibration and efficiency across targets, models, and split types',fontsize=15,fontweight='bold')
fig.tight_layout(rect=[0,0,1,.95])
save_all(fig, DIRS['fig_main']/'Figure_2_calibration_efficiency')

# Figure 3
for name,df in [('Figure_3a_candidate_counts',class_counts),('Figure_3b_risk_gate_tradeoff',risk_gate),('Figure_3c_stability',stability),('Figure_3d_decision_map_sample',decision_map)]:
    if not df.empty: df.to_csv(DIRS['src_main']/(name+'.csv'), index=False)
fig,axes=plt.subplots(2,2,figsize=(12,8.6),dpi=160); axes=axes.ravel()
ax=axes[0]
if not class_counts.empty:
    labels=class_counts['class']; vals=pd.to_numeric(class_counts['count'],errors='coerce')
    ax.barh(np.arange(len(vals)),vals,color=[COL['blue'],COL['sky'],COL['teal'],COL['purple'],COL['orange'],COL['red']][:len(vals)],edgecolor='white')
    ax.set_yticks(np.arange(len(vals))); ax.set_yticklabels(labels)
    for i,v in enumerate(vals): ax.text(v*1.01 if v>0 else 0.5,i,f'{int(v)}',va='center',fontsize=7)
style(ax,'Certificate triage produces action classes','Candidate count',''); panel(ax,'a')
ax=axes[1]
if not risk_gate.empty:
    x=np.arange(len(risk_gate)); w=.36
    ax.bar(x-w/2,risk_gate['false_elite_rate'],w,label='false-elite rate',color=COL['red'],edgecolor='white')
    ax.bar(x+w/2,risk_gate['elite_recall'],w,label='elite recall',color=COL['teal'],edgecolor='white')
    ax.set_xticks(x); ax.set_xticklabels(risk_gate['gate'],rotation=25,ha='right')
    ax.legend()
style(ax,'Risk/recall trade-off for the main target','Gate','Fraction'); panel(ax,'b')
ax=axes[2]
if not decision_map.empty:
    dm=decision_map.copy()
    ytrue='y_true'; ypred='y_pred'
    if ytrue in dm and ypred in dm:
        colors=[]
        for _,r in dm.iterrows():
            dc=str(r.get('decision_class_main',''))
            if 'positive' in dc: colors.append(COL['orange'])
            elif 'negative' in dc: colors.append(COL['blue'])
            elif 'elite' in dc: colors.append(COL['teal'])
            else: colors.append(COL['grey'])
        ax.scatter(dm[ypred],dm[ytrue],s=12,c=colors,alpha=.62,edgecolor='none',rasterized=True)
        mn=min(dm[ypred].min(),dm[ytrue].min()); mx=max(dm[ypred].max(),dm[ytrue].max()); ax.plot([mn,mx],[mn,mx],ls='--',color=COL['muted'],lw=1)
style(ax,'Uncertainty-aware decision map','Predicted uptake','True uptake'); panel(ax,'c')
ax=axes[3]
if not stability.empty:
    st=stability.copy(); x=np.arange(len(st)); w=.34
    ax.bar(x-w/2,st['jaccard_pred_mean'],w,label='predicted top-5%',color=COL['blue'],edgecolor='white')
    ax.bar(x+w/2,st['jaccard_lcb_mean'],w,label='LCB/adaptive top-5%',color=COL['teal'],edgecolor='white')
    ax.set_xticks(x); ax.set_xticklabels(st['split_type'],rotation=30,ha='right')
    ax.legend()
style(ax,'Shortlist stability across seeds','Split','Jaccard'); panel(ax,'d')
fig.suptitle('Figure 3. From top-k rankings to conformal screening certificates',fontsize=15,fontweight='bold')
fig.tight_layout(rect=[0,0,1,.95])
save_all(fig, DIRS['fig_main']/'Figure_3_screening_certificates')

# Figure 4
for name,df in [('Figure_4a_positive_top50',pos50),('Figure_4b_negative_top50',neg50),('Figure_4c_pvalues',pvals),('Figure_4d_target_anomaly_rates',anom_rates)]:
    if not df.empty: df.to_csv(DIRS['src_main']/(name+'.csv'), index=False)
fig,axes=plt.subplots(2,2,figsize=(12,8.5),dpi=160); axes=axes.ravel()
ax=axes[0]
if not pos50.empty:
    ax.scatter(np.arange(len(pos50)),pos50['residual'],s=28,color=COL['orange'],edgecolor='white',linewidth=.4,alpha=.92)
style(ax,'Top positive-surprise residuals','Rank','Residual'); panel(ax,'a')
ax=axes[1]
if not neg50.empty:
    ax.scatter(np.arange(len(neg50)),neg50['residual'],s=28,color=COL['blue'],edgecolor='white',linewidth=.4,alpha=.92)
style(ax,'Top negative-surprise residuals','Rank','Residual'); panel(ax,'b')
ax=axes[2]
if not pvals.empty and 'p_two_sided_surprise' in pvals:
    ax.hist(pd.to_numeric(pvals['p_two_sided_surprise'],errors='coerce').dropna(), bins=35, color=COL['purple'],edgecolor='white',alpha=.9)
    ax.axvline(.10,ls='--',lw=1,color=COL['muted']); ax.text(.105, ax.get_ylim()[1]*.88, 'α=0.10',fontsize=7,color=COL['muted'])
style(ax,'Two-sided conformal anomaly p-values','p-value','Count'); panel(ax,'c')
ax=axes[3]
if not anom_rates.empty:
    ar=anom_rates.sort_values('rate',ascending=False)
    ax.bar(np.arange(len(ar)),ar['rate'],color=[COL['orange'],COL['teal'],COL['blue'],COL['purple']][:len(ar)],edgecolor='white')
    ax.set_xticks(np.arange(len(ar))); ax.set_xticklabels(ar['target'],rotation=25,ha='right')
style(ax,'Positive-anomaly rate across adsorption targets','Target','Rate'); panel(ax,'d')
fig.suptitle('Figure 4. Conformal residual-anomaly atlas',fontsize=15,fontweight='bold')
fig.tight_layout(rect=[0,0,1,.95])
save_all(fig, DIRS['fig_main']/'Figure_4_anomaly_atlas')

# Figure 5
for name,df in [('Figure_5a_enrichment_top30',enrich),('Figure_5b_quintile_rates',qrate),('Figure_5c_mechanism_counts',mech_counts),('Figure_5d_external_counts',ext_counts),('Figure_5d_process_counts',proc_counts)]:
    if not df.empty: df.to_csv(DIRS['src_main']/(name+'.csv'), index=False)
fig=plt.figure(figsize=(12.5,9.2),dpi=160)
gs=fig.add_gridspec(2,3,width_ratios=[1.25,1.1,.9],height_ratios=[1,1],wspace=.42,hspace=.45)
ax=fig.add_subplot(gs[:,0])
if not enrich.empty:
    e=enrich.head(10).copy(); labels=[f"{r['group_col']} {str(r['group'])[:16]}" for _,r in e.iterrows()]
    y=np.arange(len(e)); ax.barh(y,e['odds_ratio'],color=COL['orange'],edgecolor='white')
    ax.set_yticks(y); ax.set_yticklabels(labels,fontsize=7); ax.invert_yaxis(); ax.axvline(1,ls='--',color=COL['muted'],lw=1)
style(ax,'Positive-surprise enrichment is dominated by confined/dense regimes','Odds ratio',''); panel(ax,'a')
ax=fig.add_subplot(gs[0,1])
if not qrate.empty:
    qr=qrate.copy()
    # Robust handling of column names
    desc_col = 'descriptor' if 'descriptor' in qr.columns else qr.columns[0]
    quint_col = 'quintile' if 'quintile' in qr.columns else ('group' if 'group' in qr.columns else qr.columns[1])
    rate_col = 'positive_surprise_rate' if 'positive_surprise_rate' in qr.columns else ('label_rate_in_group' if 'label_rate_in_group' in qr.columns else [c for c in qr.columns if 'rate' in c.lower()][0])
    for i,(desc,sub) in enumerate(qr.groupby(desc_col)):
        if str(desc).lower() in ['avaf','asa','ava','density','density_g_cm3','density(g/cm3)'] or i<5:
            sub=sub.copy()
            x=np.arange(len(sub))
            ax.plot(x,sub[rate_col],marker='o',lw=2,label=str(desc)[:12],color=CYCLE[i%len(CYCLE)])
    ax.legend(ncol=2,fontsize=6)
style(ax,'Positive-surprise rate by pore descriptor quintile','Quintile (low → high)','Rate'); panel(ax,'b')
ax=fig.add_subplot(gs[0,2])
if not mech_counts.empty:
    mc=mech_counts.sort_values('count')
    ax.barh(np.arange(len(mc)),mc['count'],color=COL['purple'],edgecolor='white')
    ax.set_yticks(np.arange(len(mc))); ax.set_yticklabels(mc['mechanistic_family'],fontsize=6.2)
style(ax,'Casebook mechanism families','Count',''); panel(ax,'c')
ax=fig.add_subplot(gs[1,1])
if not ext_counts.empty:
    ax.bar(np.arange(len(ext_counts)), ext_counts['count'],color=COL['teal'],edgecolor='white')
    ax.set_xticks(np.arange(len(ext_counts))); ax.set_xticklabels(ext_counts['external_match_confidence'],rotation=25,ha='right')
style(ax,'External provenance tier','Tier','Count'); panel(ax,'d')
ax=fig.add_subplot(gs[1,2])
if not proc_counts.empty:
    ax.barh(np.arange(len(proc_counts)), proc_counts['count'],color=[COL['teal'] if 'no' in str(x).lower() else COL['orange'] for x in proc_counts['process_proxy_status']],edgecolor='white')
    ax.set_yticks(np.arange(len(proc_counts))); ax.set_yticklabels(proc_counts['process_proxy_status'],fontsize=6.5)
style(ax,'Process-proxy triage','Count',''); panel(ax,'e')
fig.suptitle('Figure 5. Chemical regimes behind conformal positive surprises',fontsize=15,fontweight='bold')
fig.tight_layout(rect=[0,0,1,.95])
save_all(fig, DIRS['fig_main']/'Figure_5_chemistry_regimes')

# Structure thumbnails and Figure 6
# Map display IDs to CIF files based on slug matching
cif_files=list(DIRS['cifs'].glob('*.cif'))
def find_cif_for_id(mid):
    sm=slug(mid)
    for p in cif_files:
        if sm in slug(p.stem) or slug(p.stem).endswith(sm): return p
    # fallback contains components
    tokens=[t for t in sm.split('_') if len(t)>3]
    best=None; score=0
    for p in cif_files:
        ss=slug(p.stem); sc=sum(1 for t in tokens if t in ss)
        if sc>score: best=p; score=sc
    return best if score>=2 else None

if not casebook.empty:
    # Generate individual structure panels for all casebook entries
    for _,r in casebook.iterrows():
        mid=r['display_id']; cif=find_cif_for_id(mid)
        fig=plt.figure(figsize=(4.8,4.2),dpi=160)
        ax=fig.add_subplot(111,projection='3d')
        metrics_text=f"{str(r.get('paper_b_case_class',''))[:30]}\nresid={fmt_num(r.get('residual'),2)}; PLD={fmt_num(r.get('PLD'),2)} Å\nAVA$_f$={fmt_num(r.get('AVAf'),3)}; ASA={fmt_num(r.get('ASA'),1)}"
        render_cif_panel(ax,cif,title=str(mid)[:34],metrics_text=metrics_text,view=(25,35))
        save_all(fig, DIRS['fig_struct']/(slug(mid)+'_structure_panel'))
        # source per case
    casebook.to_csv(DIRS['src_main']/'Figure_6_structural_casebook_source.csv',index=False)

fig=plt.figure(figsize=(13,10),dpi=160)
gs=fig.add_gridspec(2,4,wspace=.06,hspace=.18)
selected=[]
priority = ['DB12-OSAVUA_clean','DB12-VAGTAA01_clean','DB12-VAGTAA_clean','DB1-Zn2O8-BTC_A-irmof8_A_No11','DB12-TIXVON_clean','DB1-Zn2O8-AZO_A-irmof8_A_No357','DB15-cds_N29_E173_opt','DB15-cds_N87_E131_opt']
if not casebook.empty:
    for mid in priority:
        hit=casebook[casebook['display_id'].eq(mid)]
        if not hit.empty: selected.append(hit.iloc[0])
    # fill remaining
    for _,r in casebook.iterrows():
        if len(selected)>=8: break
        if not any(str(x['display_id'])==str(r['display_id']) for x in selected): selected.append(r)
for i in range(8):
    ax=fig.add_subplot(gs[i//4,i%4],projection='3d')
    if i<len(selected):
        r=selected[i]; mid=r['display_id']; cif=find_cif_for_id(mid)
        role=str(r.get('paper_b_case_class','case')).replace('possible but not confident elite','possible elite').replace('ordinary high-rank contrast','ordinary contrast')
        txt=f"{role}\ntrue={fmt_num(r.get('y_true'),2)}, pred={fmt_num(r.get('y_pred'),2)}\nresid={fmt_num(r.get('residual'),2)}; PLD={fmt_num(r.get('PLD'),2)} Å\nAVA$_f$={fmt_num(r.get('AVAf'),3)}, ASA={fmt_num(r.get('ASA'),0)}"
        render_cif_panel(ax,cif,title=f"{chr(97+i)}) {str(mid)[:28]}",metrics_text=txt,view=(20+5*(i%2),35+20*(i%3)))
    else:
        ax.axis('off')
fig.suptitle('Figure 6. Structural casebook of positive surprises, fragile elites and negative controls',fontsize=15,fontweight='bold')
fig.tight_layout(rect=[0,0,1,.96])
save_all(fig, DIRS['fig_main']/'Figure_6_structural_casebook')

# SI figures
# S1 target audit summary as a figure
fig,axes=plt.subplots(1,2,figsize=(12,4.8),dpi=160)
ax=axes[0]
if not targets.empty:
    ax.bar(np.arange(len(targets)), targets['mean'], yerr=targets['std'], color=CYCLE[:len(targets)], edgecolor='white', capsize=3)
    ax.set_xticks(np.arange(len(targets))); ax.set_xticklabels(targets['target_description'],rotation=25,ha='right')
style(ax,'Target means and spread','Target','Uptake mean ± SD'); panel(ax,'a')
ax=axes[1]
if not targets.empty:
    ax.scatter(targets['median'],targets['max'],s=100,c=CYCLE[:len(targets)],edgecolor='white')
    for _,r in targets.iterrows(): ax.text(r['median'],r['max'],r['target_key'],fontsize=7)
style(ax,'Target dynamic range','Median','Maximum'); panel(ax,'b')
fig.suptitle('Figure S1. Target detection and adsorption-value summary',fontsize=13,fontweight='bold')
fig.tight_layout(rect=[0,0,1,.92])
save_all(fig, DIRS['fig_si']/'Figure_S1_target_detection_summary')
targets.to_csv(DIRS['src_si']/'Figure_S1_target_detection_summary.csv',index=False)

# S2 feature/split diagnostics
fig,axes=plt.subplots(1,2,figsize=(12,4.8),dpi=160)
ax=axes[0]
if not redundancy.empty and 'feature_family' in redundancy.columns:
    # Try generic columns
    count_col = 'n_columns' if 'n_columns' in redundancy.columns else ([c for c in redundancy.columns if 'n' in c.lower() and 'col' in c.lower()] or [None])[0]
    if count_col:
        rr=redundancy.copy().head(10)
        ax.barh(np.arange(len(rr)),pd.to_numeric(rr[count_col],errors='coerce'),color=COL['blue'],edgecolor='white')
        ax.set_yticks(np.arange(len(rr))); ax.set_yticklabels(rr['feature_family'],fontsize=7)
style(ax,'Feature-family redundancy/QC','Columns',''); panel(ax,'a')
ax=axes[1]
if not split_diag.empty:
    colors=[COL['teal'] if s=='available' else COL['red'] for s in split_diag['status']]
    ax.bar(np.arange(len(split_diag)),[1]*len(split_diag),color=colors,edgecolor='white')
    ax.set_xticks(np.arange(len(split_diag))); ax.set_xticklabels(split_diag['split_type'],rotation=30,ha='right')
    ax.set_yticks([])
    for i,r in split_diag.iterrows(): ax.text(i,.5,r['status'],ha='center',va='center',color='white',fontweight='bold',fontsize=7)
style(ax,'Split availability diagnostics','',''); panel(ax,'b')
fig.suptitle('Figure S2. Feature-family and split-design diagnostics',fontsize=13,fontweight='bold')
fig.tight_layout(rect=[0,0,1,.92])
save_all(fig, DIRS['fig_si']/'Figure_S2_feature_split_diagnostics')
if not redundancy.empty: redundancy.to_csv(DIRS['src_si']/'Figure_S2_feature_redundancy.csv',index=False)
split_diag.to_csv(DIRS['src_si']/'Figure_S2_split_diagnostics.csv',index=False)

# S3 full RMSE/S4 interval width/S5 coverage distribution copy/redraw from SI data
for si_name, title, src_rel in [('Figure_S3_full_rmse_by_target_split','Full RMSE by target and split','figure_data/si/Figure_S1_target_split_rmse.csv'),('Figure_S4_full_interval_width_by_target_split','Full interval width by target and split','figure_data/si/Figure_S2_target_interval_width.csv')]:
    df=rd(src_rel,index_col=0)
    fig,ax=plt.subplots(figsize=(6.2,4.8),dpi=160)
    if not df.empty:
        im=ax.imshow(df.values,aspect='auto',cmap=LinearSegmentedColormap.from_list('hm',['#F7FBFF',COL['sky'],COL['blue'],COL['ink']]))
        ax.set_xticks(range(df.shape[1])); ax.set_xticklabels(df.columns,rotation=30,ha='right')
        ax.set_yticks(range(df.shape[0])); ax.set_yticklabels(df.index)
        for i in range(df.shape[0]):
            for j in range(df.shape[1]): ax.text(j,i,f'{df.values[i,j]:.2g}',ha='center',va='center',fontsize=7)
        fig.colorbar(im,ax=ax,fraction=.046,pad=.04)
        df.to_csv(DIRS['src_si']/(si_name+'.csv'))
    style(ax,title,'Target','Split',grid=False); fig.tight_layout(); save_all(fig, DIRS['fig_si']/si_name)

covdist=rd('figure_data/si/Figure_S3_coverage_distribution.csv')
fig,ax=plt.subplots(figsize=(6.5,4.5),dpi=160)
if not covdist.empty:
    col=[c for c in covdist.columns if 'coverage' in c.lower()][0]
    ax.hist(pd.to_numeric(covdist[col],errors='coerce').dropna(),bins=30,color=COL['teal'],edgecolor='white')
    ax.axvline(.90,ls='--',color=COL['muted'],lw=1)
    covdist.to_csv(DIRS['src_si']/'Figure_S5_coverage_distribution.csv',index=False)
style(ax,'Coverage distribution across completed jobs','Empirical coverage','Jobs'); fig.tight_layout(); save_all(fig, DIRS['fig_si']/'Figure_S5_coverage_distribution')

# S6 filter sensitivity
fig,axes=plt.subplots(1,2,figsize=(12,4.8),dpi=160)
if not filter_sens.empty:
    fs=filter_sens.copy(); fs.to_csv(DIRS['src_si']/'Figure_S6_filter_sensitivity.csv',index=False)
    ax=axes[0]
    ax.barh(np.arange(len(fs)),fs['n_rows'],color=COL['blue'],edgecolor='white')
    ax.set_yticks(np.arange(len(fs))); ax.set_yticklabels(fs['filter'],fontsize=7)
    style(ax,'Rows retained by QC/filter','Rows',''); panel(ax,'a')
    ax=axes[1]
    cols=[c for c in fs.columns if c.startswith('rate_positive_anomaly_90')]
    if cols:
        ax.barh(np.arange(len(fs)),fs[cols[0]],color=COL['orange'],edgecolor='white')
        ax.set_yticks(np.arange(len(fs))); ax.set_yticklabels(fs['filter'],fontsize=7)
    style(ax,'Positive-surprise rate after filtering','Rate',''); panel(ax,'b')
fig.suptitle('Figure S6. Filter sensitivity and zero/inaccessible-pore robustness',fontsize=13,fontweight='bold')
fig.tight_layout(rect=[0,0,1,.92]); save_all(fig, DIRS['fig_si']/'Figure_S6_filter_sensitivity')

# S7 external/process full casebook diagnostics
fig,axes=plt.subplots(1,3,figsize=(14,4.8),dpi=160)
ax=axes[0]
if not external.empty:
    ec=external['external_match_confidence'].value_counts()
    ax.bar(np.arange(len(ec)),ec.values,color=COL['teal'],edgecolor='white')
    ax.set_xticks(np.arange(len(ec))); ax.set_xticklabels(ec.index,rotation=25,ha='right')
style(ax,'External match confidence','Tier','Count'); panel(ax,'a')
ax=axes[1]
if not process.empty:
    pc=process['process_proxy_status'].value_counts()
    ax.barh(np.arange(len(pc)),pc.values,color=[COL['teal'] if 'no' in str(x).lower() else COL['orange'] for x in pc.index],edgecolor='white')
    ax.set_yticks(np.arange(len(pc))); ax.set_yticklabels(pc.index,fontsize=7)
style(ax,'Process-proxy status','Count',''); panel(ax,'b')
ax=axes[2]
if not casebook.empty:
    cc=casebook['paper_b_case_class'].value_counts()
    ax.barh(np.arange(len(cc)),cc.values,color=COL['purple'],edgecolor='white')
    ax.set_yticks(np.arange(len(cc))); ax.set_yticklabels(cc.index,fontsize=6.5)
style(ax,'Casebook class balance','Count',''); panel(ax,'c')
fig.suptitle('Figure S7. External/provenance and process-triage diagnostics',fontsize=13,fontweight='bold')
fig.tight_layout(rect=[0,0,1,.92]); save_all(fig, DIRS['fig_si']/'Figure_S7_external_process_casebook_diagnostics')
if not external.empty: external.to_csv(DIRS['src_si']/'Figure_S7_external_match_source.csv',index=False)
if not process.empty: process.to_csv(DIRS['src_si']/'Figure_S7_process_triage_source.csv',index=False)

# S8 full structural gallery all 14
if not casebook.empty:
    n=len(casebook); cols=4; rows=math.ceil(n/cols)
    fig=plt.figure(figsize=(13,3.5*rows),dpi=140)
    gs=fig.add_gridspec(rows,cols,wspace=.06,hspace=.16)
    for i,(_,r) in enumerate(casebook.iterrows()):
        ax=fig.add_subplot(gs[i//cols,i%cols],projection='3d')
        mid=r['display_id']; cif=find_cif_for_id(mid)
        txt=f"{str(r.get('paper_b_case_class',''))[:25]}\nresid={fmt_num(r.get('residual'),2)}; PLD={fmt_num(r.get('PLD'),2)} Å\nproc={str(r.get('process_proxy_status',''))[:22]}"
        render_cif_panel(ax,cif,title=f"{i+1}. {str(mid)[:28]}",metrics_text=txt,view=(20+5*(i%2),35+12*(i%4)))
    fig.suptitle('Figure S8. Full structural casebook gallery from selected CIFs',fontsize=13,fontweight='bold')
    fig.tight_layout(rect=[0,0,1,.96]); save_all(fig, DIRS['fig_si']/'Figure_S8_full_structural_casebook_gallery')
    casebook.to_csv(DIRS['src_si']/'Figure_S8_casebook_gallery_source.csv',index=False)

# -----------------------------------------------------------------------------
# Source-data index and figure/table plan
# -----------------------------------------------------------------------------
figure_plan = [
    ['Figure 1','From ranked MOF screening to conformal discovery certificates','Conceptual workflow; target availability; split availability; output classes','Main text'],
    ['Figure 2','Calibration and efficiency across targets, models, and split types','Coverage deviation; interval width; RMSE/coverage tradeoff; adaptive-vs-standard intervals','Main text'],
    ['Figure 3','From top-k rankings to conformal screening certificates','Decision class counts; risk-gate tradeoff; decision map; seed stability','Main text'],
    ['Figure 4','Conformal residual-anomaly atlas','Positive residuals; negative residuals; p-values; target-wise anomaly rates','Main text'],
    ['Figure 5','Chemical regimes behind positive surprises','Enrichment odds ratios; pore-quintile rates; mechanism classes; external/process triage','Main text'],
    ['Figure 6','Structural casebook','Eight CIF-rendered candidate structures with metrics annotations','Main text'],
    ['Figure S1','Target detection summary','Target mean/spread and dynamic range','SI'],
    ['Figure S2','Feature/split diagnostics','Feature family redundancy and split availability','SI'],
    ['Figure S3','Full RMSE heatmap','RMSE by target and split','SI'],
    ['Figure S4','Full interval-width heatmap','Interval width by target and split','SI'],
    ['Figure S5','Coverage distribution','Empirical coverage distribution across jobs','SI'],
    ['Figure S6','Filter sensitivity','QC/filter retained rows and positive-surprise rate','SI'],
    ['Figure S7','External/process/casebook diagnostics','Match tiers, process proxy, case classes','SI'],
    ['Figure S8','Full structural gallery','All 14 selected CIF-rendered structures','SI'],
]
figplan=pd.DataFrame(figure_plan,columns=['Item','Title','Panels/data','Placement'])
save_table(figplan, DIRS['reports']/'FINAL_FIGURE_PLAN.csv', DIRS['reports']/'FINAL_FIGURE_PLAN.tex', 'Final figure plan for the conformal MOF anomaly manuscript.', 'tab:figure_plan')

table_plan = [
    ['Table 1','Condensed calibration and screening summary','Best random and grouped setups per target','Main text'],
    ['Table 2','Certificate decision classes and risk','Candidate counts and false-elite/recall gate tradeoff','Main text'],
    ['Table 3','Positive-surprise enrichment','Top enriched pore/descriptive regimes','Main text'],
    ['Table 4','Structural casebook shortlist','8–10 main structure examples with metrics/provenance/process annotations','Main text or SI depending on page limits'],
    ['Tables S1–S21','Full audit/model/split/filter/external/casebook tables','Copied and curated from v5 outputs','SI'],
]
tabplan=pd.DataFrame(table_plan,columns=['Item','Title','Data included','Placement'])
save_table(tabplan, DIRS['reports']/'FINAL_TABLE_PLAN.csv', DIRS['reports']/'FINAL_TABLE_PLAN.tex', 'Final table plan for the conformal MOF anomaly manuscript.', 'tab:table_plan')

# Copy/generate the script itself into package for reproducibility
script_text = Path(__file__).read_text(encoding='utf-8')
(DIRS['code']/'prepare_final_publication_figures_tables.py').write_text(script_text, encoding='utf-8')

# README
summary = {
    'created_at': datetime.now().isoformat(timespec='seconds'),
    'input_zip': str(Path('/mnt/data/All_Results_2_Conformal_Anomaly.zip')),
    'n_main_figures': len(list(DIRS['fig_main'].glob('*.png'))),
    'n_si_figures': len(list(DIRS['fig_si'].glob('*.png'))),
    'n_structure_panels': len(list(DIRS['fig_struct'].glob('*.png'))),
    'n_main_tables': len(list(DIRS['tables_main'].glob('*.csv'))),
    'n_si_tables': len(list(DIRS['tables_si'].glob('*'))),
    'notes': 'Main structure panels are automatically rendered from CIF coordinates and should still be manually polished in VESTA/PyMOL/OVITO for final journal submission.'
}
(DIRS['reports']/'PACKAGE_SUMMARY.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
readme = f"""
# Conformal MOF Anomaly Screening — Final Publication Package

Generated: {summary['created_at']}

This package converts the completed v5 conformal MOF anomaly-screening outputs into a high-impact-journal-ready figure/table/source-data package. It does **not** rerun the expensive ML jobs.

## Contents

- `figures/main/`: polished main-text Figures 1–6 in PNG/PDF/SVG.
- `figures/si/`: polished SI Figures S1–S8 in PNG/PDF/SVG.
- `figures/structure_panels/`: individual automatic CIF-rendered panels for each selected casebook MOF.
- `tables/main/`: final main-text tables as CSV.
- `latex_tables/main/`: final main-text tables as LaTeX.
- `tables/si/`: SI tables and copied QC/casebook tables.
- `source_data/`: source data CSVs for every newly prepared figure and table.
- `structural_case_studies/selected_cifs/`: selected CIFs copied from the v5 output.
- `code/prepare_final_publication_figures_tables.py`: standalone script used to regenerate this package from the result folder.
- `code/conformal_mof_anomaly_screening_pipeline_paperB_v5_highimpact_qc.py`: original v5 ML/QC pipeline used to generate the results.
- `reports/FINAL_FIGURE_PLAN.csv` and `reports/FINAL_TABLE_PLAN.csv`: final manuscript/SI plan.

## Suggested main-text package

1. Figure 1 — from ranked screening to conformal discovery certificates.
2. Figure 2 — calibration and efficiency across targets, models and split types.
3. Figure 3 — certificate triage and false-elite risk.
4. Figure 4 — conformal residual-anomaly atlas.
5. Figure 5 — chemical regimes behind positive surprises.
6. Figure 6 — structural casebook.

Main Tables 1–4 are prepared under `tables/main/` and `latex_tables/main/`.

## Important manual finalization step

The structural panels are automatic coordinate renderings from CIFs. They are useful for drafting and source-data traceability, but for a very high-IF journal the final structure images should still be manually polished in VESTA, PyMOL, OVITO or Mercury: pore-window view, PLD/LCD arrows, clean atom colors, and consistent orientation.

## Re-running the publication preparation

Edit `BASE`, `OUT`, and `ZIP_OUT` at the top of `code/prepare_final_publication_figures_tables.py`, then run:

```bash
python code/prepare_final_publication_figures_tables.py
```

"""
(OUT/'README.md').write_text(readme, encoding='utf-8')

# Final package manifest
rows=[]
for p in OUT.rglob('*'):
    if p.is_file():
        rows.append({'relative_path':str(p.relative_to(OUT)), 'size_bytes':p.stat().st_size})
manifest=pd.DataFrame(rows).sort_values('relative_path')
manifest.to_csv(OUT/'PACKAGE_FILE_MANIFEST.csv',index=False)

# Zip package
with zipfile.ZipFile(ZIP_OUT,'w',zipfile.ZIP_DEFLATED) as z:
    for p in OUT.rglob('*'):
        if p.is_file(): z.write(p, p.relative_to(OUT))

print(json.dumps(summary, indent=2))
print('PACKAGE', ZIP_OUT, ZIP_OUT.stat().st_size)
