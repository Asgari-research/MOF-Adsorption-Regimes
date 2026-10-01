#!/usr/bin/env python3
from __future__ import annotations
import os, re, json, shutil, zipfile, math
from pathlib import Path
from datetime import datetime
from typing import Any
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import patheffects as pe
from matplotlib.colors import LinearSegmentedColormap

BASE=Path('/mnt/data/conformal_final_work/input')
OUT=Path('/mnt/data/conformal_mof_final_publication_package')
ZIP_OUT=Path('/mnt/data/conformal_mof_final_publication_package.zip')
PIPELINE_CODE=Path('/mnt/data/conformal_mof_anomaly_screening_pipeline_paperB_v5_highimpact_qc.py')
MANIFEST_CSV=Path('/mnt/data/PUBLICATION_OUTPUT_MANIFEST.csv')
COL={'ink':'#1F2933','muted':'#64748B','grid':'#DDE6EF','blue':'#2F6B9A','sky':'#8DC1DD','teal':'#2A9D8F','green':'#5A9C6E','gold':'#E3A12E','orange':'#E76F51','red':'#B23A48','purple':'#6D5A9C','grey':'#A6AFB8','pale_blue':'#EAF4FA','pale_teal':'#E8F5F2','pale_orange':'#FFF1E8'}
CYCLE=[COL['blue'],COL['orange'],COL['teal'],COL['purple'],COL['gold'],COL['green'],COL['red'],COL['grey']]
plt.rcParams.update({'figure.facecolor':'white','axes.facecolor':'white','savefig.facecolor':'white','font.family':'DejaVu Sans','font.size':8.5,'axes.titlesize':10.5,'axes.titleweight':'bold','axes.labelsize':9,'xtick.labelsize':7.5,'ytick.labelsize':7.5,'axes.edgecolor':COL['ink'],'axes.linewidth':0.8,'grid.color':COL['grid'],'grid.alpha':0.65,'legend.frameon':False,'legend.fontsize':7,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none'})

def rd(rel, **kw):
    p=BASE/rel
    return pd.read_csv(p, **kw) if p.exists() else pd.DataFrame()
def slug(x):
    return re.sub(r'[^a-z0-9]+','_',str(x).lower()).strip('_') or 'item'
def style(ax,title='',xlabel='',ylabel='',grid=True):
    ax.set_title(title,pad=8,color=COL['ink']); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    if grid: ax.grid(True,zorder=0)
    ax.set_axisbelow(True)
    for sp in ['top','right']: ax.spines[sp].set_visible(False)
    for sp in ['left','bottom']: ax.spines[sp].set_color(COL['ink'])
    ax.tick_params(colors=COL['muted'])
def panel(ax,lab):
    t=ax.text(-.12,1.10,lab,transform=ax.transAxes,fontsize=13,fontweight='bold',color=COL['ink'],ha='left',va='top',bbox=dict(boxstyle='round,pad=.18',facecolor='white',edgecolor=COL['grid'],lw=.6),zorder=20)
    t.set_path_effects([pe.withStroke(linewidth=2.5, foreground='white')])
def save_fig(fig, base, svg=True):
    base=Path(base); base.parent.mkdir(parents=True,exist_ok=True)
    for ext in (['png','pdf','svg'] if svg else ['png','pdf']):
        fig.savefig(base.with_suffix('.'+ext), dpi=360 if ext=='png' else 240, bbox_inches='tight', facecolor='white')
    plt.close(fig)
def fmt(x,nd=2):
    try:
        if pd.isna(x): return ''
        return f'{float(x):.{nd}f}'
    except: return str(x)
def save_table(df, path):
    path.parent.mkdir(parents=True,exist_ok=True); df.to_csv(path,index=False)

# dirs
for d in ['figures/main','figures/si','source_data/si_figures','source_data/main_figures','reports','code','manifests']:
    (OUT/d).mkdir(parents=True,exist_ok=True)

# Data
casebook=rd('paper_b_impact_additions/tables/paper_b_candidate_casebook_compact.csv')
targets=rd('tables/si/table_si_detected_targets_and_availability.csv')
split_diag=rd('qc_reports/split_availability_diagnostics.csv')
redundancy=rd('qc_reports/feature_family_redundancy_report.csv')
filter_sens=rd('qc_reports/posthoc_filter_sensitivity_report.csv')
external=rd('paper_b_impact_additions/tables/paper_b_external_match_strict_validation.csv')
process=rd('paper_b_impact_additions/tables/paper_b_process_stability_triage.csv')
covdist=rd('figure_data/si/Figure_S3_coverage_distribution.csv')

# Figure 6 composite from existing individual structure panels
priority=['DB12-OSAVUA_clean','DB12-VAGTAA01_clean','DB12-VAGTAA_clean','DB1-Zn2O8-BTC_A-irmof8_A_No11','DB12-TIXVON_clean','DB1-Zn2O8-AZO_A-irmof8_A_No357','DB15-cds_N29_E173_opt','DB15-cds_N87_E131_opt']
selected=[]
if not casebook.empty:
    for mid in priority:
        hit=casebook[casebook['display_id'].eq(mid)]
        if not hit.empty: selected.append(hit.iloc[0])
    for _,r in casebook.iterrows():
        if len(selected)>=8: break
        if not any(str(x['display_id'])==str(r['display_id']) for x in selected): selected.append(r)
fig,axes=plt.subplots(2,4,figsize=(14,8.8),dpi=160)
for i,ax in enumerate(axes.ravel()):
    ax.axis('off')
    if i<len(selected):
        r=selected[i]; mid=r['display_id']; png=OUT/'figures/structure_panels'/(slug(mid)+'_structure_panel.png')
        if png.exists():
            img=plt.imread(png); ax.imshow(img)
        ax.text(0.02,0.98,chr(97+i),transform=ax.transAxes,ha='left',va='top',fontsize=13,fontweight='bold',bbox=dict(facecolor='white',edgecolor=COL['grid'],boxstyle='round,pad=.15'))
        label=f"{mid}\n{str(r.get('paper_b_case_class',''))}\ntrue={fmt(r.get('y_true'))}, pred={fmt(r.get('y_pred'))}, resid={fmt(r.get('residual'))}"
        ax.text(0.02,0.02,label,transform=ax.transAxes,ha='left',va='bottom',fontsize=6.5,bbox=dict(facecolor='white',edgecolor=COL['grid'],alpha=.86,boxstyle='round,pad=.25'))
fig.suptitle('Figure 6. Structural casebook of positive surprises, fragile elites and negative controls',fontsize=15,fontweight='bold')
fig.tight_layout(rect=[0,0,1,.96])
save_fig(fig, OUT/'figures/main/Figure_6_structural_casebook', svg=False)
if not casebook.empty: casebook.to_csv(OUT/'source_data/main_figures/Figure_6_structural_casebook_source.csv',index=False)

# SI Figure S1
fig,axes=plt.subplots(1,2,figsize=(12,4.8),dpi=160)
ax=axes[0]
if not targets.empty:
    ax.bar(np.arange(len(targets)),targets['mean'],yerr=targets['std'],color=CYCLE[:len(targets)],edgecolor='white',capsize=3)
    ax.set_xticks(np.arange(len(targets))); ax.set_xticklabels(targets['target_description'],rotation=25,ha='right')
style(ax,'Target means and spread','Target','Uptake mean ± SD'); panel(ax,'a')
ax=axes[1]
if not targets.empty:
    ax.scatter(targets['median'],targets['max'],s=100,c=CYCLE[:len(targets)],edgecolor='white')
    for _,r in targets.iterrows(): ax.text(r['median'],r['max'],r['target_key'],fontsize=7)
style(ax,'Target dynamic range','Median','Maximum'); panel(ax,'b')
fig.suptitle('Figure S1. Target detection and adsorption-value summary',fontsize=13,fontweight='bold')
fig.tight_layout(rect=[0,0,1,.92]); save_fig(fig,OUT/'figures/si/Figure_S1_target_detection_summary')
if not targets.empty: targets.to_csv(OUT/'source_data/si_figures/Figure_S1_target_detection_summary.csv',index=False)

# SI Figure S2
fig,axes=plt.subplots(1,2,figsize=(12,4.8),dpi=160)
ax=axes[0]
if not redundancy.empty:
    count_col = 'n_columns' if 'n_columns' in redundancy.columns else ([c for c in redundancy.columns if 'column' in c.lower()] or [None])[0]
    if count_col and 'feature_family' in redundancy.columns:
        rr=redundancy.head(12).copy()
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
fig.tight_layout(rect=[0,0,1,.92]); save_fig(fig,OUT/'figures/si/Figure_S2_feature_split_diagnostics')
if not redundancy.empty: redundancy.to_csv(OUT/'source_data/si_figures/Figure_S2_feature_redundancy.csv',index=False)
if not split_diag.empty: split_diag.to_csv(OUT/'source_data/si_figures/Figure_S2_split_diagnostics.csv',index=False)

# SI heatmaps
for si_name,title,src_rel in [('Figure_S3_full_rmse_by_target_split','Full RMSE by target and split','figure_data/si/Figure_S1_target_split_rmse.csv'),('Figure_S4_full_interval_width_by_target_split','Full interval width by target and split','figure_data/si/Figure_S2_target_interval_width.csv')]:
    df=rd(src_rel,index_col=0)
    fig,ax=plt.subplots(figsize=(6.2,4.8),dpi=160)
    if not df.empty:
        im=ax.imshow(df.values,aspect='auto',cmap=LinearSegmentedColormap.from_list('hm',['#F7FBFF',COL['sky'],COL['blue'],COL['ink']]))
        ax.set_xticks(range(df.shape[1])); ax.set_xticklabels(df.columns,rotation=30,ha='right')
        ax.set_yticks(range(df.shape[0])); ax.set_yticklabels(df.index)
        for ii in range(df.shape[0]):
            for jj in range(df.shape[1]): ax.text(jj,ii,f'{df.values[ii,jj]:.2g}',ha='center',va='center',fontsize=7)
        fig.colorbar(im,ax=ax,fraction=.046,pad=.04)
        df.to_csv(OUT/'source_data/si_figures'/(si_name+'.csv'))
    style(ax,title,'Target','Split',grid=False); fig.tight_layout(); save_fig(fig,OUT/'figures/si'/si_name)

# SI coverage distribution
fig,ax=plt.subplots(figsize=(6.5,4.5),dpi=160)
if not covdist.empty:
    col=next((c for c in covdist.columns if 'coverage' in c.lower()),covdist.columns[-1])
    ax.hist(pd.to_numeric(covdist[col],errors='coerce').dropna(),bins=30,color=COL['teal'],edgecolor='white')
    ax.axvline(.90,ls='--',color=COL['muted'],lw=1)
    covdist.to_csv(OUT/'source_data/si_figures/Figure_S5_coverage_distribution.csv',index=False)
style(ax,'Coverage distribution across completed jobs','Empirical coverage','Jobs')
fig.tight_layout(); save_fig(fig,OUT/'figures/si/Figure_S5_coverage_distribution')

# SI filter sensitivity
fig,axes=plt.subplots(1,2,figsize=(12,4.8),dpi=160)
if not filter_sens.empty:
    fs=filter_sens.copy()
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
    fs.to_csv(OUT/'source_data/si_figures/Figure_S6_filter_sensitivity.csv',index=False)
fig.suptitle('Figure S6. Filter sensitivity and zero/inaccessible-pore robustness',fontsize=13,fontweight='bold')
fig.tight_layout(rect=[0,0,1,.92]); save_fig(fig,OUT/'figures/si/Figure_S6_filter_sensitivity')

# SI external/process
fig,axes=plt.subplots(1,3,figsize=(14,4.8),dpi=160)
ax=axes[0]
if not external.empty:
    ec=external['external_match_confidence'].value_counts(); ax.bar(np.arange(len(ec)),ec.values,color=COL['teal'],edgecolor='white'); ax.set_xticks(np.arange(len(ec))); ax.set_xticklabels(ec.index,rotation=25,ha='right')
style(ax,'External match confidence','Tier','Count'); panel(ax,'a')
ax=axes[1]
if not process.empty:
    pc=process['process_proxy_status'].value_counts(); ax.barh(np.arange(len(pc)),pc.values,color=[COL['teal'] if 'no' in str(x).lower() else COL['orange'] for x in pc.index],edgecolor='white'); ax.set_yticks(np.arange(len(pc))); ax.set_yticklabels(pc.index,fontsize=7)
style(ax,'Process-proxy status','Count',''); panel(ax,'b')
ax=axes[2]
if not casebook.empty:
    cc=casebook['paper_b_case_class'].value_counts(); ax.barh(np.arange(len(cc)),cc.values,color=COL['purple'],edgecolor='white'); ax.set_yticks(np.arange(len(cc))); ax.set_yticklabels(cc.index,fontsize=6.5)
style(ax,'Casebook class balance','Count',''); panel(ax,'c')
fig.suptitle('Figure S7. External/provenance and process-triage diagnostics',fontsize=13,fontweight='bold')
fig.tight_layout(rect=[0,0,1,.92]); save_fig(fig,OUT/'figures/si/Figure_S7_external_process_casebook_diagnostics')
if not external.empty: external.to_csv(OUT/'source_data/si_figures/Figure_S7_external_match_source.csv',index=False)
if not process.empty: process.to_csv(OUT/'source_data/si_figures/Figure_S7_process_triage_source.csv',index=False)

# SI full structural gallery from images
if not casebook.empty:
    n=len(casebook); cols=4; rows=math.ceil(n/cols)
    fig,axes=plt.subplots(rows,cols,figsize=(14,3.4*rows),dpi=140)
    axes=np.array(axes).reshape(rows,cols)
    for ax in axes.ravel(): ax.axis('off')
    for i,(_,r) in enumerate(casebook.iterrows()):
        ax=axes[i//cols,i%cols]; mid=r['display_id']; png=OUT/'figures/structure_panels'/(slug(mid)+'_structure_panel.png')
        if png.exists(): ax.imshow(plt.imread(png))
        ax.text(.02,.98,str(i+1),transform=ax.transAxes,ha='left',va='top',fontsize=10,fontweight='bold',bbox=dict(facecolor='white',edgecolor=COL['grid'],boxstyle='round,pad=.15'))
    fig.suptitle('Figure S8. Full structural casebook gallery from selected CIFs',fontsize=13,fontweight='bold')
    fig.tight_layout(rect=[0,0,1,.96]); save_fig(fig,OUT/'figures/si/Figure_S8_full_structural_casebook_gallery',svg=False)
    casebook.to_csv(OUT/'source_data/si_figures/Figure_S8_casebook_gallery_source.csv',index=False)

# Figure/table plans
figure_plan = pd.DataFrame([
 ['Figure 1','From ranked MOF screening to conformal discovery certificates','Conceptual workflow; target availability; split availability; output classes','Main text'],
 ['Figure 2','Calibration and efficiency across targets, models, and split types','Coverage deviation; interval width; RMSE/coverage tradeoff; adaptive-vs-standard intervals','Main text'],
 ['Figure 3','From top-k rankings to conformal screening certificates','Decision class counts; risk-gate tradeoff; decision map; seed stability','Main text'],
 ['Figure 4','Conformal residual-anomaly atlas','Positive residuals; negative residuals; p-values; target-wise anomaly rates','Main text'],
 ['Figure 5','Chemical regimes behind positive surprises','Enrichment odds ratios; pore-quintile rates; mechanism classes; external/process triage','Main text'],
 ['Figure 6','Structural casebook','Eight CIF-rendered candidate structures with metrics annotations','Main text'],
 ['Figures S1–S8','SI audit/robustness/casebook gallery','Target/split/model/filter/external/structure diagnostics','SI']],columns=['Item','Title','Panels/data','Placement'])
table_plan = pd.DataFrame([
 ['Table 1','Condensed calibration and screening summary','Best random and grouped setups per target','Main text'],
 ['Table 2','Certificate decision classes and risk','Candidate counts and false-elite/recall gate tradeoff','Main text'],
 ['Table 3','Positive-surprise enrichment','Top enriched pore/descriptive regimes','Main text'],
 ['Table 4','Structural casebook shortlist','8–10 examples with metrics/provenance/process annotations','Main text or SI'],
 ['Tables S1–S21','Full audit/model/split/filter/external/casebook tables','Copied and curated from v5 outputs','SI']],columns=['Item','Title','Data included','Placement'])
save_table(figure_plan,OUT/'reports/FINAL_FIGURE_PLAN.csv'); save_table(table_plan,OUT/'reports/FINAL_TABLE_PLAN.csv')

# copy code and manifest
if PIPELINE_CODE.exists(): shutil.copy2(PIPELINE_CODE, OUT/'code'/PIPELINE_CODE.name)
# include both the finish script and big script if present
shutil.copy2(Path(__file__), OUT/'code/finish_publication_package.py')
big=Path('/mnt/data/conformal_final_work/create_final_publication_package.py')
if big.exists(): shutil.copy2(big, OUT/'code/prepare_final_publication_figures_tables_full.py')
if MANIFEST_CSV.exists(): shutil.copy2(MANIFEST_CSV, OUT/'manifests'/MANIFEST_CSV.name)

# README and summary
summary={'created_at':datetime.now().isoformat(timespec='seconds'),'n_main_figures_png':len(list((OUT/'figures/main').glob('*.png'))),'n_si_figures_png':len(list((OUT/'figures/si').glob('*.png'))),'n_structure_panels_png':len(list((OUT/'figures/structure_panels').glob('*.png'))),'n_main_tables_csv':len(list((OUT/'tables/main').glob('*.csv'))),'n_si_tables_files':len(list((OUT/'tables/si').glob('*'))),'manual_note':'Automatic CIF renders are included for drafting/source traceability; final high-IF structural panels should still be manually polished in VESTA/PyMOL/OVITO.'}
(OUT/'reports/PACKAGE_SUMMARY.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
readme=f"""# Conformal MOF Anomaly Screening — Final Publication Package

Generated: {summary['created_at']}

This package contains polished main-text and SI figures, source-data CSV files, main/SI tables, selected CIFs, the v5 pipeline code, and publication-preparation scripts.

## Main-text material
- `figures/main/`: Figures 1–6 in PNG/PDF/SVG where appropriate. Figure 6 uses automatic CIF-rendered panels in PNG/PDF.
- `tables/main/`: four main-text CSV tables.
- `latex_tables/main/`: LaTeX versions of the main tables.
- `source_data/main_figures/`: source data for each main figure panel.

## SI and reproducibility material
- `figures/si/`: Figures S1–S8.
- `tables/si/`: copied and curated SI/QC/casebook tables from the v5 results.
- `source_data/si_figures/`: source data for each SI figure.
- `structural_case_studies/selected_cifs/`: selected CIFs.
- `figures/structure_panels/`: individual automatic CIF renderings.
- `code/`: the original v5 pipeline and the figure/table preparation scripts.
- `reports/FINAL_FIGURE_PLAN.csv` and `reports/FINAL_TABLE_PLAN.csv`: recommended placement and rationale.

## Manual finalization note
The automatic CIF panels are useful and visually serviceable, but for a very high-impact journal I still recommend manually polishing the final Figure 6 structural panels in VESTA, PyMOL, OVITO, or Mercury, including pore-window orientation and PLD/LCD arrows.
"""
(OUT/'README.md').write_text(readme,encoding='utf-8')

# Package manifest and zip
rows=[]
for p in OUT.rglob('*'):
    if p.is_file(): rows.append({'relative_path':str(p.relative_to(OUT)),'size_bytes':p.stat().st_size})
pd.DataFrame(rows).sort_values('relative_path').to_csv(OUT/'PACKAGE_FILE_MANIFEST.csv',index=False)
if ZIP_OUT.exists(): ZIP_OUT.unlink()
with zipfile.ZipFile(ZIP_OUT,'w',zipfile.ZIP_DEFLATED) as z:
    for p in OUT.rglob('*'):
        if p.is_file(): z.write(p,p.relative_to(OUT))
print(json.dumps(summary,indent=2))
print(ZIP_OUT, ZIP_OUT.stat().st_size)
