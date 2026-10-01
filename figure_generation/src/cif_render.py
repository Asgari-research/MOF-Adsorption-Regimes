from pathlib import Path
import math
import re
import numpy as np
import pandas as pd

# Conservative, publication-friendly atom colors; includes Li/Cd requested by the audit.
E_COL = {
    "H":"#C7CDD3", "C":"#4B5563", "N":"#3973B7", "O":"#E76F51", "F":"#3AA17E",
    "P":"#C9892B", "S":"#E6B13C", "Cl":"#62A85A", "Br":"#8B5FA8", "I":"#7652A7",
    "Zn":"#7765A8", "Cu":"#B87333", "Ca":"#3AA6A0", "Zr":"#4F87C6", "Fe":"#B54855",
    "Mg":"#82B93E", "Al":"#9AA4AF", "Ni":"#4E9B62", "Co":"#4E72B8", "Mn":"#D77A35",
    "Li":"#A66FB5", "Cd":"#7D8793",
}
COV_RAD = {
    "H":0.31,"C":0.76,"N":0.71,"O":0.66,"F":0.57,"P":1.07,"S":1.05,"Cl":1.02,"Br":1.20,"I":1.39,
    "Zn":1.22,"Cu":1.32,"Ca":1.76,"Zr":1.54,"Fe":1.24,"Mg":1.41,"Al":1.21,"Ni":1.24,"Co":1.26,
    "Mn":1.39,"Li":1.28,"Cd":1.44,
}


def _num(token):
    token = str(token).strip().strip("'\"")
    token = re.sub(r"\([^)]*\)$", "", token)
    return float(token)


def cell_matrix(a,b,c,alpha,beta,gamma):
    al, be, ga = np.deg2rad([alpha,beta,gamma])
    va = np.array([a,0,0.])
    vb = np.array([b*np.cos(ga), b*np.sin(ga), 0.])
    cx = c*np.cos(be)
    cy = c*(np.cos(al)-np.cos(be)*np.cos(ga))/max(np.sin(ga),1e-12)
    cz = math.sqrt(max(c*c-cx*cx-cy*cy, 0.0))
    vc = np.array([cx,cy,cz])
    return np.vstack([va,vb,vc]).T


def parse_cif_listed_sites(path: Path):
    """Read unit-cell parameters and listed fractional atom sites.

    This fallback intentionally does NOT claim to expand arbitrary crystallographic
    symmetry operations. For final crystallographic renders, install OVITO/ASE and
    visually verify the result against the source CIF. The supplied handoff CIFs are
    still used directly; no generated or invented structures are substituted.
    """
    lines = path.read_text(errors="ignore").splitlines()
    vals = {}
    keys = ["_cell_length_a","_cell_length_b","_cell_length_c","_cell_angle_alpha","_cell_angle_beta","_cell_angle_gamma"]
    for line in lines:
        s = line.strip()
        for key in keys:
            if s.startswith(key):
                parts = s.split()
                if len(parts) >= 2:
                    try: vals[key] = _num(parts[1])
                    except Exception: pass
    if len(vals) < 6:
        return pd.DataFrame(), np.eye(3)
    M = cell_matrix(vals[keys[0]], vals[keys[1]], vals[keys[2]], vals[keys[3]], vals[keys[4]], vals[keys[5]])

    rows=[]
    i=0
    while i < len(lines):
        if lines[i].strip() == "loop_":
            j=i+1; headers=[]
            while j < len(lines) and lines[j].strip().startswith("_"):
                headers.append(lines[j].strip()); j+=1
            if all(k in headers for k in ["_atom_site_fract_x","_atom_site_fract_y","_atom_site_fract_z"]):
                while j < len(lines):
                    s=lines[j].strip()
                    if not s or s.startswith("_") or s=="loop_" or s.startswith("data_"):
                        break
                    parts=re.findall(r"'(?:[^']*)'|\"(?:[^\"]*)\"|\S+", s)
                    if len(parts) >= len(headers):
                        rec=dict(zip(headers,parts))
                        try:
                            raw_el=rec.get("_atom_site_type_symbol", rec.get("_atom_site_label","X"))
                            el=re.sub(r"[^A-Za-z]","",raw_el)
                            el = el[:1].upper()+el[1:2].lower() if el else "X"
                            fx,fy,fz=[_num(rec[k]) for k in ["_atom_site_fract_x","_atom_site_fract_y","_atom_site_fract_z"]]
                            frac=np.array([fx,fy,fz]); cart=M@frac
                            rows.append({"element":el,"fx":fx,"fy":fy,"fz":fz,"x":cart[0],"y":cart[1],"z":cart[2]})
                        except Exception:
                            pass
                    j+=1
                break
        i+=1
    return pd.DataFrame(rows), M


def _set_equal_3d(ax, coords, cell):
    pts = np.vstack([coords, np.zeros((1,3)), cell[:,0][None,:], cell[:,1][None,:], cell[:,2][None,:],
                     (cell[:,0]+cell[:,1]+cell[:,2])[None,:]])
    mins=pts.min(axis=0); maxs=pts.max(axis=0); center=(mins+maxs)/2; span=max(maxs-mins)*0.58
    if span <= 0: span = 1.0
    ax.set_xlim(center[0]-span, center[0]+span)
    ax.set_ylim(center[1]-span, center[1]+span)
    ax.set_zlim(center[2]-span, center[2]+span)
    try: ax.set_box_aspect((1,1,1))
    except Exception: pass


def render_cif(ax, cif_path: Path, title="", subtitle="", view=(20,-60), draw_bonds=True, title_fontsize=11.2, subtitle_fontsize=8.4):
    atoms, M = parse_cif_listed_sites(Path(cif_path))
    ax.set_axis_off()
    if atoms.empty:
        ax.text(.5,.54,"CIF parse failed",transform=ax.transAxes,ha="center",va="center",fontweight="bold")
        ax.text(.5,.42,Path(cif_path).name,transform=ax.transAxes,ha="center",va="center",fontsize=8)
        return
    coords=atoms[["x","y","z"]].to_numpy(float)
    els=atoms["element"].tolist()
    ax.view_init(elev=view[0], azim=view[1])
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.pane.set_facecolor((1,1,1,0)); axis.pane.set_edgecolor((1,1,1,0))
    ax.grid(False)

    # Draw local bonds only when the listed-site cell is not too dense. This is a
    # visual aid, not a bond-order assignment.
    if draw_bonds and len(atoms) <= 140:
        for i in range(len(atoms)):
            if els[i] == "H": continue
            for j in range(i+1,len(atoms)):
                if els[j] == "H": continue
                d=float(np.linalg.norm(coords[i]-coords[j]))
                cutoff=1.23*(COV_RAD.get(els[i],0.8)+COV_RAD.get(els[j],0.8))
                if 0.45 < d < cutoff:
                    ax.plot([coords[i,0],coords[j,0]],[coords[i,1],coords[j,1]],[coords[i,2],coords[j,2]],
                            color="#AAB3BA", lw=0.65, alpha=0.50, zorder=1)

    for el in sorted(set(els)):
        sub=atoms[atoms.element==el]
        if el == "H": size,alpha = 7,0.22
        elif el in {"C","N","O"}: size,alpha = 24,0.96
        else: size,alpha = 55,0.98
        ax.scatter(sub.x,sub.y,sub.z,s=size,c=E_COL.get(el,"#7D8793"),edgecolors="white",linewidths=.30,
                   alpha=alpha,depthshade=True,zorder=3)

    O=np.zeros(3); a,b,c=M[:,0],M[:,1],M[:,2]
    vertices=[O,a,b,c,a+b,a+c,b+c,a+b+c]
    edges=[(0,1),(0,2),(0,3),(1,4),(1,5),(2,4),(2,6),(3,5),(3,6),(4,7),(5,7),(6,7)]
    for u,v in edges:
        pu,pv=vertices[u],vertices[v]
        ax.plot([pu[0],pv[0]],[pu[1],pv[1]],[pu[2],pv[2]],color="#67727D",lw=.55,alpha=.45,zorder=2)
    _set_equal_3d(ax,coords,M)
    if title:
        ax.set_title(title,fontsize=title_fontsize,fontweight="bold",pad=4)
    if subtitle:
        ax.text2D(.5,.015,subtitle,transform=ax.transAxes,ha="center",va="bottom",fontsize=subtitle_fontsize,color="#344054")
