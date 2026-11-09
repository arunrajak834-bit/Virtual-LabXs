import os
import io
import math
import hashlib
import tempfile
from datetime import datetime, timezone
from io import BytesIO

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import streamlit.components.v1 as components

# RDKit Imports
from rdkit import Chem, DataStructs
from rdkit.Chem import (
    Draw,
    AllChem,
    Descriptors,
    Crippen,
    Lipinski,
    rdMolDescriptors
)

# 3D Visualization
import py3Dmol

# PDF Generation (ReportLab)
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.colors import HexColor
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
    Image,
    Preformatted
)
from reportlab.lib.units import inch


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Cheminformatics Virtual Laboratory v2.0",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# MODERN UI & CUSTOM CSS
# ============================================================

st.markdown("""
<style>
/* Header & Banner Styling */
.hero-banner {
    background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 60%, #1d4ed8 100%);
    color: #ffffff;
    padding: 28px 32px;
    border-radius: 16px;
    margin-bottom: 24px;
    box-shadow: 0 10px 25px -5px rgba(30, 58, 138, 0.35);
    text-align: center;
}
.hero-title {
    font-size: 36px;
    font-weight: 800;
    letter-spacing: -0.5px;
    margin-bottom: 6px;
    color: #ffffff;
}
.hero-subtitle {
    font-size: 18px;
    font-weight: 600;
    color: #93c5fd;
    margin-bottom: 4px;
}
.hero-dept {
    font-size: 14px;
    color: #cbd5e1;
}

/* Glassmorphic Feature & Metric Cards */
.info-card {
    background: rgba(255, 255, 255, 0.04);
    border: 1px solid rgba(148, 163, 184, 0.25);
    border-radius: 12px;
    padding: 18px 20px;
    margin-bottom: 14px;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
}
.card-heading {
    font-size: 16px;
    font-weight: 700;
    color: #2563eb;
    margin-bottom: 8px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* Status Badges */
.badge-pass {
    display: inline-block;
    background-color: #dcfce7;
    color: #166534;
    padding: 4px 12px;
    border-radius: 999px;
    font-weight: 700;
    font-size: 13px;
    border: 1px solid #86efac;
}
.badge-warn {
    display: inline-block;
    background-color: #fef9c3;
    color: #854d0e;
    padding: 4px 12px;
    border-radius: 999px;
    font-weight: 700;
    font-size: 13px;
    border: 1px solid #fde047;
}
.badge-fail {
    display: inline-block;
    background-color: #fee2e2;
    color: #991b1b;
    padding: 4px 12px;
    border-radius: 999px;
    font-weight: 700;
    font-size: 13px;
    border: 1px solid #fca5a5;
}
.hash-box {
    font-family: 'Courier New', monospace;
    font-size: 12px;
    background-color: #0f172a;
    color: #38bdf8;
    padding: 10px 14px;
    border-radius: 8px;
    border-left: 4px solid #38bdf8;
    word-break: break-all;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# CONSTANTS & EXAMPLE MOLECULES
# ============================================================

EXAMPLE_MOLECULES = {
    "Ethanol": "CCO",
    "Benzene": "c1ccccc1",
    "Phenol": "Oc1ccccc1",
    "Acetic Acid": "CC(=O)O",
    "Aspirin": "CC(=O)Oc1ccccc1C(=O)O",
    "Caffeine": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    "Paracetamol": "CC(=O)Nc1ccc(O)cc1",
    "Ibuprofen": "CC(C)Cc1ccc(C(C)C(=O)O)cc1",
    "Penicillin G": "CC1(C)SC2C(NC(=O)Cc3ccccc3)C(=O)N2C1C(=O)O",
    "Dopamine": "NCCc1ccc(O)c(O)c1"
}

FUNCTIONAL_GROUPS_SMARTS = {
    "Hydroxyl (-OH)": "[OX2H]",
    "Carbonyl (C=O)": "[CX3]=[OX1]",
    "Carboxylic Acid (-COOH)": "[CX3](=O)[OX2H1]",
    "Ester (-COOR)": "[#6][CX3](=O)[OX2H0][#6]",
    "Amine (-NH2 / -NR2)": "[NX3;H2,H1,H0;!$(NC=O)]",
    "Amide (-CONH-)": "[NX3][CX3](=[OX1])[#6]",
    "Aromatic Ring Atoms": "a",
    "Halogen (-F, -Cl, -Br, -I)": "[F,Cl,Br,I]",
    "Thiol / Sulfide (-SH / -S-)": "[#16X2]"
}

# Initialize Session State
if "active_smiles" not in st.session_state:
    st.session_state.active_smiles = "CC(=O)Oc1ccccc1C(=O)O"  # Default: Aspirin

if "selected_example" not in st.session_state:
    st.session_state.selected_example = "Aspirin"

if "assessment_score" not in st.session_state:
    st.session_state.assessment_score = None


def load_shared_example():
    """Load the selected example into the shared SMILES field."""
    st.session_state.active_smiles = EXAMPLE_MOLECULES[
        st.session_state.selected_example
    ]


# ============================================================
# CORE CHEMINFORMATICS ENGINE
# ============================================================

def get_molecule(smiles: str):
    """Convert SMILES into a sanitized RDKit molecule."""
    if not smiles or not smiles.strip():
        return None
    try:
        mol = Chem.MolFromSmiles(smiles.strip())
        return mol
    except Exception:
        return None


def calculate_bond_and_hybridization_info(mol):
    """
    Calculate:
    - Heavy-atom bonds
    - Sigma (σ) bonds (including implicit H)
    - Pi (π) bonds via Kekulization
    - Carbon hybridization counts (sp, sp2, sp3)
    """
    heavy_atom_bonds = mol.GetNumBonds()

    kekule_mol = Chem.Mol(mol)
    try:
        Chem.Kekulize(kekule_mol, clearAromaticFlags=True)
    except Exception:
        pass

    pi_bonds = 0
    for bond in kekule_mol.GetBonds():
        btype = bond.GetBondType()
        if btype == Chem.BondType.DOUBLE:
            pi_bonds += 1
        elif btype == Chem.BondType.TRIPLE:
            pi_bonds += 2
        elif btype == Chem.BondType.QUADRUPLE:
            pi_bonds += 3
        elif bond.GetIsAromatic():
            pi_bonds += 0.5

    mol_with_h = Chem.AddHs(Chem.Mol(mol))
    sigma_bonds = mol_with_h.GetNumBonds()
    total_bonds = sigma_bonds + pi_bonds

    sp3_c, sp2_c, sp_c = 0, 0, 0
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == "C":
            hyb = atom.GetHybridization()
            if hyb == Chem.HybridizationType.SP3:
                sp3_c += 1
            elif hyb == Chem.HybridizationType.SP2:
                sp2_c += 1
            elif hyb == Chem.HybridizationType.SP:
                sp_c += 1

    return {
        "Heavy Atom Bonds": heavy_atom_bonds,
        "Sigma Bonds": int(sigma_bonds),
        "Pi Bonds": int(pi_bonds) if float(pi_bonds).is_integer() else round(pi_bonds, 1),
        "Total Bonds": int(total_bonds) if float(total_bonds).is_integer() else round(total_bonds, 1),
        "Total Atoms Including H": mol_with_h.GetNumAtoms(),
        "sp3 Carbons": sp3_c,
        "sp2 Carbons": sp2_c,
        "sp Carbons": sp_c
    }


@st.cache_data(show_spinner=False)
def calculate_properties_cached(smiles: str):
    """Cached calculation of molecular descriptors and drug-likeness rules."""
    mol = get_molecule(smiles)
    if mol is None:
        return None

    bond_info = calculate_bond_and_hybridization_info(mol)
    molecular_formula = rdMolDescriptors.CalcMolFormula(mol)

    mw = round(Descriptors.MolWt(mol), 3)
    exact_mw = round(Descriptors.ExactMolWt(mol), 4)
    logp = round(Crippen.MolLogP(mol), 3)
    tpsa = round(rdMolDescriptors.CalcTPSA(mol), 3)
    hbd = Lipinski.NumHDonors(mol)
    hba = Lipinski.NumHAcceptors(mol)
    rot_bonds = Lipinski.NumRotatableBonds(mol)
    rings = Lipinski.RingCount(mol)
    arom_rings = rdMolDescriptors.CalcNumAromaticRings(mol)
    aliph_rings = rdMolDescriptors.CalcNumAliphaticRings(mol)
    fsp3 = round(rdMolDescriptors.CalcFractionCSP3(mol), 3)
    mr = round(Crippen.MolMR(mol), 3)
    heteroatoms = rdMolDescriptors.CalcNumHeteroatoms(mol)

    properties = {
        "Molecular Formula": molecular_formula,
        "Molecular Weight": mw,
        "Exact Molecular Weight": exact_mw,
        "LogP": logp,
        "TPSA": tpsa,
        "Hydrogen Bond Donors": hbd,
        "Hydrogen Bond Acceptors": hba,
        "Rotatable Bonds": rot_bonds,
        "Ring Count": rings,
        "Aromatic Rings": arom_rings,
        "Aliphatic Rings": aliph_rings,
        "Heteroatoms": heteroatoms,
        "Number of Atoms": mol.GetNumAtoms(),
        "Total Atoms (incl. H)": bond_info["Total Atoms Including H"],
        "Heavy Atoms": mol.GetNumHeavyAtoms(),
        "Heavy Atom Bonds": bond_info["Heavy Atom Bonds"],
        "Sigma (σ) Bonds": bond_info["Sigma Bonds"],
        "Pi (π) Bonds": bond_info["Pi Bonds"],
        "Total Bonds (σ + π)": bond_info["Total Bonds"],
        "sp3 / sp2 / sp Carbons": f"{bond_info['sp3 Carbons']} / {bond_info['sp2 Carbons']} / {bond_info['sp Carbons']}",
        "Fraction Csp3": fsp3,
        "Molar Refractivity": mr
    }

    return properties


def evaluate_druglikeness(properties: dict):
    """Evaluate Lipinski's Rule of Five and Veber's Oral Bioavailability Rule."""
    mw = properties["Molecular Weight"]
    logp = properties["LogP"]
    hbd = properties["Hydrogen Bond Donors"]
    hba = properties["Hydrogen Bond Acceptors"]
    tpsa = properties["TPSA"]
    rot = properties["Rotatable Bonds"]

    lipinski_checks = {
        "Molecular Weight ≤ 500 Da": mw <= 500,
        "LogP ≤ 5.0": logp <= 5.0,
        "H-Bond Donors (HBD) ≤ 5": hbd <= 5,
        "H-Bond Acceptors (HBA) ≤ 10": hba <= 10
    }
    violations = sum(1 for passed in lipinski_checks.values() if not passed)
    lipinski_pass = violations <= 1

    veber_checks = {
        "Rotatable Bonds ≤ 10": rot <= 10,
        "TPSA ≤ 140 Å²": tpsa <= 140
    }
    veber_pass = all(veber_checks.values())

    return {
        "lipinski_checks": lipinski_checks,
        "lipinski_violations": violations,
        "lipinski_pass": lipinski_pass,
        "veber_checks": veber_checks,
        "veber_pass": veber_pass
    }


def detect_functional_groups(mol):
    """Detect functional groups and return matching atom indices."""
    detected = {}
    for name, smarts in FUNCTIONAL_GROUPS_SMARTS.items():
        pattern = Chem.MolFromSmarts(smarts)
        if pattern is not None:
            matches = mol.GetSubstructMatches(pattern)
            if matches:
                atom_indices = sorted(list({idx for match in matches for idx in match}))
                detected[name] = {
                    "count": len(matches),
                    "atom_indices": atom_indices
                }
    return detected


def compute_gasteiger_table(mol):
    """Compute Gasteiger partial charges and atomic hybridization table."""
    mol_copy = Chem.Mol(mol)
    try:
        AllChem.ComputeGasteigerCharges(mol_copy)
    except Exception:
        pass

    rows = []
    for atom in mol_copy.GetAtoms():
        charge_str = atom.GetProp("_GasteigerCharge") if atom.HasProp("_GasteigerCharge") else "0.0"
        try:
            charge = round(float(charge_str), 4)
            if math.isnan(charge) or math.isinf(charge):
                charge = 0.0
        except Exception:
            charge = 0.0

        rows.append({
            "Atom Index": atom.GetIdx(),
            "Symbol": atom.GetSymbol(),
            "Hybridization": str(atom.GetHybridization()),
            "Is Aromatic": "Yes" if atom.GetIsAromatic() else "No",
            "Degree": atom.GetDegree(),
            "Total Hydrogens": atom.GetTotalNumHs(),
            "Gasteiger Partial Charge": charge
        })
    return pd.DataFrame(rows)


def generate_verification_hash(student_name: str, reg_no: str, smiles_str: str):
    """Generate a tamper-evident SHA-256 hash for academic report integrity."""
    timestamp_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    payload = f"{student_name.strip()}|{reg_no.strip()}|{smiles_str.strip()}|{timestamp_str}"
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest().upper()
    return digest, timestamp_str


# ============================================================
# 2D & 3D VISUALIZATION ENGINE
# ============================================================

def generate_2d_image(mol, highlight_atoms=None, size=(700, 500)):
    """Generate 2D PIL image of the molecule with optional substructure highlighting."""
    if highlight_atoms:
        return Draw.MolToImage(mol, size=size, highlightAtoms=highlight_atoms)
    return Draw.MolToImage(mol, size=size)


@st.cache_data(show_spinner=False)
def generate_3d_molecule_cached(smiles: str):
    """
    Generate a 3D conformer using ETKDGv3 and optimize geometry with MMFF94 / UFF.
    Returns (mol_block_str,xyz_str, force_field_name, minimized_energy_kcal).
    """
    mol = get_molecule(smiles)
    if mol is None:
        return None, None, "None", None

    try:
        mol3d = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        status = AllChem.EmbedMolecule(mol3d, params)

        if status != 0:
            status = AllChem.EmbedMolecule(mol3d, useRandomCoords=True, randomSeed=42)
            if status != 0:
                return None, None, "Failed", None

        ff_name = "Unoptimized"
        energy_val = None

        # Attempt MMFF94 Optimization & Energy Calculation
        try:
            if AllChem.MMFFHasAllMoleculeParams(mol3d):
                mp = AllChem.MMFFGetMoleculeProperties(mol3d, mmffVariant="MMFF94")
                ff = AllChem.MMFFGetMoleculeForceField(mol3d, mp)
                if ff is not None:
                    ff.Minimize(maxIts=400)
                    energy_val = round(float(ff.CalcEnergy()), 2)
                    ff_name = "MMFF94"
        except Exception:
            pass

        # Fallback to UFF if MMFF94 was not applicable
        if energy_val is None:
            try:
                ff = AllChem.UFFGetMoleculeForceField(mol3d)
                if ff is not None:
                    ff.Minimize(maxIts=400)
                    energy_val = round(float(ff.CalcEnergy()), 2)
                    ff_name = "UFF"
            except Exception:
                pass

        mol_block = Chem.MolToMolBlock(mol3d)
        xyz_block = Chem.MolToXYZBlock(mol3d)
        return mol_block, xyz_block, ff_name, energy_val

    except Exception:
        return None, None, "Error", None


def display_3d_structure(
    mol_block: str,
    style: str = "Ball and Stick",
    show_surface: bool = False,
    bg_color: str = "#0f172a",
    spin: bool = False,
    show_labels: bool = False
):
    """Render an interactive py3Dmol viewer inside Streamlit."""
    view = py3Dmol.view(width=880, height=560)
    view.addModel(mol_block, "mol")

    if style == "Ball and Stick":
        view.setStyle({"stick": {"radius": 0.16}, "sphere": {"scale": 0.28}})
    elif style == "Stick":
        view.setStyle({"stick": {"radius": 0.22}})
    elif style == "Space Filling (CPK)":
        view.setStyle({"sphere": {"scale": 1.0}})
    elif style == "Wireframe":
        view.setStyle({"line": {"linewidth": 2.5}})
    elif style == "Cartoon / Hybrid":
        view.setStyle({"stick": {"radius": 0.12}, "sphere": {"scale": 0.20}})

    if show_surface:
        view.addSurface(py3Dmol.VDW, {"opacity": 0.65, "color": "white"})

    if show_labels:
        mol3d = Chem.MolFromMolBlock(mol_block, removeHs=False)
        if mol3d is not None:
            conf = mol3d.GetConformer()
            for atom in mol3d.GetAtoms():
                if atom.GetSymbol() != "H":
                    pos = conf.GetAtomPosition(atom.GetIdx())
                    view.addLabel(
                        f"{atom.GetSymbol()}{atom.GetIdx()}",
                        {
                            "position": {"x": pos.x, "y": pos.y, "z": pos.z},
                            "backgroundColor": "#1e293b",
                            "fontColor": "white",
                            "fontSize": 11,
                            "backgroundOpacity": 0.75
                        }
                    )

    view.setBackgroundColor(bg_color)
    view.zoomTo()
    view.spin(spin)

    html = view._make_html()
    components.html(html, height=580)


def create_bioavailability_radar_fig(properties: dict):
    """
    Create a 6-axis Oral Bioavailability Radar Chart (normalized to drug-like optimal ranges):
    1. Size (MW <= 500)
    2. Lipophilicity (-0.7 <= LogP <= 5.0)
    3. Polarity (TPSA <= 140)
    4. H-Bond Donors (HBD <= 5)
    5. H-Bond Acceptors (HBA <= 10)
    6. Flexibility (RotBonds <= 10)
    """
    categories = [
        "Size\n(MW ≤ 500)",
        "Lipophilicity\n(LogP ≤ 5)",
        "Polarity\n(TPSA ≤ 140)",
        "H-Donors\n(HBD ≤ 5)",
        "H-Acceptors\n(HBA ≤ 10)",
        "Flexibility\n(RotB ≤ 10)"
    ]

    # Normalize each property so 1.0 is the upper threshold of the optimal zone
    norm_vals = [
        min(max(properties["Molecular Weight"] / 500.0, 0.05), 1.5),
        min(max((properties["LogP"] + 1.0) / 6.0, 0.05), 1.5),
        min(max(properties["TPSA"] / 140.0, 0.05), 1.5),
        min(max(properties["Hydrogen Bond Donors"] / 5.0, 0.05), 1.5),
        min(max(properties["Hydrogen Bond Acceptors"] / 10.0, 0.05), 1.5),
        min(max(properties["Rotatable Bonds"] / 10.0, 0.05), 1.5)
    ]

    N = len(categories)
    angles = [n / float(N) * 2 * math.pi for n in range(N)]
    angles += angles[:1]
    values = norm_vals + norm_vals[:1]
    threshold = [1.0] * (N + 1)

    fig, ax = plt.subplots(figsize=(5.5, 5.5), subplot_kw=dict(polar=True), facecolor="white")
    ax.set_theta_offset(math.pi / 2)
    ax.set_theta_direction(-1)

    plt.xticks(angles[:-1], categories, color="#1e293b", size=9, fontweight="bold")
    ax.set_rlabel_position(0)
    plt.yticks([0.5, 1.0, 1.5], ["50%", "Max Limit", "150%"], color="#64748b", size=7)
    plt.ylim(0, 1.5)

    # Optimal Drug-Like Zone (Green Shaded Polygon)
    ax.plot(angles, threshold, linewidth=1.5, linestyle="--", color="#16a34a", label="Optimal Threshold")
    ax.fill(angles, threshold, color="#22c55e", alpha=0.15)

    # Molecule Profile
    ax.plot(angles, values, linewidth=2.4, linestyle="solid", color="#2563eb", label="Molecule Profile")
    ax.fill(angles, values, color="#3b82f6", alpha=0.30)

    ax.set_title("Oral Bioavailability & Drug-Likeness Radar", size=12, fontweight="bold", pad=20, color="#0f172a")
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.12), fontsize=8)
    fig.tight_layout()
    return fig


# ============================================================
# STATIC REPORT IMAGE HELPERS
# ============================================================

def save_2d_structure_png(mol):
    """Create a temporary PNG of the 2D structure for PDF reports."""
    img = Draw.MolToImage(mol, size=(700, 500))
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    tmp.close()
    img.save(tmp.name)
    return tmp.name


def save_radar_png(properties: dict):
    """Save the bioavailability radar chart as a temporary PNG for PDF inclusion."""
    fig = create_bioavailability_radar_fig(properties)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    tmp.close()
    fig.savefig(tmp.name, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return tmp.name


def save_3d_structure_png(smiles: str):
    """
    Create a high-clarity SVD-projected 3D ball-and-stick image for the PDF report.
    """
    mol_block, _, _, _ = generate_3d_molecule_cached(smiles)
    if not mol_block:
        return None

    mol3d = Chem.MolFromMolBlock(mol_block, removeHs=False)
    if mol3d is None:
        return None

    conf = mol3d.GetConformer()
    n_atoms = mol3d.GetNumAtoms()

    coords = np.array([
        [conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z]
        for i in range(n_atoms)
    ], dtype=float)

    centered = coords - coords.mean(axis=0)
    if n_atoms >= 3 and np.linalg.matrix_rank(centered) >= 2:
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        projected = centered @ vh[:2].T
    else:
        projected = centered[:, :2]

    cpk = {
        "H": "#FFFFFF", "C": "#3F3F3F", "N": "#2F5BFF",
        "O": "#E53935", "F": "#65C466", "P": "#F39C12",
        "S": "#F4D03F", "Cl": "#27AE60", "Br": "#8E3B2F", "I": "#8E44AD"
    }
    sizes = {
        "H": 70, "C": 240, "N": 270, "O": 285, "F": 260,
        "P": 300, "S": 310, "Cl": 310, "Br": 330, "I": 350
    }

    fig, ax = plt.subplots(figsize=(8.5, 6.2), facecolor="white")
    ax.set_facecolor("white")

    for bond in mol3d.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        x1, y1 = projected[i]
        x2, y2 = projected[j]
        dx, dy = x2 - x1, y2 - y1
        length = float(np.hypot(dx, dy))
        if length == 0:
            continue

        px, py = -dy / length, dx / length
        btype = bond.GetBondType()
        if btype == Chem.BondType.DOUBLE:
            offsets = (-0.035, 0.035)
        elif btype == Chem.BondType.TRIPLE:
            offsets = (-0.055, 0.0, 0.055)
        else:
            offsets = (0.0,)

        for off in offsets:
            ax.plot(
                [x1 + px * off, x2 + px * off],
                [y1 + py * off, y2 + py * off],
                color="#707070", linewidth=3.0, solid_capstyle="round", zorder=1
            )

    for i, atom in enumerate(mol3d.GetAtoms()):
        symbol = atom.GetSymbol()
        x, y = projected[i]
        ax.scatter(
            x, y,
            s=sizes.get(symbol, 250),
            c=cpk.get(symbol, "#A0A0A0"),
            edgecolors="#222222",
            linewidths=1.0,
            zorder=3
        )
        if symbol not in ("H", "C"):
            ax.text(
                x, y, symbol,
                ha="center", va="center",
                fontsize=8, fontweight="bold",
                color="black", zorder=4
            )

    xmin, ymin = projected.min(axis=0)
    xmax, ymax = projected.max(axis=0)
    width = max(xmax - xmin, 1.0)
    height = max(ymax - ymin, 1.0)
    pad = 0.22 * max(width, height)

    ax.set_xlim(xmin - pad, xmax + pad)
    ax.set_ylim(ymin - pad, ymax + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    ax.set_title("3D Molecular Geometry (SVD Ball-and-Stick Projection)", fontsize=15, fontweight="bold", pad=14)
    fig.tight_layout(pad=1.0)

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    tmp.close()
    fig.savefig(tmp.name, dpi=300, bbox_inches="tight", facecolor="white", pad_inches=0.1)
    plt.close(fig)
    return tmp.name


# ============================================================
# INTERPRETATION & PDF REPORT GENERATORS
# ============================================================

def get_property_interpretation(property_name, value):
    """Return a concise chemical interpretation for each calculated property."""
    if property_name == "Molecular Formula":
        return "Elemental stoichiometry of the molecule."
    if property_name == "Molecular Weight":
        return "Average molecular mass weighted by natural isotopic abundance (Lipinski limit ≤ 500 g/mol)."
    if property_name == "Exact Molecular Weight":
        return "Monoisotopic mass calculated using the most abundant isotope of each element (for HRMS)."
    if property_name == "LogP":
        if isinstance(value, (int, float)):
            if value < 0:
                return "Hydrophilic character; prefers aqueous phase over octanol."
            elif value <= 3:
                return "Balanced lipophilicity; favorable for membrane permeability and solubility."
            elif value <= 5:
                return "Moderately high lipophilicity; within Lipinski's acceptable threshold (≤ 5)."
            return "High lipophilicity (> 5); potential low aqueous solubility."
    if property_name == "TPSA":
        if isinstance(value, (int, float)):
            if value < 60:
                return "Low polar surface area; high potential for blood-brain barrier and passive cell permeation."
            elif value <= 140:
                return "Moderate polarity; favorable for oral bioavailability (Veber limit ≤ 140 Å²)."
            return "High polar surface area (> 140 Å²); may limit passive membrane transport."
    if property_name == "Hydrogen Bond Donors":
        return f"{value} N-H / O-H donor group(s) available for intermolecular H-bonding (Lipinski ≤ 5)."
    if property_name == "Hydrogen Bond Acceptors":
        return f"{value} heteroatom acceptor site(s) available for H-bonding (Lipinski ≤ 10)."
    if property_name == "Rotatable Bonds":
        return f"{value} rotatable bond(s) measuring conformational flexibility (Veber limit ≤ 10)."
    if property_name == "Ring Count":
        return f"Contains {value} cyclic system(s) contributing to structural rigidity."
    if property_name == "Aromatic Rings":
        return f"{value} planar delocalized π-ring system(s) influencing π-π stacking."
    if property_name == "Aliphatic Rings":
        return f"{value} non-aromatic ring system(s)."
    if property_name == "Heteroatoms":
        return f"{value} non-carbon/non-hydrogen atom(s) contributing to reactivity and polarity."
    if property_name == "Sigma (σ) Bonds":
        return "Total single covalent framework bonds including C-H and heteroatom-H bonds."
    if property_name == "Pi (π) Bonds":
        return "Total π bonds from double/triple bonds and Kekulized aromatic systems."
    if property_name == "Fraction Csp3":
        return "Ratio of sp³ hybridized carbons to total carbons; higher values indicate 3D saturation."
    if property_name == "Molar Refractivity":
        return "Measure of steric bulk and electronic polarizability of the molecule."
    return "Structural or topological descriptor derived from the molecular graph."


def build_key_interpretation_points(properties: dict):
    """Create bulleted conclusions for PDF reports."""
    druglike = evaluate_druglikeness(properties)
    lip_status = "PASSES" if druglike["lipinski_pass"] else "VIOLATES"
    veb_status = "PASSES" if druglike["veber_pass"] else "VIOLATES"

    return [
        f"<b>• Stoichiometry & Mass:</b> Formula <b>{properties['Molecular Formula']}</b> with average MW <b>{properties['Molecular Weight']} g/mol</b> (Exact Mass: {properties['Exact Molecular Weight']} Da).",
        f"<b>• Lipophilicity & Polarity:</b> LogP = <b>{properties['LogP']}</b> and TPSA = <b>{properties['TPSA']} Å²</b>, defining partitioning between lipid bilayers and aqueous environments.",
        f"<b>• Covalent Bonding Architecture:</b> Contains <b>{properties['Sigma (σ) Bonds']} σ bonds</b> and <b>{properties['Pi (π) Bonds']} π bonds</b> (Total: {properties['Total Bonds (σ + π)']}), with C-hybridization distribution (sp³/sp²/sp) of <b>{properties['sp3 / sp2 / sp Carbons']}</b>.",
        f"<b>• Drug-Likeness Screening:</b> The molecule <b>{lip_status} Lipinski's Rule of Five</b> ({druglike['lipinski_violations']} violation(s)) and <b>{veb_status} Veber's Rule</b> for oral bioavailability."
    ]


def generate_multi_final_report(student_name, registration_number, molecule_records):
    """Create a comprehensive, cryptographic-stamped PDF report for one or multiple molecules."""
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4, leftMargin=36, rightMargin=36, topMargin=38, bottomMargin=38
    )
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "MultiReportTitle", parent=styles["Title"], fontSize=18,
        leading=22, alignment=TA_CENTER, textColor=HexColor("#0F172A")
    )
    subtitle_style = ParagraphStyle(
        "MultiReportSubTitle", parent=styles["Normal"], fontSize=10,
        leading=13, alignment=TA_CENTER, textColor=HexColor("#1E3A8A")
    )
    heading_style = ParagraphStyle(
        "MultiReportHeading", parent=styles["Heading1"], fontSize=12.5,
        leading=15, textColor=HexColor("#1E3A8A"), spaceBefore=8, spaceAfter=6
    )
    body_style = ParagraphStyle(
        "MultiReportBody", parent=styles["BodyText"], fontSize=9,
        leading=12.5, alignment=TA_JUSTIFY
    )
    small_style = ParagraphStyle(
        "MultiReportSmall", parent=styles["BodyText"], fontSize=7.2, leading=9.0
    )
    header_style = ParagraphStyle(
        "MultiReportHeader", parent=styles["BodyText"], fontSize=7.5,
        leading=9.5, textColor=colors.white, alignment=TA_CENTER
    )

    combined_smiles_str = ";".join([r[0] for r in molecule_records])
    audit_hash, timestamp_utc = generate_verification_hash(
        student_name or "Anonymous",
        registration_number or "N/A",
        combined_smiles_str
    )

    story = [
        Paragraph("SRM INSTITUTE OF SCIENCE AND TECHNOLOGY, TIRUCHIRAPPALLI", subtitle_style),
        Paragraph("CHEMINFORMATICS VIRTUAL LABORATORY", title_style),
        Spacer(1, 0.06 * inch),
    ]

    meta_data = [
        ["Student Name", student_name or "Not provided", "Date (UTC)", timestamp_utc],
        ["Registration No.", registration_number or "Not provided", "Molecules Analyzed", str(len(molecule_records))],
        ["SHA-256 Audit Seal", audit_hash, "", ""]
    ]
    meta_table = Table(meta_data, colWidths=[1.3 * inch, 2.2 * inch, 1.3 * inch, 1.9 * inch])
    meta_table.setStyle(TableStyle([
        ("SPAN", (1, 2), (3, 2)),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("BACKGROUND", (0, 0), (0, -1), HexColor("#E2E8F0")),
        ("BACKGROUND", (2, 0), (2, 1), HexColor("#E2E8F0")),
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("PADDING", (0, 0), (-1, -1), 5)
    ]))
    story += [meta_table, Spacer(1, 0.14 * inch)]

    temp_images = []
    for index, (input_smiles, mol, properties) in enumerate(molecule_records, start=1):
        if index > 1:
            story.append(PageBreak())

        story.append(Paragraph(f"Molecule {index}: {input_smiles} ({properties['Molecular Formula']})", heading_style))

        img2d = save_2d_structure_png(mol)
        img3d = save_3d_structure_png(input_smiles)
        img_radar = save_radar_png(properties)

        for p in (img2d, img3d, img_radar):
            if p:
                temp_images.append(p)

        if img2d and img3d and img_radar:
            struct_table = Table([[
                Image(img2d, width=2.15 * inch, height=1.55 * inch),
                Image(img3d, width=2.25 * inch, height=1.60 * inch),
                Image(img_radar, width=2.10 * inch, height=2.10 * inch)
            ]], colWidths=[2.2 * inch, 2.3 * inch, 2.2 * inch])
            struct_table.setStyle(TableStyle([
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ALIGN", (0, 0), (-1, -1), "CENTER")
            ]))
            story += [struct_table, Spacer(1, 0.08 * inch)]

        story.append(Paragraph("Complete Molecular Descriptors & Chemical Interpretation", heading_style))
        prop_data = [[
            Paragraph("<b>Descriptor</b>", header_style),
            Paragraph("<b>Value</b>", header_style),
            Paragraph("<b>Physicochemical Interpretation</b>", header_style)
        ]]
        for key, value in properties.items():
            prop_data.append([
                Paragraph(str(key), small_style),
                Paragraph(str(value), small_style),
                Paragraph(get_property_interpretation(key, value), small_style)
            ])

        prop_table = Table(prop_data, colWidths=[1.6 * inch, 1.0 * inch, 4.1 * inch], repeatRows=1)
        prop_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), HexColor("#1E3A8A")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("PADDING", (0, 0), (-1, -1), 3.5)
        ]))
        story += [prop_table, Spacer(1, 0.10 * inch)]

        story.append(Paragraph("Executive Summary & Drug-Likeness Assessment", heading_style))
        for point in build_key_interpretation_points(properties):
            story.append(Paragraph(point, body_style))
            story.append(Spacer(1, 0.04 * inch))

    try:
        doc.build(story)
        return buffer.getvalue()
    finally:
        buffer.close()
        for path in temp_images:
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except Exception:
                    pass


# ============================================================
# SIDEBAR NAVIGATION & STUDENT PROFILE
# ============================================================

with st.sidebar:
    st.markdown("## 🧪 Cheminformatics Lab `v2.0`")
    st.caption("Division of Chemistry | SRM IST Trichy")

    page = st.radio(
        "Navigation",
        [
            "🏠 Home & Overview",
            "📖 Theory & Concepts",
            "🧬 2D/3D Visualization & Functional Groups",
            "📊 Descriptors & Drug-Likeness Lab",
            "📈 Multi-Molecule & Similarity Analysis",
            "📝 Interactive Assessment",
            "📄 Verified Project Report (PDF)"
        ]
    )

    st.divider()
    st.markdown("### 👨‍🎓 Student Profile")
    student_name = st.text_input("Student Name", key="student_name", placeholder="e.g., Priyanshu Jha")
    registration_number = st.text_input("Registration Number", key="registration_number", placeholder="e.g., RA2611030050175")

    st.divider()
    st.markdown("### ⚡ Quick Molecule Switcher")
    st.selectbox(
        "Preset Molecules",
        list(EXAMPLE_MOLECULES.keys()),
        key="selected_example",
        on_change=load_shared_example
    )
    st.caption(f"**Active SMILES:** `{st.session_state.active_smiles}`")


# ============================================================
# PAGE 1: HOME PAGE
# ============================================================

if page == "🏠 Home & Overview":
    logo_path = os.path.join(os.path.dirname(__file__), "srm_logo.jpeg")
    if os.path.exists(logo_path):
        c1, c2, c3 = st.columns([1, 3, 1])
        with c2:
            st.image(logo_path, use_container_width=True)

    st.markdown("""
    <div class="hero-banner">
        <div class="hero-title">🧪 Cheminformatics Virtual Laboratory</div>
        <div class="hero-subtitle">SRM Institute of Science and Technology, Tiruchirappalli</div>
        <div class="hero-dept">Faculty of Engineering and Technology | School of Sciences | Division of Chemistry</div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="info-card">
            <div class="card-heading">🧬 2D & 3D Structure Engine</div>
            Interactive <b>py3Dmol</b> conformer generation with <b>MMFF94/UFF</b> force-field energy minimization, SMARTS functional group highlighting, and <code>.SDF/.XYZ</code> coordinate exports.
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="info-card">
            <div class="card-heading">💊 Drug-Likeness & ADME</div>
            Automated <b>Lipinski's Rule of Five</b> and <b>Veber's Rule</b> diagnostics, 6-axis <b>Oral Bioavailability Radar Charts</b>, Kekulized σ/π bond counting, and Gasteiger partial charges.
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="info-card">
            <div class="card-heading">🔐 Verified PDF Reports</div>
            Generate publication-quality single or multi-molecule PDF lab reports embedded with <b>SVD 3D projections</b>, <b>Bioavailability Radars</b>, and a <b>SHA-256 Cryptographic Audit Seal</b>.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### 📚 Reference Molecule Library")
    example_rows = []
    for name, smi in EXAMPLE_MOLECULES.items():
        props = calculate_properties_cached(smi)
        if props:
            example_rows.append({
                "Molecule Name": name,
                "SMILES Notation": smi,
                "Formula": props["Molecular Formula"],
                "MW (g/mol)": props["Molecular Weight"],
                "LogP": props["LogP"],
                "TPSA (Å²)": props["TPSA"],
                "σ Bonds": props["Sigma (σ) Bonds"],
                "π Bonds": props["Pi (π) Bonds"]
            })
    st.dataframe(pd.DataFrame(example_rows), use_container_width=True, hide_index=True)

    st.divider()
    fcol1, fcol2 = st.columns([2, 1])
    with fcol1:
      st.markdown("### 👨‍💻 Developed by")
st.markdown("**Arun and Sujan**")
st.markdown("B.Tech CSE with Cybersecurity")
st.markdown("SRM Institute of Science and Technology, Tiruchirappalli campus")
    with fcol2:
        st.info("💡 **Tip:** Use the **Quick Molecule Switcher** in the left sidebar or type any custom SMILES string in the laboratory modules.")
# ============================================================
# PAGE 2: THEORY PAGE
# ============================================================

elif page == "📖 Theory & Concepts":
    st.title("📖 Principles of Cheminformatics & Molecular Modeling")

    t1, t2, t3, t4 = st.tabs([
        "1. SMILES Notation",
        "2. Covalent Bonding (σ & π) & Hybridization",
        "3. Drug-Likeness (Lipinski & Veber)",
        "4. 3D Force Fields & Fingerprints"
    ])

    with t1:
        st.subheader("Simplified Molecular Input Line Entry System (SMILES)")
        st.markdown("""
        **SMILES** encodes a two-dimensional molecular graph as a linear ASCII string:
        * **Atoms:** Represented by atomic symbols (`C`, `N`, `O`, `S`, `F`, `Cl`, `Br`). Lowercase letters (`c`, `n`, `o`, `s`) denote atoms in **aromatic** rings.
        * **Bonds:** Single bonds are implicit (`CC` = ethane), double bonds use `=` (`C=O`), triple bonds use `#` (`C#N`).
        * **Branches:** Enclosed in parentheses, e.g., `CC(=O)O` for acetic acid.
        * **Rings:** Numerically indexed at ring-closure atoms, e.g., `c1ccccc1` for benzene.
        """)

    with t2:
        st.subheader("Kekulization, Sigma (σ) and Pi (π) Bond Counting")
        st.markdown(r"""
        In computational chemistry, aromatic rings are often stored with delocalized bond orders ($1.5$). To count discrete $\pi$ bonds accurately:
        1. **Kekulization:** Converts delocalized aromatic systems into alternating single and double bonds.
        2. **Sigma ($\sigma$) Bonds:** Every covalent atom-atom connection (including implicit $\text{C–H}$, $\text{O–H}$, and $\text{N–H}$ bonds) contains exactly **one $\sigma$ bond**.
        3. **Pi ($\pi$) Bonds:**
           * Each double bond contributes **$1\,\pi$ bond**.
           * Each triple bond contributes **$2\,\pi$ bonds**.
        """)

    with t3:
        st.subheader("Lipinski's Rule of Five & Veber's Rule")
        st.markdown("""
        **Lipinski's Rule of Five (Ro5)** evaluates whether a chemical compound has physicochemical properties consistent with an orally active drug in humans:
        * **Molecular Weight (MW):** $\le 500\text{ g/mol}$
        * **Octanol-Water Partition Coefficient ($\log P$):** $\le 5.0$
        * **Hydrogen Bond Donors (HBD):** $\le 5$ (sum of $\text{OH}$ and $\text{NH}$ groups)
        * **Hydrogen Bond Acceptors (HBA):** $\le 10$ (sum of $\text{N}$ and $\text{O}$ atoms)

        **Veber's Rule** adds two criteria for oral bioavailability:
        * **Topological Polar Surface Area (TPSA):** $\le 140\text{ \AA}^2$
        * **Rotatable Bonds:** $\le 10$
        """)

    with t4:
        st.subheader("3D Conformer Embedding & Morgan Fingerprints")
        st.markdown("""
        * **ETKDGv3 Algorithm:** Distance-geometry method enriched with experimental torsional angle preferences from crystal structures to generate initial 3D coordinates.
        * **MMFF94 / UFF Minimization:** Merck Molecular Force Field (MMFF94) and Universal Force Field (UFF) relax bond lengths, bond angles, and torsional strain to locate a local energy minimum (reported in $\text{kcal/mol}$).
        * **Morgan Fingerprints (ECFP4):** Circular topological fingerprints (radius 2, 2048 bits) used to quantify structural similarity between two molecules via the **Tanimoto Coefficient** ($T_c \in [0, 1]$).
        """)


# ============================================================
# PAGE 3: 2D/3D VISUALIZATION & FUNCTIONAL GROUPS
# ============================================================

elif page == "🧬 2D/3D Visualization & Functional Groups":
    st.title("🧬 2D/3D Molecular Visualization & Substructure Lab")

    col_in1, col_in2 = st.columns([2.2, 1])
    with col_in1:
        smiles = st.text_input("Enter SMILES String", key="active_smiles")
    with col_in2:
        st.selectbox(
            "Or Select Preset Molecule",
            list(EXAMPLE_MOLECULES.keys()),
            key="selected_example",
            on_change=load_shared_example
        )

    mol = get_molecule(smiles)
    if mol is None:
        st.error("❌ Invalid SMILES notation. Please check syntax (e.g., parentheses and ring numbers).")
    else:
        props = calculate_properties_cached(smiles)
        st.session_state.latest_smiles = smiles
        st.session_state.latest_properties = props

        fg_dict = detect_functional_groups(mol)

        col_left, col_right = st.columns([1.25, 1])
        with col_left:
            st.subheader("🎨 Interactive 2D Structure & Functional Group Highlighter")
            fg_options = ["None (Standard 2D View)"] + list(fg_dict.keys())
            selected_fg = st.selectbox("Highlight Detected Functional Group / Substructure:", fg_options)

            highlight_atoms = fg_dict[selected_fg]["atom_indices"] if selected_fg != "None (Standard 2D View)" else None
            img_2d = generate_2d_image(mol, highlight_atoms=highlight_atoms)
            st.image(img_2d, use_container_width=True)

            # Download 2D PNG
            img_buf = BytesIO()
            img_2d.save(img_buf, format="PNG")
            st.download_button(
                "⬇️ Download High-Res 2D Structure (.PNG)",
                data=img_buf.getvalue(),
                file_name=f"{props['Molecular Formula']}_2D.png",
                mime="image/png"
            )

        with col_right:
            st.subheader("🔬 Bonding, Hybridization & Functional Groups")
            m1, m2, m3 = st.columns(3)
            m1.metric("Formula", props["Molecular Formula"])
            m2.metric("σ Bonds", props["Sigma (σ) Bonds"])
            m3.metric("π Bonds", props["Pi (π) Bonds"])

            st.write(f"**Canonical SMILES:** `{Chem.MolToSmiles(mol)}`")
            st.write(f"**Carbon Hybridization ($sp^3 / sp^2 / sp$):** `{props['sp3 / sp2 / sp Carbons']}`")
            st.write(f"**Total Bonds ($\sigma + \pi$):** `{props['Total Bonds (σ + π)']}` | **Total Atoms (incl. H):** `{props['Total Atoms (incl. H)']}`")

            st.markdown("#### Detected Functional Groups (SMARTS)")
            if fg_dict:
                fg_rows = [
                    {"Functional Group": k, "Occurrences": v["count"], "Atom Indices": str(v["atom_indices"])}
                    for k, v in fg_dict.items()
                ]
                st.dataframe(pd.DataFrame(fg_rows), use_container_width=True, hide_index=True)
            else:
                st.info("No standard polar functional groups detected (purely aliphatic hydrocarbon).")

            with st.expander("⚡ View Atomic Gasteiger Partial Charges Table"):
                st.dataframe(compute_gasteiger_table(mol), use_container_width=True, hide_index=True)

        st.divider()
        st.subheader("🌐 Interactive 3D Conformer & Force-Field Energetics")

        c_3d1, c_3d2, c_3d3, c_3d4 = st.columns([1.5, 1.2, 1, 1])
        with c_3d1:
            vis_style = st.selectbox(
                "3D Rendering Style",
                ["Ball and Stick", "Stick", "Space Filling (CPK)", "Wireframe", "Cartoon / Hybrid"]
            )
        with c_3d2:
            bg_choice = st.selectbox(
                "Canvas Background",
                ["Dark Slate (#0f172a)", "Pure White (#ffffff)", "Midnight Black (#000000)"]
            )
            bg_hex = bg_choice.split("(")[1].replace(")", "")
        with c_3d3:
            show_surf = st.checkbox("van der Waals Surface")
            spin_mol = st.checkbox("Auto-Spin 3D Model")
        with c_3d4:
            show_lbls = st.checkbox("Show Atom Labels")

        mol_block, xyz_block, ff_name, energy_kcal = generate_3d_molecule_cached(smiles)

        if mol_block is None:
            st.error("Unable to embed 3D coordinates for this SMILES structure.")
        else:
            ecol1, ecol2, ecol3 = st.columns([1, 1, 2])
            ecol1.metric("Force Field Used", ff_name)
            ecol2.metric(
                "Minimized Potential Energy",
                f"{energy_kcal} kcal/mol" if energy_kcal is not None else "N/A"
            )
            with ecol3:
                dl1, dl2 = st.columns(2)
                with dl1:
                    st.download_button(
                        "⬇️ Download 3D Structure (.SDF/.MOL)",
                        data=mol_block.encode("utf-8"),
                        file_name=f"{props['Molecular Formula']}_3D.sdf",
                        mime="chemical/x-mdl-sdfile"
                    )
                with dl2:
                    if xyz_block:
                        st.download_button(
                            "⬇️ Download Cartesian Coords (.XYZ)",
                            data=xyz_block.encode("utf-8"),
                            file_name=f"{props['Molecular Formula']}_3D.xyz",
                            mime="chemical/x-xyz"
                        )

            display_3d_structure(
                mol_block=mol_block,
                style=vis_style,
                show_surface=show_surf,
                bg_color=bg_hex,
                spin=spin_mol,
                show_labels=show_lbls
            )


# ============================================================
# PAGE 4: DESCRIPTORS & DRUG-LIKENESS LAB
# ============================================================

elif page == "📊 Descriptors & Drug-Likeness Lab":
    st.title("📊 Complete Molecular Descriptors & Drug-Likeness Laboratory")

    col1, col2 = st.columns([2.2, 1])
    with col1:
        smiles = st.text_input("Enter SMILES Notation", key="active_smiles")
    with col2:
        st.selectbox(
            "Choose Preset Molecule",
            list(EXAMPLE_MOLECULES.keys()),
            key="selected_example",
            on_change=load_shared_example
        )

    mol = get_molecule(smiles)
    if mol is None:
        st.error("❌ Invalid SMILES notation. Please verify your input.")
    else:
        props = calculate_properties_cached(smiles)
        druglike = evaluate_druglikeness(props)

        # Top KPI Metrics
        k1, k2, k3, k4, k5, k6 = st.columns(6)
        k1.metric("MW (g/mol)", props["Molecular Weight"])
        k2.metric("LogP", props["LogP"])
        k3.metric("TPSA (Å²)", props["TPSA"])
        k4.metric("HBD / HBA", f"{props['Hydrogen Bond Donors']} / {props['Hydrogen Bond Acceptors']}")
        k5.metric("Rot. Bonds", props["Rotatable Bonds"])
        k6.metric("Fraction Csp³", props["Fraction Csp3"])

        st.divider()

        rcol1, rcol2 = st.columns([1.15, 1])
        with rcol1:
            st.subheader("💊 Drug-Likeness & Oral Bioavailability Screening")

            if druglike["lipinski_pass"]:
                st.markdown(f'<span class="badge-pass">✅ Lipinski Rule of Five: PASS ({druglike["lipinski_violations"]} violations)</span>', unsafe_allow_html=True)
            else:
                st.markdown(f'<span class="badge-fail">❌ Lipinski Rule of Five: FAIL ({druglike["lipinski_violations"]} violations)</span>', unsafe_allow_html=True)

            st.write("")
            for rule_desc, passed in druglike["lipinski_checks"].items():
                icon = "🟢" if passed else "🔴"
                st.write(f"{icon} **{rule_desc}**")

            st.write("---")
            if druglike["veber_pass"]:
                st.markdown('<span class="badge-pass">✅ Veber Oral Bioavailability Rule: PASS</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span class="badge-warn">⚠️ Veber Oral Bioavailability Rule: VIOLATED</span>', unsafe_allow_html=True)

            st.write("")
            for rule_desc, passed in druglike["veber_checks"].items():
                icon = "🟢" if passed else "🔴"
                st.write(f"{icon} **{rule_desc}**")

        with rcol2:
            radar_fig = create_bioavailability_radar_fig(props)
            st.pyplot(radar_fig, use_container_width=True)
            plt.close(radar_fig)

        st.divider()
        st.subheader("📑 Complete Quantitative Property & Interpretation Table")

        table_rows = []
        for k, v in props.items():
            table_rows.append({
                "Property / Descriptor": k,
                "Calculated Value": str(v),
                "Physicochemical Interpretation": get_property_interpretation(k, v)
            })
        df_props = pd.DataFrame(table_rows)
        st.dataframe(df_props, use_container_width=True, hide_index=True)

        dcol1, dcol2 = st.columns(2)
        with dcol1:
            csv_bytes = df_props.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download Complete Property Table (.CSV)",
                data=csv_bytes,
                file_name=f"{props['Molecular Formula']}_Properties.csv",
                mime="text/csv"
            )
        with dcol2:
            pdf_bytes = generate_multi_final_report(
                student_name, registration_number, [(smiles, mol, props)]
            )
            st.download_button(
                "📄 Download Single-Molecule Verified PDF Report",
                data=pdf_bytes,
                file_name=f"{props['Molecular Formula']}_Report.pdf",
                mime="application/pdf"
            )


# ============================================================
# PAGE 5: MULTI-MOLECULE & SIMILARITY ANALYSIS
# ============================================================

elif page == "📈 Multi-Molecule & Similarity Analysis":
    st.title("📈 Structure–Property & Tanimoto Similarity Laboratory")
    st.write("Enter one molecule per line in the format: **`Molecule Name, SMILES`**")

    default_batch = """Ethanol,CCO
Benzene,c1ccccc1
Phenol,Oc1ccccc1
Acetic Acid,CC(=O)O
Aspirin,CC(=O)Oc1ccccc1C(=O)O
Paracetamol,CC(=O)Nc1ccc(O)cc1
Ibuprofen,CC(C)Cc1ccc(C(C)C(=O)O)cc1
Caffeine,CN1C=NC2=C1C(=O)N(C(=O)N2C)C"""

    batch_input = st.text_area("Batch Molecule Input", value=default_batch, height=200)

    records = []
    mols_valid = []
    names_valid = []

    for line in batch_input.splitlines():
        if "," not in line:
            continue
        name, smi = line.split(",", 1)
        mol = get_molecule(smi.strip())
        if mol:
            p = calculate_properties_cached(smi.strip())
            dl = evaluate_druglikeness(p)
            mols_valid.append(mol)
            names_valid.append(name.strip())
            records.append({
                "Molecule": name.strip(),
                "SMILES": smi.strip(),
                "Formula": p["Molecular Formula"],
                "MW": p["Molecular Weight"],
                "LogP": p["LogP"],
                "TPSA": p["TPSA"],
                "HBD": p["Hydrogen Bond Donors"],
                "HBA": p["Hydrogen Bond Acceptors"],
                "RotBonds": p["Rotatable Bonds"],
                "σ Bonds": p["Sigma (σ) Bonds"],
                "π Bonds": p["Pi (π) Bonds"],
                "Fsp3": p["Fraction Csp3"],
                "Lipinski Pass": "Yes" if dl["lipinski_pass"] else "No"
            })

    if not records:
        st.warning("Please enter at least one valid line in the format: `Name, SMILES`")
    else:
        df_comp = pd.DataFrame(records)
        st.subheader("📋 Multi-Molecule Descriptor Matrix")
        st.dataframe(df_comp, use_container_width=True, hide_index=True)

        csv_comp = df_comp.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download Comparison Matrix (.CSV)",
            data=csv_comp,
            file_name="Multi_Molecule_Comparison.csv",
            mime="text/csv"
        )

        st.divider()
        vcol1, vcol2 = st.columns(2)

        with vcol1:
            st.subheader("🌐 Chemical Space: MW vs. LogP (Lipinski Box)")
            fig_scat, ax_s = plt.subplots(figsize=(6.2, 4.8))
            for idx, row in df_comp.iterrows():
                color = "#16a34a" if row["Lipinski Pass"] == "Yes" else "#dc2626"
                ax_s.scatter(row["MW"], row["LogP"], s=110, color=color, edgecolors="black", zorder=3)
                ax_s.annotate(
                    row["Molecule"],
                    (row["MW"], row["LogP"]),
                    xytext=(6, 5),
                    textcoords="offset points",
                    fontsize=8.5,
                    fontweight="bold"
                )
            ax_s.axvline(500, color="#ef4444", linestyle="--", linewidth=1.2, label="MW = 500 Da Limit")
            ax_s.axhline(5.0, color="#f59e0b", linestyle="--", linewidth=1.2, label="LogP = 5.0 Limit")
            ax_s.set_xlabel("Molecular Weight (g/mol)", fontweight="bold")
            ax_s.set_ylabel("LogP (Octanol/Water)", fontweight="bold")
            ax_s.set_title("Physicochemical Space Distribution", fontweight="bold")
            ax_s.grid(True, linestyle=":", alpha=0.5)
            ax_s.legend(fontsize=8)
            fig_scat.tight_layout()
            st.pyplot(fig_scat, use_container_width=True)
            plt.close(fig_scat)

        with vcol2:
            st.subheader("🧬 Morgan Fingerprint (ECFP4) Tanimoto Similarity")
            fps = [rdMolDescriptors.GetMorganFingerprintAsBitVect(m, radius=2, nBits=2048) for m in mols_valid]
            n_mols = len(fps)
            sim_matrix = np.zeros((n_mols, n_mols))
            for i in range(n_mols):
                for j in range(n_mols):
                    sim_matrix[i, j] = round(DataStructs.TanimotoSimilarity(fps[i], fps[j]), 2)

            fig_hm, ax_h = plt.subplots(figsize=(6.2, 4.8))
            im = ax_h.imshow(sim_matrix, cmap="Blues", vmin=0, vmax=1)
            ax_h.set_xticks(range(n_mols))
            ax_h.set_yticks(range(n_mols))
            ax_h.set_xticklabels(names_valid, rotation=40, ha="right", fontsize=8.5)
            ax_h.set_yticklabels(names_valid, fontsize=8.5)

            for i in range(n_mols):
                for j in range(n_mols):
                    text_color = "white" if sim_matrix[i, j] > 0.55 else "black"
                    ax_h.text(j, i, f"{sim_matrix[i, j]:.2f}", ha="center", va="center", color=text_color, fontsize=8)

            ax_h.set_title("Pairwise Tanimoto Similarity Matrix", fontweight="bold")
            fig_hm.colorbar(im, ax=ax_h, fraction=0.046, pad=0.04)
            fig_hm.tight_layout()
            st.pyplot(fig_hm, use_container_width=True)
            plt.close(fig_hm)


# ============================================================
# PAGE 6: INTERACTIVE ASSESSMENT
# ============================================================

elif page == "📝 Interactive Assessment":
    st.title("📝 Cheminformatics & Computational Chemistry Assessment")

    questions = [
        {"question": "1. What is the full form of SMILES?", "options": ["Simplified Molecular Input Line Entry System", "Standard Molecular Information Language Encoding System", "Simple Molecular Identification and Labeling System"], "answer": "Simplified Molecular Input Line Entry System", "explanation": "SMILES stands for Simplified Molecular Input Line Entry System, developed to represent chemical structures in ASCII text."},
        {"question": "2. Which SMILES string represents an aromatic benzene ring?", "options": ["c1ccccc1", "C1CCCCC1", "CC(=O)O"], "answer": "c1ccccc1", "explanation": "Lowercase 'c' denotes sp2 aromatic carbons, whereas uppercase 'C1CCCCC1' represents aliphatic cyclohexane."},
        {"question": "3. In SMILES notation, what do parentheses '()' represent?", "options": ["Branches from the main chain", "Aromatic rings", "Isotopic mass"], "answer": "Branches from the main chain", "explanation": "Parentheses enclose side chains or carbonyl oxygen branches, such as CC(=O)O."},
        {"question": "4. Which descriptor measures a molecule's partitioning between octanol and water?", "options": ["LogP", "TPSA", "Molar Refractivity"], "answer": "LogP", "explanation": "LogP is the logarithm of the octanol-water partition coefficient, measuring lipophilicity."},
        {"question": "5. According to Lipinski's Rule of Five, an orally active drug should generally have:", "options": ["Molecular Weight ≤ 500 Da and LogP ≤ 5", "Molecular Weight ≥ 800 Da and LogP ≥ 7", "More than 10 Hydrogen Bond Donors"], "answer": "Molecular Weight ≤ 500 Da and LogP ≤ 5", "explanation": "Lipinski's Ro5 specifies MW ≤ 500, LogP ≤ 5, HBD ≤ 5, and HBA ≤ 10."},
        {"question": "6. How many sigma (σ) and pi (π) bonds are present in a carbon-carbon triple bond (C≡C)?", "options": ["1 σ bond and 2 π bonds", "2 σ bonds and 1 π bond", "3 π bonds only"], "answer": "1 σ bond and 2 π bonds", "explanation": "Every covalent bond has exactly 1 σ bond; a triple bond adds 2 orthogonal π bonds."},
        {"question": "7. What does TPSA stand for in molecular descriptor analysis?", "options": ["Topological Polar Surface Area", "Total Pi Surface Area", "Thermodynamic Polarity Structure Analysis"], "answer": "Topological Polar Surface Area", "explanation": "TPSA sums the surface area contributions of polar atoms (primarily N, O, and attached hydrogens)."},
        {"question": "8. Which 3D molecular visualization style best illustrates the steric volume (van der Waals radii) of atoms?", "options": ["Space Filling (CPK)", "Wireframe", "2D Skeletal"], "answer": "Space Filling (CPK)", "explanation": "Space-filling models scale atomic spheres to their van der Waals radii to show steric bulk."},
        {"question": "9. What is the role of MMFF94 in 3D structure generation?", "options": ["Force-field energy minimization to relax bond lengths and angles", "Converting 2D PNG images into PDFs", "Counting hydrogen bond acceptors"], "answer": "Force-field energy minimization to relax bond lengths and angles", "explanation": "MMFF94 (Merck Molecular Force Field) optimizes 3D conformer geometry to a low-energy state."},
        {"question": "10. What does a Tanimoto similarity coefficient of 1.0 between two Morgan fingerprints indicate?", "options": ["Identical fingerprint bit vectors (high structural similarity)", "Completely unrelated chemical structures", "The molecule violates Lipinski's rules"], "answer": "Identical fingerprint bit vectors (high structural similarity)", "explanation": "Tanimoto similarity ranges from 0.0 (no shared bits) to 1.0 (identical bit representations)."}
    ]

    student_answers = []
    for i, q in enumerate(questions):
        ans = st.radio(q["question"], q["options"], index=None, key=f"quiz_q_{i}")
        student_answers.append(ans)

    if st.button("✅ Submit & Grade Assessment"):
        if any(a is None for a in student_answers):
            st.warning("⚠️ Please answer all 10 questions before submitting.")
        else:
            score = 0
            st.divider()
            st.subheader("📋 Detailed Answer Key & Feedback")
            for i, q in enumerate(questions):
                if student_answers[i] == q["answer"]:
                    score += 1
                    st.success(f"**Q{i+1}: Correct!** ({q['answer']}) — {q['explanation']}")
                else:
                    st.error(f"**Q{i+1}: Incorrect.** You chose *{student_answers[i]}*. Correct Answer: **{q['answer']}** — {q['explanation']}")

            pct = (score / len(questions)) * 100
            st.session_state.assessment_score = f"{score}/{len(questions)} ({pct:.0f}%)"
            st.metric("Final Assessment Score", st.session_state.assessment_score)
            if pct >= 80:
                st.balloons()


# ============================================================
# PAGE 7: VERIFIED FINAL PROJECT REPORT (PDF)
# ============================================================

elif page == "📄 Verified Project Report (PDF)":
    st.title("📄 Final Student Molecular Analysis Report Generator")
    st.write(
        "Enter one or more SMILES strings (one per line). The system will generate a single "
        "downloadable PDF containing **2D structures**, **SVD-projected 3D ball-and-stick models**, "
        "**Oral Bioavailability Radar charts**, **complete descriptor interpretations**, and a "
        "**SHA-256 Cryptographic Verification Seal**."
    )

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        rep_name = st.text_input("Student Full Name", value=st.session_state.get("student_name", ""))
    with col_s2:
        rep_reg = st.text_input("Registration Number", value=st.session_state.get("registration_number", ""))

    default_smiles_list = st.session_state.get("active_smiles", "CC(=O)Oc1ccccc1C(=O)O") + "\nCN1C=NC2=C1C(=O)N(C(=O)N2C)C"
    multi_smiles_text = st.text_area(
        "Enter SMILES Strings (one molecule per line)",
        value=default_smiles_list,
        height=140
    )

    smiles_lines = [line.strip() for line in multi_smiles_text.splitlines() if line.strip()]
    molecule_records = []
    invalid_entries = []

    for smi in smiles_lines:
        m = get_molecule(smi)
        if m is None:
            invalid_entries.append(smi)
        else:
            p = calculate_properties_cached(smi)
            molecule_records.append((smi, m, p))

    if invalid_entries:
        st.warning(f"⚠️ Invalid SMILES skipped: {', '.join(invalid_entries)}")

    if molecule_records:
        st.success(f"✅ {len(molecule_records)} valid molecule(s) queued for PDF compilation.")

        preview_df = pd.DataFrame([
            {
                "No.": idx,
                "Input SMILES": s,
                "Canonical SMILES": Chem.MolToSmiles(m),
                "Formula": p["Molecular Formula"],
                "MW (g/mol)": p["Molecular Weight"],
                "LogP": p["LogP"],
                "TPSA (Å²)": p["TPSA"],
                "σ / π Bonds": f"{p['Sigma (σ) Bonds']} / {p['Pi (π) Bonds']}"
            }
            for idx, (s, m, p) in enumerate(molecule_records, start=1)
        ])
        st.dataframe(preview_df, use_container_width=True, hide_index=True)

        preview_hash, preview_ts = generate_verification_hash(
            rep_name or "Anonymous",
            rep_reg or "N/A",
            ";".join([r[0] for r in molecule_records])
        )
        st.markdown("#### 🔐 Report Integrity Preview (SHA-256)")
        st.markdown(f'<div class="hash-box">SHA-256: {preview_hash} | UTC: {preview_ts}</div>', unsafe_allow_html=True)
        st.write("")

        if st.button("🚀 Compile & Generate Verified Multi-Molecule PDF Report"):
            with st.spinner("Rendering 2D structures, 3D SVD projections, Bioavailability Radars, and PDF tables..."):
                pdf_bytes = generate_multi_final_report(rep_name, rep_reg, molecule_records)

            st.success("🎉 PDF Report compiled successfully!")
            safe_reg = (rep_reg.strip() if rep_reg.strip() else "Student").replace(" ", "_")
            st.download_button(
                label="⬇️ Download Verified Cheminformatics Lab Report (.PDF)",
                data=pdf_bytes,
                file_name=f"Cheminformatics_Report_{safe_reg}.pdf",
                mime="application/pdf"
            )
    else:
        st.info("Please enter at least one valid SMILES string above to generate your PDF report.")

# ============================================================
# FOOTER
# ============================================================

st.divider()
st.caption("🧪 Cheminformatics Virtual Laboratory v2.0 | Built with Python, Streamlit, RDKit, py3Dmol & ReportLab | SRM IST Tiruchirappalli")
