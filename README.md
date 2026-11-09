# 🧪 Cheminformatics Virtual Laboratory (v2.0)

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![RDKit](https://img.shields.io/badge/RDKit-Cheminformatics-2563eb)](https://www.rdkit.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

An interactive **Cheminformatics and Computational Chemistry Virtual Laboratory** built with **Python, Streamlit, RDKit, py3Dmol, and ReportLab**. Designed for students, educators, and researchers to perform real-time 2D/3D molecular modeling, physicochemical descriptor analysis, ADME drug-likeness screening, and cryptographic PDF lab report generation.

---

## ✨ Key Features

* **🎨 Interactive 2D Structure & SMARTS Highlighter:** Automatically detects and highlights functional groups (Hydroxyl, Carbonyl, Carboxylic Acid, Ester, Amine, Amide, Aromatic rings, Halogens) on 2D molecular graphs and computes atomic **Gasteiger partial charges**.
* **🌐 3D Conformer Embedding & Force-Field Energetics:** Generates 3D coordinates via the **ETKDGv3** algorithm, performs geometry optimization using **MMFF94 / UFF** force fields (reporting minimized potential energy in `kcal/mol`), and supports `.SDF`, `.MOL`, and `.XYZ` coordinate exports.
* **🔬 Covalent Bonding & Hybridization Engine:** Accurately computes heavy-atom bonds, Kekulized **Sigma ($\sigma$)** and **Pi ($\pi$)** bond counts, and carbon hybridization distributions ($sp^3 / sp^2 / sp$).
* **💊 ADME & Drug-Likeness Screening:** Evaluates **Lipinski's Rule of Five (Ro5)** and **Veber's Oral Bioavailability Rule** alongside a 6-axis **Bioavailability Radar Chart**.
* **📈 Multi-Molecule & Tanimoto Similarity Lab:** Compares multiple compounds simultaneously with **MW vs. LogP Chemical Space Scatter Plots** and **Morgan Fingerprint (ECFP4, 2048-bit) Pairwise Tanimoto Similarity Heatmaps**.
* **🔐 Tamper-Evident PDF Lab Reports:** Compiles single or multi-molecule student reports featuring 2D connectivity, SVD-projected 3D ball-and-stick geometries, bioavailability radars, descriptor interpretations, and a **SHA-256 Cryptographic Audit Seal**.

---

## 🛠️ Tech Stack

* **Frontend & UI:** Streamlit, Custom Glassmorphic CSS, HTML Components
* **Cheminformatics Engine:** RDKit (`Chem`, `AllChem`, `Descriptors`, `Crippen`, `Lipinski`, `rdMolDescriptors`, `DataStructs`)
* **3D Visualization & Plotting:** py3Dmol, Matplotlib, NumPy (SVD 3D-to-2D optimal plane projection)
* **Report Generation & Integrity:** ReportLab (`platypus`), Python `hashlib` (SHA-256)

---

## 🚀 Installation & Local Setup

### 1. Clone the Repository
```bash
git clone [https://github.com/YOUR_USERNAME/cheminformatics-virtual-lab.git](https://github.com/YOUR_USERNAME/cheminformatics-virtual-lab.git)
cd cheminformatics-virtual-lab
```

### 2. Set Up Virtual Environment & Install Dependencies

#### **On Windows (PowerShell):**
```powershell
python -m venv venv
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

#### **On Windows (Command Prompt):**
```cmd
python -m venv venv
venv\Scripts\activate.bat
pip install -r requirements.txt
streamlit run app.py
```

#### **On Linux (Ubuntu/Debian) & macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

---

## 📂 Project Structure

```text
├── app.py                # Main Streamlit application & cheminformatics engine
├── requirements.txt      # Python dependencies
├── srm_logo.jpeg         # Optional institutional header banner
└── README.md             # Project documentation
```

---

## 🎓 Academic Credits

* **Curriculum & Concept:** Dr. R. Subramanian, Assistant Professor, Division of Chemistry, School of Sciences, SRM Institute of Science and Technology, Tiruchirappalli.
