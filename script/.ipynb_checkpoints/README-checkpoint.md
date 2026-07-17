# AResDB & AResKG — LLM-Powered Data Extraction Pipeline

<p align="center">
  <img src="../AResKG_resistance_prediction/figure/Figure_abstract.png" width="800">
</p>

<p align="center">
  <b>Automated construction of antibiotic resistance databases and knowledge graphs from scientific literature using Large Language Models.</b>
</p>

<p align="center">
  <a href="#installation">Installation</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#pipeline-overview">Pipeline</a> •
  <a href="#configuration">Configuration</a> •
  <a href="#data-format">Data Format</a> •
  <a href="#citation">Citation</a>
</p>

---

## 📌 Overview

This repository provides a complete, reproducible pipeline for:

1. **Retrieving** open-access scientific articles from Europe PMC
2. **Converting** PDFs and supplementary materials to machine-readable Markdown
3. **Extracting** structured antibiotic resistance data using LLMs (any OpenAI-compatible API)
4. **Integrating & cleaning** extracted records into a standardized database
5. **Constructing** a multi-relational knowledge graph (AResKG)

> **AResDB** contains 22,414 experimental samples spanning 2,769 strains, 3,379 compounds, 359 genes, 468 mutations, and 352 resistance mechanisms.
>
> **AResKG** contains 51,704,105 triples across 14 entity types and 27 relation types.

---

## 🚀 Installation

### Prerequisites

- Python ≥ 3.8
- [Conda](https://docs.conda.io/) (recommended) or pip
- An LLM API key (OpenAI, DeepSeek, or any compatible provider)

### Setup

```bash
# Clone the repository
git clone https://github.com/your-org/AResKG.git
cd AResKG/script

# Option 1: Conda (recommended)
conda env create -f environment.yaml
conda activate dataextractor

# Option 2: pip
pip install -r requirements.txt
```

### Configure API Keys

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env with your credentials
vim .env
```

Or export directly in your shell:

```bash
export LLM_API_KEY="your_api_key_here"
export LLM_API_URL="https://api.deepseek.com/v1/chat/completions"
export LLM_MODEL="deepseek-v3"
```

**Supported LLM Providers:**

| Provider | API URL | Model Examples |
|----------|---------|----------------|
| OpenAI | `https://api.openai.com/v1/chat/completions` | `gpt-4o`, `gpt-4o-mini` |
| DeepSeek | `https://api.deepseek.com/v1/chat/completions` | `deepseek-v3`, `deepseek-chat` |
| Anthropic | `https://api.anthropic.com/v1/messages` | `claude-3-opus`, `claude-3-sonnet` |
| Local (vLLM) | `http://localhost:8000/v1/chat/completions` | Any served model |
| Local (Ollama) | `http://localhost:11434/v1/chat/completions` | `llama3`, `qwen2.5` |

---

## ⚡ Quick Start

Run the full pipeline end-to-end:

```bash
cd script/

# Step 1: Download papers (requires a PMC ID list)
bash 01_paper_retrieval.sh --input ../data_demo/pmc_id_list.txt --output ../data_demo/1_download_pmc

# Step 2: Convert PDFs to Markdown
python 02_pdf_to_markdown.py --input ../data_demo/1_download_pmc --output ../data_demo/2_convert_md_pmc

# Step 3: LLM-based data extraction (requires API key)
python 03_llm_data_extraction.py \
    --input ../data_demo/2_convert_md_pmc \
    --output ../data_demo/3_response_pmc \
    --prompt prompt/p_resistance_extract.txt \
    --prompt-merge prompt/p_resistance_merge.txt

# Step 4: Integrate extraction results
python 04_data_integration.py \
    --input ../data_demo/3_response_pmc \
    --output ../data_demo/4_integrated/LLM_resistance_merged.csv

# Step 5: Post-process and standardize
python 05_data_postprocess.py \
    --input ../data_demo/4_integrated/LLM_resistance_merged.csv \
    --output ../data_demo/5_standardized/antibiotic_resistance_database.csv

# Step 6: Build knowledge graph
python 06_kg_construction.py \
    --input ../data_demo/5_standardized/antibiotic_resistance_database.csv \
    --output ../data_demo/6_knowledge_graph/antibiotic_resistance_kg.tsv
```

---

## 🔄 Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        AResDB/AResKG Construction Pipeline                   │
└─────────────────────────────────────────────────────────────────────────────┘

  ┌──────────────┐     ┌──────────────┐     ┌──────────────────────────────┐
  │  Europe PMC  │────▶│  PDF + Supp  │────▶│  Markdown (full text + tables)│
  │  (open-access)│     │  Materials   │     │                              │
  └──────────────┘     └──────────────┘     └──────────────┬───────────────┘
    01_paper_            02_pdf_to_                          │
    retrieval.sh         markdown.py                        ▼
                                             ┌──────────────────────────────┐
                                             │   LLM Data Extraction        │
                                             │   (OpenAI-compatible API)    │
                                             └──────────────┬───────────────┘
                                               03_llm_data_extraction.py
                                                            │
                                                            ▼
  ┌──────────────┐     ┌──────────────┐     ┌──────────────────────────────┐
  │  Knowledge   │◀────│ Standardized │◀────│  Integrated Raw Extractions   │
  │  Graph (TSV) │     │   Database   │     │                              │
  └──────────────┘     └──────────────┘     └──────────────────────────────┘
    06_kg_               05_data_             04_data_integration.py
    construction.py      postprocess.py
```

### Step Details

| Step | Script | Description | External Dependencies |
|------|--------|-------------|----------------------|
| 1 | `01_paper_retrieval.sh` | Download PDFs from Europe PMC | `pygetpapers` (pip) |
| 2 | `02_pdf_to_markdown.py` | PDF → Markdown conversion | `PyMuPDF`, `pymupdf4llm` |
| 3 | `03_llm_data_extraction.py` | LLM-based structured extraction | **LLM API key** ⚠️ |
| 4 | `04_data_integration.py` | Parse & merge LLM outputs | — |
| 5 | `05_data_postprocess.py` | Normalize & standardize data | ChEMBL API (optional) |
| 6 | `06_kg_construction.py` | Build KG triples | — |

---

## ⚙️ Configuration

### LLM API (Step 3)

The extraction step requires an **OpenAI-compatible API**. Configure via environment variables or command-line arguments:

```bash
# Via environment (recommended)
export LLM_API_KEY="sk-..."
export LLM_API_URL="https://api.openai.com/v1/chat/completions"
export LLM_MODEL="gpt-4o"

# Or via arguments
python 03_llm_data_extraction.py \
    --api-key "sk-..." \
    --api-url "https://api.openai.com/v1/chat/completions" \
    --model "gpt-4o" \
    --temperature 0.1 \
    --max-tokens 8192 \
    ...
```

### Extraction Prompt

The extraction prompt (`prompt/p_resistance_extract.txt`) instructs the LLM to output a structured Markdown table with 16 columns:

| Column | Description |
|--------|-------------|
| Source | Data source (PMC ID) |
| Pubmed | PubMed ID |
| Organism | Bacterial species |
| Strain | Specific strain |
| Genes | Resistance gene(s) |
| Mutation | Point mutation(s) |
| Antibiotic | Drug compound |
| Class | Antibiotic class |
| Mechanism | Resistance mechanism |
| standard_relation | =, >, <, >= |
| standard_type | MIC, IC50, MBC, etc. |
| standard_value | Numeric value |
| standard_units | μg/mL, μM, etc. |
| Susceptibility | Resistant/Sensitive |
| SMILES | Molecular structure |
| Notes | Additional notes |

You can customize the prompt for your specific extraction needs.

---

## 📁 Data Format

### Input: PMC ID List

A simple text file with one PMC ID per line:

```
PMC12189289
PMC12179561
PMC12135510
# Comments are supported
PMC11984708
```

### Output: Knowledge Graph (TSV)

Tab-separated triples file:

```tsv
Subject	Predicate	Object
Escherichia coli	has_gene	blaTEM-1
blaTEM-1	confers_resistance_to	ampicillin
ampicillin	belongs_to_class	beta-lactam
blaTEM-1	has_mutation	R164S
Escherichia coli	resistant_to	ampicillin
```

### Entity Types (14)

`Organism` · `Strain` · `Gene` · `Mutation` · `Antibiotic` · `Antibiotic_Class` · `Resistance_Mechanism` · `GO_MF` · `GO_BP` · `GO_CC` · `Measurement_ID` · `Standard_Type` · `Standard_Value` · `Standard_Unit`

### Relation Types (27)

`has_gene` · `has_mutation` · `resistant_to` · `sensitive_to` · `confers_resistance_to` · `targeted_by` · `belongs_to_class` · `has_structure` · `has_mechanism` · `upregulated_by_antibiotic` · `activates` · `represses` · `regulates` · `is_part_of` · `has` · `is_involved_in` ...

---

## 📂 Repository Structure

```
script/
├── README.md                       # This file
├── .env.example                    # API configuration template
├── requirements.txt                # Python dependencies
├── 01_paper_retrieval.sh           # Step 1: Download papers
├── 02_pdf_to_markdown.py           # Step 2: PDF → Markdown
├── 03_llm_data_extraction.py       # Step 3: LLM extraction
├── 04_data_integration.py          # Step 4: Merge results
├── 05_data_postprocess.py          # Step 5: Standardization
├── 06_kg_construction.py           # Step 6: KG construction
├── prompt/
│   ├── p_resistance_extract.txt    # Extraction prompt
│   └── p_resistance_merge.txt      # Merge prompt (for long articles)
└── utils/
    ├── __init__.py
    └── data_postprocess.py         # Cleaning & normalization utilities
```

### Related Directories

```
data_demo/                          # Example data at each pipeline stage
├── 1_download_pmc/                 # Downloaded PDFs
├── 2_convert_md_pmc/               # Converted Markdown
├── 3_response_pmc/                 # LLM extraction outputs
├── 4_integrated/                   # Merged dataset
├── 5_standardized/                 # Cleaned database (AResDB)
└── 6_knowledge_graph/              # Final KG triples (AResKG)
```

---

## 🛠️ Advanced Usage

### Using a Local LLM (vLLM / Ollama)

You can run extraction with a locally hosted model:

```bash
# Start vLLM server
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-72B-Instruct \
    --port 8000

# Run extraction against local server
python 03_llm_data_extraction.py \
    --input ../data_demo/2_convert_md_pmc \
    --output ../data_demo/3_response_pmc \
    --prompt prompt/p_resistance_extract.txt \
    --api-url http://localhost:8000/v1/chat/completions \
    --api-key "not-needed" \
    --model "Qwen/Qwen2.5-72B-Instruct"
```

### Offline Mode (Skip ChEMBL)

If you don't have internet access during post-processing:

```bash
python 05_data_postprocess.py --input ... --output ... --skip-chembl
```

### Custom Extraction Targets

Modify `prompt/p_resistance_extract.txt` to extract different information. The pipeline supports any structured Markdown table output from the LLM.

---

## 📊 Example Output

### AResDB Statistics

| Metric | Count |
|--------|-------|
| Experimental samples | 22,414 |
| Microbial strains | 2,769 |
| Antibiotic compounds | 3,379 |
| Resistance genes | 359 |
| Point mutations | 468 |
| Resistance mechanisms | 352 |

### AResKG Statistics

| Metric | Count |
|--------|-------|
| Total triples | 51,704,105 |
| Entity types | 14 |
| Relation types | 27 |
| Source articles | 2,700+ |

---

## 📄 Citation

If you use this pipeline, AResDB, or AResKG in your research, please cite:

```bibtex
@article{yang2025areskg,
  title={LLM-powered construction of antibiotic resistance datasets and knowledge graphs for accelerated resistance gene discovery},
  author={Yang, Chris Zhenyu and others},
  journal={npj Digital Medicine},
  year={2025}
}
```

---

## ⚖️ License

AResDB and AResKG were created and maintained by the Tencent AI for Life Sciences Lab.

© Tencent AI for Life Sciences Lab. All rights reserved.

Unauthorized reproduction, redistribution, or commercial use is prohibited without prior permission.
