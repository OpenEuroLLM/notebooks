# Battle Annotation

This notebook allows you to annotate battle preference data from LMSys and ComparIA.

## Installation

This project uses [uv](https://github.com/astral-sh/uv) for fast, reliable Python package management. If you don't have `uv` installed yet (check uv install instruction if you run outside of linux/mac):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then install all required packages using `uv`:

```bash
uv venv  # Create a virtual environment
source .venv/bin/activate  # activates it
uv pip install -r requirements.txt  # Install dependencies
```

Then start Jupyter:

```bash
jupyter notebook
```

and open `Annotate-Data.ipynb` in your browser.

## Usage

The notebook provides an interactive interface to:
1. Load battle preference data from LMSys and ComparIA
2. Select a language to annotate
3. Review conversation pairs and annotate preferences
4. Save annotations locally and to HuggingFace

Make sure to configure your HuggingFace repository in the notebook before starting annotation.
