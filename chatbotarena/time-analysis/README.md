# LM Arena Time Analysis

This analysis pulls historical ELO rating data from LMSys Chatbot Arena and generates comprehensive visualizations to track the evolution of language model performance over time. The analysis includes ELO ratings trends, ranking bump charts, performance comparisons between open and proprietary models, organization-level performance tracking, and volatility analysis. All visualizations are automatically saved to a `figures/` directory for easy review and sharing.

## Installation

This project uses [uv](https://github.com/astral-sh/uv) for dependency management. Install the dependencies with:

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# From the chatbotarena directory (parent of time-analysis)
cd ..
uv sync

# Or install dependencies directly
uv pip install matplotlib pandas numpy datasets huggingface-hub wandb
```

## Usage

Run the analysis script to pull the latest data and generate all visualizations:

```bash
python data_analysis.py
```

The script will:
1. Download historical ELO rating data from LMSys Chatbot Arena
2. Generate 9 different visualizations analyzing model performance
3. Save all figures to the `figures/` directory
4. Print statistical summaries to the console

## Output

All visualizations are saved in the `figures/` directory and include:
- ELO ratings over time for top models
- Ranking bump charts showing position changes
- Rating heatmaps across models and time
- Rating distribution analysis
- Top performers timeline
- Rating volatility analysis
- Organization performance comparisons
- License type performance analysis
- Open vs Proprietary model comparisons
