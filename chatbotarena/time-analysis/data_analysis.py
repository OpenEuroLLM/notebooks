import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
from pull_lmarena import load_all_elo, load_df


def plot_elo_over_time(df: pd.DataFrame, top_n: int = None, save_path: str = None):
    """Plot ELO ratings over time for all models.

    Args:
        df: DataFrame with columns: model, rating, date
        top_n: If specified, only plot the top N models by average rating
        save_path: If specified, save the plot to this path
    """
    if df.empty:
        print("No data to plot")
        return

    # Calculate average rating per model to optionally filter top models
    if top_n is not None:
        avg_ratings = df.groupby('model')['rating'].mean().sort_values(ascending=False)
        top_models = avg_ratings.head(top_n).index.tolist()
        df = df[df['model'].isin(top_models)]
        print(f"Plotting top {top_n} models by average rating")

    # Create the plot
    fig, ax = plt.subplots(figsize=(14, 8))

    # Plot each model as a line
    for model in df['model'].unique():
        model_data = df[df['model'] == model].sort_values('date')
        ax.plot(model_data['date'], model_data['rating'], marker='o', label=model, alpha=0.7, linewidth=2)

    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('ELO Rating', fontsize=12)
    ax.set_title('Model ELO Ratings Over Time', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    # plt.show()  # Disabled - only saving figures


def plot_top_models_elo(df: pd.DataFrame, top_n: int = 10, save_path: str = None):
    """Plot ELO ratings over time for top N models.

    Args:
        df: DataFrame with columns: model, rating, date
        top_n: Number of top models to plot
        save_path: If specified, save the plot to this path
    """
    plot_elo_over_time(df, top_n=top_n, save_path=save_path)


def print_statistics(df: pd.DataFrame):
    """Print basic statistics about the ELO data.

    Args:
        df: DataFrame with columns: model, rating, date
    """
    if df.empty:
        print("No data available")
        return

    print("\n" + "="*60)
    print("ELO Data Statistics")
    print("="*60)
    print(f"Total entries: {len(df)}")
    print(f"Unique models: {df['model'].nunique()}")
    print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"Number of dates: {df['date'].nunique()}")

    print("\n" + "-"*60)
    print("Top 10 Models by Average Rating:")
    print("-"*60)
    avg_ratings = df.groupby('model')['rating'].mean().sort_values(ascending=False).head(10)
    for i, (model, rating) in enumerate(avg_ratings.items(), 1):
        print(f"{i:2d}. {model:40s} {rating:8.2f}")

    print("\n" + "-"*60)
    print("Rating Statistics:")
    print("-"*60)
    print(df['rating'].describe())


def plot_ranking_bump_chart(df: pd.DataFrame, top_n: int = 15, save_path: str = None):
    """Create a bump chart showing ranking changes over time.

    This is excellent for showing relative position changes and competitive dynamics.

    Args:
        df: DataFrame with columns: model, final_ranking, date
        top_n: Number of top models to track
        save_path: If specified, save the plot to this path
    """
    if df.empty:
        print("No data to plot")
        return

    # Get models that have been in top N at any point
    top_models = df[df['final_ranking'] <= top_n]['model'].unique()
    df_top = df[df['model'].isin(top_models)].copy()

    # Pivot to get ranking over time
    pivot = df_top.pivot_table(index='date', columns='model', values='final_ranking')

    fig, ax = plt.subplots(figsize=(16, 10))

    # Plot each model's ranking over time
    for model in pivot.columns:
        model_data = pivot[model].dropna()
        ax.plot(model_data.index, model_data.values, marker='o', linewidth=2.5,
                label=model, alpha=0.8, markersize=6)

    # Invert y-axis so rank 1 is at top
    ax.invert_yaxis()
    ax.set_yticks(range(1, top_n + 1))
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Ranking Position', fontsize=12)
    ax.set_title(f'Top {top_n} Model Rankings Over Time (Bump Chart)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    # plt.show()  # Disabled - only saving figures


def plot_rating_heatmap(df: pd.DataFrame, top_n: int = 30, save_path: str = None):
    """Create a heatmap showing ratings across models and time.

    Good for pattern recognition and spotting trends.

    Args:
        df: DataFrame with columns: model, rating, date
        top_n: Number of top models to show
        save_path: If specified, save the plot to this path
    """
    if df.empty:
        print("No data to plot")
        return

    # Get top N models by average rating
    avg_ratings = df.groupby('model')['rating'].mean().sort_values(ascending=False)
    top_models = avg_ratings.head(top_n).index.tolist()
    df_top = df[df['model'].isin(top_models)].copy()

    # Pivot to get rating over time
    pivot = df_top.pivot_table(index='model', columns='date', values='rating')

    # Sort by average rating
    pivot = pivot.loc[avg_ratings.head(top_n).index]

    fig, ax = plt.subplots(figsize=(18, 12))

    # Create heatmap
    im = ax.imshow(pivot.values, aspect='auto', cmap='RdYlGn', interpolation='nearest')

    # Set ticks
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=8)
    ax.set_xticks(range(0, len(pivot.columns), max(1, len(pivot.columns)//10)))
    ax.set_xticklabels([pivot.columns[i].strftime('%Y-%m-%d') for i in range(0, len(pivot.columns), max(1, len(pivot.columns)//10))], rotation=45, ha='right', fontsize=8)

    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Model', fontsize=12)
    ax.set_title(f'ELO Rating Heatmap - Top {top_n} Models', fontsize=14, fontweight='bold')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('ELO Rating', fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    # plt.show()  # Disabled - only saving figures


def plot_rating_distribution_over_time(df: pd.DataFrame, save_path: str = None):
    """Plot how the distribution of ratings changes over time.

    Shows the spread and central tendency of all model ratings.

    Args:
        df: DataFrame with columns: rating, date
        save_path: If specified, save the plot to this path
    """
    if df.empty:
        print("No data to plot")
        return

    # Group by date and calculate statistics
    date_stats = df.groupby('date')['rating'].agg([
        ('mean', 'mean'),
        ('median', 'median'),
        ('q25', lambda x: x.quantile(0.25)),
        ('q75', lambda x: x.quantile(0.75)),
        ('min', 'min'),
        ('max', 'max')
    ]).reset_index()

    fig, ax = plt.subplots(figsize=(14, 8))

    # Plot the range
    ax.fill_between(date_stats['date'], date_stats['min'], date_stats['max'],
                     alpha=0.2, label='Full Range', color='lightblue')
    ax.fill_between(date_stats['date'], date_stats['q25'], date_stats['q75'],
                     alpha=0.4, label='25-75 Percentile', color='skyblue')
    ax.plot(date_stats['date'], date_stats['median'], linewidth=2.5,
            label='Median', color='darkblue', marker='o')
    ax.plot(date_stats['date'], date_stats['mean'], linewidth=2,
            label='Mean', color='red', linestyle='--', marker='s')

    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('ELO Rating', fontsize=12)
    ax.set_title('ELO Rating Distribution Over Time', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    # plt.show()  # Disabled - only saving figures


def plot_top_performers_timeline(df: pd.DataFrame, top_n: int = 10, save_path: str = None):
    """Show which models were in top N positions over time.

    Highlights competitive dynamics at the top of the leaderboard.

    Args:
        df: DataFrame with columns: model, final_ranking, date
        top_n: Threshold for "top" models
        save_path: If specified, save the plot to this path
    """
    if df.empty:
        print("No data to plot")
        return

    # Filter to top N at each date
    df_top = df[df['final_ranking'] <= top_n].copy()

    # Count appearances in top N
    top_counts = df_top['model'].value_counts().head(20)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))

    # Plot 1: Timeline showing which models are in top N
    for model in top_counts.index:
        model_data = df_top[df_top['model'] == model].sort_values('date')
        ax1.scatter(model_data['date'], [model] * len(model_data),
                   s=100, alpha=0.6, label=model)

    ax1.set_xlabel('Date', fontsize=12)
    ax1.set_ylabel('Model', fontsize=12)
    ax1.set_title(f'Models in Top {top_n} Over Time', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')

    # Plot 2: Bar chart of total days in top N
    ax2.barh(range(len(top_counts)), top_counts.values, color='steelblue')
    ax2.set_yticks(range(len(top_counts)))
    ax2.set_yticklabels(top_counts.index, fontsize=9)
    ax2.set_xlabel('Number of Snapshots in Top ' + str(top_n), fontsize=12)
    ax2.set_title(f'Total Appearances in Top {top_n}', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    # plt.show()  # Disabled - only saving figures


def plot_rating_volatility(df: pd.DataFrame, min_appearances: int = 5, top_n: int = 20, save_path: str = None):
    """Analyze and visualize rating volatility (stability) of models.

    Shows which models have stable vs volatile ratings.

    Args:
        df: DataFrame with columns: model, rating, date
        min_appearances: Minimum number of data points required
        top_n: Number of models to show
        save_path: If specified, save the plot to this path
    """
    if df.empty:
        print("No data to plot")
        return

    # Calculate volatility (standard deviation) and average rating per model
    model_stats = df.groupby('model').agg({
        'rating': ['mean', 'std', 'count']
    }).reset_index()
    model_stats.columns = ['model', 'avg_rating', 'volatility', 'count']

    # Filter models with enough data points
    model_stats = model_stats[model_stats['count'] >= min_appearances]

    # Sort by average rating and take top N
    model_stats = model_stats.nlargest(top_n, 'avg_rating')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

    # Plot 1: Scatter of volatility vs average rating
    scatter = ax1.scatter(model_stats['avg_rating'], model_stats['volatility'],
                         s=model_stats['count']*10, alpha=0.6, c=model_stats['avg_rating'],
                         cmap='viridis')

    for idx, row in model_stats.iterrows():
        ax1.annotate(row['model'], (row['avg_rating'], row['volatility']),
                    fontsize=7, alpha=0.7)

    ax1.set_xlabel('Average ELO Rating', fontsize=12)
    ax1.set_ylabel('Rating Volatility (Std Dev)', fontsize=12)
    ax1.set_title('Model Stability Analysis', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    cbar = plt.colorbar(scatter, ax=ax1)
    cbar.set_label('Avg Rating', fontsize=10)

    # Plot 2: Bar chart of volatility
    model_stats_sorted = model_stats.sort_values('volatility', ascending=True)
    ax2.barh(range(len(model_stats_sorted)), model_stats_sorted['volatility'],
            color='coral')
    ax2.set_yticks(range(len(model_stats_sorted)))
    ax2.set_yticklabels(model_stats_sorted['model'], fontsize=8)
    ax2.set_xlabel('Rating Volatility (Std Dev)', fontsize=12)
    ax2.set_title('Most to Least Stable Models', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    # plt.show()  # Disabled - only saving figures


def infer_organization_from_model(model_name: str) -> str:
    """Infer organization from model name using substring matching.

    Args:
        model_name: The model name

    Returns:
        Inferred organization name or None
    """
    if pd.isna(model_name):
        return None

    model_lower = str(model_name).lower()

    # Mapping of substrings to organizations (order matters - more specific first)
    org_mapping = {
        'amazon': 'Amazon',
        'athene': 'NexusFlow',
        'bard-jan-24-gemini-pro': 'Google',
        'aya': 'Cohere',
        'chatgpt': 'OpenAI',
        'gpt': 'OpenAI',
        'claude': 'Anthropic',
        'command-': 'Cohere',
        'deepseek': 'DeepSeek',
        'grok': 'xAI',
        'gemini': 'Google',
        'glm': 'Z.ai',
        'llama': 'Meta',
        'mistral': 'Mistral',
        'oasst-pythia-12b': 'Eleuther',
        'palm': 'Google',
        'phi': 'Microsoft',
        'qwen': 'Alibaba',
        'reka': 'Reka AI',
        'snowflake-arctic-instruct': 'Snowflake',
    }

    for substring, org in org_mapping.items():
        if substring in model_lower:
            return org

    return None


def infer_license_from_model(model_name: str) -> str:
    """Infer license type (Open/Proprietary) from model name using substring matching.

    Args:
        model_name: The model name

    Returns:
        'Open', 'Proprietary', or None
    """
    if pd.isna(model_name):
        return None

    model_lower = str(model_name).lower()

    # Mapping of substrings to license types (order matters - more specific first)
    model_weights = {
        "gpt-4": "Proprietary",
        "gpt-3.5": "Proprietary",
        "mistral-large": "Proprietary",
        "mistral-medium": "Proprietary",
        "mistral-next": "Proprietary",
        "mistral-7b": "Open",
        "glm-4": "Proprietary",
        "phi-3": "Open",
        "snowflake-arctic": "Open",
        "command-r": "Proprietary",
        "alpaca": "Open",
        "bard": "Proprietary",
        "chatglm": "Open",
        "claude": "Proprietary",
        "codellama": "Open",
        "dbrx": "Open",
        "deepseek": "Open",
        "dolly": "Open",
        "dolphin": "Open",
        "fastchat": "Open",
        "gemini": "Proprietary",
        "gemma": "Open",
        "gpt4all": "Open",
        "guanaco": "Open",
        "koala": "Open",
        "llama": "Open",
        "mixtral": "Open",
        "mpt": "Open",
        "nemotron": "Open",
        "nous-hermes": "Open",
        "oasst": "Open",
        "olmo": "Open",
        "openchat": "Open",
        "openhermes": "Open",
        "palm": "Proprietary",
        "qwen": "Open",
        "reka": "Proprietary",
        "solar": "Open",
        "stablelm": "Open",
        "starling": "Open",
        "stripedhyena": "Open",
        "tulu": "Open",
        "vicuna": "Open",
    }

    for substring, license_type in model_weights.items():
        if substring in model_lower:
            return license_type

    return None

def plot_organization_performance(df_elo: pd.DataFrame, df_metadata: pd.DataFrame, top_n: int = 10, save_path: str = None):
    """Plot performance of different organizations over time.

    Args:
        df_elo: DataFrame with ELO data (columns: model, rating, date)
        df_metadata: DataFrame with metadata (columns include: model, License, Organization)
        top_n: Number of top organizations to show
        save_path: If specified, save the plot to this path
    """
    if df_elo.empty or df_metadata.empty:
        print("No data to plot")
        return

    # Normalize column names to lowercase
    df_metadata_copy = df_metadata.copy()
    df_metadata_copy.columns = df_metadata_copy.columns.str.lower()

    df_elo_copy = df_elo.copy()
    df_elo_copy.model = df_elo.model.str.lower()
    df_metadata_copy["model"] = df_metadata.Model.str.lower()

    # Prepare metadata pivot
    metadata = df_metadata_copy.pivot_table(index="model", aggfunc="first", values=["license", "organization"])

    # Merge ELO data with metadata
    df_merged = df_elo_copy.merge(metadata, left_on='model', right_index=True, how='left')

    # Infer organization for unmatched models
    df_merged['organization'] = df_merged.apply(
        lambda row: row['organization'] if pd.notna(row['organization'])
        else infer_organization_from_model(row['model']),
        axis=1
    )

    # Remove rows without organization
    df_merged = df_merged[df_merged['organization'].notna()].copy()

    # Get top N organizations by average rating
    org_avg = df_merged.groupby('organization')['rating'].mean().sort_values(ascending=False)
    top_orgs = org_avg.head(top_n).index.tolist()

    # Filter to top organizations
    df_top = df_merged[df_merged['organization'].isin(top_orgs)].copy()

    # Calculate average rating per organization per date
    org_performance_avg = df_top.groupby(['date', 'organization'])['rating'].mean().reset_index()
    org_performance_avg.columns = ['date', 'organization', 'avg_rating']

    # Calculate maximum rating per organization per date
    org_performance_max = df_top.groupby(['date', 'organization'])['rating'].max().reset_index()
    org_performance_max.columns = ['date', 'organization', 'max_rating']

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 16))

    # Plot 1: Average rating over time per organization
    for org in top_orgs:
        org_data = org_performance_avg[org_performance_avg['organization'] == org]
        ax1.plot(org_data['date'], org_data['avg_rating'], marker='o',
                label=org, linewidth=2.5, alpha=0.8)

    ax1.set_xlabel('Date', fontsize=12)
    ax1.set_ylabel('Average ELO Rating', fontsize=12)
    ax1.set_title(f'Top {top_n} Organizations - Average Model Performance Over Time',
                  fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)

    # Plot 2: Maximum rating over time per organization
    for org in top_orgs:
        org_data = org_performance_max[org_performance_max['organization'] == org]
        ax2.plot(org_data['date'], org_data['max_rating'], marker='^',
                label=org, linewidth=2.5, alpha=0.8)

    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_ylabel('Maximum ELO Rating', fontsize=12)
    ax2.set_title(f'Top {top_n} Organizations - Best Model Performance Over Time',
                  fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)

    # Plot 3: Number of models per organization over time
    org_counts = df_top.groupby(['date', 'organization']).size().reset_index(name='count')

    for org in top_orgs:
        org_data = org_counts[org_counts['organization'] == org]
        ax3.plot(org_data['date'], org_data['count'], marker='s',
                label=org, linewidth=2, alpha=0.8)

    ax3.set_xlabel('Date', fontsize=12)
    ax3.set_ylabel('Number of Models', fontsize=12)
    ax3.set_title(f'Top {top_n} Organizations - Number of Models in Leaderboard',
                  fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    # plt.show()  # Disabled - only saving figures


def plot_license_performance(df_elo: pd.DataFrame, df_metadata: pd.DataFrame, save_path: str = None):
    """Plot performance of different license types over time.

    Args:
        df_elo: DataFrame with ELO data (columns: model, rating, date)
        df_metadata: DataFrame with metadata (columns include: model, License, Organization)
        save_path: If specified, save the plot to this path
    """
    if df_elo.empty or df_metadata.empty:
        print("No data to plot")
        return

    # Normalize column names to lowercase
    df_metadata_copy = df_metadata.copy()
    df_metadata_copy.columns = df_metadata_copy.columns.str.lower()

    # Prepare metadata pivot
    metadata = df_metadata_copy.pivot_table(index="model", aggfunc="first", values=["license", "organization"])

    # Merge ELO data with metadata
    df_merged = df_elo.merge(metadata, left_on='model', right_index=True, how='left')

    # Infer license for models without license metadata
    def get_license(row):
        # First check if license is known
        if pd.notna(row['license']):
            return row['license']
        # Otherwise infer from model name
        inferred = infer_license_from_model(row['model'])
        # Map to known license format
        if inferred == 'Proprietary':
            return 'Proprietary'
        elif inferred == 'Open':
            return 'Apache 2.0'  # Use Apache as default for inferred open models
        return None

    df_merged['license'] = df_merged.apply(get_license, axis=1)

    # Remove rows without license
    df_merged = df_merged[df_merged['license'].notna()].copy()

    # Simplify license categories
    def categorize_license(license_str):
        if pd.isna(license_str):
            return 'Unknown'
        license_lower = str(license_str).lower()
        if 'proprietary' in license_lower or 'closed' in license_lower:
            return 'Proprietary'
        elif 'apache' in license_lower or 'mit' in license_lower or 'bsd' in license_lower:
            return 'Permissive (Apache/MIT/BSD)'
        elif 'llama' in license_lower:
            return 'Llama License'
        elif 'cc' in license_lower or 'creative commons' in license_lower:
            return 'Creative Commons'
        elif 'gpl' in license_lower:
            return 'GPL'
        else:
            return 'Other Open Source'

    df_merged['license_category'] = df_merged['license'].apply(categorize_license)

    # Calculate statistics per license per date
    license_stats = df_merged.groupby(['date', 'license_category']).agg({
        'rating': ['mean', 'count']
    }).reset_index()
    license_stats.columns = ['date', 'license_category', 'avg_rating', 'count']

    # Get license categories with enough data
    license_counts = df_merged['license_category'].value_counts()
    licenses_to_plot = license_counts[license_counts >= 20].index.tolist()

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 16))

    # Plot 1: Average rating over time per license
    for license_cat in licenses_to_plot:
        license_data = license_stats[license_stats['license_category'] == license_cat]
        ax1.plot(license_data['date'], license_data['avg_rating'], marker='o',
                label=license_cat, linewidth=2.5, alpha=0.8)

    ax1.set_xlabel('Date', fontsize=12)
    ax1.set_ylabel('Average ELO Rating', fontsize=12)
    ax1.set_title('License Types - Average Model Performance Over Time',
                  fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best', fontsize=10)

    # Plot 2: Number of models per license over time
    for license_cat in licenses_to_plot:
        license_data = license_stats[license_stats['license_category'] == license_cat]
        ax2.plot(license_data['date'], license_data['count'], marker='s',
                label=license_cat, linewidth=2, alpha=0.8)

    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_ylabel('Number of Models', fontsize=12)
    ax2.set_title('License Types - Number of Models in Leaderboard',
                  fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='best', fontsize=10)

    # Plot 3: Distribution comparison (box plot for latest date)
    latest_date = df_merged['date'].max()
    df_latest = df_merged[df_merged['date'] == latest_date]

    license_data_for_box = [df_latest[df_latest['license_category'] == lic]['rating'].values
                            for lic in licenses_to_plot]

    bp = ax3.boxplot(license_data_for_box, labels=licenses_to_plot, patch_artist=True)

    # Color the boxes
    colors = plt.cm.Set3(np.linspace(0, 1, len(licenses_to_plot)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    ax3.set_ylabel('ELO Rating', fontsize=12)
    ax3.set_title(f'License Types - Rating Distribution (as of {latest_date.date()})',
                  fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.tick_params(axis='x', rotation=15)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    # plt.show()  # Disabled - only saving figures


def plot_open_vs_proprietary(df_elo: pd.DataFrame, df_metadata: pd.DataFrame, save_path: str = None):
    """Plot comparison between open-weight and proprietary models.

    Args:
        df_elo: DataFrame with ELO data (columns: model, rating, date, num_battles)
        df_metadata: DataFrame with metadata (columns include: model, License, Organization)
        save_path: If specified, save the plot to this path
    """
    if df_elo.empty or df_metadata.empty:
        print("No data to plot")
        return

    # Normalize column names to lowercase
    df_metadata_copy = df_metadata.copy()
    df_metadata_copy.columns = df_metadata_copy.columns.str.lower()

    # Prepare metadata pivot
    metadata = df_metadata_copy.pivot_table(index="model", aggfunc="first", values=["license", "organization"])

    # Merge ELO data with metadata
    df_merged = df_elo.merge(metadata, left_on='model', right_index=True, how='left')

    # Infer license for models without license metadata
    def get_model_type(row):
        # First check if license is known
        if pd.notna(row['license']):
            return 'Proprietary' if str(row['license']).lower() == 'proprietary' else 'Open'
        # Otherwise infer from model name
        inferred = infer_license_from_model(row['model'])
        return inferred if inferred else None

    df_merged['model_type'] = df_merged.apply(get_model_type, axis=1)

    # Remove rows without model type
    df_merged = df_merged[df_merged['model_type'].notna()].copy()

    fig, (ax2, ax3, ax1) = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    # Get all unique dates sorted and max ratings for later use
    all_dates = sorted(df_merged['date'].unique())
    max_ratings = df_merged.groupby(['date', 'model_type'])['rating'].max().reset_index()

    # Plot 1: Cumulative number of battles for each category over time
    if 'num_battles' in df_merged.columns:
        cumulative_battles = {'Proprietary': [], 'Open': []}
        total_battles = {'Proprietary': 0, 'Open': 0}

        for date in all_dates:
            date_data = df_merged[df_merged['date'] == date]

            for model_type in ['Proprietary', 'Open']:
                type_data = date_data[date_data['model_type'] == model_type]
                # Sum battles for this date and add to cumulative total
                battles_this_date = type_data['num_battles'].sum()
                total_battles[model_type] = max(battles_this_date, total_battles[model_type])
                cumulative_battles[model_type].append(total_battles[model_type])

        for model_type in ['Proprietary', 'Open']:
            ax1.plot(all_dates, cumulative_battles[model_type], marker='s',
                    label=f'{model_type} Models', linewidth=3, alpha=0.8)

        ax1.set_xlabel('Date', fontsize=12)
        ax1.set_ylabel('Cumulative Number of Battles', fontsize=12)
        ax1.set_title('Cumulative Battle Count: Open vs Proprietary Models',
                      fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='best', fontsize=11)
    else:
        ax1.text(0.5, 0.5, 'Battle count data not available',
                ha='center', va='center', fontsize=14, transform=ax1.transAxes)
        ax1.set_title('Cumulative Battle Count: Open vs Proprietary Models',
                      fontsize=14, fontweight='bold')

    # Plot 2: Best model ELO rating over time for each category
    for model_type in ['Proprietary', 'Open']:
        type_data = max_ratings[max_ratings['model_type'] == model_type]
        ax2.plot(type_data['date'], type_data['rating'], marker='o',
                label=f'Best {model_type} Model', linewidth=3, alpha=0.8)

    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_ylabel('ELO Rating', fontsize=12)
    ax2.set_title('Best Open vs Proprietary Model Performance Over Time',
                  fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='best', fontsize=11)

    # Plot 3: Win probability of best proprietary model vs best open model
    # Calculate win probability using ELO formula: P(A wins) = 1 / (1 + 10^((Rating_B - Rating_A) / 400))
    win_probabilities = []

    for date in all_dates:
        date_ratings = max_ratings[max_ratings['date'] == date]
        prop_rating = date_ratings[date_ratings['model_type'] == 'Proprietary']['rating'].values
        open_rating = date_ratings[date_ratings['model_type'] == 'Open']['rating'].values

        if len(prop_rating) > 0 and len(open_rating) > 0:
            # Probability that proprietary wins against open
            prob_prop_wins = 1 / (1 + 10**((open_rating[0] - prop_rating[0]) / 400))
            win_probabilities.append(prob_prop_wins)
        else:
            win_probabilities.append(None)

    # Filter out None values for plotting
    valid_dates = [date for date, prob in zip(all_dates, win_probabilities) if prob is not None]
    valid_probs = [prob for prob in win_probabilities if prob is not None]

    ax3.plot(valid_dates, valid_probs, marker='o', linewidth=3, alpha=0.8, color='purple')
    ax3.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='50% (Equal)')
    ax3.fill_between(valid_dates, 0.5, valid_probs,
                     where=[p >= 0.5 for p in valid_probs],
                     alpha=0.2, color='blue', label='Proprietary Favored')
    ax3.fill_between(valid_dates, 0.5, valid_probs,
                     where=[p < 0.5 for p in valid_probs],
                     alpha=0.2, color='orange', label='Open Favored')

    ax3.set_xlabel('Date', fontsize=12)
    ax3.set_ylabel('Win Probability', fontsize=12)
    ax3.set_title('Probability Best Proprietary Model Beats Best Open Model',
                  fontsize=14, fontweight='bold')
    ax3.set_ylim([0.5, 1])
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='best', fontsize=11)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    # plt.show()  # Disabled - only saving figures


def print_metadata_statistics(df_elo: pd.DataFrame, df_metadata: pd.DataFrame):
    """Print statistics about organizations and licenses.

    Args:
        df_elo: DataFrame with ELO data
        df_metadata: DataFrame with metadata
    """
    top_n = 20

    # Normalize column names to lowercase
    df_metadata_copy = df_metadata.copy()
    df_metadata_copy.columns = df_metadata_copy.columns.str.lower()

    # Prepare metadata pivot
    metadata = df_metadata_copy.pivot_table(index="model", aggfunc="first", values=["license", "organization"])

    # Merge ELO data with metadata
    df_merged = df_elo.merge(metadata, left_on='model', right_index=True, how='left')

    # Infer organization for unmatched models
    df_merged['organization'] = df_merged.apply(
        lambda row: row['organization'] if pd.notna(row['organization'])
        else infer_organization_from_model(row['model']),
        axis=1
    )

    # Infer license for models without license metadata
    def get_license(row):
        # First check if license is known
        if pd.notna(row['license']):
            return row['license']
        # Otherwise infer from model name
        inferred = infer_license_from_model(row['model'])
        # Map to known license format
        if inferred == 'Proprietary':
            return 'Proprietary'
        elif inferred == 'Open':
            return 'Apache 2.0'  # Use Apache as default for inferred open models
        return None

    df_merged['license'] = df_merged.apply(get_license, axis=1)

    print("\n" + "="*60)
    print("Organization & License Statistics")
    print("="*60)

    # Top organizations by average rating
    print(f"\nTop {top_n} Organizations by Average Model Rating:")
    print("-"*60)
    org_stats = df_merged.groupby('organization').agg({
        'rating': ['mean', 'count']
    }).round(2)
    org_stats.columns = ['Avg Rating', 'Model Count']
    org_stats = org_stats.sort_values('Avg Rating', ascending=False).head(top_n)
    print(org_stats)

    # License distribution (after inference)
    print("\nLicense Distribution (with inference):")
    print("-"*60)
    license_counts = df_merged['license'].value_counts().head(top_n)
    print(license_counts)

    # Model type distribution (Open vs Proprietary)
    print("\nModel Type Distribution:")
    print("-"*60)
    df_merged['model_type'] = df_merged['license'].apply(
        lambda x: 'Proprietary' if pd.notna(x) and str(x).lower() == 'proprietary' else 'Open' if pd.notna(x) else 'Unknown'
    )
    type_counts = df_merged['model_type'].value_counts()
    print(type_counts)


def analyze_gap_changes(df_elo: pd.DataFrame, df_metadata: pd.DataFrame, target_dates: list):
    """Analyze which models were responsible for closing the gap at specific dates.

    Args:
        df_elo: DataFrame with ELO data
        df_metadata: DataFrame with metadata
        target_dates: List of date strings in format 'YYYY-MM'
    """
    # Normalize column names to lowercase
    df_metadata_copy = df_metadata.copy()
    df_metadata_copy.columns = df_metadata_copy.columns.str.lower()

    # Prepare metadata pivot
    metadata = df_metadata_copy.pivot_table(index="model", aggfunc="first", values=["license", "organization"])

    # Merge ELO data with metadata
    df_merged = df_elo.merge(metadata, left_on='model', right_index=True, how='left')

    # Infer license for models without license metadata
    def get_model_type(row):
        if pd.notna(row['license']):
            return 'Proprietary' if str(row['license']).lower() == 'proprietary' else 'Open'
        inferred = infer_license_from_model(row['model'])
        return inferred if inferred else None

    df_merged['model_type'] = df_merged.apply(get_model_type, axis=1)
    df_merged = df_merged[df_merged['model_type'].notna()].copy()

    print("\n" + "="*60)
    print("Gap Analysis: Top Models at Key Dates")
    print("="*60)

    for target_date in target_dates:
        # Parse target date (YYYY-MM format)
        year, month = map(int, target_date.split('-'))

        # Find dates within that month
        month_data = df_merged[
            (df_merged['date'].dt.year == year) &
            (df_merged['date'].dt.month == month)
        ]

        if month_data.empty:
            print(f"\nNo data found for {target_date}")
            continue

        # Get the latest date in that month
        latest_date = month_data['date'].max()
        date_data = df_merged[df_merged['date'] == latest_date]

        # Find best proprietary and open models
        prop_data = date_data[date_data['model_type'] == 'Proprietary'].nlargest(1, 'rating')
        open_data = date_data[date_data['model_type'] == 'Open'].nlargest(1, 'rating')

        print(f"\n{target_date} (snapshot: {latest_date.date()}):")
        print("-" * 60)

        if not prop_data.empty:
            prop_model = prop_data.iloc[0]['model']
            prop_rating = prop_data.iloc[0]['rating']
            print(f"Best Proprietary: {prop_model:40s} {prop_rating:.2f}")

        if not open_data.empty:
            open_model = open_data.iloc[0]['model']
            open_rating = open_data.iloc[0]['rating']
            print(f"Best Open:        {open_model:40s} {open_rating:.2f}")

        if not prop_data.empty and not open_data.empty:
            gap = prop_rating - open_rating
            print(f"Gap:              {gap:40.2f} ELO points")

            # Calculate win probability
            prob_prop_wins = 1 / (1 + 10**((open_rating - prop_rating) / 400))
            print(f"Win Probability:  Proprietary has {prob_prop_wins*100:.1f}% chance to beat Open")


if __name__ == "__main__":
    # Create figures directory
    figures_dir = Path(__file__).parent / "figures"
    figures_dir.mkdir(exist_ok=True)

    # Load all ELO data
    print("Loading ELO data...")
    df_elo = load_all_elo()

    if df_elo.empty:
        print("No ELO data found!")
        exit(1)

    # Load metadata
    print("Loading metadata...")
    df_metadata = load_df()

    # Print statistics
    print_statistics(df_elo)
    print_metadata_statistics(df_elo, df_metadata)

    # 1. Rating over time (line chart)
    print("\n[1/8] Generating ELO ratings over time plot...")
    plot_top_models_elo(df_elo, top_n=15, save_path=str(figures_dir / "elo_ratings_over_time.png"))

    # 2. Ranking bump chart (shows position changes)
    print("\n[2/8] Generating ranking bump chart...")
    plot_ranking_bump_chart(df_elo, top_n=15, save_path=str(figures_dir / "ranking_bump_chart.png"))

    # 3. Rating heatmap
    print("\n[3/8] Generating rating heatmap...")
    plot_rating_heatmap(df_elo, top_n=30, save_path=str(figures_dir / "rating_heatmap.png"))

    # 4. Rating distribution over time
    print("\n[4/8] Generating rating distribution over time...")
    plot_rating_distribution_over_time(df_elo, save_path=str(figures_dir / "rating_distribution.png"))

    # 5. Top performers timeline
    print("\n[5/8] Generating top performers timeline...")
    plot_top_performers_timeline(df_elo, top_n=10, save_path=str(figures_dir / "top_performers_timeline.png"))

    # 6. Rating volatility analysis
    print("\n[6/8] Generating rating volatility analysis...")
    plot_rating_volatility(df_elo, min_appearances=10, top_n=20, save_path=str(figures_dir / "rating_volatility.png"))

    # 7. Organization performance over time
    print("\n[7/9] Generating organization performance plot...")
    plot_organization_performance(df_elo, df_metadata, top_n=20, save_path=str(figures_dir / "organization_performance.png"))

    # 8. License performance over time
    print("\n[8/9] Generating license performance plot...")
    plot_license_performance(df_elo, df_metadata, save_path=str(figures_dir / "license_performance.png"))

    # 9. Open vs Proprietary comparison
    print("\n[9/9] Generating open vs proprietary comparison plot...")
    plot_open_vs_proprietary(df_elo, df_metadata, save_path=str(figures_dir / "open_vs_proprietary.png"))

    print("\n" + "="*60)
    print("All visualizations generated successfully!")
    print(f"Figures saved to: {figures_dir}")
    print("="*60)

    # Analyze gap changes at key dates
    analyze_gap_changes(df_elo, df_metadata, ['2024-04', '2025-02', '2025-04'])
