"""
This is the template file for the statistics and trends assignment.
You will be expected to complete all the sections and
make this a fully working, documented file.
You should NOT change any function, file or variable names,
if they are given to you here.
Make use of the functions presented in the lectures
and ensure your code is PEP-8 compliant, including docstrings.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as ss


def plot_relational_plot(df):
    """
    Creates a scatter plot of total participation vs. total medals.
    
    Parameters:
        df (DataFrame): The dataset.
    
    Saves:
        relational_plot.png
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x=df["total_participation"], y=df["total_total"], alpha=0.7)
    ax.set_xlabel("Total Participation")
    ax.set_ylabel("Total Medals Won")
    ax.set_title("Total Participation vs. Total Medals")
    plt.grid(True)
    plt.savefig("relational_plot.png")
    plt.close()


def plot_categorical_plot(df):
    """
    Creates a bar chart of the top 10 countries by total gold medals.
    
    Parameters:
        df (DataFrame): The dataset.
    
    Saves:
        categorical_plot.png
    """
    top_countries = df.nlargest(10, "total_gold")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x=top_countries["total_gold"], y=top_countries["countries"], palette="viridis", ax=ax)
    ax.set_xlabel("Total Gold Medals")
    ax.set_ylabel("Countries")
    ax.set_title("Top 10 Countries by Total Gold Medals")
    plt.grid(axis="x", linestyle="--", alpha=0.7)
    plt.savefig("categorical_plot.png")
    plt.close()


def plot_statistical_plot(df):
    """
    Creates a correlation heatmap for numerical variables.
    
    Parameters:
        df (DataFrame): The dataset.
    
    Saves:
        statistical_plot.png
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax)
    ax.set_title("Correlation Heatmap of Numerical Features")
    plt.savefig("statistical_plot.png")
    plt.close()


def statistical_analysis(df, col: str):
    """
    Computes mean, standard deviation, skewness, and excess kurtosis for a given column.

    Parameters:
        df (DataFrame): The dataset.
        col (str): Column name for statistical analysis.

    Returns:
        tuple: (Mean, Standard Deviation, Skewness, Excess Kurtosis)
    """
    mean = df[col].mean()
    stddev = df[col].std()
    skewness = ss.skew(df[col])
    excess_kurtosis = ss.kurtosis(df[col])
    return mean, stddev, skewness, excess_kurtosis


def preprocessing(df):
    """
    Preprocesses the dataset:
    - Cleans column names by removing spaces.
    - Converts numerical-looking object columns to numeric.
    - Fills missing values with column medians.
    - Displays basic statistics.

    Parameters:
        df (DataFrame): The raw dataset.

    Returns:
        DataFrame: The cleaned dataset.
    """
    df.columns = df.columns.str.strip()

    numeric_columns = ['summer_gold', 'summer_total', 'total_gold', 'total_total']
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')

    df.fillna(df.median(numeric_only=True), inplace=True)

    print("Data Preprocessing Completed. Summary:")
    print(df.describe())

    return df


def writing(moments, col):
    """
    Prints the statistical analysis results for a selected column.

    Parameters:
        moments (tuple): Statistical moments (mean, std, skew, kurtosis).
        col (str): The column analyzed.
    """
    print(f'For the attribute {col}:')
    print(f'Mean = {moments[0]:.2f}, '
          f'Standard Deviation = {moments[1]:.2f}, '
          f'Skewness = {moments[2]:.2f}, and '
          f'Excess Kurtosis = {moments[3]:.2f}.')
    
    # Interpretation based on skewness and kurtosis values
    skewness_interpretation = "not skewed"
    kurtosis_interpretation = "mesokurtic"

    if moments[2] > 2:
        skewness_interpretation = "right-skewed"
    elif moments[2] < -2:
        skewness_interpretation = "left-skewed"

    if moments[3] > 0:
        kurtosis_interpretation = "leptokurtic (heavy-tailed)"
    elif moments[3] < 0:
        kurtosis_interpretation = "platykurtic (light-tailed)"

    print(f"The data was {skewness_interpretation} and {kurtosis_interpretation}.")


def main():
    """
    Main function to execute the full workflow:
    - Load the dataset.
    - Preprocess the data.
    - Generate plots.
    - Perform statistical analysis.
    """
    df = pd.read_csv("data.csv")
    df = preprocessing(df)

    col = "total_gold"  # Chosen column for statistical analysis

    plot_relational_plot(df)
    plot_statistical_plot(df)
    plot_categorical_plot(df)

    moments = statistical_analysis(df, col)
    writing(moments, col)


if __name__ == '__main__':
    main()
