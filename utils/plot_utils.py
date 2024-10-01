import pandas as pd
import matplotlib.pyplot as plt

def plot_dtw_score_distribution(csv_path, plot_title, similarity_threshold, output_path):
    """
    Function to plot score distribution from a CSV file and display the number of
    'very similar' audios based on a given threshold.

    Args:
    - csv_path (str): Path to the CSV file containing similarity scores.
    - plot_title (str): Title for the plot.
    - similarity_threshold (float): Threshold for defining 'very similar' scores.
    - output_path (str): Path to save the generated plot.
    """

    # Load the CSV file containing the similarity scores
    df = pd.read_csv(csv_path)

    # Extract the first two words from the plot title for x-axis label
    xlabel = " ".join(plot_title.split()[:2])

    # Plot the distribution of similarity scores
    plt.figure(figsize=(9, 6))
    plt.hist(df['similarity_score'], bins=50, alpha=0.75)
    plt.title(plot_title)
    plt.xlabel(xlabel)
    plt.ylabel('Frequency')

    # Count the number of audios that are very similar
    very_similar_count = (df['similarity_score'] < similarity_threshold).sum()

    # Display the count of very similar audios and threshold on the plot
    plt.text(0.95, 0.95, f'Very Similar Samples: {very_similar_count}\nThreshold: {similarity_threshold}',
             fontsize=9, color='black', transform=plt.gca().transAxes,
             verticalalignment='top', horizontalalignment='right')

    # Save the plot
    plt.savefig(output_path, bbox_inches='tight', format='svg')

    # Show the plot
    plt.show()


def plot_wcc_score_distribution(csv_path, plot_title, similarity_threshold, output_path):
    """
    Function to plot the correlation score distribution from a CSV file and display
    the count of 'very similar' and 'negative correlation' audios based on given thresholds.

    Args:
    - csv_path (str): Path to the CSV file containing similarity scores.
    - plot_title (str): Title for the plot.
    - similarity_threshold (float): Threshold for defining 'very similar' scores.
    - output_path (str): Path to save the generated plot.
    """

    # Load the CSV file containing the similarity scores
    df = pd.read_csv(csv_path)

    # Extract the first two words from the plot title for x-axis label
    xlabel = " ".join(plot_title.split()[:2])

    # Plot the distribution of similarity scores
    plt.figure(figsize=(9, 6))
    plt.hist(df['similarity_score'], bins=50, alpha=0.75)
    plt.title(plot_title)
    plt.xlabel(xlabel)
    plt.ylabel('Frequency')

    # Count the number of audios that are very similar
    very_similar_count = (df['similarity_score'] < similarity_threshold).sum()

    # Count the number of audios with scores beyond 1 (indicating potential negative correlation)
    negative_correlation_count = (df['similarity_score'] > 1).sum()

    # Display the counts of very similar audios and negative correlation on the plot
    plt.text(0.95, 0.95, f'Very Similar Samples: {very_similar_count}\nNegative Correlation Samples: {negative_correlation_count}\nThreshold: {similarity_threshold}',
             fontsize=9, color='black', transform=plt.gca().transAxes,
             verticalalignment='top', horizontalalignment='right')

    # Save the plot
    plt.savefig(output_path, bbox_inches='tight', format='svg')

    # Show the plot
    plt.show()

    # Print the counts for logging purposes
    print(f'Number of audios that are very similar (score < {similarity_threshold}): {very_similar_count}')
    print(f'Number of audios with negative correlation (score > 1): {negative_correlation_count}')

def plot_embed_score_distribution(csv_path, plot_title, similarity_threshold, output_path):
    """
    Function to plot the similarity score distribution from a CSV file and display
    the count of 'very similar' audios based on a given threshold.

    Args:
    - csv_path (str): Path to the CSV file containing similarity scores.
    - plot_title (str): Title for the plot.
    - similarity_threshold (float): Threshold for defining 'very similar' scores.
    - output_path (str): Path to save the generated plot.
    """

    # Load the CSV file containing the similarity scores
    df = pd.read_csv(csv_path)

    # Extract the first two words from the plot title for x-axis label
    xlabel = " ".join(plot_title.split()[:2])

    # Plot the distribution of similarity scores
    plt.figure(figsize=(9, 6))
    plt.hist(df['similarity_score'], bins=50, alpha=0.75)
    plt.title(plot_title)
    plt.xlabel(xlabel)
    plt.ylabel('Frequency')
    plt.xlim((0,0.0050))

    # Count the number of audios that are very similar
    very_similar_count = (df['similarity_score'] < similarity_threshold).sum()

    # Display the count of very similar audios on the plot
    plt.text(0.95, 0.95, f'Very Similar Samples: {very_similar_count}\nThreshold: {similarity_threshold}',
             fontsize=9, color='black', transform=plt.gca().transAxes,
             verticalalignment='top', horizontalalignment='right')

    # Save the plot
    plt.savefig(output_path, bbox_inches='tight', format='svg')

    # Show the plot
    plt.show()

    # Print the count for logging purposes
    print(f'Number of audios that are very similar (similarity score < {similarity_threshold}): {very_similar_count}')