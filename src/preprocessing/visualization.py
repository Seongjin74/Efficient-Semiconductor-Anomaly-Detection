# src/preprocessing/visualization.py

import matplotlib.pyplot as plt

def plot_target_distribution(df, target_col="Pass_Fail"):
    plt.figure(figsize=(6,6))
    labels = ['Pass', 'Fail']
    counts = df[target_col].value_counts()
    total = counts.sum()
    percentages = [f'{(count / total) * 100:.2f}%' for count in counts]
    colors = ['blue', 'red']
    plt.bar(labels, counts, color=colors)
    plt.xlabel("Outcome")
    plt.ylabel("Count")
    plt.title(f"Pass/Fail Distribution (Pass: {percentages[0]}, Fail: {percentages[1]})")
    plt.show()
