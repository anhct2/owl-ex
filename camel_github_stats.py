import matplotlib.pyplot as plt

# Replace these with the actual numbers
metrics = {
    "Stars": 1000,
    "Forks": 200,
    "Watchers": 150,
    "Open Issues": 10,
}

names = list(metrics.keys())
values = list(metrics.values())

plt.figure(figsize=(10, 5))
plt.bar(names, values)
plt.title("Camel AI - GitHub Repository Metrics")
plt.xlabel("Metric")
plt.ylabel("Count")
plt.savefig("camel_github_stats.png")  # Save the plot as an image
plt.show()