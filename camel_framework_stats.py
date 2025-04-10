import matplotlib.pyplot as plt

# Placeholder values
stars = 100
forks = 50
contributors = 10
open_issues = 5

metrics = ['Stars', 'Forks', 'Contributors', 'Open Issues']
values = [stars, forks, contributors, open_issues]

plt.figure(figsize=(10, 6))  # Adjust figure size for better readability
plt.bar(metrics, values, color=['blue', 'green', 'red', 'purple'])
plt.xlabel('Metrics')
plt.ylabel('Count')
plt.title('Camel Framework GitHub Statistics')
plt.savefig('camel_framework_stats.png')
plt.show()
