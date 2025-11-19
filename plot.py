import matplotlib.pyplot as plt

# Data
queries = [1, 5, 10, 20, 50]
times = [4.658, 17.227, 41.881, 102.294, 252.875]

# Plot with line + dots
plt.plot(queries, times, marker='o')
plt.xlabel("Number of Queries")
plt.ylabel("Time (seconds)")
plt.title("Normal RAG Query Time vs Number of Queries")
plt.grid(True)

plt.show()
