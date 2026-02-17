import matplotlib.pyplot as plt
import numpy as np

# =========================
# AND: linearly separable
# =========================
and_points = [(0,0,0),(0,1,0),(1,0,0),(1,1,1)]
colors = {0: "tab:blue", 1: "tab:orange"}

plt.figure()
for x, y, label in and_points:
    plt.scatter(x, y, c=colors[label])
    plt.text(x+0.03, y+0.03, f"({x},{y}) -> {label}", fontsize=9)

# One valid separating hyperplane: x1 + x2 = 1.5
x_vals = np.linspace(-0.1, 1.1, 200)
y_vals = 1.5 - x_vals
plt.plot(x_vals, y_vals, "k--")

plt.xlim(-0.2, 1.2)
plt.ylim(-0.2, 1.2)
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("AND: linearly separable (x1 + x2 = 1.5)")
plt.savefig("and_separable.png", bbox_inches="tight")
plt.close()

# =========================
# XOR: not linearly separable
# =========================
xor_points = [(0,0,0),(0,1,1),(1,0,1),(1,1,0)]

plt.figure()
for x, y, label in xor_points:
    plt.scatter(x, y, c=colors[label])
    plt.text(x+0.03, y+0.03, f"({x},{y}) -> {label}", fontsize=9)

# Example line: x1 = x2 (not a true separator, just for visualization)
x_vals = np.linspace(-0.1, 1.1, 200)
plt.plot(x_vals, x_vals, "k--")

plt.xlim(-0.2, 1.2)
plt.ylim(-0.2, 1.2)
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("XOR: not linearly separable (line x1 = x2 shown)")
plt.savefig("xor_not_separable.png", bbox_inches="tight")
plt.close()
