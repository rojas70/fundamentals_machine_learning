import random

# -----------------------------
# Simple Replay Buffer
# -----------------------------
class ReplayBuffer:
    def __init__(self, capacity=1000):
        self.buffer = []
        self.capacity = capacity

    def add(self, transition):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append(transition)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


# -----------------------------
# Toy "Model"
# -----------------------------
class ToyModel:
    def __init__(self):
        self.weight = 0.0  # single parameter

    def predict(self, x):
        return self.weight * x

    def update(self, grad, lr=0.1):
        self.weight -= lr * grad


# -----------------------------
# Fake Environment Step
# -----------------------------
def env_step():
    """
    Simulates one environment interaction.
    Returns (state, action, reward).
    """
    state = random.uniform(-1, 1)
    action = random.uniform(-1, 1)
    reward = -(state - action) ** 2  # arbitrary reward
    return state, action, reward


# -----------------------------
# Training Loop with UTD
# -----------------------------
def train(utd_ratio, env_steps=20, batch_size=4):
    print(f"\n=== Training with UTD ratio = {utd_ratio} ===")

    buffer = ReplayBuffer()
    model = ToyModel()

    total_updates = 0

    for step in range(env_steps):
        # 1. Collect one environment transition
        transition = env_step()
        buffer.add(transition)

        # 2. Perform UTD updates only if we have enough entries to sample a batch size
        if len(buffer) >= batch_size:
            for _ in range(utd_ratio):
                batch = buffer.sample(batch_size)

                # Fake loss: mean squared error
                grad = 0.0
                for state, action, reward in batch:
                    pred = model.predict(state)
                    target = reward
                    grad += 2 * (pred - target) * state

                # Average your error over the size of the batch
                grad /= batch_size

                # Do pack propagation
                model.update(grad)

                # Update your updates
                total_updates += 1

        print(
            f"Env step {step:2d} | "
            f"Buffer size: {len(buffer):3d} | "
            f"Model weight: {model.weight:.4f}"
        )

    print(f"Total env steps   : {env_steps}")
    print(f"Total updates     : {total_updates}")
    print(f"Effective UTD     : {total_updates / env_steps:.1f}")


# -----------------------------
# Run Experiments
# -----------------------------
if __name__ == "__main__":
    train(utd_ratio=1)
    train(utd_ratio=5)
    train(utd_ratio=10)
