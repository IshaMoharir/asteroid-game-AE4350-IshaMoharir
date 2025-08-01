import subprocess
import time
import re

train_script = "python3 rl/train.py"
eval_script = "python3 run_rl.py"




total_runs = 5
eval_rewards = []

for i in range(total_runs):
    print(f"\n Run {i+1}/{total_runs} — Training...\n")
    start = time.time()
    subprocess.run(train_script, shell=True)
    duration = time.time() - start
    print(f" Training completed in {duration:.1f} seconds.")

    print(f" Evaluating model...")
    # Run evaluation and capture the printed reward output
    result = subprocess.run(eval_script, shell=True, capture_output=True, text=True)

    # Extract reward values
    rewards = re.findall(r"Reward = (-?\d+\.?\d*)", result.stdout)
    rewards = [float(r) for r in rewards]
    eval_rewards.append(rewards)

    avg_reward = sum(rewards) / len(rewards) if rewards else 0
    print(f"✅ Evaluation complete | Episode rewards: {rewards}")
    print(f"📊 Average reward for run {i+1}: {avg_reward:.2f}")

# Summary
print("\n=== Summary of Evaluation Rounds ===")
for i, rewards in enumerate(eval_rewards, 1):
    avg = sum(rewards) / len(rewards) if rewards else 0
    print(f"Run {i}: Avg Reward = {avg:.2f} | All: {rewards}")
