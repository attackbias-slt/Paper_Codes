import os
import json
import matplotlib.pyplot as plt

JSON_DIR = "insert directory containing json files"     # directory containing JSON files
OUT_DIR  = "output directory to save plots"    # where to save the output plots
os.makedirs(OUT_DIR, exist_ok=True)

# ------------------------------
# load JSON
# ------------------------------
def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

# ------------------------------
# Function to plot loss + entropy
# ------------------------------
def plot_attack_metrics(json_name, loss_dict, entropy_dict, save_path):
    attacks = list(loss_dict.keys())
    loss_vals = [loss_dict[a] for a in attacks]
    entropy_vals = [entropy_dict[a] for a in attacks]

    fig, axes = plt.subplots(1, 2, figsize=(22, 5))
    fig.suptitle(f"Attack-wise Diagnostics ({json_name})", fontsize=22, weight="bold")

    # LOSS
    axes[0].bar(attacks, loss_vals, color="#c46e52", alpha=0.85)
    axes[0].set_title("Loss", fontsize=16)
    axes[0].set_ylabel("Loss")
    axes[0].tick_params(axis='x', rotation=45)

    # ENTROPY
    axes[1].bar(attacks, entropy_vals, color="#356fa6", alpha=0.85)
    axes[1].set_title("Entropy", fontsize=16)
    axes[1].set_ylabel("Entropy")
    axes[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()

# locate JSON FILES in the directory

    files = [f for f in os.listdir(JSON_DIR) if f.endswith(".json")]
    print(f"Found {len(files)} JSON files.")

    for file in files:
      json_path = os.path.join(JSON_DIR, file)
      data = load_json(json_path)

       # Direct access — NO epochs
      loss_dict = data["loss"]
      entropy_dict = data["entropy"]

      save_path = os.path.join(OUT_DIR, file.replace(".json", "_plot.png"))
      plot_attack_metrics(file.replace(".json", ""), loss_dict, entropy_dict, save_path)

      print(f"✓ Saved {save_path}")
