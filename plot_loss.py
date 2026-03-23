import os
import re
import sys
import matplotlib.pyplot as plt

def parse_log(log_path):
    train_re = re.compile(
        r'epoch[:=]\s*(\d+)\s+loss[:=]\s*([+-]?\d+(?:\.\d*)?(?:[eE][+-]?\d+)?)'
    )
    val_re = re.compile(
        r'(?:val[_ ]?loss|validation[_ ]?loss)[:=]\s*'
        r'([+-]?\d+(?:\.\d*)?(?:[eE][+-]?\d+)?)'
    )
    
    train_epochs = []
    train_losses = []
    val_epochs   = []
    val_losses   = []
    
    # diagnostics: print any lines containing 'val'
    print("··· sample lines containing 'val' from log:")
    with open(log_path) as fin:
        for i, l in enumerate(fin):
            if 'val' in l.lower() and i < 20:
                print(f"  {i:03d}: {l.strip()}")
        fin.seek(0)
    
    with open(log_path, 'r') as f:
        for line in f:
            m = train_re.search(line)
            if m:
                ep = int(m.group(1))
                loss = float(m.group(2))
                train_epochs.append(ep)
                train_losses.append(loss)
            m2 = val_re.search(line)
            if m2:
                # assume val_epoch == last train_epoch
                if train_epochs:
                    val_epochs.append(train_epochs[-1])
                    val_losses.append(float(m2.group(1)))
    
    if not val_epochs:
        print("⚠️  No validation-loss matches found. Adjust your log tags or regex.")
    return train_epochs, train_losses, val_epochs, val_losses

def plot_losses(train_epochs, train_losses, val_epochs, val_losses, out_png):
    plt.figure(figsize=(6,4))
    # lines only, no marker symbols
    plt.plot(train_epochs, train_losses, '-', label='Train Loss')
    if val_epochs:
        plt.plot(val_epochs, val_losses, '--', label='Val Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.yscale('log')        # log‐scale on loss axis
    plt.title("Training vs. Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    print(f"Saved plot to {out_png}")
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python plot_loss.py <logfile>")
        sys.exit(1)

    log_file = sys.argv[1]
    train_epochs, train_losses, val_epochs, val_losses = parse_log(log_file)

    # derive output path next to the log file
    folder = os.path.dirname(log_file) or "."
    name = os.path.splitext(os.path.basename(log_file))[0]
    out_png = os.path.join(folder, f"{name}_loss_curve.png")

    plot_losses(train_epochs, train_losses, val_epochs, val_losses, out_png=out_png)