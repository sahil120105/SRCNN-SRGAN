import matplotlib.pyplot as plt
import os

def main():
    methods = ['Bicubic', 'SRCNN', 'Real-ESRGAN']
    psnr = [28.76, 26.21, 25.62]
    ssim = [0.8073, 0.7948, 0.7489]
    latency = [2.1, 19.1, 99.3]

    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")
    os.makedirs(output_dir, exist_ok=True)

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    # 1. PSNR Plot
    plt.figure(figsize=(6, 4))
    bars = plt.bar(methods, psnr, color=colors)
    plt.title('Mean PSNR (higher is better)')
    plt.ylabel('PSNR (dB)')
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.2, f'{yval:.2f}', ha='center', va='bottom', fontweight='bold')
    plt.ylim(0, max(psnr) + 5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'psnr_graph.png'), dpi=150)
    plt.close()

    # 2. SSIM Plot
    plt.figure(figsize=(6, 4))
    bars = plt.bar(methods, ssim, color=colors)
    plt.title('Mean SSIM (higher is better)')
    plt.ylabel('SSIM')
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f'{yval:.4f}', ha='center', va='bottom', fontweight='bold')
    plt.ylim(0, 1.0)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ssim_graph.png'), dpi=150)
    plt.close()

    # 3. Latency Plot
    plt.figure(figsize=(6, 4))
    bars = plt.bar(methods, latency, color=colors)
    plt.title('Mean Latency (lower is better)')
    plt.ylabel('Latency (ms)')
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 2, f'{yval:.1f}ms', ha='center', va='bottom', fontweight='bold')
    plt.ylim(0, max(latency) + 20)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'latency_graph.png'), dpi=150)
    plt.close()

    print("Graphs saved to:", output_dir)

if __name__ == "__main__":
    main()
