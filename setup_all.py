"""
CollectIQ — One-click setup.

Runs the complete pipeline build:
  1. Generate synthetic AR dataset (50 customers, 24 months, ~7000 invoices)
  2. Train customer segmenter (k-means, 5 clusters)
  3. Train late-payment XGBoost classifier
  4. Save eval metrics for the README

Usage:
    python setup_all.py
"""
import json
import sys
from pathlib import Path

# Allow `python setup_all.py` from project root
sys.path.insert(0, str(Path(__file__).resolve().parent))


def main():
    print("=" * 70)
    print("CollectIQ — One-Click Setup")
    print("=" * 70)

    # ---- 1. Synthetic data ----
    print("\n[1/3] Generating synthetic AR dataset...")
    from src.data_generation.generate_synthetic_ar import generate_all
    summary = generate_all(num_customers=50, months=24)

    # ---- 2. Segmentation ----
    print("\n[2/3] Training customer segmenter (k-means)...")
    from src.segmentation.customer_segmenter import main as seg_main
    seg_main()

    # ---- 3. Late-payment classifier ----
    print("\n[3/3] Training late-payment XGBoost classifier...")
    from src.prediction.late_payment_model import train_and_evaluate
    train_and_evaluate()

    # ---- Summary ----
    data_dir = Path(__file__).resolve().parent / "data"
    metrics = json.load(open(data_dir / "eval" / "classifier_metrics.json"))

    print("\n" + "=" * 70)
    print("✅ Setup complete!")
    print("=" * 70)
    print(f"\nDataset:")
    for k, v in summary.items():
        print(f"  {k:30}  {v}")
    print(f"\nLate-payment classifier:")
    for k, v in metrics.items():
        print(f"  {k:10}  {v:.3f}")
    print(f"\n📁 Models saved to: {data_dir / 'models'}")
    print(f"\n▶  Next step:")
    print(f"   1. Set GROQ_API_KEY in .env  (free at https://console.groq.com)")
    print(f"   2. Run:  python app_gradio.py")


if __name__ == "__main__":
    main()
