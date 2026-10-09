"""Train, compare and save the efficiency models.   Usage:  python train_model.py"""
import time

from src.data import load_base, training_frame
from src.modeling import train_all


def main():
    t0 = time.time()
    df = training_frame(load_base())
    out = train_all(df, progress=lambda m, f: print(f"[{f:4.0%}] {m}"))
    m = out["metrics"]
    print(f"\nPrimary model : {m['primary']}")
    for r in m["models"]:
        print(f"  {r['model']:<22} acc={r['accuracy']:.4f}  macroF1={r['macro_f1']:.4f}  cv={r['cv_mean']:.4f}±{r['cv_std']:.4f}")
    print(f"Early-warning (sensor+network only): acc={m['early']['accuracy']:.4f} vs majority baseline {m['early']['majority_baseline']:.4f}")
    print(f"Rule agreement: {m['rule_agreement']:.5f}\nDone in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
