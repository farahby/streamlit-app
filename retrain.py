import os, json
import pandas as pd
from supabase import create_client
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error

SUPABASE_URL  = os.environ["SUPABASE_URL"]
SUPABASE_KEY  = os.environ["SUPABASE_KEY"]
SCORED_CSV    = "normalized_alerts/scored_findings.csv"
MAE_PATH      = "mae_history.json"
MIN_LABELS    = 10
MAE_THRESHOLD = 0.02

def load_feedback():
    sb  = create_client(SUPABASE_URL, SUPABASE_KEY)
    res = sb.table("analyst_feedback").select("*").execute()
    return pd.DataFrame(res.data)

def retrain(fb, scored):
    merged = scored.merge(fb[["id", "analyst_score"]], on="id", how="inner")
    if len(merged) < MIN_LABELS:
        print(f"Only {len(merged)} labels — minimum is {MIN_LABELS}, skipping")
        return False

    features = [c for c in [
        "epss_score", "asset_criticality", "in_kev",
        "has_exploit", "internet_facing", "tool_count", "confidence_score"
    ] if c in merged.columns]

    X = merged[features].fillna(0)
    y = merged["analyst_score"]

    xgb = XGBRegressor(n_estimators=100, max_depth=4, random_state=42)
    lgb = LGBMRegressor(n_estimators=100, max_depth=4, random_state=42)
    xgb.fit(X, y)
    lgb.fit(X, y)

    preds   = (xgb.predict(X) + lgb.predict(X)) / 2
    mae_new = mean_absolute_error(y, preds)
    mae_prev = json.load(open(MAE_PATH)).get("last_mae", 999) if os.path.exists(MAE_PATH) else 999

    print(f"MAE previous: {mae_prev:.4f} | MAE new: {mae_new:.4f}")

    if mae_new >= mae_prev - MAE_THRESHOLD:
        print("No improvement — rollback guard triggered, not deploying")
        return False

    scored.loc[scored["id"].isin(merged["id"]), "risk_score"] = preds
    scored.to_csv(SCORED_CSV, index=False)
    json.dump({"last_mae": mae_new, "label_count": len(merged)}, open(MAE_PATH, "w"))
    print(f"Deployed — {len(merged)} labels used, MAE {mae_new:.4f}")
    return True

if __name__ == "__main__":
    fb     = load_feedback()
    scored = pd.read_csv(SCORED_CSV)
    retrain(fb, scored)
