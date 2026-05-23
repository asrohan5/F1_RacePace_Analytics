# train_pipeline.py
import sys
from src.logger import logging as log
from src.exception import CustomException


def run_train_pipeline():
    try:
        log.info("=" * 60)
        log.info("TRAIN PIPELINE — START")
        log.info("=" * 60)

        # ── Step 1: Data Ingestion ──
        log.info("\n[1/5] Data Ingestion")
        from src.components.data_ingestion import run_ingestion
        laps_df, tel_df = run_ingestion()
        log.info(f"  Laps: {laps_df.shape} | Telemetry: {tel_df.shape}")

        # ── Step 2: EDA ──
        # EDA is exploratory — run it once manually via eda.py
        # It is not part of the automated train pipeline
        log.info("\n[2/5] EDA — skipped in pipeline (run eda.py manually)")

        # ── Step 3: Data Transformation ──
        log.info("\n[3/5] Data Transformation")
        from src.components.data_transformation import run_transformation
        train_df, val_df, test_df, feature_cols = run_transformation()
        log.info(f"  Train: {train_df.shape} | Val: {val_df.shape} | Test: {test_df.shape}")

        # ── Step 4: Model Training ──
        log.info("\n[4/5] Model Training")
        from src.components.model_trainer import run_model_trainer
        (best_reg, best_clf,
         reg_results, clf_results,
         best_reg_sc, best_clf_sc,
         reg_sc_results, clf_sc_results) = run_model_trainer()

        log.info("  Full dataset best models saved.")
        log.info("  Same-compound models saved.")

        # ── Step 5: Model Validation ──
        log.info("\n[5/5] Model Validation")
        from src.components.model_validation import run_validation
        cv_results, val_with_pred, curve_results = run_validation()

        log.info(f"  Regressor  CV MAE : "
                 f"{cv_results['reg_cv_mae'].mean():.4f}s "
                 f"± {cv_results['reg_cv_mae'].std():.4f}s")
        log.info(f"  Classifier CV F1  : "
                 f"{cv_results['clf_cv_f1'].mean():.4f} "
                 f"± {cv_results['clf_cv_f1'].std():.4f}")

        log.info("=" * 60)
        log.info("TRAIN PIPELINE — COMPLETE")
        log.info("Run model_diagnostics.py separately to generate all plots.")
        log.info("=" * 60)

    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    run_train_pipeline()