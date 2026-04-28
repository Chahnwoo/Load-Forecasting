#!/usr/bin/env python3
"""Appends Prophet and TFT sections to notebooks/model_comparison.ipynb."""
import json, os, sys

NB_PATH = os.path.join(os.path.dirname(__file__), "..", "notebooks", "model_comparison.ipynb")

with open(NB_PATH) as f:
    nb = json.load(f)

def md(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source}

def code(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source,
    }

new_cells = [

    # ── Prophet ──────────────────────────────────────────────────────────────
    md(
        "## 9. Prophet — seasonal decomposition baseline\n\n"
        "Facebook Prophet decomposes the series into **trend + seasonality + holidays + regressors**. "
        "We fit one Prophet model per region with weather regressors added.\n\n"
        "Install: `pip install prophet`"
    ),

    code(
        "# Prophet training – one model per region\n"
        "# requires: pip install prophet\n"
        "try:\n"
        "    from prophet import Prophet\n"
        "    _prophet_available = True\n"
        "except ImportError:\n"
        "    _prophet_available = False\n"
        "    print('prophet not installed. Run: pip install prophet')\n"
        "\n"
        "if _prophet_available:\n"
        "    PROPHET_REGRESSORS = [\n"
        "        'temperature_2m', 'apparent_temperature', 'relative_humidity_2m',\n"
        "        'precipitation', 'cloud_cover', 'wind_speed_10m', 'shortwave_radiation',\n"
        "        'cdd_65f', 'hdd_65f', 'is_weekend', 'US_federal_holidays',\n"
        "        'load_previous_week',\n"
        "    ]\n"
        "    AVAIL_REGRESSORS = [c for c in PROPHET_REGRESSORS if c in train_df.columns]\n"
        "    print(f'Available regressors: {AVAIL_REGRESSORS}')\n"
        "\n"
        "    prophet_models = {}\n"
        "    prophet_preds = {}\n"
        "\n"
        "    for region, grp_train in train_df.groupby('region'):\n"
        "        m = Prophet(\n"
        "            changepoint_prior_scale=0.05,\n"
        "            seasonality_prior_scale=10.0,\n"
        "            daily_seasonality=True,\n"
        "            weekly_seasonality=True,\n"
        "            yearly_seasonality=True,\n"
        "        )\n"
        "        for reg in AVAIL_REGRESSORS:\n"
        "            m.add_regressor(reg)\n"
        "\n"
        "        df_fit = grp_train[['time_utc', 'load_mw'] + AVAIL_REGRESSORS].copy()\n"
        "        df_fit = df_fit.rename(columns={'time_utc': 'ds', 'load_mw': 'y'})\n"
        "        df_fit['ds'] = pd.to_datetime(df_fit['ds']).dt.tz_localize(None)\n"
        "        m.fit(df_fit)\n"
        "        prophet_models[region] = m\n"
        "        print(f'  Fitted Prophet for {region}')\n"
        "\n"
        "    for region, grp_valid in valid_df.groupby('region'):\n"
        "        df_pred = grp_valid[['time_utc'] + AVAIL_REGRESSORS].copy()\n"
        "        df_pred = df_pred.rename(columns={'time_utc': 'ds'})\n"
        "        df_pred['ds'] = pd.to_datetime(df_pred['ds']).dt.tz_localize(None)\n"
        "        forecast = prophet_models[region].predict(df_pred)\n"
        "        prophet_preds[region] = forecast['yhat'].values\n"
        "\n"
        "    all_preds = np.concatenate([\n"
        "        prophet_preds[r] for r in valid_df['region'].unique() if r in prophet_preds\n"
        "    ])\n"
        "    all_actual = valid_df.sort_values(['region', 'time_utc'])['load_mw'].values\n"
        "    mask = all_actual > 0\n"
        "    prophet_mape = np.mean(np.abs(all_preds[mask] - all_actual[mask]) / all_actual[mask]) * 100\n"
        "    prophet_rmse = np.sqrt(np.mean((all_preds - all_actual) ** 2))\n"
        "    print(f'Prophet  MAPE: {prophet_mape:.2f}%   RMSE: {prophet_rmse:.1f} MW')\n"
    ),

    code(
        "# Plot Prophet forecast vs actual for first region\n"
        "if _prophet_available:\n"
        "    region0 = valid_df['region'].unique()[0]\n"
        "    grp = valid_df[valid_df['region'] == region0].sort_values('time_utc')\n"
        "    preds = prophet_preds[region0]\n"
        "\n"
        "    fig, ax = plt.subplots(figsize=(14, 4))\n"
        "    ax.plot(grp['time_utc'].values, grp['load_mw'].values, label='Actual', lw=1)\n"
        "    ax.plot(grp['time_utc'].values, preds, label='Prophet', lw=1, alpha=0.8)\n"
        "    ax.set_title(f'Prophet vs Actual — {region0}')\n"
        "    ax.set_ylabel('Load (MW)')\n"
        "    ax.legend()\n"
        "    plt.tight_layout()\n"
        "    plt.show()\n"
    ),

    # ── TFT ──────────────────────────────────────────────────────────────────
    md(
        "## 10. Temporal Fusion Transformer (TFT)\n\n"
        "Giacomazzi et al. (2023) apply TFT to building-level energy forecasting and report **2.43% MAPE**. "
        "TFT uses an LSTM encoder + multi-head attention decoder with variable-selection networks that "
        "learn which features matter most at each horizon.\n\n"
        "Install: `pip install pytorch-forecasting lightning`\n\n"
        "**Note:** TFT requires integer `time_idx` (no gaps). We create one from the sorted hourly index."
    ),

    code(
        "# TFT setup — check dependencies\n"
        "try:\n"
        "    import pytorch_forecasting  # noqa\n"
        "    import lightning  # noqa\n"
        "    _tft_available = True\n"
        "    print('pytorch-forecasting and lightning are available')\n"
        "except ImportError as e:\n"
        "    _tft_available = False\n"
        "    print(f'TFT dependencies missing: {e}')\n"
        "    print('Install with: pip install pytorch-forecasting lightning')\n"
    ),

    code(
        "# Build TFT dataset\n"
        "if _tft_available:\n"
        "    from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer\n"
        "    from pytorch_forecasting.data import GroupNormalizer\n"
        "    from pytorch_forecasting.metrics import MAPE as TFT_MAPE\n"
        "    import lightning as L\n"
        "\n"
        "    LOOKBACK = 24\n"
        "    HORIZON  = 1\n"
        "\n"
        "    TFT_TIME_VARYING_KNOWN = [\n"
        "        'temperature_2m', 'apparent_temperature', 'relative_humidity_2m',\n"
        "        'precipitation', 'cloud_cover', 'wind_speed_10m', 'shortwave_radiation',\n"
        "        'cdd_65f', 'hdd_65f',\n"
        "    ]\n"
        "    TFT_STATIC_CATS = ['region']\n"
        "    TFT_TIME_VARYING_UNKNOWN = ['load_mw']\n"
        "\n"
        "    def make_tft_df(df):\n"
        "        out = df.sort_values(['region', 'time_utc']).copy()\n"
        "        out['time_idx'] = out.groupby('region').cumcount()\n"
        "        out['region'] = out['region'].astype(str)\n"
        "        for c in TFT_TIME_VARYING_KNOWN:\n"
        "            if c not in out.columns:\n"
        "                out[c] = 0.0\n"
        "        out['load_mw'] = out['load_mw'].astype(float)\n"
        "        return out\n"
        "\n"
        "    tft_train_df = make_tft_df(train_df)\n"
        "    tft_valid_df = make_tft_df(valid_df)\n"
        "    # offset valid time_idx so it continues from train\n"
        "    max_train_idx = tft_train_df.groupby('region')['time_idx'].max()\n"
        "    tft_valid_df['time_idx'] = tft_valid_df.apply(\n"
        "        lambda r: r['time_idx'] + max_train_idx[r['region']] + 1, axis=1\n"
        "    )\n"
        "\n"
        "    all_tft = pd.concat([tft_train_df, tft_valid_df], ignore_index=True)\n"
        "    max_train_tidx = tft_train_df['time_idx'].max()\n"
        "\n"
        "    training_cutoff = max_train_tidx\n"
        "    avail_known = [c for c in TFT_TIME_VARYING_KNOWN if c in all_tft.columns]\n"
        "\n"
        "    tft_dataset = TimeSeriesDataSet(\n"
        "        all_tft[all_tft['time_idx'] <= training_cutoff],\n"
        "        time_idx='time_idx',\n"
        "        target='load_mw',\n"
        "        group_ids=['region'],\n"
        "        min_encoder_length=LOOKBACK // 2,\n"
        "        max_encoder_length=LOOKBACK,\n"
        "        min_prediction_length=1,\n"
        "        max_prediction_length=HORIZON,\n"
        "        static_categoricals=['region'],\n"
        "        time_varying_known_reals=avail_known,\n"
        "        time_varying_unknown_reals=['load_mw'],\n"
        "        target_normalizer=GroupNormalizer(groups=['region'], transformation='softplus'),\n"
        "        add_relative_time_idx=True,\n"
        "        add_target_scales=True,\n"
        "        add_encoder_length=True,\n"
        "    )\n"
        "\n"
        "    valid_dataset = TimeSeriesDataSet.from_dataset(\n"
        "        tft_dataset, all_tft, predict=True, stop_randomization=True\n"
        "    )\n"
        "\n"
        "    train_loader = tft_dataset.to_dataloader(train=True,  batch_size=128, num_workers=0)\n"
        "    valid_loader = valid_dataset.to_dataloader(train=False, batch_size=128, num_workers=0)\n"
        "    print(f'TFT train batches: {len(train_loader)}, valid batches: {len(valid_loader)}')\n"
    ),

    code(
        "# Train TFT\n"
        "if _tft_available:\n"
        "    tft_model = TemporalFusionTransformer.from_dataset(\n"
        "        tft_dataset,\n"
        "        learning_rate=1e-3,\n"
        "        hidden_size=64,\n"
        "        attention_head_size=4,\n"
        "        dropout=0.1,\n"
        "        hidden_continuous_size=32,\n"
        "        loss=TFT_MAPE(),\n"
        "        log_interval=50,\n"
        "        reduce_on_plateau_patience=3,\n"
        "    )\n"
        "    print(f'TFT parameters: {sum(p.numel() for p in tft_model.parameters()):,}')\n"
        "\n"
        "    trainer = L.Trainer(\n"
        "        max_epochs=10,\n"
        "        accelerator='auto',\n"
        "        enable_progress_bar=True,\n"
        "        gradient_clip_val=0.1,\n"
        "        logger=False,\n"
        "        enable_checkpointing=False,\n"
        "    )\n"
        "    trainer.fit(tft_model, train_dataloaders=train_loader, val_dataloaders=valid_loader)\n"
        "    print('TFT training complete')\n"
    ),

    code(
        "# Evaluate TFT\n"
        "if _tft_available:\n"
        "    raw_preds, idx = tft_model.predict(valid_loader, return_index=True, return_x=False)\n"
        "    tft_pred_vals = raw_preds.numpy().flatten()\n"
        "\n"
        "    # Align predictions with actual values\n"
        "    tft_actual = valid_df.sort_values(['region', 'time_utc'])['load_mw'].values\n"
        "    min_len = min(len(tft_pred_vals), len(tft_actual))\n"
        "    tft_pred_vals = tft_pred_vals[:min_len]\n"
        "    tft_actual = tft_actual[:min_len]\n"
        "\n"
        "    mask = tft_actual > 0\n"
        "    tft_mape = np.mean(np.abs(tft_pred_vals[mask] - tft_actual[mask]) / tft_actual[mask]) * 100\n"
        "    tft_rmse = np.sqrt(np.mean((tft_pred_vals - tft_actual) ** 2))\n"
        "    print(f'TFT   MAPE: {tft_mape:.2f}%   RMSE: {tft_rmse:.1f} MW')\n"
    ),

    # ── Final summary ─────────────────────────────────────────────────────────
    md(
        "## 11. Full model comparison summary\n\n"
        "| Model | Avg MAPE (2025) | Avg RMSE (MW) | Notes |\n"
        "|---|---|---|---|\n"
        "| GAM (global) | 7.9% | ~900 | Best interpretable; per-hour GAM would improve further |\n"
        "| Per-hour GAM | ~6–7% est. | — | Fan & Hyndman approach; more expensive |\n"
        "| XGBoost | ~12% | ~1 100 | Strong tree baseline |\n"
        "| LSTM | ~15% | ~1 200 | Needs more epochs / tuning |\n"
        "| Transformer | 37% | ~634 | Best RMSE but predicts near mean → bad MAPE |\n"
        "| Prophet | see above | — | Good for trend/seasonality decomposition |\n"
        "| TFT | target ~2–5% | — | Giacomazzi 2023: 2.43% on building load |\n\n"
        "**Performance target**: 3–5% MAPE is achievable with per-hour GAM or well-tuned TFT/LightGBM."
    ),
]

nb["cells"].extend(new_cells)

with open(NB_PATH, "w") as f:
    json.dump(nb, f, indent=1)

print(f"Done. Notebook now has {len(nb['cells'])} cells.")
