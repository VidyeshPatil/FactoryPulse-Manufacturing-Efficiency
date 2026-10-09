# FactoryPulse

AI-based classification of manufacturing efficiency (High / Medium / Low) from sensor, production and 6G network data,
with a live Streamlit monitor, explainability, and the ability to add new readings.

## Run it
```bash
pip install -r requirements.txt
streamlit run app.py
```
Trained models ship in `models/`. If they are missing or were built with another scikit-learn version, the app retrains them on first launch (about a minute).
To retrain manually: `python train_model.py`. To redraw the artwork: `python make_assets.py`.

## Tabs
| Tab | What it does |
|---|---|
| Live Monitor (first) | Traffic-light status of the latest reading, Pause/Slow/Normal/Fast controls, live session counter, alert level, incoming readings list and rolling class mix. A "Deep dive" section below adds KPIs, trends, the 50-machine status board and alerts. |
| Efficiency Dashboard | Sliders for every metric, prediction with confidence, reasons, sensitivity charts, save as a new entry. |
| Machine Insights | Fleet ranking, per-machine trend, weekly class mix, hour/weekday heat map, movers, profiles. |
| Explainability | Global drivers, plain-language rules, one-reading explanation, "why did efficiency change" attribution. |
| Operational Monitoring | Efficiency by operation mode and network quality; network vs sensor impact. |
| Live Simulator | Add a reading, upload a CSV, inject demo faults, manage saved entries (`data/new_entries.csv`). |
| Data Explorer | Browse readings, Model lab (comparison, confusion matrix, EDA, retrain), executive summary download. |

Filters (machines, mode, network quality, time window, hours) live in the collapsed sidebar: click the arrow at the top left.
Hover over the small "i" icons for plain-English explanations.

## Important finding
In `Thales_Group_Manufacturing.csv` the label is set by two columns: **Low** if `Error_Rate_% > 5` or `Production_Speed_units_per_hr <= 200`;
**High** if speed `> 400` and error `<= 2`; otherwise **Medium** (99.998 % agreement). Sensor and 6G network metrics do not predict the label
(sensor + network-only model: ROC-AUC 0.50). The app shows this openly in the Model Lab, Explain and Operations tabs.

## Layout
```
app.py            entry point (sidebar filters, caching, tabs)
tabs/             one module per tab
src/              config, features, data, modeling, charts, ui, narrative, report
models/           trained models + metrics.json
assets/           pipeline and logo artwork
data/             source CSV; new_entries.csv is created when you add readings
```
Requires Streamlit 1.37 or newer (uses `st.fragment(run_every=...)` for the live updates).

Offline check (no browser): `python tests/smoke_test.py` runs every tab against stand-in Streamlit/Plotly modules.
