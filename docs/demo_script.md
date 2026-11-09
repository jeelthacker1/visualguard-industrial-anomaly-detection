# Reproducible demonstration walkthrough

Use the actual trained E01 bottle artifact. Do not replace predictions with scripted numbers.
This is a recording storyboard, not a claim that a narrated video has been recorded.

1. **0–15 seconds:** show the README question and clarify normal-only anomaly detection.
2. **15–35 seconds:** open the bottle split and calibration metadata. Point out that official
   test rows do not populate the bank or choose the threshold.
3. **35–55 seconds:** explain global embeddings versus intermediate patches using the diagram.
4. **55–85 seconds:** run a local normal test image and an anomalous test image in Streamlit.
   Show the raw score, fixed threshold, model ID, latency and localization map. A heatmap is
   a deviation signal, not a causal explanation.
5. **85–105 seconds:** use `reports/E01/predictions.csv` to show a missed defect or false alarm
   when present. Do not cherry-pick only high-confidence successes.
6. **105–125 seconds:** show `reports/RESULTS.md` with all categories, runtime and memory.
7. **125–145 seconds:** demonstrate invalid input rejection, mention model/category limits,
   and state the original contributions and external components.

Dataset-image recordings should remain local unless the intended publication complies with
MVTec's attribution/non-commercial/share-alike terms. Public repository evidence uses numerical
tables and owned diagrams. A screen capture of the empty dashboard or synthetic fixtures is
safe to describe only as software UI evidence, never industrial accuracy evidence.
