# Model and artifact storage
Keep code, configs, docs and small reviewed result summaries in Git. Store large datasets, model weights, raw prediction dumps and complete run folders in team-controlled Drive/lab storage with a second copy. Do not commit credentials or signed download URLs.

Current model/run: https://drive.google.com/drive/folders/1c6NBxvgDwOQYrOMTDYjklne3NVDWJyOj
Owner must grant teammates access; the link alone does not guarantee permissions.

Download the COMPLETE full_v1 folder to runs/full_v1 locally and a second safe location. Keep best.pth (selected epoch 17), last.pth (optimizer/EMA/RNG/offset state), flame.onnx, config.json, history.json, export.json, model_contract.json, operating_metrics.json and training.log. Keep validation_predictions.json and examples for failure analysis. Verify flame.onnx SHA256 against results/full_v1_summary.json, then inspect checkpoints in the pinned environment. A Drive listing is not a verified local backup. The local backup remains pending as of 28 October.

ONNX alone supports inference but cannot resume training. For continuation transfer the complete run folder, unchanged manifests/code/recipe and use --resume; never run two writers. A changed experiment must use a new output/config.

A tagged GitHub Release with explicit model checksums and provenance can be considered later for a team-approved frozen model; Git LFS is another optional storage policy. Neither is needed now. Baseline weights are NOT uploaded to ordinary Git. Do not publish data or training assets through a release without reviewing their licences/access rules.

Only after the GitHub push and independent model/checkpoint backup are verified may this project's temporary cloud copies be removed. Keep the lab/raw dataset and chosen models. Do not delete unrelated Drive files or empty account-wide Trash.
