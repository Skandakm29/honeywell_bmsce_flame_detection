# Dataset and frozen splits
Official FASDD_CV V9 record: https://www.scidb.cn/en/file?id=85bad3972ed60e0a20a790032c0d85fb
Public download: https://china.scidb.cn/download?fileId=85bad3972ed60e0a20a790032c0d85fb
Wang et al., An open flame and smoke detection dataset for deep learning in remote sensing based fire detection: https://doi.org/10.1080/10095020.2024.2347922
Licence: CC BY-SA 4.0; preserve attribution and modification records and review source/redistribution conditions. No raw images or annotations are committed here.

Archive: FASDD_CV.zip, 12,326,081,298 bytes, MD5 da5e1410e13113fc9d605567aea54508. Keep an untouched laptop/lab/USB copy. Do not upload it to ordinary Git. Team Drive may hold a controlled working copy; arrange access through the owner, not public sharing by default.

For the exact baseline, get the existing honeywell_training.zip from the project owner (My Drive/HoneywellTraining). It contains frozen manifests and original initialization weights; its SHA256 is in configs/assets.json. Run scripts/restore_assets.py on it, then scripts/extract_local.py with the dataset archive. Only named assets are restored; obsolete bundled code/docs are not copied. The handoff is a frozen asset carrier, not the current code source.

To audit a new raw copy independently, scripts/audit_local.py --help documents options. It verifies archive/labels/decoding and screens cross-split duplicates. Regenerating manifests must be treated as a new data version unless all four hashes in configs/assets.json match. Never silently replace baseline manifests. Full audit summary is in results/data_audit_summary.json; detailed duplicate/quarantine records remain with external data assets. The full audit cache is not required for training.

Current manifests: data/fasdd/train.json, val.json, validation_full.json, test.json. Image paths are relative to repository root; keep extraction layout unchanged. Test priority in the data audit is not a test-performance evaluation. No held-out test predictions have been run.
