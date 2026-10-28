# Honeywell BMSCE | Vision-Based Flame Detection and Monitoring

Collaborative development repository for the Honeywell project focused on vision-based flame detection and monitoring using computer vision, AI models, and edge deployment technologies.

**Repository:** [honeywell_bmsce_flame_detection](https://github.com/Skandakm29/honeywell_bmsce_flame_detection)

---

## Project Overview

The project focuses on developing a vision-based flame detection pipeline, exploring AI-based detection approaches, video-stream processing, model integration, and deployment on supported edge hardware.

The repository will serve as the central workspace for implementation, experimentation, testing, and technical documentation.

### Technologies Under Exploration

- **Computer Vision:** OpenCV
- **Model Development and Interoperability:** ONNX
- **Edge AI Development:** TI Edge AI Studio
- **Video Processing:** GStreamer
- **Version Control:** Git and GitHub

The final technology stack and implementation details will follow the project's technical requirements and agreed architecture.

---

## Current implementation (28 October 2026)
The full_v1 baseline completed 20 epochs on Colab T4. Selected epoch 17: AP50 62.69%, AP50:95 32.92% on a 2,000-image validation subset. This is a baseline, not production acceptance. Held-out testing and TI/video integration remain pending.

- [Start here: teammate / AI-agent handoff](docs/PROJECT_STATUS.md)
- [Setup and execution](docs/SETUP.md)
- [Dataset and frozen splits](docs/DATASET.md)
- [Model storage / backup](docs/ARTIFACTS.md)
- [Algorithm rationale and papers](docs/ALGORITHM_DECISION_RECORD.md)
- [Pipeline plan](docs/IMPLEMENTATION_PLAN.md)
- [Full epoch metrics](results/full_v1_history.json)

Repository folders: scripts/ training/audit/extraction; tests/ focused checks; configs/ reproducible recipe and asset hashes; third_party/ pinned TI source/licence; docs/ current documentation; results/ small evidence summaries; models/ documentation only. src/ is reserved for future runtime modules. No dataset, weights or Python environment is committed.

## Development Workflow

We will use a **feature-branch workflow** to keep `main` stable and make integration manageable.

### 1. Branching Strategy

Create a separate branch for each feature, fix, or substantial change.

| Prefix | Purpose | Example |
|---|---|---|
| `feature/` | New functionality | `feature/flame-detection` |
| `fix/` | Bug fixes | `fix/frame-processing` |
| `refactor/` | Code restructuring | `refactor/model-integration` |
| `test/` | Test-related changes | `test/detection-pipeline` |
| `docs/` | Documentation | `docs/architecture` |

Keep branches focused on a specific task rather than combining unrelated changes.

### 2. Sync Before Development

Update your local `main` before creating a new feature branch.

```bash
git switch main
git pull origin main
git switch -c feature/your-feature
```

Replace `feature/your-feature` with a descriptive branch name.

### 3. Commit Standards

Use clear, meaningful commit messages.

```text
feat: add flame detection inference
fix: handle invalid video frames
refactor: modularize preprocessing pipeline
test: add detection evaluation tests
docs: document model integration
perf: optimize frame processing
```

Each commit should represent a logical unit of work. Avoid vague messages such as `update`, `changes`, or `final code`.

### 4. Push Your Branch

```bash
git add <files>
git commit -m "feat: describe your change"
git push -u origin feature/your-feature
```

Review staged changes before committing. Do not include unrelated modifications or generated files.

### 5. Pull Requests and Integration

All changes intended for `main` should be submitted through a pull request.

Before opening a PR:

- Ensure the implementation is complete for the scope of the change.
- Run the relevant tests and verify the expected behaviour.
- Check for accidental changes, debug code, and unnecessary dependencies.
- Update documentation where required.

Each PR should include:

- **Context:** What problem or requirement does this address?
- **Implementation:** What changed, and why?
- **Validation:** What tests or experiments were performed?
- **Limitations:** Any known issues or follow-up work.

Keep PRs focused and reasonably sized. Request review from the relevant teammate or project maintainer before merging.

---

## Engineering Guidelines

### Code Quality
- Keep components modular, reusable, and independently testable.
- Follow consistent naming conventions and formatting.
- Avoid unnecessary duplication and tightly coupled modules.
- Handle invalid inputs and runtime failures appropriately.

### Model and Pipeline Integration
- Document model inputs, outputs, preprocessing, and postprocessing.
- Keep model-specific logic separate from video acquisition and application logic where practical.
- Record relevant evaluation metrics and experimental assumptions.
- Validate integration changes against the agreed interface before merging.

### Dependencies and Reproducibility
- Document dependency changes in the appropriate dependency file.
- Avoid machine-specific paths and hardcoded environment assumptions.
- Keep configuration separate from implementation where appropriate.
- Ensure experiments can be reproduced using documented steps.

### Large Files and Sensitive Data
- Do not commit credentials, access tokens, private keys, or confidential data.
- Coordinate before adding large datasets, video recordings, or model weights.
- Use the storage approach agreed upon by the team for large artifacts.
- Keep `.gitignore` updated to exclude generated files and local environments.

---

## Collaboration Principles

- Coordinate module ownership and interfaces before parallel implementation.
- Communicate changes that may affect other components.
- Keep pull requests focused and reviewable.
- Discuss architectural changes before introducing dependencies or changing shared interfaces.
- Document important technical decisions and unresolved issues.
- Keep experimental implementations separate from stable, integrated functionality.

---

## Standard Contribution Flow

```text
Sync with main
     â†“
Create feature branch
     â†“
Implement and validate
     â†“
Commit changes
     â†“
Push branch
     â†“
Open pull request
     â†“
Review and resolve feedback
     â†“
Merge into main
```

**Branch protection:** The `main` branch should be protected against direct pushes where repository settings and team permissions permit. Integrate changes through reviewed pull requests.

---

## Project Documentation

Implementation-specific instructions, architecture diagrams, model details, hardware requirements, and evaluation procedures should be maintained in `docs/` as they are established.

Keep this README focused on the project overview and contribution workflow. Avoid documenting unconfirmed implementation decisions as finalized requirements.

---

**Project:** Honeywell BMSCE — Vision-Based Flame Detection and Monitoring
**Repository:** [GitHub](https://github.com/Skandakm29/honeywell_bmsce_flame_detection)
