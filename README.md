# Honeywell BMSCE | Vision-Based Flame Detection and Monitoring

Collaborative development repository for the Honeywell project focused on vision-based flame detection and monitoring using computer vision, AI models, and edge deployment technologies.

**Repository:** [honeywell_bmsce_flame_detection](https://github.com/Skandakm29/honeywell_bmsce_flame_detection)

--

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

## Repository Structure

The repository will evolve as development progresses. Keep the code modular and organize new components according to their functionality.

```text
.
├── src/              # Core implementation
├── tests/            # Unit and integration tests
├── models/           # Model documentation and integration
├── configs/          # Configuration files
├── scripts/          # Utility and execution scripts
├── docs/             # Architecture and technical documentation
├── results/          # Evaluation summaries and relevant outputs
├── requirements.txt  # Python dependencies, if applicable
├── .gitignore
└── README.md
```

Avoid committing generated artifacts, large datasets, model weights, or environment-specific files unless explicitly agreed upon.

---

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
     ↓
Create feature branch
     ↓
Implement and validate
     ↓
Commit changes
     ↓
Push branch
     ↓
Open pull request
     ↓
Review and resolve feedback
     ↓
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
