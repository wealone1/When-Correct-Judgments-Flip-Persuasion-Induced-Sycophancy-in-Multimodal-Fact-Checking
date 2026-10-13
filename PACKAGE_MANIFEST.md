# Submission Package Manifest

## Included

- Four formal experiment entry points.
- Eight runtime source modules required by those entry points.
- Five formal experiment configurations.
- Frozen 7,700-row baseline manifest.
- Frozen 3,599-row seven-model shared-correct manifest and persuasion inputs.
- Frozen 255-row diagnostic manifest and audit metadata.
- Four regression-test modules covering the submitted experiment paths.
- Minimal installation and reproduction documentation.

## Deliberately excluded

- Model checkpoints and Hugging Face caches.
- MMFakeBench images and raw dataset archives.
- Generated predictions, reports, spreadsheets, and figures.
- Prompt-generation scripts.
- Dataset download, screening, selection, and review-site code.
- Result aggregation/Excel/paper-material helper scripts.
- Pilot v0-v2 and the retired animal-category experiments.
- Local synchronization and server launch scripts.
- Personal machine paths, usernames, tokens, and credentials.

The in-run `aggregate` subcommands remain part of the main experiment entry points, but no separate aggregation or paper-export utility is shipped.

