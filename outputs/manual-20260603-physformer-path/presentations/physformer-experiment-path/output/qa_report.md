# QA Report

## Build

- Final PPTX: `physformer-experiment-path-report.pptx`
- Slide count: 14
- Language: Simplified Chinese with technical English terms preserved
- Authoring path: local artifact-tool PPTX export
- External media: none; figures/charts are editable PPT-native shapes

## Source Basis

- ARA overview, exploration tree, claims, and staging observations
- Evidence tables for V3-V5 aggregate/component metrics
- C08 variance-decomposition analysis
- Phase A / Phase B local design and status documents
- Local A1/c23 metrics JSON where available

## Self-Review

- Checked the rendered contact sheet for visual rhythm and repeated-template issues.
- Inspected key slides individually after first export.
- Fixed two medium-severity defects:
  - Slide 12: A5 bar overlapped the right interpretation rail.
  - Slide 13: B1 finetune range wrapped into the metric label.
- Re-exported the PPTX and previews after fixes.

## Verification

- PPTX package contains 14 slide XML files.
- Rendered previews generated for all 14 slides.
- Contact sheet generated and inspected.
- No missing media dependencies because the deck uses native shapes.

## Known Limitations

- Speaker notes are not embedded; the deck is designed as a concise oral-report visual spine.
- A5 detailed local metrics files were not present locally; the A5 +7.5% value is taken from the crystallized ARA/Phase B record.
