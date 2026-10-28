# Pinned TI detector source
Minimal unchanged 2D modules copied from TexasInstruments/edgeai-yolox at
cbdd9d85def4e55e3fbbf38c155e183e90ed5cc6. The adjacent LICENSE applies to vendor files.
Source: https://github.com/TexasInstruments/edgeai-yolox/tree/cbdd9d85def4e55e3fbbf38c155e183e90ed5cc6
The project adapter in scripts/flame_model.py loads only these modules, avoiding unrelated pose imports.
Do not substitute upstream YOLOX-Nano: the TI model uses different layers/activations.
No project-wide licence grant is inferred from the vendor licence; the team must decide its own licence.
