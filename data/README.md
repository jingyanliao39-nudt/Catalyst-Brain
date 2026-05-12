# Data files

Put the OC metadata file used by the agent here:

- `data/processed/updated_sid_to_details.pkl`

The default config points to this relative path. If your metadata is stored elsewhere, edit `paths.metadata_path` in the YAML config.

FairChem bulk and adsorbate database files are expected at:

- `fairchem-forked/src/fairchem/data/oc/databases/pkls/bulks.pkl`
- `fairchem-forked/src/fairchem/data/oc/databases/pkls/adsorbates.pkl`

Those files were not present in the source workspace, so this project keeps the directory and documents where to place them.
