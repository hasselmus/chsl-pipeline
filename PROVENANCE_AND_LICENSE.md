# CHSL-v1 — Provenance and licensing register

This file is the sole externally named reference register for this private research repository, outside the isolated raw-ingest adapters. It is **not** imported by analysis, processing, plotting or scenario exports. No raw or processed data products are committed or redistributed.

## 1. Seshat historical institutional data

- Origin: Seshat Global History Databank, `seshatdb/Equinox_Data`.
- Workbook: `Equinox_on_GitHub_June9_2022.xlsx`, snapshot dated 9 June 2022, publicly published 14 June 2022.
- Repository: https://github.com/seshatdb/Equinox_Data
- Archival DOI: https://doi.org/10.5281/zenodo.6642229
- **Conflicting repository notices**: README and repository description expressly specify Creative Commons Attribution–NonCommercial–ShareAlike (CC BY-NC-SA), without stating a version; the repository's `LICENSE` file contains Creative Commons CC0 1.0 Universal. These statements should not be silently reconciled in favour of the more permissive notice. For private analytical operation, conservatively treat the dataset as subject to attribution, non-commercial and share-alike conditions unless the provider clarifies its terms.
- Credit requested by provider: Seshat Global History Databank; the data release identified by DOI above. Historical citations embedded in source records remain in the locally stored `data/raw/citation_crosswalk.json`.
- Repository software is separately licensed if the repository owner chooses to add a software licence; that choice does not relicense upstream historical data.

## 2. Temperature reconstructions

- PAGES2k Common Era Surface Temperature Reconstructions, Neukom, R., Barboza, L. A., Erb, M. P., Shi, F., Emile-Geay, J., Evans, M. N., et al. (2019), NOAA/WDS Paleoclimatology – PAGES2k Common Era Surface Temperature Reconstructions. NOAA NCEI. https://doi.org/10.25921/tkxp-vn12
- Associated article: Neukom et al. (2019), *Consistent multidecadal variability in global temperature reconstructions and simulations over the Common Era*, Nature Geoscience. https://doi.org/10.1038/s41561-019-0400-0
- Archive: https://www.ncei.noaa.gov/pub/data/paleo/pages2k/neukom2019temp/
- Follow NOAA archival citation instructions, identify subset and access date. Verify any applicable file-level terms when downloading.

## 3. Tree-ring drought reconstructions

- Old World Drought Atlas (OWDA): Cook, E. R., Seager, R., Kushnir, Y., et al. (2015), *Old World megadroughts and pluvials during the Common Era*, Science Advances 1(10), e1500561. https://doi.org/10.1126/sciadv.1500561
  Information: https://www.drought.gov/data-maps-tools/old-world-drought-atlas
- North American Drought Atlas (NADA): Cook, E. R., Woodhouse, C. A., Eakin, C. M., Meko, D. M., Stahle, D. W. (2004), *Long-Term Aridity Changes in the Western United States*, Science 306(5698), 1015–1018. https://doi.org/10.1126/science.1102586. Check the precise version/associated bibliographic record for the file downloaded.
- Monsoon Asia Drought Atlas (MADA): Cook, E. R., Anchukaitis, K. J., Buckley, B. M., D'Arrigo, R. D., Jacoby, G. C., Wright, W. E. (2010), *Asian Monsoon Failure and Megadrought During the Last Millennium*, Science 328(5977), 486–489. https://doi.org/10.1126/science.1185188
- For all three: obtain original NCEI archive release notices and file-specific terms; maintain release URL, access date and SHA-256 in the **local** download manifest.

## 4. Regional gridded population and land use

- HYDE 3.3, History Database of the Global Environment, produced by K. Klein Goldewijk, A. Beusen, J. Doelman and E. Stehfest with the PBL Netherlands Environmental Assessment Agency / Utrecht University.
- Source documentation and access: https://landuse.sites.uu.nl/hyde-project/
- HYDE 3.3 data describe total population and gridded agricultural land use at 5-arc-minute spatial resolution. Baseline, lower and upper estimates exist; do not silently discard compatible-state bounds where available.
- Standard published paper for previous version: Klein Goldewijk, K., Beusen, A., Doelman, J., Stehfest, E. (2017), *Anthropogenic land-use estimates for the Holocene – HYDE 3.2*, Earth System Science Data 9, 927–953. https://doi.org/10.5194/essd-9-927-2017. This paper documents version 3.2, **not** an independently verified 3.3 publication.
- Confirm precise licence and terms on the actual archive obtained; no blanket CC BY 4.0 conclusion is inferred from secondary repackagings.

## 5. Geographic crosswalks and polygons

- Original historical-geography names, basin joins, source polygons and bibliographic details remain under ignored `data/raw/`, or inside the raw-ingest translation boundary. Any externally named polygon product (including Cliopatria) must have its own provenance record and licence check before local use.
- The processed CHSL-v1 ledger retains only a stable geometric boundary identifier and content hash, never upstream geographic names or source file paths.

## Operating controls

- No downloads, extracted raw records, processed Parquet panels, statistical result files or generated sidecars are tracked in Git.
- Attribution and licence conditions continue to apply when required by source terms, even where data are not redistributed.
- Internal scenario prompting uses generated CHSL-v1 outputs only. It does not read this provenance register.
- Publication, sharing, or commercial use of derived data requires a separate licence and attribution review. This file does not constitute legal advice.
