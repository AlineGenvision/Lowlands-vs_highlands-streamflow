# Mres Project AI4ER

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10841513.svg)](https://doi.org/10.5281/zenodo.10841513) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Project Description

In this study, we demonstrate that machine learning techniques can predict abyssal MOC strength using only satellite-observable variables. We train a suite of models for this task using the "Estimating the Circulation and Climate of the Ocean" (ECCO) state estimate, obtaining state-of-the-art performance. We incorporate the "Australian Community Climate and Earth System Simulator Ocean Model" (ACCESS), a high-resolution numerical ocean circulation model; and observational "Rapid Climate Change-Meridional Overturning Circulation and Heatflux Array" (RAPID) data, a cross-basin sensor array that directly measures the Atlantic MOC strength. Our experiments indicate an approximately linear relationship between satellite-observable variables and abyssal MOC strength. We additionally demonstrate the utility of observational data for predicting long-range oceanic dependencies through the integration of RAPID, and show that a deep learning model is able to accurately capture latitude-invariant features for MOC strength prediction. Through these experiments, we present a methodology for predicting abyssal circulation, which will be instrumental in informing climate policy and empowering further oceanographic research. 

Please see the final report ([`assets/gtc_report_FINAL.pdf`](assets/gtc_report_FINAL.pdf)) for full details.

This work was carried out as part of the [Artificial Intelligence for Environmental Risks](https://ai4er-cdt.esc.cam.ac.uk/) (AI4ER) Centre for Doctoral Training Masters of Research (Mres) Project, which ran from April, 2024 to June, 2024.

## Documentation

Please see the included [`DOCUMENTATION.md`](DOCUMENTATION.md) file for an overview of the repository structure. We include [tables](https://github.com/ai4er-cdt/OTP/blob/main/DOCUMENTATION.md#reproducing-report-figures-and-tables) that direct the user towards the appropriate notebook for reproducing each table and figure in the final report.

-----

## Author

<td><img src="assets/alinevd.jpg" alt="Aline Van Driessche" style="border-radius: 50%; width: 80px; height: 80px;"></td>
<td><a href="mailto:av656@cam.ac.uk">Aline Van Driessche</a></td>

## Acknowledgements

I would like to thank my day-to-day MRes supervisor, Robert Rouse, as well as our general project coordinator, Prof. Emily Shuckburgh. Their guidance and support were vital to the success of this project. I also extend my gratitude to the AI4ER support staff, Annabelle Scott and Adriana Dote, for providing the necessary infrastructure and support.

-----

## License and Citation

If you use the code in this repository, please consider citing it--see the [`citation.cff`](citation.cff) file or use the "Cite this repository" function on the right sidebar. All code is under the MIT license--see the [`LICENSE`](LICENSE) file.

-----

## Data Availability

### ERA5 

All the ERA data extracted can be downloaded from the Copernicus Climate Data Service.nodo repository linked in the badge directly above. To facilitate interpretation of the code in this repository, we have copied over dataset metadata--see [`DATA_README.md`](DATA_README.md). Raw ECCO data is available for download from [NASA PO.DAAC](https://podaac.jpl.nasa.gov/); see the table in `DATA_README.md` for the PO.DAAC entries that we used.

### NRFA

ACCESS data was provided to us directly by colleagues at the National Oceanography Centre (NOC; Southampton, UK) and is not publicly available but may be provided upon reasonable request. Team member emails are linked above.

-----

<p align="middle">
  <a href="https://ai4er-cdt.esc.cam.ac.uk/"><img src="assets/ai4er_logo.jpg" width="15%"/></a>
  <a href="https://www.cam.ac.uk/"><img src="assets/cambridge_logo.jpg" width="56%"/></a>
</p>