# Mres Project AI4ER

[![DOI](https://zenodo.org/badge/811090945.svg)](https://zenodo.org/doi/10.5281/zenodo.12581614)[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Project Description

Climate change significantly impacts hydrology, altering precipitation patterns, river flows, and water availability. Accurate streamflow predictions are necessary to mitigate flood risks and manage water resources. Directly measuring streamflow is challenging and resource-intensive, hence the rapid evolution of machine learning (ML) models based on measurable inputs like precipitation and temperature. This study examines the performance differences of an existing Neural Network (NN) and Gaussian Processes (GPs) between highland and lowland regions across the UK.

Snowmelt is identified as a critical factor for highland catchments, which was previously unaccounted for. Other experiments reveal that model performance is fundamentally limited by the ERA5 dataset, which has notable biases and limitations, especially in highland regions. GPs underscore the importance of snowmelt and input variable significance but also face challenges in highland areas, highlighting that a model’s effectiveness is tied to data quality.

This research emphasizes the need for better representations of highland catchment dynamics and addressing biases in widely used datasets like ERA5. While more accurate, localized datasets can improve model performance, they are impractical for developing a universal model applicable across diverse conditions. Developing such a universal model is crucial to improve streamflow predictions and manage water resources effectively.

Please see the final report ([`assets/mres_report.pdf`](assets/gtc_report_FINAL.pdf)) for full details.

This work was carried out as part of the [Artificial Intelligence for Environmental Risks](https://ai4er-cdt.esc.cam.ac.uk/) (AI4ER) Centre for Doctoral Training Masters of Research (Mres) Project, which ran from April, 2024 to June, 2024.

## Documentation

Please see the included [`DOCUMENTATION.md`](DOCUMENTATION.md) file for an overview of the repository structure. We include [tables](https://github.com/ai4er-cdt/OTP/blob/main/DOCUMENTATION.md#reproducing-report-figures-and-tables) that direct the user towards the appropriate notebook for reproducing each table and figure in the final report.

-----

## Author

<td><img src="assets/alinevd.jpg" alt="Aline Van Driessche" style="border-radius: 50%; width: 80px; height: 80px;"></td>
<td><a href="mailto:av656@cam.ac.uk">Aline Van Driessche</a></td>

## Acknowledgements

I would like to thank my day-to-day MRes supervisor, Robert Rouse, for giving me the opportunity to learn so many new things in a short timeframe and for his constant encouragement and insightful feedback. I am also greatful for the general oversight provided by Prof. Emily Shuckburgh, their combined support was vital to the success of this project. Additionally, I am grateful to the AI4ER support staff, Annabelle Scott and Adriana Dote, for creating a supportive environment and providing the necessary encouragement throughout the project.

-----

## License and Citation

If you use the code in this repository, please consider citing it--see the [`citation.cff`](citation.cff) file or use the "Cite this repository" function on the right sidebar. All code is under the MIT license--see the [`LICENSE`](LICENSE) file.

-----

## Data Availability

### ERA5 

All the ERA data extracted can be downloaded from the Copernicus Climate Data Service. The Documentation contains a table with the specific input variables used to download data from as well ERA5 as ERA5-land. This data can be obtained through the Coopernicus CDS API, https://cds.climate.copernicus.eu/. The input variables are preprocessed in the correct manner after running ‘assembly.py’ for the ERA5 data, and 'assembly_HR' for the ERA5-land data.

-----

### NRFA

The initial catchment information (shapefile with catchment boundary, elevationg etc) are downloaded from https://nrfa.ceh.ac.uk/. Also the the target river streamflow values and precipitation per catchment can be obtained from there. Similarl to the ERA5 data, this data is processed through 'assembly.py'.

-----

<p align="middle">
  <a href="https://ai4er-cdt.esc.cam.ac.uk/"><img src="assets/ai4er_logo.jpg" width="15%"/></a>
  <a href="https://www.cam.ac.uk/"><img src="assets/cambridge_logo.jpg" width="56%"/></a>
</p>