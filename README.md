# CrimeGraph

In the CrimeGraph project, I will explore training a graph deep learning  model to forecast the probability of crime events in the City of Baltimore, using data provided by [SpotCrime](https://spotcrime.com/#what-is-spotcrime). SpotCrime maintains a uniquely comprehensive database of geocoded crime reports obtained from public safety databases and news reports from across the country. They were kind enough to grant me access to 23 years of data documenting crime events in the Baltimore.

I plan to adopt the Gated Localised Diffusion (GLDnet) architecture described in the 2020 paper _[Graph Deep Learning Model for Network-based Predictive Hotspot Mapping of Sparse Spatio-Temporal Events](https://discovery.ucl.ac.uk/id/eprint/10085742/)_.

Models with this design can learn from the spatial patterns of crime events by representing intersecting city streets as a graph of nodes and edges.

## Work in Progress

- [Street Graph Feature Engineering](https://www.subelsky.com/crimegraph/street_graph_feature_engineering/)
- [Event Timestep Feature Engineering](https://www.subelsky.com/crimegraph/event_timestep_feature_engineering/)
- [Graph Deep Learning Network Architecture](https://www.subelsky.com/crimegraph/model_development/)
- [Model Training and Hyperparameter Tuning](https://www.subelsky.com/crimegraph/model_training_and_tuning/)

## Technical Details

The project is written in Python 3.10 and uses these libraries:

* networkx
* geopandas
* shapely
* matplotlib
* sklearn

## Tools

* Jupyter Notebook
* Quarto (for web publishing)

## License

[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)

## Questions

Email me at contact@subelsky.com.