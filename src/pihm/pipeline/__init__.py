"""Run pipeline: stages, records, campaigns, the summary table and the results catalogue.

* :mod:`~pihm.pipeline.records`: the record schema (strict JSON, NPZ, provenance).
* :mod:`~pihm.pipeline.spec`: run specifications, and where each run's record lives.
* :mod:`~pihm.pipeline.stages`: :func:`~pihm.pipeline.stages.run`, and the ``"build"`` and
  ``"encode"`` stages.
* :mod:`~pihm.pipeline.pillar2`: the ``"stateprep"`` and ``"readout"`` stages.
* :mod:`~pihm.pipeline.campaign`: YAML campaigns expanded into runs, and a local executor.
* :mod:`~pihm.pipeline.summary`: the ``summary.csv`` the analysis reads.
* :mod:`~pihm.pipeline.results`: each campaign's results by name, and the selections of
  records the analyses share.
* :mod:`~pihm.pipeline.catalogue`: every run the campaigns expect, under a readable key, with
  its status.
"""

__all__: list[str] = []
