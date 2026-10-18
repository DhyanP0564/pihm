API reference
=============

Every public module, one page each. The package is layered: each group below imports only from
the groups above it, so reading top to bottom follows the method from the basis to the pipeline.

Most work needs only a handful of entry points:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - To
     - Use
   * - Load a problem from the ladder
     - :func:`pihm.problems.ladder.problem`
   * - State a problem of your own
     - :class:`pihm.problems.spec.Problem`
   * - Assemble its residual, in either form
     - :func:`pihm.assemble.assemble`
   * - Find the ground state classically
     - :func:`pihm.classical.physical_reference`
   * - Read a state out as a field
     - :func:`pihm.classical.decode`
   * - Compile the residual to circuits
     - :func:`pihm.circuits.assemble.encode_linear`
   * - Check a circuit against its operator
     - :func:`pihm.verify.verify_exact`, :func:`pihm.verify.verify_probe`
   * - Count qubits and gates
     - :func:`pihm.resources.count_resources`
   * - Prepare the ground state
     - :func:`pihm.qsp.prepare.t1`, :func:`~pihm.qsp.prepare.t2`, :func:`~pihm.qsp.prepare.t3`
   * - Run one specified run, or a campaign
     - :func:`pihm.pipeline.stages.run`, :mod:`pihm.pipeline.campaign`
   * - Read collected results back
     - :class:`pihm.pipeline.results.Results`

.. toctree::
   :maxdepth: 1

   core
   problems
   assemble
   classical
   circuits
   verify
   qsp
   pipeline
   plotting

The package itself holds only its version:

.. autodata:: pihm.__version__
