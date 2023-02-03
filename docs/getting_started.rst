.. _getting_started:

Getting started
=======================

First install the ``GGPymanager`` (see :ref:`install`) and then import the module.

.. code-block:: python
    :linenos:

    import ggpymanager

Creating a new catalog
----------------------

To create a new catalog of GRAL simulations, initialize the variables ``catalog_path``, ``config_path``, and ``sim_path``:

.. code-block:: python
    :linenos:

    catalog_path = "<path to the GRAMM catalog directory>"
    config_path = "<path to the GRAL config directory>"
    sim_path = "<path to the GRAL catalog directory>"

Then initialize the ``Catalog`` in write mode with ``read_only=False`` and initialize the simulations:

.. code-block:: python
    :linenos:

    catalog = ggpymanager.Catalog(
        catalog_path,
        config_path,
        sim_path,
        read_only=False,
    )

    catalog.init_simulations()

If there are already some simulations in the directory that are finished or that terminated with an error, it is possible to get a summary of the status of the simulations with  ``get_info()`` or to get more detailed information by checking the status of each simulation:

.. code-block:: python
    :linenos:

    # Summary
    info_dict = catalog.get_info()

    # Detailed information
    detailed_info_dict = {
        "init": 0,
        "error": 0,
        "finished": 0,
        "running": 0,
    }
    for sim in catalog.simulations:
        detailed_info_dict[sim.status.name] += 1

The GRAL simulations can be started in parralel with ``n_processes`` as the number of parallel GRAL runs and ``n_limit`` as the number of simulations to be computed. If ``n_limit`` is ``None`` all initialized simulations are run.

.. code-block:: python
    :linenos:

    catalog.run_simulations(n_processes, n_limit)

.. warning::

    The number of processes ``n_processes`` is not the number of total CPU processors. The total number of processors used is the product of ``n_processes`` and the number of CPU processes in the config directory in the file ``Max_Proc.txt``.

When GRAL simulations are finished, the concentration fields are stored in binary files. To improve the performance of reading out the concentration fields, it is more efficient to store the data in sparse matrices from ``scipy``. To save the concentration fields of finished GRAL simulations with ``n_processes`` parallel CPU processes run:

.. code:: python
    :linenos:

    catalog.save_simulations_as_npz(n_processes)


Reading an existing catalog
---------------------------
To read an existing catalog of GRAL simulations, initialize the variables ``catalog_path``, ``config_path``, and ``sim_path``:

.. code-block:: python
    :linenos:

    catalog_path = "<path to the GRAMM catalog directory>"
    config_path = "<path to the GRAL config directory>"
    sim_path = "<path to the GRAL catalog directory>"

Then initialize the ``Reader`` which has the ``Catalog`` class as the base class:

.. code-block:: python
    :linenos:

    reader = ggpymanager.Reader(
        catalog_path,
        config_path,
        sim_path,
    )

Now, the reader can be used to get the concentration files with ``sim_id`` as a valid simulation id:

.. code-block:: python
    :linenos:

    sim_id = 100
    con_dict = reader.get_concentration(sim_id)

The ``cond_dict`` is a dictionary of the GRAL source groups and emission heights with the keys "hxx" with "h" as an integer for the height layer starting from 1 and "xx" the number of the source group as two digits also starting from 1.

So the first height layer of the fifth source group can be accessed with:

.. code-block:: python
    :linenos:

    height = 1
    source_group = 5
    key = "{}{:02}".format(height, source_group)
    con_array = con_dict[key]

.. note::

    The concentration fields are in units :math:`\mu g m^{-3}`.

