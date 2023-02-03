Getting started
=======================

First install the ``GGPymanager`` (see :ref:`install`) and then import the module.

.. code-block:: python
    :linenos:

    import ggpymanager

Reading an existing catalog
---------------------------
To read an existing catalog, initialize the variables ``catalog_path``, ``config_path``, and ``sim_path``:

.. code-block:: python
    :linenos:

    catalog_path = "<path to the GRAMM catalog directory>"
    config_path = "<path to the GRAL config directory>"
    sim_path = "<path to the GRAL catalog directory>"

Then initialize the ``Reader``:

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

Creating a new catalog
----------------------
