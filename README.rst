A python package to streamline a machine learning accelerated approach to transforming between action-angle coordinates and phase-space coordinates in galactic potentials.
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Architecture
------------
The purpose of the subpackages are as follows

- ``orbitflows.flow``: Contains class for constructing a flow in pytorch. 
    - the main class for the normalizing flow is ``GsympNetFlow``
- ``orbitflows.models``: Contains classes for that wrap around the normalizing flow model 
    - the models differ by the technique they use to transform between action-angle coordinates and phase-space coordinates in different ways.
    - for example, models that map between a toy hamiltonian and a target hamiltonian will have different methods than models that map directly between action-angle coordinates and phase-space coordinates, hence they have different classes.
    - each model is a subclass of ``Model``, ensuring that all models have methods to calculate the hamiltonian as a function of action-angle coordinates, transforming between the coordinate systems with the model, the frequency, and orbit integration using the model.
    - models that train the normalizing flow to transform between a toy system to a target system are subclasses of ``MappingModel``
    - ``HamiltonianMappingModel`` transforms trains the normalizing flow to transform between the phase-space coordinates of a toy system to phase-space coordinates in the target system
- ``orbitflows.utils``: Contains utility functions that are used across the package.
- ``orbitflows.train``: Contains the functions needed training the machine learning models, including prewritten loss functions.




License
-------

This project is Copyright (c) Gabriel Pfaffman and licensed under
the terms of the BSD 3-Clause license. This package is based upon
the `Openastronomy packaging guide <https://github.com/OpenAstronomy/packaging-guide>`_
which is licensed under the BSD 3-clause licence. See the licenses folder for
more information.

Contributing
------------

We love contributions! orbitflows is open source,
built on open source, and we'd love to have you hang out in our community.

**Imposter syndrome disclaimer**: We want your help. No, really.

There may be a little voice inside your head that is telling you that you're not
ready to be an open source contributor; that your skills aren't nearly good
enough to contribute. What could you possibly offer a project like this one?

We assure you - the little voice in your head is wrong. If you can write code at
all, you can contribute code to open source. Contributing to open source
projects is a fantastic way to advance one's coding skills. Writing perfect code
isn't the measure of a good developer (that would disqualify all of us!); it's
trying to create something, making mistakes, and learning from those
mistakes. That's how we all improve, and we are happy to help others learn.

Being an open source contributor doesn't just mean writing code, either. You can
help out by writing documentation, tests, or even giving feedback about the
project (and yes - that includes giving feedback about the contribution
process). Some of these contributions may be the most valuable to the project as
a whole, because you're coming to the project with fresh eyes, so you can see
the errors and assumptions that seasoned contributors have glossed over.

Note: This disclaimer was originally written by
`Adrienne Lowe <https://github.com/adriennefriend>`_ for a
`PyCon talk <https://www.youtube.com/watch?v=6Uj746j9Heo>`_, and was adapted by
orbitflows based on its use in the README file for the
`MetPy project <https://github.com/Unidata/MetPy>`_.
