# nimbleGPT

A Jax/Flax re-implementation of minGPT, with multiple implementations which increase in
speed but decrease in readability.

# Project Structure

- `model.py` contains the simplest (and slowest) implementation.
- `jmodel.py` extends the model to allow for jitted text generation (model.py is already
jittable, but only for fixed sized inputs. jmodel.py implements padding to allow inputs
of any size, as required for generating text).