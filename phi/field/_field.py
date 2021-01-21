from __future__ import annotations

import copy
from abc import ABC

from phi import math
from phi.geom import Geometry
from phi.math import Shape, Tensor, Extrapolation
from phi.math._shape import SPATIAL_DIM, BATCH_DIM, CHANNEL_DIM


class Field:

    @property
    def shape(self) -> Shape:
        """
        Returns a shape with the following properties

        * The spatial dimension names match the dimensions of this Field
        * The batch dimensions match the batch dimensions of this Field
        * The channel dimensions match the channels of this Field
        """
        raise NotImplementedError()

    @property
    def spatial_rank(self) -> int:
        """
        Spatial rank of the field (1 for 1D, 2 for 2D, 3 for 3D).
        This is equal to the spatial rank of the `data`.
        """
        return self.shape.spatial.rank

    def sample_in(self, geometry: Geometry, reduce_channels=()) -> Tensor:
        """
        Approximates the mean field value inside the volume of the geometry (batch).

        For small volumes, the value at the volume's center may be sampled.
        The batch dimensions of the geometry are matched with this Field.
        Spatial dimensions can be used to sample a grid of geometries.

        The default implementation of this method samples this Field at the center point of the geometry.

        :param geometry: single or batched Geometry object
        :param reduce_channels: (optional) dimension of `points` to be reduced against the vector dimension of this Field.
        Causes the components of this field to be sampled at different locations.
        The result is the same as `math.channel_stack([component.sample_at(p) for component, p in zip(field.unstack('vector'), points.unstack(reduce)])`

        :return: sampled values
        """
        return self.sample_at(geometry.center, reduce_channels)

    def sample_at(self, points: Tensor, reduce_channels=()) -> Tensor:
        """
        Sample this field at the world-space locations (in physical units) given by points.
        Points must have a single channel dimension named `vector`.
        It may additionally contain any number of batch and spatial dimensions, all treated as batch dimensions.

        :param points: world-space locations
        :param reduce_channels: (optional) dimension of `points` to be reduced against the vector dimension of this Field.
        Causes the components of this field to be sampled at different locations.
        The result is the same as `math.channel_stack([component.sample_at(p) for component, p in zip(field.unstack('vector'), points.unstack(reduce)])`

        :return: sampled values
        """
        raise NotImplementedError(self)

    def at(self, representation: SampledField) -> SampledField:
        """
        Samples this field at the sample points of `representation`.
        The result will approximate the values of this field on the data structure of `representation`.

        Unlike Field.sample_at(), this method returns a Field object, not a Tensor.

        :param representation: Field object defining the sample points. The values of `representation` are ignored.
        :return: Field object of same type as `representation`
        """
        elements = representation.elements
        resampled = self.sample_in(elements, reduce_channels=elements.shape.non_channel.without(representation.shape).names)
        extrap = self.extrapolation if isinstance(self, SampledField) else representation.extrapolation
        return representation._op1(lambda old: extrap if isinstance(old, math.extrapolation.Extrapolation) else resampled)

    def __rshift__(self, other):
        return self.at(other)

    def unstack(self, dimension: str) -> tuple:
        """
        Unstack the field along one of its dimensions.
        The dimension can be batch, spatial or channel.

        :param dimension: name of the dimension to unstack, must be part of `self.shape`
        :return: tuple of Fields
        """
        raise NotImplementedError()

    def dimension(self, name):
        return _FieldDim(self, name)

    def __getattr__(self, name: str) -> _FieldDim:
        if name.startswith('_'):
            raise AttributeError(f"'{type(self)}' object has no attribute '{name}'")
        return _FieldDim(self, name)

    def __mul__(self, other):
        return self._op2(other, lambda d1, d2: d1 * d2)

    __rmul__ = __mul__

    def __truediv__(self, other):
        return self._op2(other, lambda d1, d2: d1 / d2)

    def __rtruediv__(self, other):
        return self._op2(other, lambda d1, d2: d2 / d1)

    def __sub__(self, other):
        return self._op2(other, lambda d1, d2: d1 - d2)

    def __rsub__(self, other):
        return self._op2(other, lambda d1, d2: d2 - d1)

    def __add__(self, other):
        return self._op2(other, lambda d1, d2: d1 + d2)

    __radd__ = __add__

    def __pow__(self, power, modulo=None):
        return self._op2(power, lambda f, p: f ** p)

    def __neg__(self):
        return self._op1(lambda x: -x)

    def __gt__(self, other):
        return self._op2(other, lambda x, y: x > y)

    def __ge__(self, other):
        return self._op2(other, lambda x, y: x >= y)

    def __lt__(self, other):
        return self._op2(other, lambda x, y: x < y)

    def __le__(self, other):
        return self._op2(other, lambda x, y: x <= y)

    def __abs__(self):
        return self._op1(lambda x: abs(x))

    def _op1(self, operator) -> Field:
        """
        Perform an operation on the data of this field.

        :param operator: function that accepts tensors and extrapolations and returns objects of the same type and dimensions
        :return: Field of same type
        """
        raise NotImplementedError()

    def _op2(self, other, operator) -> Field:
        raise NotImplementedError()


class SampledField(Field):

    def __init__(self, elements: Geometry, values: Tensor or float or int, extrapolation: math.Extrapolation):
        """
        Base class for fields that are sampled at specific locations such as grids or point clouds.

        :param elements: Geometry object specifying the sample points and sizes
        :param values: values corresponding to elements
        :param extrapolation: values outside elements
        """
        assert isinstance(extrapolation, (Extrapolation, tuple, list)), extrapolation
        assert isinstance(elements, Geometry), elements
        self._elements = elements
        self._values = math.tensor(values)
        self._extrapolation = extrapolation
        self._shape = elements.shape.non_channel & self._values.shape.non_spatial

    @property
    def elements(self) -> Geometry:
        """
        Returns a geometrical representation of the discretized volume elements.
        The result is a tuple of Geometry objects, each of which can have additional spatial (but not batch) dimensions.

        For grids, the geometries are boxes while particle fields may be represented as spheres.

        If this Field has no discrete points, this method returns an empty geometry.

        :return: Geometry with all batch/spatial dimensions of this Field. Staggered sample points are modelled using extra batch dimensions.
        """
        return self._elements

    @property
    def points(self) -> Tensor:
        return self.elements.center

    @property
    def values(self) -> Tensor:
        return self._values

    @property
    def extrapolation(self) -> Extrapolation:
        return self._extrapolation

    @property
    def shape(self) -> Shape:
        return self._shape

    def sample_at(self, points, reduce_channels=()) -> Tensor:
        raise NotImplementedError()

    def unstack(self, dimension: str) -> tuple:
        values = self._values.unstack(dimension)
        return tuple(self._with(v) for i, v in enumerate(values))

    def _op1(self, operator) -> Field:
        values = operator(self.values)
        extrapolation_ = operator(self._extrapolation)
        return self._with(values=values, extrapolation=extrapolation_)

    def _op2(self, other, operator) -> Field:
        if isinstance(other, Field):
            other_values = other.sample_in(self._elements)
            values = operator(self._values, other_values)
            extrapolation_ = operator(self._extrapolation, other.extrapolation)
            return self._with(values, extrapolation_)
        else:
            other = math.as_tensor(other)
            values = operator(self._values, other)
            return self._with(values)

    def __getitem__(self, item):
        values = self._values[item]
        return self._with(values)

    def _with(self, values: Tensor = None, extrapolation: math.Extrapolation = None):
        copied = copy.copy(self)
        SampledField.__init__(copied, self._elements, values if values is not None else self._values, extrapolation if extrapolation is not None else self._extrapolation)
        return copied


class _FieldDim:

    def __init__(self, field: Field, name: str):
        self.field = field
        self.name = name

    @property
    def exists(self):
        return self.name in self.field.shape

    def __str__(self):
        return self.name

    def unstack(self, size: int or None = None):
        if size is None:
            return self.field.unstack(self.name)
        else:
            if self.exists:
                unstacked = self.field.unstack(self.name)
                assert len(unstacked) == size, f"Size of dimension {self.name} does not match {size}."
                return unstacked
            else:
                return (self.field,) * size

    @property
    def size(self):
        return self.field.shape.get_size(self.name)

    @property
    def dim_type(self):
        return self.field.shape.get_type(self.name)

    @property
    def is_spatial(self):
        return self.dim_type == SPATIAL_DIM

    @property
    def is_batch(self):
        return self.dim_type == BATCH_DIM

    @property
    def is_channel(self):
        return self.dim_type == CHANNEL_DIM

    def __getitem__(self, item):
        return self.field.unstack(self.name)[item]