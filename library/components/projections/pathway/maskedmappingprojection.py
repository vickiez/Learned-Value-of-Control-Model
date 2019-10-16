# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# *******************************************  AutoAssociativeProjection ***********************************************

"""
.. _Masked_MappingProjection_Overview:

Overview
--------

A MaskedMappingProjection is a subclass of `MappingProjection` that applies a specified mask array
(either additively, multiplicatively, or exponentially) to the MappingProjection's `matrix <MappingProjection.matrix>`
each time the MappingProjection is executed.  The mask is assigned a `ParameterState` and can thus be modulated
by a `ControlMechanism <ControlMechanism>`.

.. _Masked_MappingProjection_Creation:

Creating a MaskedMappingProjection
----------------------------------

A MaskedMappingProjection is created in the same way as a `MappingProjection`, with the exception that its constructor
includes **mask** and **mask_operation** arguments that can be used to configure the mask and how it is applied to the
Projection's `matrix <MaskedMappingProjection.matrix>`.  The **mask** attribute must be an array that has the same
shape as the `matrix <MaskedMappingProjection.matrix>` attribute, and the **mask_operation** argument must be the
keyword *ADD*, *MULTIPLY*, or *EXPONENTIATE* (see `Masked_MappingProjection_Execution` below).

.. _Masked_MappingProjection_Structure:

Structure
---------

A MaskedMappingProjection is identical to a MappingProjection, with the addition of `mask
<MaskedMappingProjection.mask>` and `mask_operation <MaskedMappingProjection.mask_operation>` attributes.

.. _Masked_MappingProjection_Execution:

Execution
---------

A MaskedMappingProjection executes in the same manner as a `MappingProjection`, with the exception that,
each time the Projection is executed, its `mask <MaskedMappingProjection.mask>` is applied to its `matrix
<MaskedMappingProjection.matrix>` parameter as specified by its `mask_operation
<MaskedMappingProjection.mask_operation>` attribute, before generating the Projection's `value
<MaskedMappingProjection.value>`.

.. _Masked_MappingProjection_Class_Reference:

Class Reference
---------------

"""
import numbers

import numpy as np
import typecheck as tc

from psyneulink.core.components.component import parameter_keywords
from psyneulink.core.components.functions.transferfunctions import get_matrix
from psyneulink.core.components.projections.pathway.mappingprojection import MappingProjection
from psyneulink.core.components.projections.projection import projection_keywords
from psyneulink.core.components.shellclasses import Mechanism
from psyneulink.core.components.states.outputstate import OutputState
from psyneulink.core.globals.context import ContextFlags
from psyneulink.core.globals.keywords import DEFAULT_MATRIX, FUNCTION_PARAMS, MASKED_MAPPING_PROJECTION, MATRIX
from psyneulink.core.globals.parameters import Parameter
from psyneulink.core.globals.preferences.componentpreferenceset import is_pref_set
from psyneulink.core.globals.preferences.preferenceset import PreferenceLevel

__all__ = [
    'MaskedMappingProjection', 'MaskedMappingProjectionError',
    'MASK', 'MASK_OPERATION', 'ADD', 'MULTIPLY', 'EXPONENTIATE'
]

MASK = 'mask'
MASK_OPERATION = 'mask_operation'
ADD = 'add'
MULTIPLY = 'multiply'
EXPONENTIATE = 'exponentiate'

parameter_keywords.update({MASKED_MAPPING_PROJECTION})
projection_keywords.update({MASKED_MAPPING_PROJECTION})

class MaskedMappingProjectionError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value

class MaskedMappingProjection(MappingProjection):
    """
    MaskedMappingProjection(     \
        sender=None,             \
        receiver=None,           \
        matrix=DEFAULT_MATRIX,   \
        mask=None,               \
        mask_operation=MULTIPLY  \
        params=None,             \
        name=None,               \
        prefs=None)

    Implement MappingProjection the `matrix <MaskedMappingProjection.matrix>` of which can be masked on each execution.

    Arguments
    ---------

    sender : Optional[OutputState or Mechanism]
        specifies the source of the Projection's input. If a Mechanism is specified, its
        `primary OutputState <OutputState_Primary>` will be used. If it is not specified, it will be assigned in
        the context in which the Projection is used.

    receiver : Optional[InputState or Mechanism]
        specifies the destination of the Projection's output.  If a Mechanism is specified, its
        `primary InputState <InputState_Primary>` will be used. If it is not specified, it will be assigned in
        the context in which the Projection is used.

    matrix : list, np.ndarray, np.matrix, function or keyword : default DEFAULT_MATRIX
        the matrix used by `function <MaskedMappingProjection.function>` (default: `LinearCombination`) to transform
        the value of the `sender <MaskedMappingProjection.sender>`.

    mask : int, float, list, np.ndarray or np.matrix : default None
        specifies a mask to be applied to the `matrix <MaskedMappingProjection.matrix>` each time the Projection is
        executed, in a manner specified by the **mask_operation** argument.

    mask_operation : ADD, MULTIPLY, or EXPONENTIATE : default MULTIPLY
        specifies the manner in which the `mask <MaskedMappingProjection.mask>` is applied to the `matrix
        <MaskedMappingProjection.matrix>` each time the Projection is executed.

    params : Dict[param keyword: param value] : default None
        a `parameter dictionary <ParameterState_Specification>` that can be used to specify the parameters for
        the Projection, its function, and/or a custom function and its parameters. By default, it contains an entry for
        the Projection's default assignment (`LinearCombination`).  Values specified for parameters in the dictionary
        override any assigned to those parameters in arguments of the constructor.

    name : str : default AutoAssociativeProjection-<index>
        a string used for the name of the AutoAssociativeProjection. When an AutoAssociativeProjection is created by a
        RecurrentTransferMechanism, its name is assigned "<name of RecurrentTransferMechanism> recurrent projection"
        (see `Registry <LINK>` for conventions used in naming, including for default and duplicate names).

    prefs : Optional[PreferenceSet or specification dict : Projection_Base.classPreferences]
        the `PreferenceSet` for the MappingProjection; if it is not specified, a default is assigned using
        `classPreferences` defined in __init__.py (see `PreferenceSet <LINK>` for details).

    Attributes
    ----------

    componentType : MASKED_MAPPING_PROJECTION

    sender : OutputState
        identifies the source of the Projection's input.

    receiver: InputState
        identifies the destination of the Projection.

    matrix : 2d np.ndarray
        matrix used by `function <AutoAssociativeProjection.function>` to transform input from the `sender
        <MappingProjection.sender>` to the value provided to the `receiver <AutoAssociativeProjection.receiver>`.

    mask : int, float, list, np.ndarray or np.matrix : default None
        mask applied to the `matrix <MaskedMappingProjection.matrix>` each time the Projection is executed,
        in a manner specified by `mask_operation <MaskedMappingProjection.mask_operation>`.

        mask_operation : ADD, MULTIPLY, or EXPONENTIATE : default MULTIPLY
        determines the manner in which the `mask <MaskedMappingProjection.mask>` is applied to the `matrix
        <MaskedMappingProjection.matrix>` when the Projection is executed.

    value : np.ndarray
        Output of AutoAssociativeProjection, transmitted to `variable <InputState.variable>` of `receiver`.

    name : str
        a string used for the name of the AutoAssociativeProjection (see `Registry <LINK>` for conventions used in
        naming, including for default and duplicate names).

    prefs : PreferenceSet or specification dict : Projection_Base.classPreferences
        the `PreferenceSet` for AutoAssociativeProjection (see :doc:`PreferenceSet <LINK>` for details).
    """

    componentType = MASKED_MAPPING_PROJECTION
    className = componentType
    suffix = " " + className

    class Parameters(MappingProjection.Parameters):
        """
            Attributes
            ----------

                variable
                    see `variable <MaskedMappingProjection.variable>`

                    :default value: numpy.array([[0]])
                    :type: numpy.ndarray

                mask
                    see `mask <MaskedMappingProjection.mask>`

                    :default value: None
                    :type:

                mask_operation
                    see `mask_operation <MaskedMappingProjection.mask_operation>`

                    :default value: `MULTIPLY`
                    :type: str

                matrix
                    see `matrix <MaskedMappingProjection.matrix>`

                    :default value: `AUTO_ASSIGN_MATRIX`
                    :type: str

        """
        variable = np.array([[0]])    # function is always LinearMatrix that requires 1D input
        mask = None
        mask_operation = MULTIPLY

        def _validate_mask_operation(self, mode):
            options = {ADD, MULTIPLY, EXPONENTIATE}
            if mode in options:
                # returns None indicating no error message (this is a valid assignment)
                return None
            else:
                # returns error message
                return 'not one of {0}'.format(options)

    classPreferenceLevel = PreferenceLevel.TYPE

    # necessary?
    paramClassDefaults = MappingProjection.paramClassDefaults.copy()

    @tc.typecheck
    def __init__(self,
                 sender=None,
                 receiver=None,
                 matrix=DEFAULT_MATRIX,
                 mask:tc.optional(tc.any(int,float,list,np.ndarray,np.matrix))=None,
                 mask_operation:tc.enum(ADD, MULTIPLY, EXPONENTIATE)=MULTIPLY,
                 function=None,
                 params=None,
                 name=None,
                 prefs: is_pref_set = None):

        params = self._assign_args_to_param_dicts(mask=mask,
                                                  mask_operation=mask_operation,
                                                  function_params={MATRIX: matrix},
                                                  params=params)

        super().__init__(sender=sender,
                         receiver=receiver,
                         matrix=matrix,
                         function=function,
                         params=params,
                         name=name,
                         prefs=prefs)


    def _validate_params(self, request_set, target_set=None, context=None):
        """Validate **mask** argument"""

        super()._validate_params(request_set=request_set,
                                 target_set=target_set,
                                 context=context)

        if MASK in target_set and target_set[MASK]:
            mask = target_set[MASK]
            if isinstance(mask, (int, float)):
                return
            mask_shape = np.array(mask).shape
            matrix = get_matrix(self.user_params[FUNCTION_PARAMS][MATRIX],
                                len(self.sender.defaults.value), len(self.receiver.defaults.value))
            matrix_shape = matrix.shape
            if mask_shape != matrix_shape:
                raise MaskedMappingProjectionError("Shape of the {} for {} ({}) "
                                                   "must be the same as its {} ({})".
                                                   format(repr(MASK), self.name, mask_shape,
                                                          repr(MATRIX), matrix_shape))

    def _update_parameter_states(self, execution_id=None, runtime_params=None, context=None):

        # Update parameters first, to be sure mask that has been updated if it is being modulated
        #  and that it is applied to the updated matrix param
        super()._update_parameter_states(execution_id=execution_id, runtime_params=runtime_params, context=context)

        mask = self.parameters.mask.get(execution_id)
        mask_operation = self.parameters.mask_operation.get(execution_id)
        matrix = self.parameters.matrix.get(execution_id)
        # Apply mask to matrix using mask_operation
        if mask is not None:
            if mask_operation is ADD:
                matrix += mask
            elif mask_operation is MULTIPLY:
                matrix *= mask
            elif mask_operation is EXPONENTIATE:
                matrix **= mask

        self.parameters.matrix.set(matrix, execution_id)
