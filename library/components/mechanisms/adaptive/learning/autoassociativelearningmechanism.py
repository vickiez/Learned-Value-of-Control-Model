# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# *************************************  AutoAssociativeLearningMechanism **********************************************

"""
.. _AutoAssociativeLearningMechanism_Overview:

Overview
--------

An AutoAssociativeLearningMechanism is a subclass of `LearningMechanism`, modified for use with a
`RecurrentTransferMechanism` to train its `recurrent_projection <RecurrentTransferMechanism.recurrent_projection>`.

.. _AutoAssociativeLearningMechanism_Creation:

Creating an AutoAssociativeLearningMechanism
--------------------------------------------

An AutoAssociativeLearningMechanism can be created directly by calling its constructor, but most commonly it is
created automatically when a RecurrentTransferMechanism is `configured for learning <Recurrent_Transfer_Learning>`,
(identified in its `activity_source <AutoAssociativeLearningMechanism.activity_source>` attribute).

.. _AutoAssociativeLearningMechanism_Structure:

Structure
---------

An AutoAssociativeLearningMechanism is identical to a `LearningMechanism` in all respects except the following:

  * it has only a single *ACTIVATION_INPUT* `InputState`, that receives a `MappingProjection` from an `OutputState` of
    the `RecurrentTransferMechanism` with which it is associated (identified by the `activity_source
    <AutoAssociativeLearningMechanism.activity_source>`);

  * it has a single *LEARNING_SIGNAL* `OutputState` that sends a `LearningProjection` to the `matrix
    <AutoAssociativeProjection.matrix>` parameter of an 'AutoAssociativeProjection` (typically, the
    `recurrent_projection <RecurrentTransferMechanism.recurrent_projection>` of a RecurrentTransferMechanism),
    but not an *ERROR_SIGNAL* OutputState.

  * it has no `input_source <LearningMechanism.input_source>`, `output_source <LearningMechanism.output_source>`,
    or `error_source <LearningMechanism.error_source>` attributes;  instead, it has a single `activity_source
    <AutoAssociativeLearningMechanism.activity_source>` attribute that identifies the source of the activity vector
    used by the Mechanism's `function <AutoAssociativeLearningProjection.function>`.

  * its `function <AutoAssociativeLearningMechanism.function>` takes as its `variable <Function_Base.variable>`
    a list or 1d np.array of numeric entries, corresponding in length to the AutoAssociativeLearningMechanism's
    *ACTIVATION_INPUT* InputState; and it returns a `learning_signal <LearningMechanism.learning_signal>`
    (a weight change matrix assigned to the Mechanism's *LEARNING_SIGNAL* OutputState), but not necessarily an
    `error_signal <LearningMechanism.error_signal>`.

  * its `learning_rate <AutoAssociativeLearningMechanism.learning_rate>` can be specified as a 1d or 2d array (or
    matrix) to scale the contribution made, respectively, by individual elements or connections among them,
    to the weight change matrix;  as with a standard `LearningMechanism`, a scalar can also be specified to scale
    the entire weight change matrix (see `learning_rate <AutoAssociativeLearningMechanism.learning_rate>` for
    additional details).

.. _AutoAssociativeLearningMechanism_Learning:

Execution
---------

An AutoAssociativeLearningMechanism executes in the same manner as standard `LearningMechanism`, with two exceptions:
* 1) its execution can be enabled or disabled by setting the `learning_enabled
  <RecurrentTransferMechanism.learning_enabled>` attribute of the `RecurrentTransferMechanism` with which it is
  associated (identified in its `activity_source <AutoAssociativeLearningMechanism.activity_source>` attribute).
* 2) it is executed during the `execution phase <System_Execution>` of the System's execution.  Note that this is
  different from the behavior of supervised learning algorithms (such as `Reinforcement` and `BackPropagation`),
  that are executed during the `learning phase <System_Execution>` of a System's execution


.. _AutoAssociativeLearningMechanism_Class_Reference:

Class Reference
---------------

"""

import numpy as np
import typecheck as tc

from psyneulink.core.components.component import parameter_keywords
from psyneulink.core.components.functions.function import ModulationParam, _is_modulation_param, is_function_type
from psyneulink.core.components.functions.learningfunctions import Hebbian
from psyneulink.core.components.mechanisms.adaptive.learning.learningmechanism import ACTIVATION_INPUT, LearningMechanism, LearningTiming, LearningType
from psyneulink.core.components.mechanisms.processing.objectivemechanism import ObjectiveMechanism
from psyneulink.core.components.projections.projection import Projection_Base, projection_keywords
from psyneulink.core.globals.context import ContextFlags
from psyneulink.core.globals.keywords import AUTOASSOCIATIVE_LEARNING_MECHANISM, CONTROL_PROJECTIONS, INPUT_STATES, LEARNING, LEARNING_PROJECTION, LEARNING_SIGNAL, NAME, OUTPUT_STATES, OWNER_VALUE, VARIABLE
from psyneulink.core.globals.parameters import Parameter
from psyneulink.core.globals.preferences.componentpreferenceset import is_pref_set
from psyneulink.core.globals.preferences.preferenceset import PreferenceLevel
from psyneulink.core.globals.utilities import is_numeric, parameter_spec

__all__ = [
    'AutoAssociativeLearningMechanism', 'AutoAssociativeLearningMechanismError', 'DefaultTrainingMechanism',
    'input_state_names', 'output_state_names',
]

# Parameters:

parameter_keywords.update({LEARNING_PROJECTION, LEARNING})
projection_keywords.update({LEARNING_PROJECTION, LEARNING})

input_state_names =  [ACTIVATION_INPUT]
output_state_names = [LEARNING_SIGNAL]

DefaultTrainingMechanism = ObjectiveMechanism

class AutoAssociativeLearningMechanismError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value

    def __str__(self):
        return repr(self.error_value)


class AutoAssociativeLearningMechanism(LearningMechanism):
    """
    AutoAssociativeLearningMechanism(              \
        variable,                                  \
        function=Hebbian,                          \
        learning_rate=None,                        \
        learning_signals=LEARNING_SIGNAL,          \
        modulation=ModulationParam.ADDITIVE,       \
        params=None,                               \
        name=None,                                 \
        prefs=None)

    Implements a `LearningMechanism` that modifies the `matrix <MappingProjection.matrix>` parameter of an
    `AutoAssociativeProjection` (typically the `recurrent_projection <RecurrentTransferMechanism.recurrent_projection>`
    of a `RecurrentTransferMechanism`).


    Arguments
    ---------

    variable : List or 2d np.array : default None
        it must have a single item that corresponds to the value required by the AutoAssociativeLearningMechanism's
        `function <AutoAssociativeLearningMechanism.function>`;  it must each be compatible (in number and type)
        with the `value <InputState.value>` of the Mechanism's `InputState <LearningMechanism_InputStates>` (see
        `variable <AutoAssociativeLearningMechanism.variable>` for additional details).

    learning_signals : List[parameter of Projection, ParameterState, Projection, tuple[str, Projection] or dict] \
    : default None
        specifies the `matrix <AutoAssociativeProjection.matrix>` to be learned (see `learning_signals
        <LearningMechanism.learning_signals>` for details of specification).

    modulation : ModulationParam : default ModulationParam.ADDITIVE
        specifies the default form of modulation used by the AutoAssociativeLearningMechanism's LearningSignals,
        unless they are `individually specified <LearningSignal_Specification>`.

    function : LearningFunction or function : default Hebbian
        specifies the function used to calculate the AutoAssociativeLearningMechanism's `learning_signal
        <AutoAssociativeLearningMechanism.learning_signal>` attribute.  It must take as its **variable** argument a
        list or 1d array of numeric values (the "activity vector") and return a list, 2d np.array or np.matrix
        representing a square matrix with dimensions that equal the length of its variable (the "weight change
        matrix").

    learning_rate : float : default None
        specifies the learning rate for the AutoAssociativeLearningMechanism. (see `learning_rate
        <AutoAssociativeLearningMechanism.learning_rate>` for details).

    params : Dict[param keyword: param value] : default None
        a `parameter dictionary <ParameterState_Specification>` that specifies the parameters for the
        Projection, its function, and/or a custom function and its parameters. By default, it contains an entry for
        the Projection's default `function <LearningProjection.function>` and parameter assignments.  Values specified
        for parameters in the dictionary override any assigned to those parameters in arguments of the constructor.

    name : str : default see `name <AutoAssociativeLearningMechanism.name>`
        specifies the name of the AutoAssociativeLearningMechanism.

    prefs : PreferenceSet or specification dict : default Mechanism.classPreferences
        specifies the `PreferenceSet` for the AutoAssociativeLearningMechanism; see `prefs <AutoAssociativeLearningMechanism.prefs>` for details.


    Attributes
    ----------

    COMMENT:
        componentType : LEARNING_MECHANISM
    COMMENT

    variable : 2d np.array
        has a single item, that serves as the template for the input required by the AutoAssociativeLearningMechanism's
        `function <AutoAssociativeLearningMechanism.function>`, corresponding to the `value
        <OutputState.value>` of the `activity_source <AutoAssociativeLearningMechanism.activity_source>`.

    activity_source : OutputState
        the `OutputState` that is the `sender <AutoAssociativeProjection.sender>` of the `AutoAssociativeProjection`
        that the Mechanism trains.

    input_states : ContentAddressableList[OutputState]
        has a single item, that contains the AutoAssociativeLearningMechanism's single *ACTIVATION_INPUT* `InputState`.

    primary_learned_projection : AutoAssociativeProjection
        the `Projection` with the `matrix <AutoAssociativeProjection.matrix>` parameter being trained by the
        AutoAssociativeLearningMechanism.  It is always the first Projection listed in the
        AutoAssociativeLearningMechanism's `learned_projections <AutoAssociativeLearningMechanism.learned_projections>`
        attribute.

    learned_projections : List[MappingProjection]
        all of the `AutoAssociativeProjections <AutoAssociativeProjection>` modified by the
        AutoAssociativeLearningMechanism;  the first item in the list is always the `primary_learned_projection
        <AutoAssociativeLearningMechanism.primary_learned_projection>`.

    function : LearningFunction or function : default Hebbian
        the function used to calculate the `learning_signal <AutoAssociativeLearningMechanism.learning_signal>`
        (assigned to the AutoAssociativeLearningMechanism's `LearningSignal(s) <LearningMechanism_LearningSignal>`).
        It's `variable <Function_Base.variable>` must be a list or 1d np.array of numeric entries, corresponding in
        length to the AutoAssociativeLearningMechanism's *ACTIVATION_INPUT* (`primary <InputState_Primary>`) InputState.

    learning_rate : float, 1d or 2d np.array, or np.matrix of numeric values : default None
        determines the learning rate used by the AutoAssociativeLearningMechanism's `function
        <AutoAssociativeLearningMechanism.function>` to scale the weight change matrix it returns. If it is a scalar, it is used to multiply the weight change matrix;  if it is a 2d array or matrix,
        it is used to Hadamard (elementwise) multiply the weight matrix (allowing the contribution of individual
        *connections* to be scaled);  if it is a 1d np.array, it is used to Hadamard (elementwise) multiply the input
        to the `function <AutoAssociativeLearningMechanism.function>` (i.e., the `value <InputState.value>` of the
        AutoAssociativeLearningMechanism's *ACTIVATION_INPUT* `InputState <AutoAssociativeLearningMechanism_Structure>`,
        allowing the contribution of individual *units* to be scaled). If specified, the value supersedes the
        learning_rate assigned to any `Process` or `System` to which the AutoAssociativeLearningMechanism belongs.
        If it is `None`, then the `learning_rate <Process.learning_rate>` specified for the Process to which the
        AutoAssociativeLearningMechanism belongs belongs is used;  and, if that is `None`, then the `learning_rate
        <System.learning_rate>` for the System to which it belongs is used. If all are `None`, then the
        `default_learning_rate <LearningFunction.default_learning_rate>` for the `function
        <AutoAssociativeLearningMechanism.function>` is used (see `learning_rate <LearningMechanism_Learning_Rate>`
        for additional details).

    learning_signal : 2d ndarray or matrix of numeric values
        the value returned by `function <AutoAssociativeLearningMechanism.function>`, that specifies
        the changes to the weights of the `matrix <AutoAssociativeProjection.matrix>` parameter for the
        AutoAssociativeLearningMechanism's`learned_projections <AutoAssociativeLearningMechanism.learned_projections>`;
        It is assigned as the value of the AutoAssociativeLearningMechanism's `LearningSignal(s)
        <LearningMechanism_LearningSignal>` and, in turn, its `LearningProjection(s) <LearningProjection>`.

    learning_signals : List[LearningSignal]
        list of all of the `LearningSignals <LearningSignal>` for the AutoAssociativeLearningMechanism, each of which
        sends one or more `LearningProjections <LearningProjection>` to the `ParameterState(s) <ParameterState>` for
        the `matrix <AutoAssociativeProjection.matrix>` parameter of the `AutoAssociativeProjection(s)
        <AutoAssociativeProjection>` trained by the AutoAssociativeLearningMechanism.  Although in most instances an
        AutoAssociativeLearningMechanism is used to train a single AutoAssociativeProjection, like a standard
        `LearningMechanism`, it can be assigned additional LearningSignals and/or LearningProjections to train
        additional ones;  in such cases, the `value <LearningSignal>` for all of the LearningSignals is the
        the same:  the AutoAssociativeLearningMechanism's `learning_signal
        <AutoAssociativeLearningMechanism.learning_signal>` attribute, based on its `activity_source
        <AutoAssociativeLearningMechanism.activity_source>`.  Since LearningSignals are `OutputStates
        <OutputState>`, they are also listed in the AutoAssociativeLearningMechanism's `output_states
        <AutoAssociativeLearningMechanism.output_states>` attribute.

    learning_projections : List[LearningProjection]
        list of all of the LearningProjections <LearningProjection>` from the AutoAssociativeLearningMechanism, listed
        in the order of the `LearningSignals <LearningSignal>` to which they belong (that is, in the order they are
        listed in the `learning_signals <AutoAssociativeLearningMechanism.learning_signals>` attribute).

    output_states : ContentAddressableList[OutputState]
        list of the AutoAssociativeLearningMechanism's `OutputStates <OutputState>`, beginning with its
        `learning_signals <AutoAssociativeLearningMechanism.learning_signals>`, and followed by any additional
        (user-specified) `OutputStates <OutputState>`.

    output_values : 2d np.array
        the first item is the `value <OutputState.value>` of the LearningMechanism's `learning_signal
        <AutoAssociativeLearningMechanism.learning_signal>`, followed by the `value <OutputState.value>`\\s
        of any additional (user-specified) OutputStates.

    modulation : ModulationParam
        the default form of modulation used by the AutoAssociativeLearningMechanism's `LearningSignal(s)
        <LearningMechanism_LearningSignal>`, unless they are `individually specified <LearningSignal_Specification>`.

    name : str
        the name of the AutoAssociativeLearningMechanism; if it is not specified in the **name** argument of the
        constructor, a default is assigned by MechanismRegistry (see `Naming` for conventions used for default and
        duplicate names).

    prefs : PreferenceSet or specification dict
        the `PreferenceSet` for the AutoAssociativeLearningMechanism; if it is not specified in the **prefs** argument
        of the constructor, a default is assigned using `classPreferences` defined in __init__.py (see
        `PreferenceSet <LINK>` for details).
    """

    componentType = AUTOASSOCIATIVE_LEARNING_MECHANISM
    className = componentType
    suffix = " " + className

    class Parameters(LearningMechanism.Parameters):
        """
            Attributes
            ----------

                learning_rate
                    see `learning_rate <AutoAssociativeLearningMechanism.learning_rate>`

                    :default value: None
                    :type:

                learning_signals
                    see `learning_signals <AutoAssociativeLearningMechanism.learning_signals>`

                    :default value: None
                    :type:

                modulation
                    see `modulation <AutoAssociativeLearningMechanism.modulation>`

                    :default value: ModulationParam.ADDITIVE
                    :type: `ModulationParam`

        """
        learning_signals = None
        modulation = ModulationParam.ADDITIVE

    classPreferenceLevel = PreferenceLevel.TYPE

    learning_type = LearningType.UNSUPERVISED
    learning_timing = LearningTiming.EXECUTION_PHASE

    paramClassDefaults = Projection_Base.paramClassDefaults.copy()
    paramClassDefaults.update({
        CONTROL_PROJECTIONS: None,
        INPUT_STATES:input_state_names,
        OUTPUT_STATES:[{NAME:LEARNING_SIGNAL,  # NOTE: This is the default, but is overridden by any LearningSignal arg
                        VARIABLE: (OWNER_VALUE,0)}
                       ]})

    @tc.typecheck
    def __init__(self,
                 default_variable:tc.any(list, np.ndarray),
                 size=None,
                 function:is_function_type=Hebbian,
                 learning_signals:tc.optional(list) = None,
                 modulation:tc.optional(_is_modulation_param)=ModulationParam.ADDITIVE,
                 learning_rate:tc.optional(parameter_spec)=None,
                 params=None,
                 name=None,
                 prefs:is_pref_set=None):

        # Assign args to params and functionParams dicts 
        params = self._assign_args_to_param_dicts(function=function,
                                                  learning_signals=learning_signals,
                                                  params=params)

        # # USE FOR IMPLEMENTATION OF deferred_init()
        # # Store args for deferred initialization
        # self.init_args = locals().copy()
        # self.init_args['context'] = self
        # self.init_args['name'] = name

        # # Flag for deferred initialization
        # self.context.initialization_status = ContextFlags.DEFERRED_INIT
        # self.initialization_status = ContextFlags.DEFERRED_INIT

        # self._learning_rate = learning_rate

        super().__init__(default_variable=default_variable,
                         size=size,
                         function=function,
                         modulation=modulation,
                         learning_rate=learning_rate,
                         params=params,
                         name=name,
                         prefs=prefs,
                         context=ContextFlags.CONSTRUCTOR)

    def _parse_function_variable(self, variable, execution_id=None, context=None):
        return variable
    
    def _instantiate_attributes_after_function(self, context=None):
        super(AutoAssociativeLearningMechanism, self)._instantiate_attributes_after_function(context=context)
        # KAM 2/27/19 added the line below to set the learning rate of the hebbian learning function to the learning
        # rate value passed into RecurrentTransfermechanism
        if self.learning_rate:
            self.function.learning_rate = self.learning_rate

    def _validate_variable(self, variable, context=None):
        """Validate that variable has only one item: activation_input.
        """

        # Skip LearningMechanism._validate_variable in call to super(), as it requires variable to have 3 items
        variable = super(LearningMechanism, self)._validate_variable(variable, context)

        # # MODIFIED 9/22/17 NEW: [HACK] JDC: 6/29/18 -> CAUSES DEFAULT variable [[0]] OR ANYTHING OF size=1 TO FAIL
        # if np.array(np.squeeze(variable)).ndim != 1 or not is_numeric(variable):
        # MODIFIED 6/29/18 NEWER JDC: ALLOW size=1, AND DEFER FAILURE TO LearningFunction IF enbale_learning=True
        if np.array(variable)[0].ndim != 1 or not is_numeric(variable):
        # MODIFIED 9/22/17 END
            raise AutoAssociativeLearningMechanismError("Variable for {} ({}) must be "
                                                        "a list or 1d np.array containing only numbers".
                                                        format(self.name, variable))
        return variable

    def _execute(
        self,
        variable=None,
        execution_id=None,
        runtime_params=None,
        context=None
    ):
        """Execute AutoAssociativeLearningMechanism. function and return learning_signal

        :return: (2D np.array) self.learning_signal
        """

        # COMPUTE LEARNING SIGNAL (note that function is assumed to return only one value)
        # IMPLEMENTATION NOTE:  skip LearningMechanism's implementation of _execute
        #                       as it assumes projections from other LearningMechanisms
        #                       which are not relevant to an autoassociative projection
        learning_signal = super(LearningMechanism, self)._execute(
            variable=variable,
            execution_id=execution_id,
            runtime_params=runtime_params,
            context=context
        )

        if self.parameters.context.get(execution_id).initialization_status != ContextFlags.INITIALIZING and self.reportOutputPref:
            print("\n{} weight change matrix: \n{}\n".format(self.name, self.parameters.learning_signal.get(execution_id)))

        # # TEST PRINT
        # if not self.context.initialization_status == ContextFlags.INITIALIZING:
        #     if self.context.composition:
        #         time = self.context.composition.scheduler_processing.clock.simple_time
        #     else:
        #         time = self.current_execution_time
        #     print("\nEXECUTED AutoAssociative LearningMechanism [CONTEXT: {}]\nTRIAL:  {}  TIME-STEP: {}".
        #         format(self.context.flags_string,
        #                time.trial,
        #                # self.pass_,
        #                time.time_step))
        #     print("{} weight change matrix: \n{}\n".format(self.name, self.learning_signal))

        value = np.array([learning_signal])

        self.parameters.value.set(value, execution_id, override=True)

        return value

    def _update_output_states(self, execution_id=None, runtime_params=None, context=None):
        '''Update the weights for the AutoAssociativeProjection for which this is the AutoAssociativeLearningMechanism

        Must do this here, so it occurs after LearningMechanism's OutputState has been updated.
        This insures that weights are updated within the same trial in which they have been learned
        '''

        super()._update_output_states(execution_id, runtime_params, context)

        from psyneulink.core.components.process import Process
        if self.parameters.learning_enabled.get(execution_id) and self.parameters.context.get(execution_id).composition and not isinstance(self.parameters.context.get(execution_id).composition, Process):
            learned_projection = self.activity_source.recurrent_projection
            learned_projection.execute(execution_id=execution_id, context=ContextFlags.LEARNING)
            learned_projection.parameters.context.get(execution_id).execution_phase = ContextFlags.IDLE

    @property
    def activity_source(self):
        return self.input_state.path_afferents[0].sender.owner
