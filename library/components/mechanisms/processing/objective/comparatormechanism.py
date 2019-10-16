# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

# *********************************************  ComparatorMechanism ***************************************************

"""

Overview
--------

A ComparatorMechanism is a subclass of `ObjectiveMechanism` that receives two inputs (a sample and a target), compares
them using its `function <ComparatorMechanism.function>`, and places the calculated discrepancy between the two in its
*OUTCOME* `OutputState <ComparatorMechanism.output_state>`.

.. _ComparatorMechanism_Creation:

Creating a ComparatorMechanism
------------------------------

ComparatorMechanisms are generally created automatically when other PsyNeuLink components are created (such as
`LearningMechanisms <LearningMechanism_Creation>`).  A ComparatorMechanism can also be created directly by calling
its constructor.  Its **sample** and **target** arguments are used to specify the OutputStates that provide the
sample and target inputs, respectively (see `ObjectiveMechanism_Monitored_States` for details concerning their
specification, which are special versions of an ObjectiveMechanism's **monitor** argument).  When the
ComparatorMechanism is created, two InputStates are created, one each for its sample and target inputs (and named,
by default, *SAMPLE* and *TARGET*). Each is assigned a MappingProjection from the corresponding OutputState specified
in the **sample** and **target** arguments.

It is important to recognize that the value of the *SAMPLE* and *TARGET* InputStates must have the same length and type,
so that they can be compared using the ComparatorMechanism's `function <ComparatorMechanism.function>`.  By default,
they use the format of the OutputStates specified in the **sample** and **target** arguments, respectively,
and the `MappingProjection` to each uses an `IDENTITY_MATRIX`.  Therefore, for the default configuration, the
OutputStates specified in the **sample** and **target** arguments must have values of the same length and type.
If these differ, the **input_states** argument can be used to explicitly specify the format of the ComparatorMechanism's
*SAMPLE* and *TARGET* InputStates, to insure they are compatible with one another (as well as to customize their
names, if desired).  If the **input_states** argument is used, *both* the sample and target InputStates must be
specified.  Any of the formats for `specifying InputStates <InputState_Specification>` can be used in the argument.
If values are assigned for the InputStates, they must be of equal length and type.  Their types must
also be compatible with the value of the OutputStates specified in the **sample** and **target** arguments.  However,
the length specified for an InputState can differ from its corresponding OutputState;  in that case, by default, the
MappingProjection created uses a `FULL_CONNECTIVITY` matrix.  Thus, OutputStates of differing lengths can be mapped
to the sample and target InputStates of a ComparatorMechanism (see the `example <ComparatorMechanism_Example>` below),
so long as the latter are of the same length.  If a projection other than a `FULL_CONNECTIVITY` matrix is needed, this
can be specified using the *PROJECTION* entry of a `State specification dictionary <State_Specification>` for the
InputState in the **input_states** argument.

.. _ComparatorMechanism_Structure:

Structure
---------

A ComparatorMechanism has two `input_states <ComparatorMechanism.input_states>`, each of which receives a
`MappingProjection` from a corresponding OutputState specified in the **sample** and **target** arguments of its
constructor.  The InputStates are listed in the Mechanism's `input_states <ComparatorMechanism.input_States>` attribute
and named, respectively, *SAMPLE* and *TARGET*.  The OutputStates from which they receive their projections (specified
in the **sample** and **target** arguments) are listed in the Mechanism's `sample <ComparatorMechanism.sample>` and
`target <ComparatorMechanism.target>` attributes as well as in its `monitor <ComparatorMechanism.monitor>` attribute.
The ComparatorMechanism's `function <ComparatorMechanism.function>` compares the value of the sample and target
InputStates.  By default, it uses a `LinearCombination` function, assigning the sample InputState a `weight
<LinearCombination.weight>` of *-1* and the target a `weight <LinearCombination.weight>` of *1*, so that the sample
is subtracted from the target.  However, the `function <ComparatorMechanism.function>` can be customized, so long as
it is replaced with one that takes two arrays with the same format as its inputs and generates a similar array as its
result. The result is assigned as the value of the Comparator Mechanism's *OUTCOME* (`primary <OutputState_Primary>`)
OutputState.

.. _ComparatorMechanism_Function:

Execution
---------

When a ComparatorMechanism is executed, it updates its input_states with the values of the OutputStates specified
in its **sample** and **target** arguments, and then uses its `function <ComparatorMechanism.function>` to
compare these.  By default, the result is assigned to the `value <ComparatorMechanism.value>` of its *OUTCOME*
`output_state <ComparatorMechanism.output_state>`, and as the first item of the Mechanism's
`output_values <ComparatorMechanism.output_values>` attribute.

.. _ComparatorMechanism_Example:

Example
-------

.. _ComparatorMechanism_Default_Input_Value_Example:

*Formatting InputState values*

The **default_variable** argument can be used to specify a particular format for the SAMPLE and/or TARGET InputStates
of a ComparatorMechanism.  This can be useful when one or both of these differ from the format of the
OutputState(s) specified in the **sample** and **target** arguments. For example, for `Reinforcement Learning
<Reinforcement>`, a ComparatorMechanism is used to monitor an action selection Mechanism (the sample), and compare
this with a reinforcement signal (the target).  In the example below, the action selection Mechanism is a
`TransferMechanism` that uses the `SoftMax` function (and the `PROB <Softmax.PROB>` as its output format) to select
an action.  This generates a vector with a single non-zero value (the selected action). Because the output is a vector,
specifying it as the ComparatorMechanism's **sample** argument will generate a corresponding InputState with a vector
as its value.  This will not match the reward signal specified in the ComparatorMechanism's **target** argument, the
value of which is a single scalar.  This can be dealt with by explicitly specifying the format for the SAMPLE and
TARGET InputStates in the **default_variable** argument of the ComparatorMechanism's constructor, as follows::

    >>> import psyneulink as pnl
    >>> my_action_selection_mech = pnl.TransferMechanism(size=5,
    ...                                                  function=pnl.SoftMax(output=pnl.PROB))

    >>> my_reward_mech = pnl.TransferMechanism()

    >>> my_comparator_mech = pnl.ComparatorMechanism(default_variable = [[0],[0]],
    ...                                              sample=my_action_selection_mech,
    ...                                              target=my_reward_mech)

Note that ``my_action_selection_mechanism`` is specified to take an array of length 5 as its input, and therefore
generate one of the same length as its `primary output <OutputState_Primary>`.  Since it is assigned as the **sample**
of the ComparatorMechanism, by default this will create a *SAMPLE* InputState of length 5, that will not match the
length of the *TARGET* InputState (the default for which is length 1).  This is taken care of, by specifying the
**default_variable** argument as an array with two single-value arrays (corresponding to the *SAMPLE* and *TARGET*
InputStates). (In this example, the **sample** and **target** arguments are specified as Mechanisms since,
by default, each has only a single (`primary <OutputState_Primary>`) OutputState, that will be used;  if either had
more than one OutputState, and one of those was desired, it would have had to be specified explicitly in the
**sample** or **target** argument).

.. _ComparatorMechanism_Class_Reference:

Class Reference
---------------

"""

from collections import Iterable

import numpy as np
import typecheck as tc

from psyneulink.core.components.functions.combinationfunctions import LinearCombination
from psyneulink.core.components.mechanisms.mechanism import Mechanism_Base
from psyneulink.core.components.mechanisms.processing.objectivemechanism import ObjectiveMechanism
from psyneulink.core.components.shellclasses import Mechanism
from psyneulink.core.components.states.inputstate import InputState
from psyneulink.core.components.states.outputstate import OutputState, PRIMARY, StandardOutputStates
from psyneulink.core.components.states.state import _parse_state_spec
from psyneulink.core.globals.context import ContextFlags
from psyneulink.core.globals.keywords import COMPARATOR_MECHANISM, FUNCTION, INPUT_STATES, NAME, OUTCOME, SAMPLE, TARGET, VARIABLE, kwPreferenceSetName
from psyneulink.core.globals.parameters import Parameter
from psyneulink.core.globals.preferences.componentpreferenceset import is_pref_set, kpReportOutputPref
from psyneulink.core.globals.preferences.preferenceset import PreferenceEntry, PreferenceLevel
from psyneulink.core.globals.utilities import is_numeric, is_value_spec, iscompatible, kwCompatibilityLength, kwCompatibilityNumeric, recursive_update
from psyneulink.core.globals.utilities import safe_len

__all__ = [
    'COMPARATOR_OUTPUT', 'ComparatorMechanism', 'ComparatorMechanismError', 'MSE', 'SSE',
]

SSE = 'SSE'
MSE = 'MSE'


class COMPARATOR_OUTPUT():
    """
    .. _ComparatorMechanism_Standard_OutputStates:

    `Standard OutputStates <OutputState_Standard>` for `ComparatorMechanism`

    .. _COMPARATOR_MECHANISM_SSE

    *SSE*
        the value of the sum squared error of the Mechanism's function

    .. _COMPARATOR_MECHANISM_MSE

    *MSE*
        the value of the mean squared error of the Mechanism's function

    """
    SSE = SSE
    MSE = MSE


class ComparatorMechanismError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value

    def __str__(self):
        return repr(self.error_value)


class ComparatorMechanism(ObjectiveMechanism):
    """
    ComparatorMechanism(                                \
        sample,                                         \
        target,                                         \
        input_states=[SAMPLE,TARGET]                    \
        function=LinearCombination(weights=[[-1],[1]],  \
        output_states=OUTCOME                           \
        params=None,                                    \
        name=None,                                      \
        prefs=None)

    Subclass of `ObjectiveMechanism` that compares the values of two `OutputStates <OutputState>`.

    COMMENT:
        Description:
            ComparatorMechanism is a subtype of the ObjectiveMechanism Subtype of the ProcssingMechanism Type
            of the Mechanism Category of the Component class.
            By default, it's function uses the LinearCombination Function to compare two input variables.
            COMPARISON_OPERATION (functionParams) determines whether the comparison is subtractive or divisive
            The function returns an array with the Hadamard (element-wise) differece/quotient of target vs. sample,
                as well as the mean, sum, sum of squares, and mean sum of squares of the comparison array

        Class attributes:
            + componentType (str): ComparatorMechanism
            + classPreference (PreferenceSet): Comparator_PreferenceSet, instantiated in __init__()
            + classPreferenceLevel (PreferenceLevel): PreferenceLevel.SUBTYPE
            + class_defaults.variable (value):  Comparator_DEFAULT_STARTING_POINT // QUESTION: What to change here
            + paramClassDefaults (dict): {FUNCTION_PARAMS:{COMPARISON_OPERATION: SUBTRACTION}}

        Class methods:
            None

        MechanismRegistry:
            All instances of ComparatorMechanism are registered in MechanismRegistry, which maintains an
              entry for the subclass, a count for all instances of it, and a dictionary of those instances
    COMMENT

    Arguments
    ---------

    sample : OutputState, Mechanism, value, or string
        specifies the value to compare with the `target` by the `function <ComparatorMechanism.function>`.

    target :  OutputState, Mechanism, value, or string
        specifies the value with which the `sample` is compared by the `function <ComparatorMechanism.function>`.

    input_states :  List[InputState, value, str or dict] or Dict[] : default [SAMPLE, TARGET]
        specifies the names and/or formats to use for the values of the sample and target InputStates;
        by default they are named *SAMPLE* and *TARGET*, and their formats are match the value of the OutputStates
        specified in the **sample** and **target** arguments, respectively (see `ComparatorMechanism_Structure`
        for additional details).

    function :  Function, function or method : default Distance(metric=DIFFERENCE)
        specifies the `function <Comparator.function>` used to compare the `sample` with the `target`.

    output_states :  List[OutputState, value, str or dict] or Dict[] : default [OUTCOME]
        specifies the OutputStates for the Mechanism;

    params :  Optional[Dict[param keyword: param value]]
        a `parameter dictionary <ParameterState_Specification>` that can be used to specify the parameters for
        the Mechanism, its function, and/or a custom function and its parameters. Values specified for parameters in
        the dictionary override any assigned to those parameters in arguments of the
        constructor.

    name : str : default see `name <ComparatorMechanism.name>`
        specifies the name of the ComparatorMechanism.

    prefs : PreferenceSet or specification dict : default Mechanism.classPreferences
        specifies the `PreferenceSet` for the ComparatorMechanism; see `prefs <ComparatorMechanism.prefs>` for details.


    Attributes
    ----------

    COMMENT:
    default_variable : Optional[List[array] or 2d np.array]
    COMMENT

    sample : OutputState
        determines the value to compare with the `target` by the `function <ComparatorMechanism.function>`.

    target : OutputState
        determines the value with which `sample` is compared by the `function <ComparatorMechanism.function>`.

    input_states : ContentAddressableList[InputState, InputState]
        contains the two InputStates named, by default, *SAMPLE* and *TARGET*, each of which receives a
        `MappingProjection` from the OutputStates referenced by the `sample` and `target` attributes
        (see `ComparatorMechanism_Structure` for additional details).

    function : CombinationFunction, function or method
        used to compare the `sample` with the `target`.  It can be any PsyNeuLink `CombinationFunction`,
        or a python function that takes a 2d array with two items and returns a 1d array of the same length
        as the two input items.

    value : 1d np.array
        the result of the comparison carried out by the `function <ComparatorMechanism.function>`.

    output_state : OutputState
        contains the `primary <OutputState_Primary>` OutputState of the ComparatorMechanism; the default is
        its *OUTCOME* OutputState, the value of which is equal to the `value <ComparatorMechanism.value>`
        attribute of the ComparatorMechanism.

    output_states : ContentAddressableList[OutputState]
        contains, by default, only the *OUTCOME* (primary) OutputState of the ComparatorMechanism.

    output_values : 2d np.array
        contains one item that is the value of the *OUTCOME* OutputState.

    name : str
        the name of the ComparatorMechanism; if it is not specified in the **name** argument of the constructor, a
        default is assigned by MechanismRegistry (see `Naming` for conventions used for default and duplicate names).

    prefs : PreferenceSet or specification dict
        the `PreferenceSet` for the ComparatorMechanism; if it is not specified in the **prefs** argument of the
        constructor, a default is assigned using `classPreferences` defined in __init__.py (see :doc:`PreferenceSet
        <LINK>` for details).


    """
    componentType = COMPARATOR_MECHANISM

    classPreferenceLevel = PreferenceLevel.SUBTYPE
    # These will override those specified in TypeDefaultPreferences
    classPreferences = {
        kwPreferenceSetName: 'ComparatorCustomClassPreferences',
        kpReportOutputPref: PreferenceEntry(False, PreferenceLevel.INSTANCE)}

    class Parameters(ObjectiveMechanism.Parameters):
        """
            Attributes
            ----------

                variable
                    see `variable <ComparatorMechanism.variable>`

                    :default value: numpy.array([[0], [0]])
                    :type: numpy.ndarray
                    :read only: True

                function
                    see `function <ComparatorMechanism.function>`

                    :default value: `LinearCombination`(offset=0.0, operation=sum, scale=1.0, weights=numpy.array([[-1], [ 1]]))
                    :type: `Function`

                sample
                    see `sample <ComparatorMechanism.sample>`

                    :default value: None
                    :type:

                target
                    see `target <ComparatorMechanism.target>`

                    :default value: None
                    :type:

        """
        # By default, ComparatorMechanism compares two 1D np.array input_states
        variable = Parameter(np.array([[0], [0]]), read_only=True)
        function = Parameter(LinearCombination(weights=[[-1], [1]]), stateful=False, loggable=False)
        sample = None
        target = None

    # ComparatorMechanism parameter and control signal assignments):
    paramClassDefaults = Mechanism_Base.paramClassDefaults.copy()

    standard_output_states = ObjectiveMechanism.standard_output_states.copy()

    standard_output_states.extend([{NAME: SSE,
                                    FUNCTION: lambda x: np.sum(x*x)},
                                   {NAME: MSE,
                                    FUNCTION: lambda x: np.sum(x * x) / safe_len(x)}])

    @tc.typecheck
    def __init__(self,
                 default_variable=None,
                 sample: tc.optional(tc.any(OutputState, Mechanism_Base, dict, is_numeric, str))=None,
                 target: tc.optional(tc.any(OutputState, Mechanism_Base, dict, is_numeric, str))=None,
                 function=LinearCombination(weights=[[-1], [1]]),
                 output_states:tc.optional(tc.any(str, Iterable))=(OUTCOME,),
                 params=None,
                 name=None,
                 prefs:is_pref_set=None,
                 **input_states # IMPLEMENTATION NOTE: this is for backward compatibility
                 ):

        input_states = self._merge_legacy_constructor_args(sample, target, default_variable, input_states)

        # Default output_states is specified in constructor as a tuple rather than a list
        # to avoid "gotcha" associated with mutable default arguments
        # (see: bit.ly/2uID3s3 and http://docs.python-guide.org/en/latest/writing/gotchas/)
        if isinstance(output_states, (str, tuple)):
            output_states = list(output_states)

        # IMPLEMENTATION NOTE: The following prevents the default from being updated by subsequent assignment
        #                     (in this case, to [OUTCOME, {NAME= MSE}]), but fails to expose default in IDE
        # output_states = output_states or [OUTCOME, MSE]

        # Create a StandardOutputStates object from the list of stand_output_states specified for the class
        if not isinstance(self.standard_output_states, StandardOutputStates):
            self.standard_output_states = StandardOutputStates(self,
                                                               self.standard_output_states,
                                                               indices=PRIMARY)

        super().__init__(# monitor=[sample, target],
                         monitor=input_states,
                         function=function,
                         output_states=output_states.copy(), # prevent default from getting overwritten by later assign
                         params=params,
                         name=name,
                         prefs=prefs,
                         context=ContextFlags.CONSTRUCTOR)

    def _validate_params(self, request_set, target_set=None, context=None):
        """If sample and target values are specified, validate that they are compatible
        """

        if INPUT_STATES in request_set and request_set[INPUT_STATES] is not None:
            input_states = request_set[INPUT_STATES]

            # Validate that there are exactly two input_states (for sample and target)
            num_input_states = len(input_states)
            if num_input_states != 2:
                raise ComparatorMechanismError("{} arg is specified for {} ({}), so it must have exactly 2 items, "
                                               "one each for {} and {}".
                                               format(INPUT_STATES,
                                                      self.__class__.__name__,
                                                      len(input_states),
                                                      SAMPLE,
                                                      TARGET))

            # Validate that input_states are specified as dicts
            if not all(isinstance(input_state,dict) for input_state in input_states):
                raise ComparatorMechanismError("PROGRAM ERROR: all items in input_state args must be converted to dicts"
                                               " by calling State._parse_state_spec() before calling super().__init__")

            # Validate length of variable for sample = target
            if VARIABLE in input_states[0]:
                # input_states arg specified in standard state specification dict format
                lengths = [len(input_state[VARIABLE]) for input_state in input_states]
            else:
                # input_states arg specified in {<STATE_NAME>:<STATE SPECIFICATION DICT>} format
                lengths = [len(list(input_state_dict.values())[0][VARIABLE]) for input_state_dict in input_states]

            if lengths[0] != lengths[1]:
                raise ComparatorMechanismError("Length of value specified for {} InputState of {} ({}) must be "
                                               "same as length of value specified for {} ({})".
                                               format(SAMPLE,
                                                      self.__class__.__name__,
                                                      lengths[0],
                                                      TARGET,
                                                      lengths[1]))

        elif SAMPLE in request_set and TARGET in request_set:

            sample = request_set[SAMPLE]
            if isinstance(sample, InputState):
                sample_value = sample.value
            elif isinstance(sample, Mechanism):
                sample_value = sample.input_value[0]
            elif is_value_spec(sample):
                sample_value = sample
            else:
                sample_value = None

            target = request_set[TARGET]
            if isinstance(target, InputState):
                target_value = target.value
            elif isinstance(target, Mechanism):
                target_value = target.input_value[0]
            elif is_value_spec(target):
                target_value = target
            else:
                target_value = None

            if sample is not None and target is not None:
                if not iscompatible(sample, target, **{kwCompatibilityLength: True,
                                                       kwCompatibilityNumeric: True}):
                    raise ComparatorMechanismError("The length of the sample ({}) must be the same as for the target ({})"
                                                   "for {} {}".
                                                   format(len(sample),
                                                          len(target),
                                                          self.__class__.__name__,
                                                          self.name))

        super()._validate_params(request_set=request_set,
                                 target_set=target_set,
                                 context=context)

    def _merge_legacy_constructor_args(self, sample, target, default_variable=None, input_states=None):

        # USE sample and target TO CREATE AN InputState specfication dictionary for each;
        # DO SAME FOR InputStates argument, USE TO OVERWRITE ANY SPECIFICATIONS IN sample AND target DICTS
        # TRY tuple format AS WAY OF PROVIDED CONSOLIDATED variable AND OutputState specifications

        sample_dict = _parse_state_spec(owner=self,
                                        state_type=InputState,
                                        state_spec=sample,
                                        name=SAMPLE)

        target_dict = _parse_state_spec(owner=self,
                                        state_type=InputState,
                                        state_spec=target,
                                        name=TARGET)

        # If either the default_variable arg or the input_states arg is provided:
        #    - validate that there are exactly two items in default_variable or input_states list
        #    - if there is an input_states list, parse it and use it to update sample and target dicts
        if input_states:
            input_states = input_states[INPUT_STATES]
            # print("type input_states = {}".format(type(input_states)))
            if not isinstance(input_states, list):
                raise ComparatorMechanismError("If an \'{}\' argument is included in the constructor for a {} "
                                               "it must be a list with two {} specifications.".
                                               format(INPUT_STATES, ComparatorMechanism.__name__, InputState.__name__))

        input_states = input_states or default_variable

        if input_states is not None:
            if len(input_states)!=2:
                raise ComparatorMechanismError("If an \'input_states\' arg is "
                                               "included in the constructor for "
                                               "a {}, it must be a list with "
                                               "exactly two items (not {})".
                                               format(ComparatorMechanism.__name__, len(input_states)))

            sample_input_state_dict = _parse_state_spec(owner=self,
                                                        state_type=InputState,
                                                        state_spec=input_states[0],
                                                        name=SAMPLE,
                                                        value=None)

            target_input_state_dict = _parse_state_spec(owner=self,
                                                        state_type=InputState,
                                                        state_spec=input_states[1],
                                                        name=TARGET,
                                                        value=None)

            sample_dict = recursive_update(sample_dict, sample_input_state_dict)
            target_dict = recursive_update(target_dict, target_input_state_dict)

        return [sample_dict, target_dict]
