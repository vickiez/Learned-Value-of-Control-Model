# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# ********************************************  NeuralNetCFA ***************************************************

"""

NeuralNetCFA
-------------

A `NeuralNetCFA` is a subclass of `CompositionFunctionApproximator` that parameterizes a set of `regression_weights
<NeuralNetCFA.regression_weights>` over trials to predict the `net_outcome <ControlMechanism.net_outcome>` for a
`Composition` (or part of one) controlled by an `OptimizationControlMechanism`.

Its `adapt <CompositionFunctionApproximator.adapt>` method updates its `regression_weights
<NeuralNetCFA.regression_weights>`, based on a set of `feature_values <OptimizationControlMechanism.feature_values>`,
a `control_allocation <ControlMechanism.control_allocation>`, and the `net_outcome <ControlMechanism.net_outcome>`
they produced, passed to it from an `OptimizationControlMechanism`.

Its `evaluate <CompositionFunctionApproximator.evaluate>` method calls its `update_weights
<NeuralNetCFA.regresssion_function>` to generate and return a predicted `net_outcome
<ControlMechanism.net_outcome>` for a given set of `feature_values <OptimizationControlMechanism.feature_values>`
and a `control_allocation <ControlMechanism.control_allocation>` provided by an `OptimizationControlMechanism`.

COMMENT:
.. note::
  The NeuralNetCFA's `update_weights <NeuralNetCFA.update_weights>` is provided the `feature_values
  <OptimizationControlMechanism.feature_values>` and `net_outcome <ControlMechanism.net_outcome>` from the
  *previous* trial to update its parameters.  Those are then used to determine the `control_allocation
  <ControlMechanism.control_allocation>` predicted to yield the greatest `EVC <OptimizationControlMechanism_EVC>`
  based on the `feature_values <OptimizationControlMechanism.feature_values>` for the current trial.
COMMENT

"""
import itertools
import numpy as np
import typecheck as tc
# VZ CHANGE
import torch
import torch.nn.functional as F

from enum import Enum
from itertools import product

from psyneulink.core.components.functions.learningfunctions import BayesGLM
from psyneulink.core.components.states.modulatorysignals.controlsignal import ControlSignal
from psyneulink.core.components.states.state import _parse_state_spec
from psyneulink.core.compositions.compositionfunctionapproximator import CompositionFunctionApproximator
from psyneulink.core.globals.keywords import ALL, CONTROL_SIGNALS, DEFAULT_VARIABLE, VARIABLE
from psyneulink.core.globals.parameters import Parameter
from psyneulink.core.globals.utilities import get_deepcopy_with_shared, powerset, tensor_power

__all__ = ['PREDICTION_TERMS', 'PV', 'NeuralNetCFA']

REGRESSION_WEIGHTS = 'REGRESSION_WEIGHTS'
PREDICTION_TERMS = 'prediction_terms'
PREDICTION_WEIGHT_PRIORS = 'prediction_weight_priors'


class PV(Enum):
# class PV(AutoNumberEnum):
    '''PV()
    Specifies terms used to compute `vector <PredictionVector.vector>` attribute of `PredictionVector`.

    Attributes
    ----------

    F
        Main effect of `feature_values <OptimizationControlMechanism_Feature_values>`.
    C
        Main effect of `values <ControlSignal.value>` of `control_signals <ControlMechanism.control_signals>`.
    FF
        Interaction among `feature_values <OptimizationControlMechanism_Feature_values>`.
    CC
        Interaction among `values <ControlSignal.value>` of `control_signals <ControlMechanism.control_signals>`.
    FC
        Interaction between `feature_values <OptimizationControlMechanism_Feature_values>` and
        `values <ControlSignal.value>` of `control_signals <ControlMechanism.control_signals>`.
    FFC
        Interaction between interactions of `feature_values
        <OptimizationControlMechanism_Feature_values>` and `values <ControlSignal.value>` of
        `control_signals <ControlMechanism.control_signals>`.
    FCC
        Interaction between `feature_values <OptimizationControlMechanism_Feature_values>` and
        interactions among `values <ControlSignal.value>` of `control_signals <ControlMechanism.control_signals>`.
    FFCC
        Interaction between interactions of `feature_values
        <OptimizationControlMechanism_Feature_values>` and interactions among `values
        <ControlSignal.value>` of `control_signals <ControlMechanism.control_signals>`.
    COST
        Main effect of `costs <ControlSignal.cost>` of `control_signals <ControlMechanism.control_signals>`.
    '''
    # F =    auto()
    # C =    auto()
    # FF =   auto()
    # CC =   auto()
    # FC =   auto()
    # FFC =  auto()
    # FCC =  auto()
    # FFCC = auto()
    # COST = auto()
    F =    0
    C =    1
    FF =   2
    CC =   3
    FC =   4
    FFC =  5
    FCC =  6
    FFCC = 7
    COST = 8


class NeuralNetCFAError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value


# VZ CHANGE - new class
class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)   # hidden layer
        self.predict = torch.nn.Linear(n_hidden, n_output)   # output layer

    def forward(self, x):
        x = F.relu(self.hidden(x))      # activation function for hidden layer
        x = self.predict(x)             # linear output
        return x


    # to use as x in run_net
    def get_nn_features(self,features):
        return torch.from_numpy(features)


    # to use as y in run_net
    def get_nn_output(self,net_outcome):
        return net_outcome[0][0]


    def run_net(self,x,y):
        for t in range(6):
            prediction = self.forward(x)
            print("prediction: ", prediction)
            loss_func = torch.nn.MSELoss()
            loss = loss_func(prediction, y)

            optimizer = torch.optim.SGD(self.parameters(), lr=0.1)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(self.predict.weight)
        print("final: ",self.predict.weight)
        return self.predict.weight.detach().numpy()[0]


class NeuralNetCFA(CompositionFunctionApproximator):
    '''Parameterizes weights of a `update_weights <RegressorCFA.update_weights>` used by its `evaluate
    <CompositionFunctionApproximator.evaluate>` method to predict the `net_outcome <ControlMechanism.net_outcome>`
    for a `Composition` (or part of one) controlled by an `OptimiziationControlMechanism`, from a set of `feature_values
    <OptimizationControlMechanism.feature_values>` and a `control_allocation <ControlMechanism.control_allocation>`
    provided by the OptimiziationControlMechanism.

    The `feature_values <OptimiziationControlMechanism.feature_values>` and `control_allocation
    <ControlMechanism.control_allocation>` passed to the RegressorCFA's `adapt <RegressorCFA.adapt>` method,
    and provided as the input to its `update_weights <RegressorCFA.update_weights>`, are represented in the
    `vector <PredictionVector.vector>` attribute of a `PredictionVector` assigned to the RegressorCFA`s
    `prediction_vector <RegressorCFA.prediction_vector>` attribute.  The  `feature_values
    <OptimizationControlMechanism.feature_values>` are assigned to the features field of the
    `prediction_vector <RegressorCFA.prediction_vector>`, and the `control_allocation
    <ControlMechanism_control_allocation>` is assigned to the control_allocation field of the `prediction_vector
    <RegressorCFA.prediction_vector>`.  The `prediction_vector <RegressorCFA.prediction_vector>` may also contain
    fields for the `costs ControlMechanism.costs` associated with the `control_allocation
    <ControlMechanism.control_allocation>` and for interactions among those terms.

    The `regression_weights <RegressorCFA.regression_weights>` returned by the `update_weights
    <RegressorCFA.update_weights>` are used by the RegressorCFA's `evaluate <RegressorCFA.evaluate>` method to
    predict the `net_outcome <ControlMechanism.net_outcome>` from the
    `prediction_vector <RegressorCFA.prediction_vector>`.

    '''

    class Parameters(CompositionFunctionApproximator.Parameters):
        """
            Attributes
            ----------

                prediction_vector
                    see `prediction_vector <NeuralNetCFA.prediction_vector>`

                    :default value: None
                    :type:

                previous_state
                    see `previous_state <NeuralNetCFA.previous_state>`

                    :default value: None
                    :type:

                regression_weights
                    see `regression_weights <NeuralNetCFA.regression_weights>`

                    :default value: None
                    :type:

                update_weights
                    see `update_weights <NeuralNetCFA.update_weights>`

                    :default value: `BayesGLM`
                    :type: `Function`

        """
        # VZ CHANGE
        # update_weights = None
        update_weights = Parameter(BayesGLM, stateful=False, loggable=False)
        prediction_vector = None
        previous_state = None
        regression_weights = None
        # neural_net = None


    def __init__(self,
                 name=None,
                 update_weights=BayesGLM,
                 prediction_terms:tc.optional(list)=None):
        # VZ CHANGE ^ removed "update_weights=BayesGLM," from args
        '''

        Arguments
        ---------

        update_weights : LearningFunction, function or method : default BayesGLM
            parameterizes the `regression_weights <NeuralNetCFA.regression_weights>` used by the `evaluate
            <NeuralNetCFA.evaluate>` method to improve its prediction of `net_outcome <ControlMechanism.net_outcome>`
            from a given set of `feature_values <OptimiziationControlMechanism.feature_values>` and a
            `control_allocation <ControlMechanism.control_allocation>` provided by an `OptimiziationControlMechanism`.
            It must take a 2d array as its first argument, the first item of which is an array the same length of the
            `vector <PredictionVector.prediction_vector>` attribute of its `prediction_vector
            <RegressorCFA.prediction_vector>`, and the second item a 1d array containing a scalar
            value that it tries predict.

        prediction_terms : List[PV] : default [PV.F, PV.C, PV.COST]
            terms to be included in (and thereby determines the length of) the `vector
            <PredictionVector.prediction_vector>` attribute of the `prediction_vector <RegressorCFA.prediction_vector>`;
            items are members of the `PV` enum; the default is [`F <PV.F>`, `C <PV.C>` `FC <PV.FC>`, `COST <PV.COST>`].
            If `None` is specified, the default values will automatically be assigned.

        Attributes
        ----------

        update_weights : LearningFunction, function or method
            parameterizes the `regression_weights <NeuralNetCFA.regression_weights>` used by the `evaluate
            <NeuralNetCFA.evaluate>` method to improve prediction of `net_outcome <ControlMechanism.net_outcome>`
            from a given set of `feature_values <OptimiziationControlMechanism.feature_values>` and a
            `control_allocation <ControlMechanism.control_allocation>` provided by an `OptimiziationControlMechanism`;
            its result is assigned as the value of the `regression_weights <RegressorCFA.regression_weights>` attribute.

        prediction_terms : List[PV]
            terms included in `vector <PredictionVector.prediction_vector>` attribute of the
            `prediction_vector <RegressorCFA.prediction_vector>`;  items are members of the `PV` enum; the
            default is [`F <PV.F>`, `C <PV.C>` `FC <PV.FC>`, `COST <PV.COST>`].

        prediction_vector : PredictionVector
            represents and manages values in its `vector <PredictionVector.vector>` attribute that are used by
            `evaluate <RegressorCFA.evaluate>`, along with `regression_weights <RegressorCFA.regression_weights>` to
            make its prediction.  The values contained in the `vector <PredictionVector.vector>` attribute are
            determined by `prediction_terms <RegressorCFA.prediction_terms>`.

        regression_weights : 1d array
            result returned by `update_weights <RegressorCFA.update_weights>, and used by
            `evaluate <RegressorCFA.evaluate>` method together with `prediction_vector <RegressorCFA.prediction_vector>`
            to generate predicted `net_outcome <OptimiziationControlMechanism.net_outcome>`.
        '''

        # VZ CHANGE vv
        # self.update_weights = update_weights

        self._instantiate_prediction_terms(prediction_terms)
        # VZ CHANGE 
        self.neural_net = Net(n_feature=17, n_hidden=17, n_output=1)
        super().__init__(name=name)

        # VZ CHANGE
        # super().__init__(name=name, update_weights=update_weights)

    def _instantiate_prediction_terms(self, prediction_terms):

        # # MODIFIED 11/9/18 OLD:
        # prediction_terms = prediction_terms or [PV.F,PV.C,PV.FC, PV.COST]
        # if ALL in prediction_terms:
        #     prediction_terms = list(PV.__members__.values())
        # MODIFIED 11/9/18 NEW: [JDC]
        # FIX: FOR SOME REASON prediction_terms ARE NOT GETTING PASSED INTACT (ARE NOT RECOGNIZED IN AS MEMBERS OF PV)
        #      AND SO THEY'RE FAILING IN _validate_params
        #      EVEN THOUGH THEY ARE FINE UNDER EXACTLY THE SAME CONDITIONS IN LVOCCONTROLMECHANISM
        #      THIS IS A HACK TO FIX THE PROBLEM:
        if prediction_terms:
            if ALL in prediction_terms:
                self.prediction_terms = list(PV.__members__.values())
            else:
                terms = prediction_terms.copy()
                self.prediction_terms = []
                for term in terms:
                    self.prediction_terms.append(PV[term.name])
        # MODIFIED 11/9/18 END
            for term in self.prediction_terms:
                if not term in PV:
                    raise NeuralNetCFAError("{} specified in {} arg of {} is not a member of the {} enum".
                                                    format(repr(term.name),repr(PREDICTION_TERMS),
                                                           self.__class__.__name__, PV.__name__))
        else:
            self.prediction_terms = [PV.F,PV.C,PV.COST]


    # def initialize(self, owner):
    def initialize(self, features_array, control_signals):
        '''Assign owner and instantiate `prediction_vector <RegressorCFA.prediction_vector>`

        Must be called before RegressorCFA's methods can be used.
        '''

        prediction_terms = self.prediction_terms
        self.prediction_vector = self.PredictionVector(features_array, control_signals, prediction_terms)


        # Assign parameters to update_weights
        update_weights_default_variable = [self.prediction_vector.vector, np.zeros(1)]
        if isinstance(self.update_weights, type):
            self.update_weights = \
                self.update_weights(default_variable=update_weights_default_variable)
        else:
            self.update_weights.reinitialize({DEFAULT_VARIABLE: update_weights_default_variable})


    # # private class for neural net; add file to this class and initialize it to be a global neural net
    # # to which you feed next trial input iteratively 
    # class NeuralNet():


    # # private function, use in place of update_weights BayesGLM
    # def neuralnet_weights(inputs):
    #     features = inputs[0]
    #     target = inputs[1]


    def adapt(self, feature_values, control_allocation, net_outcome, execution_id=None):
        '''Update `regression_weights <RegressorCFA.regression_weights>` so as to improve prediction of
        **net_outcome** from **feature_values** and **control_allocation**.'''
        # VZ CHANGE
        print("my neural net: ", self.neural_net)

        print("adapt fed net outcome: ", net_outcome)
        print("de-array: ", net_outcome[0][0])

        prediction_vector = self.parameters.prediction_vector.get(execution_id)
        previous_state = self.parameters.previous_state.get(execution_id)
        print("previous state: ", previous_state)

        output = torch.tensor(net_outcome[0], dtype=torch.float32)
        print("output tensor: ", output)
        print("output tensor: ", output.type())

        if previous_state is not None:
            # Update regression_weights
            # VZ CHANGE
            # regression_weights = self.update_weights([previous_state, net_outcome], execution_id=execution_id)
            # regression_weights = get_weights(torch.from_numpy(previous_state), net_outcome[0][0])
            features = torch.from_numpy(previous_state)
            features.to(torch.float32)
            print("feat tensor: ", features)
            print("feat tensor: ", features.type())
            regression_weights = Net.run_net(self.neural_net, features.float(), output)
            print("NET COMPUTED W: ", regression_weights)
            # Update vector with current feature_values and control_allocation and store for next trial
            prediction_vector.update_vector(control_allocation, feature_values, execution_id)
            previous_state = prediction_vector.vector
        else:
            # Initialize vector and control_signals on first trial
            # Note:  initialize vector to 1's so that learning_function returns specified priors
            # FIX: 11/9/19 LOCALLY MANAGE STATEFULNESS OF ControlSignals AND costs
            # prediction_vector.reference_variable = control_allocation
            previous_state = np.full_like(prediction_vector.vector, 0)
            features = torch.from_numpy(previous_state)
            features.to(torch.float32)
            print("feat tensor: ", features)
            print("feat tensor: ", features.type())
            # print("prev state yo: ", previous_state)
            # VZ CHANGE
            regression_weights = Net.run_net(self.neural_net, features.float(), output)
            print("NET COMPUTED W: ", regression_weights)
            # regression_weights = get_weights(torch.from_numpy(previous_state), net_outcome[0][0])
            # regression_weights = self.update_weights([previous_state, 0], execution_id=execution_id)

        self._set_multiple_parameter_values(
            execution_id,
            previous_state=previous_state,
            prediction_vector=prediction_vector,
            regression_weights=regression_weights,
            override=True
        )

    # FIX: RENAME AS _EXECUTE_AS_REP ONCE SAME IS DONE FOR COMPOSITION
    # def evaluate(self, control_allocation, num_samples, reinitialize_values, feature_values, context):
    def evaluate(self, feature_values, control_allocation, num_estimates, context, execution_id=None):
        '''Update prediction_vector <RegressorCFA.prediction_vector>`,
        then multiply by regression_weights.

        Uses the current values of `regression_weights <RegressorCFA.regression_weights>` together with
        values of **control_allocation** and **feature_values** arguments to generate predicted `net_outcome
        <OptimiziationControlMechanism.net_outcome>`.

        .. note::
            If this method is assigned as the `objective_funtion of a `GradientOptimization` `Function`,
            it is differentiated using `autograd <https://github.com/HIPS/autograd>`_\\.grad().
        '''

        predicted_outcome=0

        # # VZ
        # print("Control alloc: ", control_allocation)

        prediction_vector = self.parameters.prediction_vector.get(execution_id)
        count = num_estimates if num_estimates else 1
        # print("count: ", count)
        for i in range(count):
            terms = self.prediction_terms
            vector = prediction_vector.compute_terms(control_allocation, execution_id=execution_id)
            # FIX: THIS SHOULD GET A SAMPLE RATHER THAN JUST USE THE ONE RETURNED FROM ADAPT METHOD
            #      OR SHOULD MULTIPLE SAMPLES BE DRAWN AND AVERAGED AT END OF ADAPT METHOD?
            #      I.E., AVERAGE WEIGHTS AND THEN OPTIMIZE OR OTPIMZE FOR EACH SAMPLE OF WEIGHTS AND THEN AVERAGE?

            weights = self.parameters.regression_weights.get(execution_id)
            print('weights: ', weights)
            net_outcome = 0

            # #VZ
            # print("CFA vector.items: ", vector.items())
            # print("CFA weights: ", weights)

            for term_label, term_value in vector.items():
                if term_label in terms:
                    pv_enum_val = term_label.value
                    item_idx = prediction_vector.idx[pv_enum_val]
                    net_outcome += np.sum(term_value.reshape(-1) * weights[item_idx])
                    # # VZ
                    # print("CFA item_idx: ", item_idx)
                    # print("CFA term label: ", term_label)
                    # print("CFA term_value: ", term_value.reshape(-1))
                    # print("CFA weights 2: ", weights[item_idx])
                    # print("CFA add to outcome: ", np.sum(term_value.reshape(-1) * weights[item_idx]))
            predicted_outcome+=net_outcome
        predicted_outcome/=count
        # VZ
        print('predicted outcome: ', predicted_outcome)
        print()
        return predicted_outcome

    @property
    def _dependent_components(self):
        return list(itertools.chain(
            [self.update_weights]
        ))

    class PredictionVector():
        '''Maintain a `vector <PredictionVector.vector>` of terms for a regression model specified by a list of
        `specified_terms <PredictionVector.specified_terms>`.

        Terms are maintained in lists indexed by the `PV` Enum and, in "flattened" form within fields of a 1d
        array in `vector <PredictionVector.vector>` indexed by slices listed in the `idx <PredicitionVector.idx>`
        attribute.

        Arguments
        ---------

        feature_values : 2d nparray
            arrays of features to assign as the `PV.F` term of `terms <PredictionVector.terms>`.

        control_signals : List[ControlSignal]
            list containing the `ControlSignals <ControlSignal>` of an `OptimizationControlMechanism`;
            the `variable <ControlSignal.variable>` of each is assigned as the `PV.C` term of `terms
            <PredictionVector.terms>`.

        specified_terms : List[PV]
            terms to include in `vector <PredictionVector.vector>`; entries must be members of the `PV` Enum.

        Attributes
        ----------

        specified_terms : List[PV]
            terms included as predictors, specified using members of the `PV` Enum.

        terms : List[ndarray]
            current value of ndarray terms, some of which are used to compute other terms. Only entries for terms in
            `specified_terms <specified_terms>` are assigned values; others are assigned `None`.

        num : List[int]
            number of arrays in outer dimension (axis 0) of each ndarray in `terms <PredictionVector.terms>`.
            Only entries for terms in `specified_terms <PredictionVector.specified_terms>` are assigned values;
            others are assigned `None`.

        num_elems : List[int]
            number of elements in flattened array for each ndarray in `terms <PredictionVector.terms>`.
            Only entries for terms in `specified_terms <PredictionVector.specified_terms>` are assigned values;
            others are assigned `None`.

        self.labels : List[str]
            label of each item in `terms <PredictionVector.terms>`. Only entries for terms in  `specified_terms
            <PredictionVector.specified_terms>` are assigned values; others are assigned `None`.

        vector : ndarray
            contains the flattened array for all ndarrays in `terms <PredictionVector.terms>`.  Contains only
            the terms specified in `specified_terms <PredictionVector.specified_terms>`.  Indices for the fields
            corresponding to each term are listed in `idx <PredictionVector.idx>`.

        idx : List[slice]
            indices of `vector <PredictionVector.vector>` for the flattened version of each nd term in
            `terms <PredictionVector.terms>`. Only entries for terms in `specified_terms
            <PredictionVector.specified_terms>` are assigned values; others are assigned `None`.

        '''

        _deepcopy_shared_keys = ['control_signal_functions', '_compute_costs']

        def __init__(self, feature_values, control_signals, specified_terms):

            # Get variable for control_signals specified in constructor
            control_allocation = []
            for c in control_signals:
                if isinstance(c, ControlSignal):
                    try:
                        v = c.variable
                    except:
                        v = c.defaults.variable
                elif isinstance(c, type):
                    if issubclass(c, ControlSignal):
                        v = c.class_defaults.variable
                    else:  # If a class other than ControlSignal was specified, typecheck should have found it
                        assert False, "PROGRAM ERROR: unrecognized specification for {} arg of {}: {}".\
                                                      format(repr(CONTROL_SIGNALS), self.name, c)
                else:
                    state_spec_dict = _parse_state_spec(state_type=ControlSignal, owner=self, state_spec=c)
                    v = state_spec_dict[VARIABLE]
                    v = v or ControlSignal.defaults.variable
                control_allocation.append(v)
            self.control_signal_functions = [c.function for c in control_signals]
            self._compute_costs = [c.compute_costs for c in control_signals]

            def get_intrxn_labels(x):
                return list([s for s in powerset(x) if len(s)>1])

            def error_for_too_few_terms(term):
                spec_type = {'FF':'feature_values', 'CC':'control_signals'}
                raise NeuralNetCFAError("Specification of {} for {} arg of {} "
                                                "requires at least two {} be specified".
                                                format('PV.'+term, repr(PREDICTION_TERMS), self.name, spec_type(term)))

            F = PV.F.value
            C = PV.C.value
            FF = PV.FF.value
            CC = PV.CC.value
            FC = PV.FC.value
            FFC = PV.FFC.value
            FCC = PV.FCC.value
            FFCC = PV.FFCC.value
            COST = PV.COST.value

            # RENAME THIS AS SPECIFIED_TERMS
            self.specified_terms = specified_terms
            self.terms = [None] * len(PV)
            self.idx =  [None] * len(PV)
            self.num =  [None] * len(PV)
            self.num_elems =  [None] * len(PV)
            self.labels = [None] * len(PV)

            # MAIN EFFECT TERMS (unflattened)

            # Feature_values
            self.terms[F] = f = feature_values
            self.num[F] = len(f)  # feature_values are arrays
            self.num_elems[F] = len(f.reshape(-1)) # num of total elements assigned to vector
            self.labels[F] = ['f'+str(i) for i in range(0,len(f))]

            # Placemarker until control_signals are instantiated
            self.terms[C] = c = np.array([[0]] * len(control_allocation))
            self.num[C] = len(c)
            self.num_elems[C] = len(c.reshape(-1))
            self.labels[C] = ['c'+str(i) for i in range(0,len(control_allocation))]

            # Costs
            # Placemarker until control_signals are instantiated
            self.terms[COST] = cst = np.array([[0]] * len(control_allocation))
            self.num[COST] = self.num[C]
            self.num_elems[COST] = len(cst.reshape(-1))
            self.labels[COST] = ['cst'+str(i) for i in range(0,self.num[COST])]

            # INTERACTION TERMS (unflattened)

            # Interactions among feature vectors
            if any(term in specified_terms for term in [PV.FF, PV.FFC, PV.FFCC]):
                if len(f) < 2:
                    self.error_for_too_few_terms('FF')
                self.terms[FF] = ff = np.array(tensor_power(f, levels=range(2,len(f)+1)))
                self.num[FF] = len(ff)
                self.num_elems[FF] = len(ff.reshape(-1))
                self.labels[FF]= get_intrxn_labels(self.labels[F])

            # Interactions among values of control_signals
            if any(term in specified_terms for term in [PV.CC, PV.FCC, PV.FFCC]):
                if len(c) < 2:
                    self.error_for_too_few_terms('CC')
                self.terms[CC] = cc = np.array(tensor_power(c, levels=range(2,len(c)+1)))
                self.num[CC]=len(cc)
                self.num_elems[CC] = len(cc.reshape(-1))
                self.labels[CC] = get_intrxn_labels(self.labels[C])

            # feature-control interactions
            if any(term in specified_terms for term in [PV.FC, PV.FCC, PV.FFCC]):
                self.terms[FC] = fc = np.tensordot(f, c, axes=0)
                self.num[FC] = len(fc.reshape(-1))
                self.num_elems[FC] = len(fc.reshape(-1))
                self.labels[FC] = list(product(self.labels[F], self.labels[C]))

            # feature-feature-control interactions
            if any(term in specified_terms for term in [PV.FFC, PV.FFCC]):
                if len(f) < 2:
                    self.error_for_too_few_terms('FF')
                self.terms[FFC] = ffc = np.tensordot(ff, c, axes=0)
                self.num[FFC] = len(ffc.reshape(-1))
                self.num_elems[FFC] = len(ffc.reshape(-1))
                self.labels[FFC] = list(product(self.labels[FF], self.labels[C]))

            # feature-control-control interactions
            if any(term in specified_terms for term in [PV.FCC, PV.FFCC]):
                if len(c) < 2:
                    self.error_for_too_few_terms('CC')
                self.terms[FCC] = fcc = np.tensordot(f, cc, axes=0)
                self.num[FCC] = len(fcc.reshape(-1))
                self.num_elems[FCC] = len(fcc.reshape(-1))
                self.labels[FCC] = list(product(self.labels[F], self.labels[CC]))

            # feature-feature-control-control interactions
            if PV.FFCC in specified_terms:
                if len(f) < 2:
                    self.error_for_too_few_terms('FF')
                if len(c) < 2:
                    self.error_for_too_few_terms('CC')
                self.terms[FFCC] = ffcc = np.tensordot(ff, cc, axes=0)
                self.num[FFCC] = len(ffcc.reshape(-1))
                self.num_elems[FFCC] = len(ffcc.reshape(-1))
                self.labels[FFCC] = list(product(self.labels[FF], self.labels[CC]))

            # Construct "flattened" vector based on specified terms, and assign indices (as slices)
            i=0
            for t in range(len(PV)):
                if t in [t.value for t in specified_terms]:
                    self.idx[t] = slice(i, i + self.num_elems[t])
                    i += self.num_elems[t]

            self.vector = np.zeros(i)

        def __call__(self, terms:tc.any(PV, list))->tc.any(PV, tuple):
            '''Return subvector(s) for specified term(s)'''
            if not isinstance(terms, list):
                return self.idx[terms.value]
            else:
                return tuple([self.idx[pv_member.value] for pv_member in terms])

        __deepcopy__ = get_deepcopy_with_shared(shared_keys=_deepcopy_shared_keys)

        # FIX: 11/9/19 LOCALLY MANAGE STATEFULNESS OF ControlSignals AND costs
        def update_vector(self, variable, feature_values=None, execution_id=None):
            '''Update vector with flattened versions of values returned from the `compute_terms
            <PredictionVector.compute_terms>` method of the `prediction_vector
            <RegressorCFA.prediction_vector>`.

            Updates `vector <PredictionVector.vector>` with current values of variable and, optionally,
            and feature_values.

            '''
            # print("reg cfa UPDATE VECT var: ", variable) # = LAST CTRL SIG ALLOC
            # # FIX: 11/9/19 LOCALLY MANAGE STATEFULNESS OF ControlSignals AND costs
            # if reference_variable is not None:
            #     self.reference_variable = reference_variable

            if feature_values is not None:
                self.terms[PV.F.value] = np.array(feature_values)
            # FIX: 11/9/19 LOCALLY MANAGE STATEFULNESS OF ControlSignals AND costs
            computed_terms = self.compute_terms(np.array(variable), execution_id=execution_id)

            # Assign flattened versions of specified terms to vector
            for k, v in computed_terms.items():
                if k in self.specified_terms:
                    self.vector[self.idx[k.value]] = v.reshape(-1)

        def compute_terms(self, control_allocation, execution_id=None):
            '''Calculate interaction terms.

            Results are returned in a dict; entries are keyed using names of terms listed in the `PV` Enum.
            Values of entries are nd arrays.
            '''

            # FIX: 11/9/19 LOCALLY MANAGE STATEFULNESS OF ControlSignals AND costs
            # ref_variables = ref_variables or self.reference_variable
            # self.reference_variable = ref_variables

            terms = self.specified_terms
            computed_terms = {}

            # No need to calculate features, so just get values
            computed_terms[PV.F] = f = self.terms[PV.F.value]

            # VZ
            # print("PV.F features: ", self.terms[PV.F.value])
            print("Reg Cfa ctrl alloc: ", control_allocation)

            # Compute value of each control_signal from its variable
            c = [None] * len(control_allocation)
            for i, var in enumerate(control_allocation):
                c[i] = self.control_signal_functions[i](var, execution_id=execution_id)
            computed_terms[PV.C] = c = np.array(c)

            # VZ
            print("Reg Cfa PV.C ctrl sig: ", np.array(c))

            # Compute costs for new control_signal values
            if PV.COST in terms:
                # computed_terms[PV.COST] = -(np.exp(0.25*c-3))
                # computed_terms[PV.COST] = -(np.exp(0.25*c-3) + (np.exp(0.25*np.abs(c-self.control_signal_change)-3)))
                costs = [None] * len(c)
                for i, val in enumerate(c):
                    # MODIFIED 11/9/18 OLD:
                    costs[i] = -(self._compute_costs[i](val, execution_id=execution_id))
                    # # MODIFIED 11/9/18 NEW: [JDC]
                    # costs[i] = -(self._compute_costs[i](val, ref_variables[i]))
                    # MODIFIED 11/9/18 END
                computed_terms[PV.COST] = np.array(costs)

            # VZ
            print("Reg Cfa PV.COST: ", np.array(costs))

            # Compute terms interaction that are used
            if any(term in terms for term in [PV.FF, PV.FFC, PV.FFCC]):
                computed_terms[PV.FF] = ff = np.array(tensor_power(f, range(2, self.num[PV.F.value]+1)))
            if any(term in terms for term in [PV.CC, PV.FCC, PV.FFCC]):
                computed_terms[PV.CC] = cc = np.array(tensor_power(c, range(2, self.num[PV.C.value]+1)))
            if any(term in terms for term in [PV.FC, PV.FCC, PV.FFCC]):
                computed_terms[PV.FC] = np.tensordot(f, c, axes=0)
            if any(term in terms for term in [PV.FFC, PV.FFCC]):
                computed_terms[PV.FFC] = np.tensordot(ff, c, axes=0)
            if any(term in terms for term in [PV.FCC, PV.FFCC]):
                computed_terms[PV.FCC] = np.tensordot(f,cc,axes=0)
            if PV.FFCC in terms:
                computed_terms[PV.FFCC] = np.tensordot(ff,cc,axes=0)

            return computed_terms
