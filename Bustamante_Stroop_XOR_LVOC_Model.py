# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# ********************************************* Bustamante_Stroop_XOR_LVOC_Model ***************************************

'''
Implements a model of the `Stroop XOR task
<https://scholar.google.com/scholar?hl=en&as_sdt=0%2C31&q=laura+bustamante+cohen+musslick&btnG=>`_
using a version of the `Learned Value of Control Model
<https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1006043&rev=2>`_
'''

import psyneulink as pnl
import numpy as np
from build_input import xor_dict
from pandas import DataFrame
import pandas as pd 
import csv


np.random.seed(0)

def get_log_dict(mechanism):

  log_dict = mechanism.log.nparray_dictionary()
  print("log dict: ", log_dict)
  print(mechanism.log.csv())

  mechanism_dict = log_dict['Stroop XOR Model']
  
  df = DataFrame(index=[0], columns=['Pass', 'Run', 'Time_step', 'Trial', 
    'allocation_policy', 'value', 'variable'])

  # arrays = [np.array([x[0] for x in mechanism_dict[key]]) for key in ('Run', 'Trial', 'Pass', 'Time_step')]
  # # arrays.extend([np.squeeze(mechanism_dict['allocation_policy'][:, :, i]) for i in range(layer_size)])
  # arrays.extend([np.squeeze(mechanism_dict['value'][:, :, i]) for i in range(layer_size)])
  # # arrays.extend([np.squeeze(mechanism_dict['variable'][:, :, i]) for i in range(layer_size)])
  # table = np.stack(arrays, axis=1)

  # if first:
  #   df = pandas.DataFrame(table, columns=['Run', 'Trial', 'Pass', 'Time_step'] +
  #                                                    [f'{log_layer.name}_{i}' for i in range(layer_size)])
  #   first = False

  # else:
  #   df = pandas.DataFrame(table[:, -1 * layer_size:], columns=[f'{log_layer.name}_{i}' 
  #     for i in range(layer_size)])

  for key in ['Pass', 'Run', 'Time_step', 'Trial']:
    # print("mechdict[key]: ", key)
    df.loc[0][key] = mechanism_dict[key]

  for key in ['allocation_policy', 'value', 'variable']:
    # value_array = mechanism_dict[key]

    # for item in value_array:
    new_value = np.squeeze(mechanism_dict[key])
    # print("new value: ", new_value)      

    df.loc[0][key] = new_value
    # print("new value: ", new_value)
    # print("key: ", key)

    # print(df)

  return df


def w_fct(stim, color_control):
    '''function for word_task, to modulate strength of word reading based on 1-strength of color_naming ControlSignal'''
    return stim * (1-color_control)
w_fct_UDF = pnl.UserDefinedFunction(custom_function=w_fct, color_control=1)


def objective_function(v):
    '''function used for ObjectiveMechanism of lvoc
     v[0] = output of DDM: [probability of color naming, probability of word reading]
     v[1] = reward:        [color naming rewarded, word reading rewarded]
     '''
    return np.sum(v[0]*v[1])


color_stim = pnl.TransferMechanism(name='Color Stimulus', size=8)
word_stim = pnl.TransferMechanism(name='Word Stimulus', size=8)

color_task = pnl.TransferMechanism(name='Color Task')
word_task = pnl.ProcessingMechanism(name='Word Task', function=w_fct_UDF)

reward = pnl.TransferMechanism(name='Reward', size=2)

task_decision = pnl.DDM(name='Task Decision',
            # function=pnl.NavarroAndFuss,
            output_states=[pnl.DDM_OUTPUT.PROBABILITY_UPPER_THRESHOLD,
                           pnl.DDM_OUTPUT.PROBABILITY_LOWER_THRESHOLD])

task_decision.set_log_conditions('func_drift_rate')
task_decision.set_log_conditions('mod_drift_rate')
task_decision.set_log_conditions('PROBABILITY_LOWER_THRESHOLD')
task_decision.set_log_conditions('PROBABILITY_UPPER_THRESHOLD')
print("task decision loggables: ", task_decision.loggable_items)

lvoc = pnl.LVOCControlMechanism(name='LVOC ControlMechanism',
                                feature_predictors={pnl.SHADOW_EXTERNAL_INPUTS:[color_stim, word_stim]},
                                objective_mechanism=pnl.ObjectiveMechanism(name='LVOC ObjectiveMechanism',
                                                                           monitored_output_states=[task_decision,
                                                                                                    reward],
                                                                           function=objective_function),
                                prediction_terms=[pnl.PV.FC, pnl.PV.COST],
                                terminal_objective_mechanism=True,

                                # learning_function=pnl.BayesGLM(mu_0=0, sigma_0=0.1),
                                learning_function=pnl.BayesGLM(mu_0=-0.17, sigma_0=0.11),

                                # function=pnl.GradientOptimization(
                                #         convergence_criterion=pnl.VALUE,
                                #         convergence_threshold=0.001,
                                #         step_size=1,
                                #         annealing_function= lambda x,y : x / np.sqrt(y),
                                #         # direction=pnl.ASCENT
                                # ),

                                function=pnl.GridSearch,

                                # function=pnl.OptimizationFunction,

                                # control_signals={'COLOR CONTROL':[(pnl.SLOPE, color_task),
                                #                                    ('color_control', word_task)]}
                                # control_signals={pnl.NAME:'COLOR CONTROL',
                                #                  pnl.PROJECTIONS:[(pnl.SLOPE, color_task),
                                #                                   ('color_control', word_task)],
                                #                  pnl.COST_OPTIONS:[pnl.ControlSignalCosts.INTENSITY,
                                #                                    pnl.ControlSignalCosts.ADJUSTMENT],
                                #                  pnl.INTENSITY_COST_FUNCTION:pnl.Exponential(rate=0.25, bias=-3),
                                #                  pnl.ADJUSTMENT_COST_FUNCTION:pnl.Exponential(rate=0.25,bias=-3)}
                                control_signals=pnl.ControlSignal(projections=[(pnl.SLOPE, color_task),
                                                                               ('color_control', word_task)],
                                                                  # function=pnl.ReLU,
                                                                  function=pnl.Logistic,
                                                                  cost_options=[pnl.ControlSignalCosts.INTENSITY,
                                                                                pnl.ControlSignalCosts.ADJUSTMENT],
                                                                  intensity_cost_function=pnl.Exponential(rate=0.25,
                                                                                                          bias=-3),
                                                                  adjustment_cost_function=pnl.Exponential(rate=0.25,
                                                                                                           bias=-3),
                                                                  allocation_samples=[i/2 for i in list(range(0,50,1))]
                                                                  )
                                )

lvoc.set_log_conditions('value')
lvoc.set_log_conditions('variable')
# print("loggable: ", lvoc.loggable_items)

lvoc.reportOutputPref=True
c = pnl.Composition(name='Stroop XOR Model')
c.add_c_node(color_stim)
c.add_c_node(word_stim)
c.add_c_node(color_task, required_roles=pnl.CNodeRole.ORIGIN)
c.add_c_node(word_task, required_roles=pnl.CNodeRole.ORIGIN)
c.add_c_node(reward)
c.add_c_node(task_decision)
c.add_projection(sender=color_task, receiver=task_decision)
c.add_projection(sender=word_task, receiver=task_decision)
c.add_c_node(lvoc)

# c.show_graph()

# 200 trials * 30 subjs = 6000 size
df = DataFrame(index=np.arange(6000), columns=['Subject', 'Pass', 'Run', 'Time_step', 'Trial', 
    'allocation_policy', 'value', 'variable'])
old_df = DataFrame()


def run():
  c.run(inputs=input_dict) #num_trials = 

# first = True
# myfile = open('xxx_mod_drift.csv', 'w')
myfile = open('xxx_not_real.csv', 'w')
# for i in range(len(xor_dict)):
for i in range(2):
  input_dict = {color_stim: xor_dict[i][0],
              word_stim: xor_dict[i][1],
              color_task: xor_dict[i][2],
              word_task: xor_dict[i][3],
              reward:    xor_dict[i][4]}
  # def run():
  #   c.run(inputs=input_dict) #num_trials = 

  import timeit
  duration = timeit.timeit(run, number=1) #number=2

  # this_df = get_log_dict(lvoc)

  # task_dict = task_decision.log.nparray_dictionary()
  # task_dict2 = task_dict['Stroop XOR Model']

  print(lvoc.log.csv())
  # print(task_decision.log.csv())

  print('\n')
  print('Subject: ', i+1)
  print('--------------------')
  print('ControlSignal variables: ', [sig.parameters.variable.get(c) for sig in lvoc.control_signals])
  print('ControlSignal values: ', [sig.parameters.value.get(c) for sig in lvoc.control_signals])
  print('features: ', lvoc.parameters.feature_values.get(c))
  print('lvoc: ', lvoc.compute_EVC([sig.parameters.variable.get(c) for sig in lvoc.control_signals], execution_id=c))
  print('time: ', duration)
  print('--------------------\n')

csv_dict = lvoc.log.csv()
task_dict = task_decision.log.csv()
myfile.write(task_dict) # write final task dict to csv
myfile.close()
# old_df.to_csv("Log_Test3.csv", sep='\t')


