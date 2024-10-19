import torch
import numpy as np
from mlagents_envs.environment import UnityEnvironment, BaseEnv, ActionTuple
from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
import random

# All agent indexes are

environmentChannel = EnvironmentParametersChannel()
# engineChannel = EnvironmentParametersChannel() for now I can only have one channel at a time
env = UnityEnvironment(file_name=None, side_channels=[environmentChannel])
# channel.set_float_parameter("parameter_1", 2.0)
env.reset()

print(f"env.behavior_specs: {env.behavior_specs} So first element will be our behaviorName")
behaviorName = list(env.behavior_specs)[0]
specs = env.behavior_specs[behaviorName]
print(f"All specs: {specs}")

nContinuousActions = specs.action_spec.continuous_size
print(f"We have {nContinuousActions} continuous action size")
print(f"We have {specs.action_spec.discrete_size} branches of {specs.action_spec.discrete_branches} continuous action sizes respectively")

# try:
nEpisodes = 5
for i in range(1, nEpisodes + 1):
    done = False
    episodicReward = 0.0
    env.reset()
    decisionSteps, terminalSteps = env.get_steps(behaviorName)
    print(f"agent_id: {decisionSteps.agent_id}, and first element is {decisionSteps.agent_id[0]}")
    agentID = decisionSteps.agent_id[0] # I dont get that part yet
    state = decisionSteps.obs[agentID]
    # print(f"So agent {agentID} is in state {state}")
    while not done:
        # gotta expand it so shape is not () but (1). Maybe not needed for more
        actionContinuous = np.expand_dims(np.random.randint(0, 1, dtype='int32'), 0).reshape(-1, 1)
        action = ActionTuple(continuous=actionContinuous)
        env.set_actions(behaviorName, action)
        env.step()
        decisionSteps, terminalSteps = env.get_steps(behaviorName)
        # print(f"decisionSteps: {decisionSteps[agentID]}")
        nextState = decisionSteps.obs[agentID]
        if agentID in terminalSteps:
            print(f"decisionSteps: {terminalSteps[agentID]}")
            done = True
            nextState = terminalSteps.obs[agentID]
            print(f"Total episodic reward: {episodicReward}")

        stepReward = decisionSteps.reward
        episodicReward += stepReward
        print(f"Got reward: {stepReward}")
        state = nextState
env.close()