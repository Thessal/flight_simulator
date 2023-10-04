import airline
import rl
if __name__=="__main__":
    print("Hello World")

cfg = {
    "std_delay":5,
    "late_threshold":15,
    "holding_period":120,
    "timestep":5,
    "capacity":10
    }

# To interact with your custom AEC environment, use the following code:

# import aec_rps

# env = aec_rps.env(render_mode="human")
# env.reset(seed=42)

# for agent in env.agent_iter():
#     observation, reward, termination, truncation, info = env.last()

#     if termination or truncation:
#         action = None
#     else:
#         # this is where you would insert your policy
#         action = env.action_space(agent).sample()

#     env.step(action)
# env.close()