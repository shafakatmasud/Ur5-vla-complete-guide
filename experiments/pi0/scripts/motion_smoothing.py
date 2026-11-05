"""Pi0-only arm smoothing. Gripper channels and hardware guards stay explicit."""
import math


class InvalidPolicyTargets(ValueError):
    pass


def smooth_action_chunk(actions, observed, window=5):
    """Validate the unsmoothed predictions, then average arm channels only."""
    if not actions or len(observed) != 6 or window < 1 or window % 2 != 1:
        raise InvalidPolicyTargets('Invalid Pi0 smoothing input')
    if any(len(a) != 7 or not all(math.isfinite(v) for v in a) for a in actions):
        raise InvalidPolicyTargets('Pi0 predicted non-finite or incorrectly shaped targets')
    if max(abs(a-b) for a,b in zip(actions[0][:6], observed)) > .10:
        raise InvalidPolicyTargets('Unsmoothed Pi0 first target exceeds the existing 0.10-rad startup envelope')
    if any(abs(actions[k][j]-actions[k-1][j]) > .15
           for k in range(1,len(actions)) for j in range(6)):
        raise InvalidPolicyTargets('Unsmoothed Pi0 targets exceed the existing 0.15-rad step envelope')
    radius=window//2
    result=[]
    for k,action in enumerate(actions):
        local=actions[max(0,k-radius):min(len(actions),k+radius+1)]
        result.append([sum(a[j] for a in local)/len(local) for j in range(6)]+[action[6]])
    return result


def smooth_motion_step(command, velocity, filtered_target, target, dt,
                       max_velocity, max_acceleration, filter_time_constant):
    """Continuous target/velocity filtering with the existing acceleration cap.

    A 1 mrad soft deadband rejects small reversals. Proportional arrival slows
    near the goal instead of asking to cover the remaining distance in one tick.
    """
    alpha=1-math.exp(-dt/filter_time_constant)
    beta=1-math.exp(-dt/.10)
    filtered=[];positions=[];velocities=[]
    for q,v,f,t in zip(command,velocity,filtered_target,target):
        f=f+alpha*(t-f)
        error=f-q
        error=math.copysign(max(0.,abs(error)-.001),error)
        desired=max(-max_velocity,min(max_velocity,error/.50))
        requested=v+beta*(desired-v)
        dv=max_acceleration*dt
        new_v=max(v-dv,min(v+dv,requested))
        filtered.append(f);velocities.append(new_v);positions.append(q+new_v*dt)
    return filtered,positions,velocities
