import numpy as np

def collect_trajectories(env, agent, batch_size, render=False):
    batch = {'states':[], 'actions':[], 'rews_list':[], \
             'dones': [], 'images': []}
    for _ in range(batch_size):
        obs, info = env.reset() 
        if isinstance(obs, tuple):  # gym v0.26+ return (state, info)
            obs = obs[0]
        traj_rewards = []
        image_obs = []
        done = False
        t = 0
        while not done:
            if render:
                img = env.render()  # [H,W,C]
                if isinstance(img, list):
                    img = img[0]
                # image_obs.append(
                #     cv2.resize(img, dsize=(250, 250), interpolation=cv2.INTER_CUBIC)
                # )  # [T,H,W,C]
                image_obs.append(
                    img
                )  # [T,H,W,C]

            action = agent.sample_action(obs)
            # action_env = action.detach().cpu().numpy().reshape(-1)
            step_result = env.step(action)
            if len(step_result) == 5:
                next_obs, reward, terminated, truncated, info = step_result
            else:
                next_obs, reward, terminated, info = step_result
                truncated = False
            # next_obs, reward, terminated, truncated, _ = env.step(action.cpu().numpy())
            done = terminated or truncated

            batch['states'].append(obs)
            batch['actions'].append(action)
            batch['dones'].append(done)
            traj_rewards.append(reward)
            obs = next_obs
            t += 1
        batch['rews_list'].append(traj_rewards)
        if render:
            batch['images'].append(np.array(image_obs, dtype=np.uint8))
            
    batch['rewards'] = np.concatenate(batch['rews_list'])  # [N*T]
    batch['states'] = np.stack(batch['states'])  # [N*T, state_dim]
    batch['actions'] = np.stack(batch['actions'])  # [N*T, action_dim]
    batch['next_state'] = np.array(next_obs)  # [state_dim], last state

    if render:
        max_T = max(frames.shape[0] for frames in batch['images'])
        padded = []
        for frames in batch['images']:
            arr = np.array(frames)
            T = arr.shape[0]
            if T < max_T:
                pad = np.repeat(arr[-1][None], max_T - T, axis=0)
                arr = np.concatenate([arr, pad], axis=0)
            padded.append(arr)
        batch['images'] = np.stack(padded)  # [N,T,H,W,C]
        # print(batch['images'].shape)
    return batch