import torch

def planner(rssm, reward_model, iterations=10, H=12, num_candidates=1000, topk=100, action_low = -2, action_high=2, device=None):

    action_size = rssm.action_size

    means = torch.zeros((H, action_size), device=device)
    stds = torch.ones((H, action_size), device=device)

    for it in range(0, iterations):

        with torch.no_grad():  
            actions = means.unsqueeze(0) + stds.unsqueeze(0) * torch.rand((num_candidates, H, action_size), device=device)
            actions = actions.to(device)
            actions = torch.clamp(actions, min=action_low, max=action_high)
            out = rssm.imagine_ahead(actions)

            ss = out['ss'] # shape: (B, H, stoch)
            hs = out['hs'] # shape: (B, H, deter)

            latents = torch.cat((hs,ss), dim=-1) # shape: (B, H, D) with D=stoch+deter
            squeeze_latents = latents.view((num_candidates*H), -1)

            rewards = reward_model(squeeze_latents).view((num_candidates, H)).sum(dim=-1) # shape: (B,)
            topk_indexes = torch.argsort(rewards, descending=True)[:topk]
        
        topk_actions = actions[topk_indexes] # shape: (topk, H, action_size)

        new_means = topk_actions.mean(dim=0)
        new_stds = topk_actions.std(dim=0)

        means = new_means
        stds = new_stds

    mean_action = means[0].clamp(action_low, action_high)

    return mean_action