import torch
from typing import Optional, Tuple


def planner(rssm, reward_model, init_state: dict, action_size: int,
            horizon: int = 12,
            num_candidates: int = 1000,
            num_elite: int = 100,
            iterations: int = 5,
            action_low: Optional[float] = -2.0,
            action_high: Optional[float] = 2.0,
            device: Optional[torch.device] = None,
            epsilon: float = 1e-4) -> torch.Tensor:
    """
    Cross-Entropy Method (CEM) planner in latent space.

    Args:
        rssm: RSSM model instance with 'imagine_ahead(actions, init_state)'
              method that takes actions shaped (B, H, action_size) and returns
              dict containing 'hs' and 'ss' (both (B, H, dim)).
        reward_model: model mapping concatenated latent feats -> scalar reward
                      (expects input shape (B*H, D) and returns (B*H, 1)).
        init_state: dict with keys 'h' and 's' giving initial deterministic and
                    stochastic states (shape (batch, dim) where batch should
                    be 1 normally, but will be ignored and replaced by the
                    candidate batch size inside the planner).
        action_size: dimensionality of action vector.

    Keyword args:
        horizon: planning horizon (H).
        num_candidates: number of candidate action sequences sampled per
                        iteration.
        num_elite: number of top candidates used to update the sampling
                   distribution.
        iterations: number of CEM iterations.
        action_low/action_high: scalar bounds applied per action dimension.
        device: torch device to run planning on. If None, inferred from
                init_state tensors.
        epsilon: small value added to std for numerical stability.

    Returns:
        torch.Tensor: planned first action (action_size,).
    """

    # action size
    action_size = rssm.action_size

    # device and dtype
    if device is None:
        if isinstance(init_state, dict) and 'h' in init_state:
            device = init_state['h'].device
        else:
            device = torch.device('cpu')
    dtype = torch.float32

    # bounds for actions
    if action_low is None:
        action_low = -2.0
    if action_high is None:
        action_high = 2.0

    # init beliefs over action sequences (mean and std for the action sequence normal distrib)
    mean = torch.zeros((horizon, action_size), device=device, dtype=dtype)
    std = torch.ones((horizon, action_size), device=device, dtype=dtype)

    # optimization loop
    for it in range(iterations):
        # sample 'num_candidates' action sequences of length 'horizon'
        # from the current belief: shape (num_candidates, horizon, action_size)
        samples = mean.unsqueeze(0) + std.unsqueeze(0) * torch.randn((num_candidates, horizon, action_size), device=device, dtype=dtype)
        samples = samples.clamp(action_low, action_high) # ensure actions are within bounds

        # evaluate each candidate by imagining ahead in latent space
        with torch.no_grad():
            # rssm.imagine_ahead expects actions shaped (B, H, action_size) (here B=num_candidates)
            imagined = rssm.imagine_ahead(samples, init_state=init_state)

            hs = imagined['hs']  # (B, H, deter)
            ss = imagined['ss']  # (B, H, stoch)

            # concat latent features and compute rewards
            latent = torch.cat([hs, ss], dim=-1)  # (B, H, D) (D = deter + stoch)
            B, H, D = latent.shape
            flat = latent.reshape(B * H, D)

            # compute rewards for all imagined steps
            # reward_model expects input shape (B*H, D) and returns (B*H, 1)
            rewards = reward_model(flat).view(B, H) # (B, H)

            # sum rewards across horizon
            returns = rewards.sum(dim=1)  # (B,)

        # select top k candidates (highest returns)
        topk = min(num_elite, num_candidates)
        _, elite_idxs = torch.topk(returns, topk, largest=True)
        elites = samples[elite_idxs]  # (topk, H, action_size) (normally topk = num_elite << num_candidates = B)
        
        # new mean is the mean of the elites, std is their std
        new_mean = elites.mean(dim=0)
        new_std = elites.std(dim=0) + epsilon

        # update (current) belief (mean and std) based on elites
        mean = new_mean
        std = new_std

    # final action sequence: use mean (best estimate)
    planned_seq = mean.clamp(action_low, action_high)

    # return first action of the planned sequence (actually return the mean)
    return planned_seq[0].detach()
