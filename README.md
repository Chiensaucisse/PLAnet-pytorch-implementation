# PLAnet PyTorch Implementation

This repository is a PyTorch re-implementation of the **Deep Planning Network (PlaNet)** algorithm proposed by Hafner et al. [1]. 

PlaNet learns a latent dynamics model from image observations and uses it for planning in a latent space to solve control tasks. This implementation specifically targets the `Pendulum-v1` environment using the `gymnasium` library.

## 🛠️ Installation

Ensure you have Python 3.8+ installed. You can install the required dependencies using pip:

```bash
pip install torch numpy gymnasium[classic_control] opencv-python matplotlib tqdm tensorboard

```

## 🚀 Usage

### Training

To train the model on `Pendulum-v1`, run the `main.py` script. This will collect random seed episodes, train the RSSM, and iteratively plan using the learned model.

```bash
python main.py --num_train_step 500 --batch_size 32

```

**Common Arguments:**

* `--lr`: Learning rate (default: `1e-3`)
* `--num_train_step`: Total training steps (default: `1000`)
* `--batch_size`: Batch size for training (default: `32`)
* `--stochastic_dim`: Size of the stochastic latent variable (default: `30`)
* `--deter_dim`: Size of the deterministic recurrent state (default: `200`)
* `--C`: Number of update steps per iteration (default: `100`)
* `--L`: Sequence chunk length for training (default: `50`)

### Evaluation

To evaluate a trained model, use `evaluation.py`. This loads the weights and runs the agent in the environment, saving visualization videos.

```bash
python evaluation.py --load_path weights/checkpoint.pth

```

## 📂 Project Structure

* **`main.py`**: The entry point for training. Manages the data collection, training loop, and logging.
* **`model.py`**: Contains the core architecture:
* `ConvEncoder` & `ConvDecoder`: Image processing.
* `RSSM`: The transition model (Prior/Posterior networks, GRU cell).
* `RewardModel`: Predicts rewards from latent states.


* **`planner.py`**: Implements the Cross-Entropy Method  for planning action sequences in the latent space.
* **`memory.py`**: `ReplayBuffer` implementation for storing and sampling episode chunks.
* **`evaluation.py`**: Script to load a checkpoint and evaluate performance/visualize reconstructions.
* **`utils.py`**: Helper functions for image preprocessing, KL divergence calculation, and video saving.

## 📊 Outputs and Visualization

* **Logs:** Training metrics (Loss, Reward) are saved to `runs/planet_pendulum` and can be viewed with Tensorboard:
```bash
tensorboard --logdir runs

```


* **Checkpoints:** Model weights are saved to `weights/checkpoint.pth`.
* **Videos:** During evaluation/training, the model saves `actual.mp4` (ground truth) and `decoded.mp4` (model imagination) to the `videos/` directory to visualize how well the model "imagines" the trajectory.

## 📚 References

[1] Hafner, D., Lillicrap, T., Fischer, I., Villegas, R., Ha, D., Lee, H., & Davidson, J. (2019, May). Learning latent dynamics for planning from pixels. In International conference on machine learning (pp. 2555-2565). PMLR.
