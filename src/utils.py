from pathlib import Path

import matplotlib.pyplot as plt
import torch
from PIL import Image


# Plot functions credits:
# Ben Moseley https://github.com/benmoseley/harmonic-oscillator-pinn
def plot_result(training_step: int, x: torch.Tensor, y: torch.Tensor, x_data: torch.Tensor, y_data: torch.Tensor, yh: torch.Tensor, xp: None | torch.Tensor = None) -> None:
    """Plot the exact solution, neural network prediction, and training data.

    This function visualizes the exact oscillator solution, the predicted
    solution from a neural network, and the training data points. Optionally,
    physics-informed training locations can also be plotted.

    Args:
        training_step (int): Current training step index (for annotation).
        x (torch.Tensor): Time or input values for the exact solution.
        y (torch.Tensor): Exact solution values.
        x_data (torch.Tensor): Input values of the training data.
        y_data (torch.Tensor): Target values of the training data.
        yh (torch.Tensor): Predicted values from the neural network.
        xp (torch.Tensor | None, optional): Physics-informed loss training locations. Defaults to None.

    Returns:
        None: Displays the plot.

    """
    # Move tensors on cpu as per numpy requirements
    x = x.cpu()
    y = y.cpu()
    x_data = x_data.cpu()
    y_data = y_data.cpu()
    yh = yh.cpu()

    max_y = torch.max(y.max(), yh.max())
    min_y = torch.min(y.min(), yh.min())
    max_x = torch.max(x.max(), x_data.max())
    min_x = torch.min(x.min(), x_data.min())

    plt.figure(figsize=(8,4))
    plt.plot(x,y, color="grey", linewidth=2, alpha=0.8, label="Exact solution")
    plt.plot(x,yh, color="tab:blue", linewidth=4, alpha=0.8, label="Neural network prediction")
    plt.scatter(x_data, y_data, s=60, color="tab:orange", alpha=0.4, label="Training data")
    if xp is not None:
        plt.scatter(xp, -0*torch.ones_like(xp), s=60, color="tab:green", alpha=0.4, label="Physics loss training locations")
    legend_ = plt.legend(loc=(1.01,0.34), frameon=False, fontsize="large")
    plt.setp(legend_.get_texts(), color="k")
    plt.xlim(min_x - 0.05, max_x + 0.05)
    plt.ylim(min_y - 0.1, max_y + 0.1)
    plt.text(0.3 * max_x.item(), max_y.item() - 0.2, f"Training step: {training_step + 1}", fontsize="xx-large", color="k")
    plt.axis("off")


def save_gif_pil(outfile: Path, files: list[Path], fps: int = 5, loop: int = 0) -> None:
    """Create and save a GIF from a sequence of image files.

    Args:
        outfile (Path): Output GIF file path.
        files (list[Path]): List of image file paths to include in the GIF.
        fps (int, optional): Frames per second. Defaults to 5.
        loop (int, optional): Number of times the GIF should loop (0 = infinite). Defaults to 0.

    Returns:
        None

    """
    imgs = [Image.open(file) for file in files]
    imgs[0].save(fp=outfile, format="GIF", append_images=imgs[1:], save_all=True, duration=int(len(files)/fps), loop=loop)


def remove_files(files: list[Path]) -> None:
    """Delete a list of files from the filesystem.

    Args:
        files (list[Path]): List of file paths to remove.

    Returns:
        None

    """
    for file in files:
        Path(file).unlink()
