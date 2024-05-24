import torch
import matplotlib.pyplot as plt

def plot_attention_map(attention_map: torch.Tensor, title, x_label="X", y_label="Y", save_path=None, plot_type="default"):
        """ Plots an attention map using matplotlib.pyplot 
                Arguments:
                        attention_map: Tensor - The attention map to plot. Shape: (H, W)
                        title: str - The title of the plot
                        x_label: str (optional) - The x-axis label
                        y_label: str (optional) - The y-axis label
                        save_path: str (optional) - The path to save the plot
                        plot_type: str (optional) - The type of plot to create. Options: 'default', 'num', or any matplotlib colormap name. https://matplotlib.org/stable/gallery/color/colormap_reference.html
                Returns:
                        None
        """

        # Convert attention map to numpy array
        attention_map = attention_map.detach().cpu().numpy()

        # Create figure and axis
        fig, ax = plt.subplots()

        cmap_name = 'viridis'
        match plot_type:
                case 'default':
                        cmap_name = 'viridis'
                case 'num':
                        cmap_name = 'tab20c'
                case _:
                        cmap_name = plot_type 
        # Plot the attention map
        ax.imshow(attention_map, cmap=cmap_name, interpolation='nearest')
        if plot_type == 'num':
                elements = list(set(attention_map.flatten()))
                labels = [f"{x}" for x in elements]
                fig.legend(elements, labels, loc='lower left')

        # Set title and labels
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

        # Save the plot if save_path is provided
        if save_path:
                plt.savefig(save_path)
        
        plt.close(fig)