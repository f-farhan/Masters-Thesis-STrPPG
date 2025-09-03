import matplotlib.pyplot as plt
import numpy as np
import matplotlib.widgets as widgets
import os

output_dir="C:/ff/FINAL OUTPUTS/RGB/16/GREEN/Cross Polarization/heatmaps"
snr_map_path= "C:/ff/FINAL OUTPUTS/RGB/16/GREEN/Cross Polarization/npy_files/snr_map_0.npy"


def interactive_snr_viewer(snr_map_path, save_dir=None, title="Interactive SNR Heatmap"):
    """
    Displays an interactive SNR heatmap viewer with sliders for vmin and vmax, and a save button.

    Parameters:
    - snr_map_path: str, path to the .npy file containing the SNR map
    - save_dir: str or None, directory to save the current view if Save button is clicked (optional)
    - title: str, title of the plot window
    """
    if not os.path.isfile(snr_map_path):
        raise FileNotFoundError(f"SNR map file not found: {snr_map_path}")

    snr_map = np.load(snr_map_path)

    fig, ax = plt.subplots()
    vmin_init = np.percentile(snr_map, 5)
    vmax_init = np.percentile(snr_map, 95)
    im = ax.imshow(snr_map, cmap='jet', vmin=vmin_init, vmax=vmax_init)
    cbar = plt.colorbar(im, ax=ax)
    plt.title(title)
    plt.subplots_adjust(bottom=0.3)

    # Sliders
    axcolor = 'lightgoldenrodyellow'
    ax_vmin = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)
    ax_vmax = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)

    s_vmin = widgets.Slider(ax_vmin, 'vmin', -20, 15, valinit=vmin_init)
    s_vmax = widgets.Slider(ax_vmax, 'vmax', -20, 15, valinit=vmax_init)

    def update(val):
        im.set_clim(s_vmin.val, s_vmax.val)
        fig.canvas.draw_idle()

    s_vmin.on_changed(update)
    s_vmax.on_changed(update)

    # Save button
    save_ax = plt.axes([0.75, 0.02, 0.18, 0.06])
    save_button = widgets.Button(save_ax, 'Save View', color='lightgray', hovercolor='0.975')

    def save_view(event):
        if save_dir is None:
            save_path = os.path.join(os.path.dirname(snr_map_path), "interactive_snr_heatmap_saved.png")
        else:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, "interactive_snr_heatmap_saved.png")

        fig.savefig(save_path, dpi=300)
        print(f"💾 Interactive view saved to: {save_path}")

    save_button.on_clicked(save_view)
    plt.show()
interactive_snr_viewer(snr_map_path, save_dir=output_dir)