# visualizers/lsb_plane.py

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path


class LSBVisualizer:
    """
    Generates visual representations of LSB steganography.

    Why visualization matters:
        When an image contains LSB steganography, the LSB plane shows a
        distinct pattern — the embedded region appears as structured noise
        (horizontal stripes or uniform texture) while the clean region
        shows the natural image structure. This visual signal is often
        unambiguous even when statistical scores are inconclusive.

    Available outputs:
        - LSB plane image (per channel or combined)
        - Entropy heatmap (block-level entropy across the image)
        - Side-by-side comparison (original vs stego LSB planes)
    """

    BLOCK_SIZE = 16  # pixels per block for entropy heatmap

    def _extract_lsb_plane(self, image: np.ndarray) -> np.ndarray:
        """
        Extract the LSB plane from an RGB image.
        Returns a grayscale array where 0 = LSB off, 255 = LSB on.
        """
        lsb = (image & 1).astype(np.uint8) * 255
        # Convert to grayscale by averaging channels
        return lsb.mean(axis=2).astype(np.uint8)

    def _entropy_heatmap(self, image: np.ndarray) -> np.ndarray:
        """
        Compute a block-level LSB entropy heatmap.
        Each cell represents the Shannon entropy of the LSB plane
        within a BLOCK_SIZE x BLOCK_SIZE region.

        Returns a 2D float array of entropy values in [0, 1].
        """
        h, w, _ = image.shape
        lsb = (image & 1).astype(np.float32)

        rows = h // self.BLOCK_SIZE
        cols = w // self.BLOCK_SIZE

        heatmap = np.zeros((rows, cols), dtype=np.float32)

        for r in range(rows):
            for c in range(cols):
                block = lsb[
                    r * self.BLOCK_SIZE: (r + 1) * self.BLOCK_SIZE,
                    c * self.BLOCK_SIZE: (c + 1) * self.BLOCK_SIZE,
                    :,
                ].flatten()

                p1 = np.mean(block)
                p0 = 1.0 - p1

                if p1 == 0.0 or p1 == 1.0:
                    heatmap[r, c] = 0.0
                else:
                    heatmap[r, c] = -(p0 * np.log2(p0) + p1 * np.log2(p1))

        return heatmap

    def save_lsb_plane(
        self, image: np.ndarray, output_path: str, title: str = "LSB Plane"
    ) -> None:
        """
        Save the LSB plane of an image as a grayscale PNG.

        Args:
            image       : RGB image array (H x W x 3, uint8)
            output_path : path to save the output image
            title       : title shown on the plot
        """
        lsb_plane = self._extract_lsb_plane(image)

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(lsb_plane, cmap="gray", vmin=0, vmax=255)
        ax.set_title(title, fontsize=12)
        ax.axis("off")

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[VISUALIZER] LSB plane saved to {output_path}")

    def save_entropy_heatmap(
        self, image: np.ndarray, output_path: str, title: str = "LSB Entropy Heatmap"
    ) -> None:
        """
        Save an entropy heatmap of the image's LSB plane.
        High-entropy regions (bright) suggest embedded content.
        Low-entropy regions (dark) suggest natural image structure.

        Args:
            image       : RGB image array (H x W x 3, uint8)
            output_path : path to save the output image
            title       : title shown on the plot
        """
        heatmap = self._entropy_heatmap(image)

        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(heatmap, cmap="hot", vmin=0.0, vmax=1.0)
        ax.set_title(title, fontsize=12)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="LSB Entropy")

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[VISUALIZER] Entropy heatmap saved to {output_path}")

    def save_comparison(
        self,
        original: np.ndarray,
        stego: np.ndarray,
        output_path: str,
    ) -> None:
        """
        Save a side-by-side comparison of original vs stego LSB planes
        alongside their entropy heatmaps.

        Layout:
            [Original LSB Plane] [Stego LSB Plane]
            [Original Heatmap  ] [Stego Heatmap  ]

        Args:
            original    : clean image array (H x W x 3, uint8)
            stego       : stego image array (H x W x 3, uint8)
            output_path : path to save the output image
        """
        orig_lsb  = self._extract_lsb_plane(original)
        stego_lsb = self._extract_lsb_plane(stego)
        orig_heat = self._entropy_heatmap(original)
        stego_heat= self._entropy_heatmap(stego)

        fig = plt.figure(figsize=(12, 10))
        gs  = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.25)

        # Row 0: LSB planes
        ax0 = fig.add_subplot(gs[0, 0])
        ax0.imshow(orig_lsb, cmap="gray", vmin=0, vmax=255)
        ax0.set_title("Original — LSB Plane", fontsize=11)
        ax0.axis("off")

        ax1 = fig.add_subplot(gs[0, 1])
        ax1.imshow(stego_lsb, cmap="gray", vmin=0, vmax=255)
        ax1.set_title("Stego — LSB Plane", fontsize=11)
        ax1.axis("off")

        # Row 1: Entropy heatmaps
        ax2 = fig.add_subplot(gs[1, 0])
        im2 = ax2.imshow(orig_heat, cmap="hot", vmin=0.0, vmax=1.0)
        ax2.set_title("Original — Entropy Heatmap", fontsize=11)
        ax2.axis("off")
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

        ax3 = fig.add_subplot(gs[1, 1])
        im3 = ax3.imshow(stego_heat, cmap="hot", vmin=0.0, vmax=1.0)
        ax3.set_title("Stego — Entropy Heatmap", fontsize=11)
        ax3.axis("off")
        plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

        fig.suptitle(
            "LSB Steganography Visual Analysis",
            fontsize=13,
            fontweight="bold",
            y=1.01,
        )

        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[VISUALIZER] Comparison saved to {output_path}")