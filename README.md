# Decompressor-for-Playcanvas-compressed-ply
A lightweight tool to decompress .compressed.ply files exported from PlayCanvas, restoring them to standard .ply format for use in 3D Gaussian Splatting pipelines or visualization tools.

## Get started

Install numpy

```bash
pip install numpy
```

Decompresses a PlayCanvas-exported .compressed.ply file and outputs a standard .ply file suitable for 3D processing or visualization.
```bash
python decompress.py [input.compressed.ply] [output.ply]
```

Converts a PlayCanvas .compressed.ply file into a .splat format optimized for 3D Gaussian Splatting, using multithreading for faster processing.
```bash
python convert_compressed_ply_to_splat.py [input.compressed.ply] [output.splat] [num_threads]
```

## Acknowledgements

Inspired by:
- Thank to [Playcanvas](https://github.com/playcanvas) for the compress `ply` to `.compressed.ply` tool from: https://github.com/playcanvas/splat-transform
- Thank to [Vincent Lecrubier](https://github.com/vincent-lecrubier-skydio) for the convert `.ply` to `.splat` tool: https://github.com/vincent-lecrubier-skydio/react-three-fiber-gaussian-splat/blob/main/convert_ply_to_splat.py

