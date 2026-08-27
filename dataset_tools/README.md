# Preparing Your Own Dataset

NeRF<sup>2</sup> currently supports three tasks: RFID spatial-spectrum prediction,
BLE RSSI prediction, and MIMO CSI prediction. This document describes the exact
on-disk formats expected by `dataloader.py`.

## Common requirements

- Store every position as Cartesian coordinates `[x, y, z]` in one shared world
  coordinate system. Gateway/base-station positions and transmitter positions
  must use the same origin, axes, and unit.
- CSV files must include a header row. Data rows that describe the same
  measurement must have the same order across files.
- Set `path.datadir` in the corresponding file under [`configs/`](../configs/)
  to the new dataset directory.
- `train_index.txt` and `test_index.txt` are optional. If either is missing,
  NeRF<sup>2</sup> creates a random 80/20 split when the runner starts. Because
  this split is not seeded, save both files if you need reproducible results.
- Choose `render.near`, `render.far`, and `render.scale_worldsize` consistently
  with the coordinate unit and the physical size of the scene. In the BLE and
  MIMO loaders, positions are divided by `scale_worldsize` before training.

The three formats use different index conventions: RFID spectrum filenames are
**1-based**, whereas BLE and MIMO sample indices are **0-based**.

## RFID spatial-spectrum dataset

### Directory layout

```text
data/RFID/my_scene/
|-- gateway_info.yml
|-- tx_pos.csv
|-- spectrum/
|   |-- 1.png
|   |-- 2.png
|   `-- ...
|-- train_index.txt       # optional; spectrum stems such as 1, 2, ...
`-- test_index.txt        # optional
```

`gateway_info.yml` contains the receiver-array position and orientation. The
orientation is a quaternion in SciPy's `[x, y, z, w]` order:

```yaml
gateway1:
  position: [0.0, 0.0, 1.5]
  orientation: [0.0, 0.0, 0.0, 1.0]
```

`tx_pos.csv` contains one transmitter position per spectrum:

```csv
x,y,z
1.20,0.75,1.00
1.50,0.75,1.00
```

Spectrum filenames must be positive integer stems. In particular, `1.png`
uses the first row of `tx_pos.csv`, `2.png` uses the second row, and so on. Do
not use zero-based filenames for this dataset.

Each spectrum must be a single-channel PNG. Pixel values are interpreted as
normalized spectrum intensity in `[0, 1]` after division by 255. The supplied Bartlett
generator produces a `90 x 360` image: rows correspond to elevation angles
from 1 to 90 degrees, and columns correspond to azimuth angles from 0 to 359
degrees. The current loader constructs the corresponding ray grid from 1 to
360 degrees in azimuth, so helper-generated spectra have a one-degree circular
offset relative to the ray directions. Keep this convention for compatibility
with the current code, or update both grids together for a new dataset.

### Generating spectra from array phases

[`gen_spectrum.py`](gen_spectrum.py) converts one phase vector into a Bartlett
spatial spectrum. Before using it:

1. Replace `ANT_LOC` with the `[x, y, z]` coordinates of your antenna elements,
   in the same order as the measured phase vector.
2. Pass the carrier frequency in hertz to `Bartlett(frequency=...)`.
3. Express every measured phase in radians and provide exactly one value per
   antenna element.
4. Save the generated spectra as grayscale `uint8` PNG files named `1.png`,
   `2.png`, and so forth.

The script includes a runnable 16-antenna example:

```bash
python dataset_tools/gen_spectrum.py
```

For a collection of measurements, adapt the example's final block into a loop
and save each output under the dataset's `spectrum/` directory. Avoid applying
an image colormap: the loader expects a two-dimensional grayscale array.

## BLE RSSI dataset

### Directory layout

```text
data/BLE/my_scene/
|-- gateway_position.yml
|-- tx_pos.csv
|-- gateway_rssi.csv
|-- train_index.txt       # optional; zero-based row indices
`-- test_index.txt        # optional
```

`gateway_position.yml` maps gateway names to their positions. YAML insertion
order defines the gateway order used by the loader:

```yaml
gateway1: [0.0, 0.0, 1.5]
gateway2: [8.0, 0.0, 1.5]
gateway3: [8.0, 6.0, 1.5]
```

`tx_pos.csv` has shape `[N, 3]`, with one transmitter position per measurement:

```csv
x,y,z
1.20,0.75,1.00
1.50,0.75,1.00
```

`gateway_rssi.csv` has shape `[N, G]`, where `G` is the number of gateways.
Its columns must follow the same order as the entries in
`gateway_position.yml`:

```csv
gateway1,gateway2,gateway3
-61.2,-74.8,-100
-59.7,-72.1,-85.4
```

RSSI values are in dBm. Use exactly `-100` for a missing measurement; the
loader removes those transmitter/gateway pairs. Consequently, a real reading
of exactly `-100` dBm cannot be represented without changing `dataloader.py`.
The number of rows in `tx_pos.csv` and `gateway_rssi.csv` must match.

## MIMO CSI dataset

### Directory layout

```text
data/MIMO/my_scene/
|-- base-station.yml
|-- csidata.npy
|-- train_index.txt       # optional; zero-based sample indices
`-- test_index.txt        # optional
```

`base-station.yml` contains an ordered list of base-station antenna positions.
The current loader expects at least two positions:

```yaml
base_station:
  - [0.00, 0.00, 2.50]
  - [0.08, 0.00, 2.50]
  - [0.16, 0.00, 2.50]
```

`csidata.npy` must be a NumPy complex array with shape `[N, G, 52]`, where:

- `N` is the number of measurement samples;
- `G` equals the number of positions listed under `base_station`; and
- the last axis stores 26 uplink subcarriers followed by the corresponding 26
  downlink subcarriers.

For example:

```python
import numpy as np

# uplink and downlink: complex arrays with shape [N, G, 26]
csi = np.concatenate([uplink, downlink], axis=-1).astype(np.complex64)
np.save("data/MIMO/my_scene/csidata.npy", csi)
```

The loader normalizes the complete array by its maximum magnitude, then uses
the uplink CSI as input and the downlink CSI as the label. Real and imaginary
parts are separated internally; do not split them in the saved file.

## Dataset split files

Each split file contains one index per line. RFID indices are spectrum filename
stems, while BLE and MIMO indices are zero-based row/sample numbers:

```text
0
3
5
```

Keep at least two indices in each split file. With only one line, NumPy loads a
scalar but the current dataset classes expect a one-dimensional index array.

To regenerate an automatic split, remove both index files and start the runner.
For a deliberate spatial split, create the files yourself so that nearby test
positions do not accidentally leak into the training set.

## Configure and run

Copy the closest example in [`configs/`](../configs/), then update at least the
data and log paths, scene scale, near/far bounds, and experiment name. For
example:

```yaml
path:
  expname: my-ble-scene
  datadir: data/BLE/my_scene/
  logdir: logs/BLE/
```

Start training with the matching dataset type:

```bash
# RFID spectrum
python nerf2_runner.py --mode train --config configs/rfid-spectrum.yml --dataset_type rfid --gpu 0

# BLE RSSI
python nerf2_runner.py --mode train --config configs/ble-rssi.yml --dataset_type ble --gpu 0

# MIMO CSI
python nerf2_runner.py --mode train --config configs/mimo-csi.yml --dataset_type mimo --gpu 0
```

Before a long run, check that all CSV row counts, YAML entry counts, spectrum
filenames, and NumPy dimensions agree. The current implementation loads the
selected data into memory, and an RFID spectrum contributes one training item
per pixel (`90 x 360 = 32,400` items for the default resolution), so begin with
a small subset when validating a new format.
