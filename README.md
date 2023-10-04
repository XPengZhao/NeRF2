# NeRF $^\textsf{2}$ : Neural Radio-Frequency Radiance Fields

Thank you for your interest in our work. We're excited to announce that we will gradually release the code and dataset for this project.



## Datasets





## Running

### Training the model

```bash
python nerf2_runner.py --mode train --config configs/spectrum.yml --gpu 0
```



### Inference the model

```bash
python nerf2_runner.py --mode test --config configs/spectrum.yml --gpu 0
```



## To-Do List

- [ ] NeRF2 for FDD MIMO channel prediction
- [ ] Release more datasets
- [ ] Instruction of preparing own datasets
- [ ] Pytorch Distributed training to speed up the code




Please stay tuned for updates and feel free to reach out if you have any questions or need further information.



## Acknowledgment

Some code snippets are borrowed from [nerf-pytorch](https://github.com/yenchenlin/nerf-pytorch) and [NeuS](https://github.com/Totoro97/NeuS).

